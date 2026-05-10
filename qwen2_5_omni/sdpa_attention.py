from typing import Optional
import torch
import math
import time
from transformers.utils import is_torch_xpu_available, logging
from transformers.utils.import_utils import is_torch_greater_or_equal
from typing import Optional

logger = logging.get_logger(__name__)


_is_torch_greater_or_equal_than_2_5 = is_torch_greater_or_equal("2.5", accept_dev=True)
_is_torch_greater_or_equal_than_2_8 = is_torch_greater_or_equal("2.8", accept_dev=True)
_is_torch_xpu_available = is_torch_xpu_available()


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def use_gqa_in_sdpa(attention_mask: Optional[torch.Tensor], key: torch.Tensor) -> bool:
    # GQA can only be used under the following conditions
    # 1.cuda
    #   - torch version >= 2.5
    #   - attention_mask is None (otherwise it will fall back to the math kernel)
    #   - key is not a torch.fx.Proxy (otherwise it will fail with a tracing error)
    # 2.xpu
    #   - torch version >= 2.8
    #   - key is not a torch.fx.Proxy (otherwise it will fail with a tracing error)
    if _is_torch_xpu_available:
        return _is_torch_greater_or_equal_than_2_8 and not isinstance(key, torch.fx.Proxy)
    return _is_torch_greater_or_equal_than_2_5 and attention_mask is None and not isinstance(key, torch.fx.Proxy)


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    bias = None,
    bias_matrix = None,
    gated_fc=False,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`sdpa` attention does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )
    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups"):
        if not use_gqa_in_sdpa(attention_mask, key):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        else:
            sdpa_kwargs = {"enable_gqa": True}

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        # The last condition is for encoder (decoder) models which specify this by passing their own `is_causal` flag
        # This is mainly due to those models having mixed implementations for encoder, decoder, and encoder-decoder attns
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    # attn_output_official = torch.nn.functional.scaled_dot_product_attention(
    #     query,
    #     key,
    #     value,
    #     attn_mask=attention_mask,
    #     dropout_p=dropout,
    #     scale=scaling,
    #     is_causal=is_causal,
    #     **sdpa_kwargs,
    # )

    attn_output = bias_scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
        bias=bias,
        bias_matrix=bias_matrix,
        gated_fc=gated_fc,
        **sdpa_kwargs,
    )
    #########################################
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def bias_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: Optional[torch.Tensor] = None,                  # 新增 bias 参数
    bias_matrix = None,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    training: bool = True,
    gated_fc=False
) -> torch.Tensor:
    """
    替代 torch.nn.functional.scaled_dot_product_attention 的版本，支持 post-softmax bias 调整。
    """
    # Step 0: GQA 处理
    if enable_gqa and key.size(1) != query.size(1):
        factor = query.size(1) // key.size(1)
        if key.size(1) * factor != query.size(1):
            raise ValueError(
                f"Incompatible GQA heads: query heads {query.size(1)} vs key heads {key.size(1)}"
            )
        key = key.repeat_interleave(factor, dim=1)
        value = value.repeat_interleave(factor, dim=1)

    # Step 1: scale factor
    head_dim = query.size(-1)
    scale_factor = (1.0 / math.sqrt(head_dim)) if (scale is None) else scale

    # 为数值稳定性，将主要计算转为 float32
    q = query.float()
    k = key.float()
    v = value.float()

    # Step 2: 计算注意力得分 QK^T * scale
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor  # shape: (B, H, q_len, k_len)
    
    if bias_matrix is not None:
        if bias_matrix.dim() == 3:
            bias_matrix = bias_matrix.unsqueeze(1)

        # 匹配 QK logits 的 scale（最关键增强）
        # scale_bias = bias_matrix
        # attn_scores = attn_scores + scale_bias
        attn_scores = attn_scores * (1.0 + bias_matrix)
        
        # q = attn_scores.size(-2) - 1
        # orig = attn_scores[0, 0, q].detach().cpu()
        # biasv = scale_bias[0, 0, q if scale_bias.size(-2) > 1 else 0].detach().cpu()
        # print("\n===== DEBUG ATTENTION + BIAS DIFF =====")
        # print(f"Scale factor: {scale_factor:.4f}")
        # print(f"orig attn_scores: mean={orig.mean():.4f}, min={orig.min():.4f}, max={orig.max():.4f}")
        # print(f"scale_bias:      mean={biasv.mean():.4f}, min={biasv.min():.4f}, max={biasv.max():.4f}")
        # print(f"orig[:20]  -> {orig[:20].tolist()}")
        # print(f"bias[:20]  -> {biasv[:20].tolist()}")
        # print(f"orig+bias[:20] -> {(orig + biasv)[:20].tolist()}")
        # print("========================================\n")

    # Step 3: mask & causal
    if is_causal:
        q_len, k_len = attn_scores.size(-2), attn_scores.size(-1)
        causal_mask = torch.ones((q_len, k_len), dtype=torch.bool, device=attn_scores.device).tril()
        attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))
        else:
            attn_scores = attn_scores + attn_mask

    # Step 4: 手动数值稳定 softmax
    max_scores = attn_scores.max(dim=-1, keepdim=True).values
    attn_scores = attn_scores - max_scores
    exp_scores = torch.exp(attn_scores)
    exp_scores = torch.where(torch.isnan(exp_scores), torch.zeros_like(exp_scores), exp_scores)
    sum_exp = exp_scores.sum(dim=-1, keepdim=True)
    attn_probs = exp_scores / (sum_exp + 1e-9)

    # Step 5: 加入 bias 调整（post‐softmax）
    if bias is not None:
        # bias 可为 shape (batch, num_heads, q_len, k_len) 或 (batch, q_len, k_len)
        if bias.dim() == 3:
            # (batch, q_len, k_len) → (batch, 1, q_len, k_len) 再 broadcast
            bias = bias.unsqueeze(1)
        # 防止 bias 值过小或过大
        # bias = torch.clamp(bias, min=0.0, max=10.0)
        # 将 attn_mask 同步屏蔽位置的 bias
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            # attn_mask: (batch, 1, q_len, k_len)
            bias = bias.masked_fill(~attn_mask, 0.0)
        elif attn_mask is not None:
            bias = bias.masked_fill(attn_mask == float('-inf'), 0.0)
        # 检查无效值
        if torch.isnan(bias).any() or torch.isinf(bias).any():
            raise ValueError("Bias contains NaN or Inf!")
        # 乘法调整 + 再归一化
        attn_probs = attn_probs * bias
        # plot_attention_bias_only(
        #     bias,
        #     save_dir="/projects_vol/gp_aseschng/wang.tianrui/codes/temp"
        # )
        attn_probs = attn_probs / (attn_probs.sum(dim=-1, keepdim=True) + 1e-9)

    # Step 6: dropout
    if dropout_p > 0.0 and training:
        attn_probs = torch.nn.functional.dropout(attn_probs, p=dropout_p)

    # Step 7: 输出
    attn_output = torch.matmul(attn_probs, v)  # shape: (B, H, q_len, head_dim)
    # cast 回原 dtype
    attn_output = attn_output.to(query.dtype)
    return attn_output


import os
# Debug-only plotting helpers below. Keep viz deps optional.
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    _HAS_VIZ = True
except ImportError:
    matplotlib = plt = sns = None
    _HAS_VIZ = False

def plot_attention_bias_only(
    bias,
    save_dir="./bias_only/",
    dpi=300,
    cmap="coolwarm"
):
    """
    只绘制 attention bias，不做任何 token 解码、区域标注等。
    bias: Tensor [B, 1, L, L]
    """
    if not _HAS_VIZ:
        raise ImportError("plot_attention_bias_only() requires matplotlib + seaborn.")
    os.makedirs(save_dir, exist_ok=True)

    B = bias.shape[0]

    for b in range(B):
        cur_bias = bias[b, 0].cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cur_bias, cmap=cmap, cbar=True, ax=ax)

        plt.title(f"Attention Bias — Sample {b}")
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"attention_bias.png")
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")
