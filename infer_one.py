"""Single-sample end-to-end inference for CEAeval.

Pipeline:

    novel_context + target_line
        └─► [Qwen3-8B]  predicts ideal {emotion, rhythm, intonation, sound_effects}
                    │
                    ▼
    ideal + audio ─► [Qwen2.5-Omni-Thinker + gated / dynamic attn bias]
                    │
                    ▼
            <think>…</think><score>X.X</score>
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from data_processor import (
    Qwen2_5OmniProcessor,
    build_scorer_inputs,
    build_scorer_messages,
    make_bias_config,
    normalise_novel_context,
)
from hf_utils import CEAEVAL_HF_REPO, resolve_model_source
from qwen3_ideal_labeler import (
    DEFAULT_MAX_CTS,
    DEFAULT_QWEN3_PATH,
    load_qwen3,
    predict_ideal,
    predict_ideal_voted,
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_scorer(args):
    """Load the Qwen2.5-Omni-Thinker scorer."""
    from qwen2_5_omni.modeling_qwen2_5_omni import (
        DynamicAttenQwen2_5omnithinker,
        GatedAttenQwen2_5omnithinker,
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if args.gated_fc:
        ModelCls = GatedAttenQwen2_5omnithinker
    elif args.dynamic_fac:
        ModelCls = DynamicAttenQwen2_5omnithinker
    else:
        ModelCls = Qwen2_5OmniThinkerForConditionalGeneration

    src = resolve_model_source(args.model_name_or_path)
    print(f"[scorer] Loading {ModelCls.__name__} from {src}")
    model = ModelCls.from_pretrained(
        src,
        torch_dtype=dtype,
        device_map="auto",
    )

    processor = Qwen2_5OmniProcessor.from_pretrained(src)

    # Align embedding with tokenizer size.  Released checkpoints already have
    # resized embeddings, so this is normally a no-op.
    emb_size = model.get_input_embeddings().weight.shape[0]
    vocab_size = len(processor.tokenizer)
    if emb_size != vocab_size:
        print(f"[scorer] Resizing token embeddings: {emb_size} -> {vocab_size}")
        model.resize_token_embeddings(vocab_size)

    for tok in ["<|AUDIO|>", "<think>", "</think>", "<score>", "</score>",
                "<focus_audio>", "</focus_audio>"]:
        if tok not in processor.tokenizer.get_vocab():
            print(f"[scorer][WARN] special token missing: {tok}")

    model.eval()
    return model, processor


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@torch.inference_mode()
def generate_once(model, processor, inputs, bias_config, gen_kwargs):
    model.bias_config = bias_config

    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
              for k, v in inputs.items()}

    out = model.generate(**inputs, **gen_kwargs)
    seq = out.sequences if hasattr(out, "sequences") else out
    gen = seq[:, inputs["input_ids"].size(1):]

    stripped = processor.batch_decode(
        gen, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    with_special = processor.batch_decode(
        gen, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0].strip()
    return stripped, with_special


# ---------------------------------------------------------------------------
# Ideal-label resolution (Qwen3-8B or user-supplied)
# ---------------------------------------------------------------------------
def resolve_ideal(args) -> dict:
    """Return a ``{emotion, rhythm, intonation, sound_effects}`` dict.

    If ``--skip_qwen3`` is set, pull the values directly from CLI flags.
    Otherwise invoke the expressive planner (Qwen3-8B) on
    (novel_context, target_line).  By default this uses the multi-context
    voting strategy from Appendix C of the paper (enumerate CTS=1..15 and
    joint-vote across 4-tuples); pass ``--no_vote`` for the legacy
    single-query behaviour.
    """
    if args.skip_qwen3:
        if not all([args.emotion, args.rhythm, args.intonation]):
            raise SystemExit(
                "--skip_qwen3 requires --emotion, --rhythm, --intonation "
                "(and optionally --sound_effects)."
            )
        return {
            "emotion":       args.emotion,
            "rhythm":        args.rhythm,
            "intonation":    args.intonation,
            "sound_effects": args.sound_effects or ("正常说话" if args.lang == "zh" else "normal speaking"),
        }

    if not args.target_line:
        raise SystemExit("--target_line is required when --skip_qwen3 is not set.")

    # For voting we want the raw line structure; `args.novel_context` is a
    # string, so split on newlines and let the voter peel off preceding-line
    # prefixes of size 1..max_cts.
    if args.vote:
        lines = [ln.strip() for ln in args.novel_context.splitlines() if ln.strip()]
        if not lines:
            print("[qwen3][WARN] novel_context is empty — voting will have a single CTS=0 query.")

        labeler = load_qwen3(args.qwen3_model_path, dtype=args.qwen3_dtype)
        ideal, per_cts, chosen_cts, raw = predict_ideal_voted(
            labeler,
            lines,
            args.target_line,
            lang=args.lang,
            max_cts=args.max_cts,
            seed=args.qwen3_seed,
            temperature=args.qwen3_temperature,
            top_p=args.qwen3_top_p,
            max_attempts=args.qwen3_max_attempts,
            verbose=True,
        )
        if ideal is None:
            raise SystemExit(
                f"[qwen3] Voted planner failed to produce a valid plan "
                f"across CTS=1..{args.max_cts}.  Last raw output:\n{raw}"
            )
        total_votes = sum(1 for v in per_cts.values() if v is not None)
        print(f"[qwen3] voted ideal ({args.lang}) = "
              f"{json.dumps(ideal, ensure_ascii=False)}  "
              f"[chosen_cts={chosen_cts}, votes={total_votes}/{len(per_cts)}]")
    else:
        ctx = normalise_novel_context(args.novel_context)
        if not ctx:
            print("[qwen3][WARN] novel_context is empty — predictions will be less reliable.")

        labeler = load_qwen3(args.qwen3_model_path, dtype=args.qwen3_dtype)
        ideal, raw = predict_ideal(
            labeler,
            ctx,
            args.target_line,
            lang=args.lang,
            seed=args.qwen3_seed,
            temperature=args.qwen3_temperature,
            top_p=args.qwen3_top_p,
            max_attempts=args.qwen3_max_attempts,
            verbose=True,
        )
        if ideal is None:
            raise SystemExit(
                f"[qwen3] Failed to parse a valid JSON ideal-label block after "
                f"{args.qwen3_max_attempts} attempts.  Last raw output:\n{raw}"
            )
        print(f"[qwen3] ideal ({args.lang}) = {json.dumps(ideal, ensure_ascii=False)}")

    # Free Qwen3 from GPU so we can load the scorer next.
    del labeler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ideal


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="CEAeval — single-sample end-to-end inference")

    # scorer
    p.add_argument("--model_name_or_path", default=None,
                   help=f"Local checkpoint directory, or a HF repo id.  "
                        f"Defaults to ${{CEAEVAL_HF_REPO}} "
                        f"({CEAEVAL_HF_REPO}); will be downloaded on first use.")

    # inputs
    p.add_argument("--audio_path",    required=True)
    p.add_argument("--novel_context", default="",
                   help="Long-form novel context around the target line.  "
                        "Pass a string (use bash $(cat file) or quoted text). "
                        "Required unless --skip_qwen3 is used.")
    p.add_argument("--target_line",   default="",
                   help="The line that is spoken in the audio, e.g. "
                        "\"张三说：你好。\".  Required unless --skip_qwen3.")
    p.add_argument("--lang", choices=["zh", "en"], default="en",
                   help="Prompt language for the scorer AND for Qwen3 output.")

    # Qwen3-8B (stage-1 ideal labeler)
    p.add_argument("--skip_qwen3", action="store_true",
                   help="Skip Qwen3 and use the --emotion/--rhythm/... flags directly.")
    p.add_argument("--qwen3_model_path", default=None,
                   help=f"Local path or HF repo id.  Defaults to ${{QWEN3_MODEL_PATH}} "
                        f"or {DEFAULT_QWEN3_PATH}.")
    p.add_argument("--qwen3_dtype", choices=["bf16", "fp16", "fp32"], default="fp16")
    p.add_argument("--qwen3_seed",        type=int,   default=42)
    p.add_argument("--qwen3_temperature", type=float, default=0.7)
    p.add_argument("--qwen3_top_p",       type=float, default=0.9)
    p.add_argument("--qwen3_max_attempts", type=int,  default=10)

    # expressive-planner voting (Appendix C of the paper)
    p.add_argument("--vote", dest="vote", action="store_true",
                   help="Multi-context voting: query the planner at "
                        "CTS=1..--max_cts and joint-vote across the "
                        "4-tuple outputs (paper default).")
    p.add_argument("--no_vote", dest="vote", action="store_false",
                   help="Disable voting; issue a single planner query.")
    p.set_defaults(vote=True)
    p.add_argument("--max_cts", type=int, default=DEFAULT_MAX_CTS,
                   help="Maximum cumulative context size to enumerate "
                        "during voting.  Paper uses 15.")

    # direct ideal override (only used with --skip_qwen3)
    p.add_argument("--emotion",       default=None)
    p.add_argument("--rhythm",        default=None)
    p.add_argument("--intonation",    default=None)
    p.add_argument("--sound_effects", default=None)

    # generation
    p.add_argument("--max_new_tokens",     type=int,   default=1024)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--sys_prompt",
                   default="你是一个用于评估语音表达质量的模型。"
                           "|You are a model designed to evaluate the quality of speech expressiveness.")

    # attention-bias switches (leave the defaults for the released model)
    p.add_argument("--gated_fc",         dest="gated_fc",       action="store_true",  default=True)
    p.add_argument("--no-gated_fc",      dest="gated_fc",       action="store_false")
    p.add_argument("--dynamic_fac",      dest="dynamic_fac",    action="store_true",  default=True)
    p.add_argument("--no-dynamic_fac",   dest="dynamic_fac",    action="store_false")
    p.add_argument("--inner_strength",   dest="inner_strength", action="store_true",  default=True)
    p.add_argument("--no-inner_strength",dest="inner_strength", action="store_false")
    p.add_argument("--focus_strength",    type=float, default=-1.0)
    p.add_argument("--suppress_strength", type=float, default=-2.0)
    p.add_argument("--instruct_strength", type=float, default=-3.0)

    p.add_argument("--output_json", default=None,
                   help="Optional path to dump the JSON result.")
    return p


def main():
    args = build_argparser().parse_args()
    print(f"[args]\n{json.dumps(vars(args), indent=2, ensure_ascii=False)}")

    if not os.path.isfile(args.audio_path):
        raise FileNotFoundError(f"audio not found: {args.audio_path}")

    # 1. Resolve ideal labels (Qwen3 or CLI)
    ideal = resolve_ideal(args)

    # 2. Scorer
    model, processor = load_scorer(args)
    bias_config = make_bias_config(
        processor,
        gated_fc=args.gated_fc,
        dynamic_fac=args.dynamic_fac,
        inner_strength=args.inner_strength,
        focus_strength=args.focus_strength,
        suppress_strength=args.suppress_strength,
        instruct_strength=args.instruct_strength,
    )

    msgs = build_scorer_messages(
        audio_path=args.audio_path,
        ideal=ideal,
        lang=args.lang,
        sys_prompt=args.sys_prompt,
    )
    inputs = build_scorer_inputs(processor, msgs)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        num_beams=1,
        repetition_penalty=args.repetition_penalty,
        return_dict_in_generate=True,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    text, text_sp = generate_once(model, processor, inputs, bias_config, gen_kwargs)

    print("\n===== PREDICTION =====")
    print(f"[stripped]    {text}")
    print(f"[with_spt]    {text_sp}")

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({
                "audio_path":   args.audio_path,
                "target_line":  args.target_line,
                "lang":         args.lang,
                "ideal":        ideal,
                "prediction": {
                    "text":              text,
                    "text_with_special": text_sp,
                },
            }, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] Wrote {args.output_json}")


if __name__ == "__main__":
    main()
