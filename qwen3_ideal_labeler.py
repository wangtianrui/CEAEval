"""Stage-1 of the CEAEval inference pipeline: the *Expressive Planner*.

Given the ``context`` (the narrative / dialogue lines preceding /
surrounding the target speech) together with the ``target_line`` (the
sentence that is actually going to be spoken in the audio), this module
prompts a local Qwen3 chat model to predict the *ideal expressive plan*
the audio *should* exhibit:

    {
      "emotion":       "悲伤 / 愤怒 / 温柔 / 惊讶 / 平静 / ...",
      "rhythm":        "轻快型 / 凝重型 / 低沉型 / 高亢型 / 舒缓型 / 紧张型",
      "intonation":    "平直调 / 上扬调 / 弯曲调 / 下降调",
      "sound_effects": "正常说话 / 电话转录 / 远场 / 断断续续 / 内心独白 ..."
    }

That plan is then handed to the Qwen2.5-Omni-Thinker judge
(``infer_one.py`` / ``infer.py``) as the ``ideal`` dictionary.

Multi-context **voting** (Appendix C of the paper)
--------------------------------------------------
For each target utterance the planner is queried **multiple times** with
cumulative context windows ranging from CTS=1 up to CTS=15 preceding
lines. Each query yields a four-tuple
``(emotion, rhythm, intonation, sound_effects)``. The final plan is the
combination that appears most frequently across all CTS variants; in the
event of a tie the combination predicted under the **longest** context
span wins. This favours expressive plans that are stable across varying
amounts of narrative context.

See :func:`predict_ideal_voted` for the public voting API and
:func:`predict_ideal` for the single-CTS primitive.

Two surface forms are produced:
* a Chinese plan (``predict_ideal*(..., lang="zh")``)
* an English one (``predict_ideal*(..., lang="en")``)

Both share the same prompt structure; they only differ in the vocab used
for the JSON values.

The module can also be invoked as a script; see ``__main__`` at the
bottom.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed as hf_set_seed


# ---------------------------------------------------------------------------
# Default local path for Qwen3-8B.  Override with ``--qwen3_model_path`` or
# by passing ``model_path`` to :func:`load_qwen3`.
# ---------------------------------------------------------------------------
DEFAULT_QWEN3_PATH = os.environ.get("QWEN3_MODEL_PATH", "Qwen/Qwen3-8B")

# Maximum context size used by the voted planner.  Matches the
# 1..15 range reported in Sec. 3.3.1 / Appendix C of the paper.
DEFAULT_MAX_CTS = 15


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
_ZH_PROMPT = """你是一名专业的配音指导与语音表现分析师，熟悉文学作品中的情感表达、语音韵律控制和音效设计。
现在请你基于给定的小说语境 (context) 与目标台词 (target line)，预测该台词在有声书合成中应呈现的情感、韵律、音调和音效特征。

请综合考虑以下因素：
1. 语境中的人物关系、叙事背景、语气暗示；
2. 台词中的情绪转折、语义焦点与潜在心理状态；
3. 不同场景下的语音表达策略（如电话转录、远场录音、低声独白等）。

---

【输入】
Context（语境）：
{context}

Target Line（目标台词）：
{target_line}

---

【输出要求】
请严格按照以下 JSON 格式输出，不添加任何额外文字说明：

{{
  "emotion": "悲伤 / 愤怒 / 温柔 / 惊讶 / 平静 / 兴奋 / 忧郁 / 紧张 / 感动 / 调侃 / 思考 等",
  "rhythm": "轻快型 / 凝重型 / 低沉型 / 高亢型 / 舒缓型 / 紧张型",
  "intonation": "平直调 / 上扬调 / 弯曲调 / 下降调",
  "sound_effects": "正常说话 / 电话转录 / 远场 / 断断续续 / 内心独白 等"
}}
"""

_EN_PROMPT = """You are a professional voice-over director and speech-expression analyst, specialised
in the emotional delivery, prosodic control and sound-design of narrated fiction.
Given the novel context and the target line below, predict the ideal expressive
attributes the target line should exhibit when rendered as audiobook speech.

Consider:
1. Character relationships, narrative setting and subtle tonal cues in the context;
2. Emotional pivots, semantic focus and latent mental state in the target line;
3. Scene-specific delivery strategies (phone transcript, far-field recording,
   intimate monologue, etc).

---

[Input]
Context:
{context}

Target Line:
{target_line}

---

[Output]
Reply *strictly* in the JSON form below with no extra prose:

{{
  "emotion": "Sad / Angry / Gentle / Surprised / Calm / Excited / Melancholic / Nervous / Moved / Teasing / Thoughtful / ...",
  "rhythm": "Lively type / Solemn / Low type / High-pitched / Relaxed type / Nervous type",
  "intonation": "Flat tone / Rising tone / Curved tone / Falling tone",
  "sound_effects": "Normal speaking / Phone transcript / Far-field / Intermittent / Inner monologue / ..."
}}
"""


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@dataclass
class Qwen3Labeler:
    """Thin wrapper around a loaded Qwen3 chat model."""
    tokenizer: any
    model: any
    model_path: str

    def call(self, prompt: str, *, seed: int = 42, temperature: float = 0.7,
             top_p: float = 0.9, max_new_tokens: int = 1024) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # transformers>=4.50 removed the ``generator=`` kwarg on .generate(),
        # so drive reproducibility via the global seed instead.
        hf_set_seed(seed)

        do_sample = not (temperature == 0)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        return self.tokenizer.decode(
            output_ids[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True,
        )


def load_qwen3(
    model_path: Optional[str] = None,
    dtype: str = "fp16",
    device_map: str = "auto",
) -> Qwen3Labeler:
    """Load a Qwen3-8B (or compatible) chat model from ``model_path``.

    ``model_path`` may be a local directory (preferred) or a HF hub repo id.
    """
    if model_path is None:
        model_path = os.environ.get("QWEN3_MODEL_PATH", DEFAULT_QWEN3_PATH)

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.float16)

    is_local = os.path.isdir(model_path)
    kwargs = dict(local_files_only=is_local)

    print(f"[qwen3] Loading tokenizer from {model_path} "
          f"(local_files_only={is_local})")
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)

    print(f"[qwen3] Loading model from {model_path} "
          f"(dtype={dtype}, device_map={device_map})")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        **kwargs,
    )
    return Qwen3Labeler(tokenizer=tokenizer, model=model, model_path=model_path)


# ---------------------------------------------------------------------------
# Prompt / parse helpers
# ---------------------------------------------------------------------------
def _context_to_lines(ctx: Union[str, Iterable[str], None]) -> List[str]:
    """Normalise an arbitrary ``context`` input into an ordered list of
    non-empty lines, preserving the original ordering.

    Accepts a string (split on newlines) or any iterable of strings.
    """
    if ctx is None:
        return []
    if isinstance(ctx, str):
        lines = ctx.splitlines()
    else:
        lines = []
        for x in ctx:
            if x is None:
                continue
            s = str(x)
            # allow a list where each item itself contains newlines
            lines.extend(s.splitlines() if "\n" in s else [s])
    return [ln.strip() for ln in lines if ln is not None and str(ln).strip()]


def _lines_to_prompt_block(lines: List[str]) -> str:
    return "\n".join(lines).strip()


def _coerce_context(ctx: Union[str, Iterable[str], None]) -> str:
    """Legacy helper kept for backward-compat.  Always returns a flat string."""
    return _lines_to_prompt_block(_context_to_lines(ctx))


def cumulative_contexts(
    ctx: Union[str, Iterable[str], None],
    max_cts: int = DEFAULT_MAX_CTS,
) -> List[Tuple[int, List[str]]]:
    """Build the ``CTS = 1 .. max_cts`` cumulative context windows used by
    the voted planner (Appendix C of the paper).

    For each CTS value, we take the *last* ``CTS`` lines of ``ctx`` (the
    lines immediately preceding the target line), matching the preference
    described in Appendix B.

    Returns a list of ``(cts, lines)`` pairs, deduplicated so that if
    ``len(ctx) < max_cts`` we stop growing the window once it covers the
    entire context (avoiding duplicated votes for identical prompts).
    """
    lines = _context_to_lines(ctx)
    if not lines:
        # context-free fallback (CTS=0): single query with no preceding lines.
        return [(0, [])]
    out: List[Tuple[int, List[str]]] = []
    seen_len = -1
    for cts in range(1, max_cts + 1):
        take = lines[-cts:] if cts <= len(lines) else lines
        if len(take) == seen_len:
            # context already fully included at a smaller CTS; further
            # enlarging it would only produce identical prompts and bias
            # the vote towards this single answer.
            continue
        seen_len = len(take)
        out.append((cts, take))
    return out


def build_prompt(context: Union[str, Iterable[str]], target_line: str,
                 lang: str = "zh") -> str:
    ctx_block = _coerce_context(context)
    tmpl = _ZH_PROMPT if lang == "zh" else _EN_PROMPT
    return tmpl.format(context=ctx_block, target_line=target_line)


def parse_json_block(text: str) -> Optional[dict]:
    """Extract the first JSON block (``{...}``) from Qwen3's free-form reply."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        # Try a simple quote-style fix-up (common failure mode: trailing
        # commas, smart quotes, half-width / full-width mix).
        cleaned = (
            blob.replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2018", "'").replace("\u2019", "'")
                .replace(",\n}", "\n}").replace(", }", " }")
        )
        try:
            return json.loads(cleaned)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
REQUIRED_KEYS = ("emotion", "rhythm", "intonation", "sound_effects")


def predict_ideal(
    labeler: Qwen3Labeler,
    context: Union[str, Iterable[str]],
    target_line: str,
    *,
    lang: str = "zh",
    seed: int = 42,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_attempts: int = 10,
    retry_sleep_s: float = 1.0,
    verbose: bool = False,
) -> Tuple[Optional[dict], str]:
    """Single-CTS primitive.

    Return the parsed ``{emotion, rhythm, intonation, sound_effects}``
    dict for the given ``context`` / ``target_line``.

    Returns ``(parsed, raw_text)``.  ``parsed`` is ``None`` if Qwen3 did
    not produce a valid JSON block within ``max_attempts`` tries.
    """
    prompt = build_prompt(context, target_line, lang=lang)
    raw = ""
    for attempt in range(1, max_attempts + 1):
        raw = labeler.call(
            prompt,
            seed=seed + attempt - 1,  # vary seed across retries
            temperature=temperature,
            top_p=top_p,
        )
        parsed = parse_json_block(raw)
        if parsed and all(k in parsed for k in REQUIRED_KEYS):
            return parsed, raw
        if verbose:
            print(f"[qwen3] attempt {attempt}/{max_attempts} — parse failed, retrying")
        time.sleep(retry_sleep_s)
    return None, raw


# ---------------------------------------------------------------------------
# Joint voting across cumulative context sizes (Appendix C of the paper)
# ---------------------------------------------------------------------------
_HYPHEN_RE = re.compile(r"(\w)\s*-\s*(\w)")


def _fix_hyphen_spacing(text: str) -> str:
    # canonicalise "Low - paced" → "Low-paced" so that surface variants
    # do not fragment the vote.
    return _HYPHEN_RE.sub(r"\1-\2", text)


def _canonical_combo(ideal: Dict[str, Any], lang: str) -> Optional[Tuple[str, str, str, str]]:
    """Reduce a parsed ideal dict to a hashable 4-tuple used as vote key."""
    try:
        emotion    = _fix_hyphen_spacing(str(ideal["emotion"]).strip())
        rhythm     = _fix_hyphen_spacing(str(ideal["rhythm"]).strip())
        intonation = _fix_hyphen_spacing(str(ideal["intonation"]).strip())
    except (KeyError, TypeError):
        return None
    effect_raw = ideal.get("sound_effects")
    if effect_raw is None or str(effect_raw).strip() == "":
        effect_raw = "正常说话" if lang == "zh" else "Normal speaking"
    effect = _fix_hyphen_spacing(str(effect_raw).strip())
    return (emotion, rhythm, intonation, effect)


def predict_ideal_voted(
    labeler: Qwen3Labeler,
    context: Union[str, Iterable[str]],
    target_line: str,
    *,
    lang: str = "zh",
    max_cts: int = DEFAULT_MAX_CTS,
    seed: int = 42,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_attempts: int = 10,
    retry_sleep_s: float = 1.0,
    verbose: bool = False,
) -> Tuple[Optional[dict], Dict[int, Optional[dict]], Optional[int], str]:
    """Expressive planner with multi-context voting (Appendix C).

    For each ``CTS`` in ``1 .. max_cts`` (stopping once the context is
    fully covered), query the planner with the corresponding cumulative
    context window.  Each returned four-tuple
    ``(emotion, rhythm, intonation, sound_effects)`` casts one vote.

    The selected plan is the combination with:
        1. the highest vote count across CTS variants; ties broken by
        2. the largest CTS at which it was observed.

    Returns ``(voted_ideal, per_cts, chosen_cts, raw_last)`` where:

    * ``voted_ideal`` — the winning ``{emotion, rhythm, intonation,
      sound_effects}`` dict (copied verbatim from the per-CTS prediction
      that selected it), or ``None`` if every CTS query failed to parse.
    * ``per_cts``    — ``{cts: ideal_or_None}`` for every CTS queried.
    * ``chosen_cts`` — CTS of the rollout the winning plan was taken from.
    * ``raw_last``   — raw text of the last Qwen3 reply (for debugging).
    """
    windows = cumulative_contexts(context, max_cts=max_cts)

    per_cts: Dict[int, Optional[dict]] = {}
    # combo -> {count, max_cts, ideal_at_max_cts, cts_seen}
    combo_stats: Dict[Tuple[str, str, str, str], Dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "max_cts": -1, "ideal": None, "cts_seen": []}
    )

    raw_last = ""
    # space seeds out so retries across different CTS don't collide.
    for i, (cts, lines) in enumerate(windows):
        prompt_ctx = _lines_to_prompt_block(lines) if lines else (
            "(no preceding context)" if lang == "en" else "（无前文语境）"
        )
        ideal, raw = predict_ideal(
            labeler,
            prompt_ctx,
            target_line,
            lang=lang,
            seed=seed + i * 1_000,
            temperature=temperature,
            top_p=top_p,
            max_attempts=max_attempts,
            retry_sleep_s=retry_sleep_s,
            verbose=verbose,
        )
        raw_last = raw
        per_cts[cts] = ideal
        if ideal is None:
            if verbose:
                print(f"[vote] CTS={cts:>2} parse failed")
            continue

        combo = _canonical_combo(ideal, lang=lang)
        if combo is None:
            if verbose:
                print(f"[vote] CTS={cts:>2} missing required keys, skipped")
            continue

        stats = combo_stats[combo]
        stats["count"] += 1
        stats["cts_seen"].append(cts)
        if cts > stats["max_cts"]:
            stats["max_cts"] = cts
            stats["ideal"] = ideal  # store the dict produced at the largest CTS

        if verbose:
            print(f"[vote] CTS={cts:>2} -> {combo}")

    if not combo_stats:
        return None, per_cts, None, raw_last

    # Winner: highest count, then largest CTS (matches test_voting.py).
    best_combo, best_info = max(
        combo_stats.items(),
        key=lambda kv: (kv[1]["count"], kv[1]["max_cts"]),
    )
    voted_ideal = best_info["ideal"]
    chosen_cts  = best_info["max_cts"]

    if verbose:
        total_votes = sum(s["count"] for s in combo_stats.values())
        print(f"[vote] winner (votes={best_info['count']}/{total_votes}, "
              f"max_cts={chosen_cts}): {best_combo}")

    return voted_ideal, per_cts, chosen_cts, raw_last


def predict_ideal_both(
    labeler: Qwen3Labeler,
    context: Union[str, Iterable[str]],
    target_line: str,
    *,
    vote: bool = True,
    max_cts: int = DEFAULT_MAX_CTS,
    **kwargs,
) -> Tuple[Optional[dict], Optional[dict]]:
    """Convenience wrapper that returns ``(zh_ideal, en_ideal)``.

    By default performs multi-context voting (Appendix C of the paper).
    Set ``vote=False`` to fall back to a single CTS-∞ query per language.
    """
    if vote:
        zh, _, _, _ = predict_ideal_voted(
            labeler, context, target_line,
            lang="zh", max_cts=max_cts, **kwargs,
        )
        en, _, _, _ = predict_ideal_voted(
            labeler, context, target_line,
            lang="en", max_cts=max_cts, **kwargs,
        )
        return zh, en
    zh, _ = predict_ideal(labeler, context, target_line, lang="zh", **kwargs)
    en, _ = predict_ideal(labeler, context, target_line, lang="en", **kwargs)
    return zh, en


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Qwen3-8B expressive-planner predictor for CEAEval "
                    "(with multi-context voting, Appendix C)."
    )
    p.add_argument("--qwen3_model_path", default=None,
                   help=f"Local path or HF repo id.  Defaults to "
                        f"${{QWEN3_MODEL_PATH}} or {DEFAULT_QWEN3_PATH}.")
    p.add_argument("--input_json",
                   help="List-of-dicts JSON.  Each item must carry "
                        "`novel_context` (str or list[str]) and `target_line`.")
    p.add_argument("--context",    default=None, help="Single-sample mode: context string.")
    p.add_argument("--target_line", default=None, help="Single-sample mode: target line.")
    p.add_argument("--lang", choices=["zh", "en", "both"], default="both")
    p.add_argument("--output_json", default=None,
                   help="If set, write back augmented JSON with `ideal_zh`/`ideal_en`.")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="fp16")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p",       type=float, default=0.9)
    p.add_argument("--max_attempts", type=int,  default=10,
                   help="Per-CTS parse retry budget.")

    # voting controls
    p.add_argument("--vote", dest="vote", action="store_true",
                   help="Multi-context voting (Appendix C of the paper). "
                        "This is the default.")
    p.add_argument("--no_vote", dest="vote", action="store_false",
                   help="Disable voting; issue a single query per language.")
    p.set_defaults(vote=True)
    p.add_argument("--max_cts", type=int, default=DEFAULT_MAX_CTS,
                   help="Maximum cumulative context size to enumerate when "
                        "voting (1..max_cts).  Paper uses 15.")
    p.add_argument("--emit_per_cts", action="store_true",
                   help="Also emit the per-CTS ideal dicts (key: "
                        "`ideal_{lang}_per_cts`) alongside the voted plan.")
    return p


def _run_one(labeler: Qwen3Labeler, ctx, target_line, *, lang: str,
             args, verbose: bool) -> Tuple[Optional[dict], Dict[int, Optional[dict]], Optional[int]]:
    """Run a single (ctx, target_line) through the planner — voted or not."""
    if args.vote:
        ideal, per_cts, chosen_cts, _ = predict_ideal_voted(
            labeler, ctx, target_line,
            lang=lang, max_cts=args.max_cts,
            seed=args.seed, temperature=args.temperature,
            top_p=args.top_p, max_attempts=args.max_attempts,
            verbose=verbose,
        )
        return ideal, per_cts, chosen_cts
    ideal, _raw = predict_ideal(
        labeler, ctx, target_line,
        lang=lang, seed=args.seed, temperature=args.temperature,
        top_p=args.top_p, max_attempts=args.max_attempts,
        verbose=verbose,
    )
    return ideal, {}, None


def main():
    args = _build_argparser().parse_args()

    if (args.context is None) == (args.input_json is None):
        raise SystemExit(
            "Provide exactly one of: --input_json  OR  (--context AND --target_line)."
        )
    if args.context is not None and args.target_line is None:
        raise SystemExit("--target_line is required when --context is given.")

    labeler = load_qwen3(args.qwen3_model_path, dtype=args.dtype)

    langs = ("zh", "en") if args.lang == "both" else (args.lang,)

    if args.input_json is None:
        # single-sample mode
        result: Dict[str, Any] = {}
        for lang in langs:
            ideal, per_cts, chosen_cts = _run_one(
                labeler, args.context, args.target_line,
                lang=lang, args=args, verbose=True,
            )
            result[f"ideal_{lang}"] = ideal
            if args.vote:
                result[f"ideal_{lang}_chosen_cts"] = chosen_cts
                if args.emit_per_cts:
                    result[f"ideal_{lang}_per_cts"] = {
                        str(k): v for k, v in per_cts.items()
                    }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        return

    # batch mode
    with open(args.input_json, "r", encoding="utf-8") as f:
        items = json.load(f)

    out: List[dict] = []
    for i, item in enumerate(items):
        # Prefer `novel_context_lines` (list[str]) so that voting can
        # enumerate exact preceding-line windows.
        ctx = (item.get("novel_context_lines")
               or item.get("novel_context")
               or item.get("novel_context_text")
               or "")
        tgt = item.get("target_line") or item.get("spoken_content") or ""
        if not tgt:
            print(f"[{i+1}/{len(items)}] skip (no target_line)")
            out.append(item)
            continue

        new_item = dict(item)
        for lang in langs:
            ideal, per_cts, chosen_cts = _run_one(
                labeler, ctx, tgt, lang=lang, args=args, verbose=False,
            )
            new_item[f"ideal_{lang}"] = ideal
            if args.vote:
                new_item[f"ideal_{lang}_chosen_cts"] = chosen_cts
                if args.emit_per_cts:
                    new_item[f"ideal_{lang}_per_cts"] = {
                        str(k): v for k, v in per_cts.items()
                    }
        out.append(new_item)
        msg = (f"[{i+1}/{len(items)}] "
               f"zh={new_item.get('ideal_zh')!s:.80}")
        if args.lang in ("both", "en"):
            msg += f"  en={new_item.get('ideal_en')!s:.80}"
        print(msg)

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[DONE] wrote {args.output_json}")


if __name__ == "__main__":
    main()
