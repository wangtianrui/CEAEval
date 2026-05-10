"""Batch inference entry point for CEAeval.

Reads a list-of-dicts JSON / JSONL where each item provides:
    - one of: ``audio`` (abs or relative path) / ``audio_path`` /
      ``meta.segment_file`` (resolved under ``--input_audio_dir``)
    - ``novel_context`` (str or list[str])
    - ``target_line``  (str)
    - (optional) ``ideal_zh`` / ``ideal_en`` -- if pre-computed, Qwen3 is skipped
    - (optional) ``lang``                    -- overrides the global --lang

For each item we:
  1. resolve ideal labels (pre-computed or via Qwen3-8B),
  2. run the Qwen2.5-Omni-Thinker scorer,
  3. append the prediction and write one JSON line to ``result.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from typing import Optional, Tuple

import torch
from tqdm import tqdm

from data_processor import (
    build_scorer_inputs,
    build_scorer_messages,
    make_bias_config,
    normalise_novel_context,
    slugify_name,
)
from hf_utils import CEAEVAL_HF_REPO
from infer_one import generate_once, load_scorer
from qwen3_ideal_labeler import (
    DEFAULT_MAX_CTS,
    DEFAULT_QWEN3_PATH,
    Qwen3Labeler,
    load_qwen3,
    predict_ideal,
    predict_ideal_voted,
)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def _read_samples(path: str):
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_audio(item: dict, *, input_audio_dir: Optional[str]) -> str:
    # Priority: explicit absolute path keys, then meta-based layout.
    for k in ("audio_path", "audio"):
        if item.get(k):
            p = item[k]
            if os.path.isabs(p) or input_audio_dir is None:
                return p
            return os.path.join(input_audio_dir, p)

    meta = item.get("meta") or {}
    seg = meta.get("segment_file")
    story = meta.get("story_name")
    if seg and input_audio_dir and story:
        return os.path.join(input_audio_dir, slugify_name(story), seg)
    if seg and input_audio_dir:
        return os.path.join(input_audio_dir, seg)

    raise KeyError(
        f"Cannot locate audio for item: {json.dumps(item, ensure_ascii=False)[:200]}"
    )


def _resolve_ideal_for_item(
    item: dict,
    *,
    lang: str,
    labeler: Optional[Qwen3Labeler],
    qwen3_kwargs: dict,
    vote: bool = True,
    max_cts: int = DEFAULT_MAX_CTS,
) -> Tuple[dict, Optional[int]]:
    """Resolve the expressive plan for ``item``.

    Returns ``(ideal, chosen_cts)`` where ``chosen_cts`` is the CTS the
    voted plan was taken from (``None`` if voting disabled or ideal was
    already pre-computed).
    """
    # Pre-computed plan?
    key_map = {"zh": ("ideal_zh",), "en": ("ideal_en",)}
    for k in key_map[lang]:
        v = item.get(k)
        if isinstance(v, dict) and v.get("emotion"):
            return v, item.get(f"{k}_chosen_cts")
    # Otherwise: Qwen3 or error
    if labeler is None:
        raise RuntimeError(
            f"item has no pre-computed ideal_{lang} and Qwen3 labeler is disabled.  "
            "Either add ideal_zh/ideal_en to the JSON or drop --skip_qwen3."
        )
    # Preserve list structure so the voter can enumerate preceding-line windows.
    ctx = (item.get("novel_context_lines")
           or item.get("novel_context")
           or item.get("novel_context_text")
           or "")
    tgt = item.get("target_line") or item.get("spoken_content") or ""
    if not tgt:
        raise RuntimeError("target_line / spoken_content missing from item")

    if vote:
        ideal, _per_cts, chosen_cts, _raw = predict_ideal_voted(
            labeler, ctx, tgt, lang=lang,
            max_cts=max_cts, **qwen3_kwargs,
        )
        if ideal is None:
            raise RuntimeError(f"Qwen3 voted planner failed to produce valid JSON for: {tgt}")
        return ideal, chosen_cts

    ctx_str = normalise_novel_context(ctx)
    ideal, _raw = predict_ideal(labeler, ctx_str, tgt, lang=lang, **qwen3_kwargs)
    if ideal is None:
        raise RuntimeError(f"Qwen3 failed to produce valid JSON for: {tgt}")
    return ideal, None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="CEAeval — batch inference")

    # scorer
    p.add_argument("--model_name_or_path", required=True,
                   help="Path to the CEAeval scorer checkpoint.")

    # data
    p.add_argument("--input_file_name", required=True,
                   help="Test JSON / JSONL.")
    p.add_argument("--input_audio_dir", default=None,
                   help="Root dir used when items use relative audio paths.")
    p.add_argument("--output_dir",      required=True)
    p.add_argument("--lang", choices=["zh", "en"], default="en")

    # qwen3
    p.add_argument("--skip_qwen3", action="store_true",
                   help="Require every item to carry ideal_zh/ideal_en already.")
    p.add_argument("--qwen3_model_path", default=None,
                   help=f"Local path or HF repo id.  Defaults to {DEFAULT_QWEN3_PATH}.")
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
                   help="Disable voting; issue a single planner query per sample.")
    p.set_defaults(vote=True)
    p.add_argument("--max_cts", type=int, default=DEFAULT_MAX_CTS,
                   help="Maximum cumulative context size to enumerate "
                        "during voting.  Paper uses 15.")

    # generation
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--sys_prompt",
                   default="你是一个用于评估语音表达质量的模型。"
                           "|You are a model designed to evaluate the quality of speech expressiveness.")

    # bias switches (match training)
    p.add_argument("--gated_fc",       dest="gated_fc",       action="store_true",  default=True)
    p.add_argument("--no-gated_fc",    dest="gated_fc",       action="store_false")
    p.add_argument("--dynamic_fac",    dest="dynamic_fac",    action="store_true",  default=True)
    p.add_argument("--no-dynamic_fac", dest="dynamic_fac",    action="store_false")
    p.add_argument("--inner_strength", dest="inner_strength", action="store_true",  default=True)
    p.add_argument("--no-inner_strength", dest="inner_strength", action="store_false")

    p.add_argument("--model_output_key", default="model_prediction")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite result.jsonl if it already exists (default: append).")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = build_argparser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[args]\n{json.dumps(vars(args), indent=2, ensure_ascii=False)}")

    samples = _read_samples(args.input_file_name)
    print(f"[data] loaded {len(samples)} samples from {args.input_file_name}")

    # ----------------------------------------------------------------------
    # Stage-0: figure out whether we need Qwen3 at all.
    # ----------------------------------------------------------------------
    lang = args.lang
    need_qwen3 = not args.skip_qwen3 and any(
        not isinstance(s.get(f"ideal_{lang}"), dict) for s in samples
    )
    labeler: Optional[Qwen3Labeler] = None
    qwen3_kwargs = dict(
        seed=args.qwen3_seed,
        temperature=args.qwen3_temperature,
        top_p=args.qwen3_top_p,
        max_attempts=args.qwen3_max_attempts,
    )
    if need_qwen3:
        labeler = load_qwen3(args.qwen3_model_path, dtype=args.qwen3_dtype)

    # ----------------------------------------------------------------------
    # Stage-1: pre-fill ideals for every item (done up front so we can fully
    # release Qwen3 from GPU before loading the 7B scorer).
    # ----------------------------------------------------------------------
    print("[stage-1] resolving ideal labels …")
    resolved_items = []
    for i, item in enumerate(tqdm(samples, desc="ideal")):
        it = deepcopy(item)
        item_lang = it.get("lang") or lang
        chosen_cts = None
        try:
            ideal, chosen_cts = _resolve_ideal_for_item(
                it, lang=item_lang, labeler=labeler, qwen3_kwargs=qwen3_kwargs,
                vote=args.vote, max_cts=args.max_cts,
            )
        except Exception as e:
            print(f"  [{i+1}] FAILED: {e}", file=sys.stderr)
            it["_error"] = str(e)
            ideal = None
        it[f"ideal_{item_lang}"] = ideal
        if chosen_cts is not None:
            it[f"ideal_{item_lang}_chosen_cts"] = chosen_cts
        it["_lang"] = item_lang
        resolved_items.append(it)

    if labeler is not None:
        del labeler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----------------------------------------------------------------------
    # Stage-2: load the scorer and run generation.
    # ----------------------------------------------------------------------
    print("[stage-2] loading scorer …")
    model, processor = load_scorer(args)
    bias_config = make_bias_config(
        processor,
        gated_fc=args.gated_fc,
        dynamic_fac=args.dynamic_fac,
        inner_strength=args.inner_strength,
    )
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        num_beams=1,
        repetition_penalty=args.repetition_penalty,
        return_dict_in_generate=True,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

    out_path = os.path.join(args.output_dir, "result.jsonl")
    mode = "w" if args.overwrite or not os.path.exists(out_path) else "a"
    if os.path.exists(out_path) and mode == "a":
        print(f"[out] appending to existing {out_path}")
    fout = open(out_path, mode, encoding="utf-8")
    print(f"[out] writing predictions to {out_path}")

    for i, it in enumerate(tqdm(resolved_items, desc="score")):
        if it.get("_error"):
            fout.write(json.dumps(it, ensure_ascii=False) + "\n")
            fout.flush()
            continue

        try:
            audio = _resolve_audio(it, input_audio_dir=args.input_audio_dir)
            if not os.path.isfile(audio):
                raise FileNotFoundError(audio)
        except Exception as e:
            it["_error"] = f"audio resolve failed: {e}"
            fout.write(json.dumps(it, ensure_ascii=False) + "\n")
            fout.flush()
            continue

        item_lang = it.get("_lang", lang)
        ideal = it[f"ideal_{item_lang}"]
        msgs = build_scorer_messages(audio, ideal=ideal, lang=item_lang,
                                     sys_prompt=args.sys_prompt)
        inputs = build_scorer_inputs(processor, msgs)

        try:
            text, text_sp = generate_once(model, processor, inputs,
                                          bias_config, gen_kwargs)
        except Exception as e:
            it["_error"] = f"generate failed: {e}"
            fout.write(json.dumps(it, ensure_ascii=False) + "\n")
            fout.flush()
            continue

        it[args.model_output_key]              = text
        it[args.model_output_key + "_wspt"]    = text_sp
        it["_resolved_audio_path"]             = audio
        fout.write(json.dumps(it, ensure_ascii=False) + "\n")
        fout.flush()

    fout.close()
    print(f"[done] wrote predictions to {out_path}")


if __name__ == "__main__":
    main()
