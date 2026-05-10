"""Run the bundled CEAeval example samples end-to-end.

For each entry in ``examples/infer_samples.json`` we:
  1. use Qwen3-8B to predict the ideal expressive labels from
     ``novel_context_lines + spoken_content`` (unless ``--skip_qwen3``),
  2. feed that ideal + audio into the Qwen2.5-Omni-Thinker scorer,
  3. write a line to ``examples/predictions.jsonl``.

Usage:
    python examples/run_examples.py \
        --model_name_or_path /path/to/ceaeval_models \
        [--qwen3_model_path  /path/to/Qwen3-8B] \
        [--samples           examples/infer_samples.json] \
        [--output            examples/predictions.jsonl] \
        [--data_root         /path/to/ceaeval_data] \
        [--lang              en] \
        [--skip_qwen3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from typing import Optional

import torch
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_processor import (  # noqa: E402
    build_scorer_inputs,
    build_scorer_messages,
    make_bias_config,
    normalise_novel_context,
)
from hf_utils import CEAEVAL_HF_REPO, download_test_datas  # noqa: E402
from infer_one import generate_once, load_scorer  # noqa: E402
from qwen3_ideal_labeler import (  # noqa: E402
    DEFAULT_MAX_CTS,
    DEFAULT_QWEN3_PATH,
    load_qwen3,
    predict_ideal,
    predict_ideal_voted,
)


def build_argparser():
    p = argparse.ArgumentParser(description="Run the bundled CEAeval example samples")

    # scorer
    p.add_argument("--model_name_or_path", default=None,
                   help=f"Local checkpoint dir or HF repo id.  "
                        f"Defaults to {CEAEVAL_HF_REPO}.")

    # qwen3
    p.add_argument("--skip_qwen3", action="store_true",
                   help="Reuse ideal_{lang} already present in the JSON instead "
                        "of running Qwen3.")
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
                   help="Maximum cumulative context size for voting. "
                        "Paper uses 15.")

    # data / io
    p.add_argument("--samples",  default=None,
                   help="Path to the sample JSON.  If omitted, the 10-sample "
                        "`test_datas/infer_samples.json` is fetched from "
                        f"{CEAEVAL_HF_REPO}.")
    p.add_argument("--output",   default=os.path.join(REPO_ROOT, "examples", "predictions.jsonl"))
    p.add_argument("--data_root", default=None,
                   help="Root dir relative audio paths are resolved against.  "
                        "Auto-filled when --samples is auto-fetched.")
    p.add_argument("--lang", choices=["zh", "en"], default="en")

    # generation
    p.add_argument("--max_new_tokens",     type=int,   default=1024)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--sys_prompt",
                   default="你是一个用于评估语音表达质量的模型。"
                           "|You are a model designed to evaluate the quality of speech expressiveness.")

    # bias switches
    p.add_argument("--gated_fc",       dest="gated_fc",       action="store_true",  default=True)
    p.add_argument("--no-gated_fc",    dest="gated_fc",       action="store_false")
    p.add_argument("--dynamic_fac",    dest="dynamic_fac",    action="store_true",  default=True)
    p.add_argument("--no-dynamic_fac", dest="dynamic_fac",    action="store_false")
    p.add_argument("--inner_strength", dest="inner_strength", action="store_true",  default=True)
    p.add_argument("--no-inner_strength", dest="inner_strength", action="store_false")
    return p


def resolve_audio(entry: dict, data_root: Optional[str]) -> str:
    p = entry.get("audio_path") or entry.get("audio")
    if not p:
        raise KeyError(f"entry {entry.get('example_id')} has no audio_path")
    if os.path.isabs(p):
        return p
    if not data_root:
        raise KeyError(f"entry {entry.get('example_id')} uses a relative "
                       f"audio_path ({p}) but --data_root was not provided")
    return os.path.join(data_root, p)


def main():
    args = build_argparser().parse_args()

    if not os.path.isfile(args.samples):
        raise SystemExit(f"samples JSON not found: {args.samples}")

    with open(args.samples, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"[data] loaded {len(samples)} examples from {args.samples}")

    # ------------------------------------------------------------------
    # Stage-1 — Qwen3 ideal labels (optional)
    # ------------------------------------------------------------------
    need_qwen3 = not args.skip_qwen3
    labeler = None
    if need_qwen3:
        t0 = time.time()
        labeler = load_qwen3(args.qwen3_model_path, dtype=args.qwen3_dtype)
        print(f"[qwen3] loaded in {time.time()-t0:.1f}s")

    resolved = []
    print("[stage-1] resolving ideal labels …")
    for i, entry in enumerate(tqdm(samples, desc="ideal")):
        e = deepcopy(entry)
        key = f"ideal_{args.lang}"
        if args.skip_qwen3:
            if not isinstance(e.get(key), dict):
                print(f"  [{i+1}] WARN no pre-computed {key}; skipping Qwen3 but "
                      "this sample has no ideal — it will be flagged as error.")
                e["_error"] = f"missing {key} and --skip_qwen3 is set"
                resolved.append(e)
                continue
            resolved.append(e)
            continue

        ctx = (e.get("novel_context_lines")
               or e.get("novel_context")
               or e.get("novel_context_text"))
        tgt = e.get("spoken_content") or e.get("target_line") or ""
        if not tgt:
            e["_error"] = "no target_line / spoken_content"
            resolved.append(e)
            continue

        if args.vote:
            ideal, per_cts, chosen_cts, raw = predict_ideal_voted(
                labeler, ctx, tgt, lang=args.lang,
                max_cts=args.max_cts,
                seed=args.qwen3_seed,
                temperature=args.qwen3_temperature,
                top_p=args.qwen3_top_p,
                max_attempts=args.qwen3_max_attempts,
                verbose=False,
            )
            if ideal is None:
                e["_error"] = f"qwen3 voted-parse failed; raw={raw[:300]}"
            else:
                e[key] = ideal
                e[f"{key}_chosen_cts"] = chosen_cts
                e[f"{key}_per_cts"] = {str(k): v for k, v in per_cts.items()}
        else:
            # Single-query fallback: flatten list context to string to
            # match the legacy prompt format.
            ctx_str = normalise_novel_context(ctx)
            ideal, raw = predict_ideal(
                labeler, ctx_str, tgt, lang=args.lang,
                seed=args.qwen3_seed,
                temperature=args.qwen3_temperature,
                top_p=args.qwen3_top_p,
                max_attempts=args.qwen3_max_attempts,
                verbose=False,
            )
            if ideal is None:
                e["_error"] = f"qwen3 parse failed; raw={raw[:300]}"
            else:
                e[key] = ideal
        resolved.append(e)

    if labeler is not None:
        del labeler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Stage-2 — scorer
    # ------------------------------------------------------------------
    print("[stage-2] loading scorer …")
    t0 = time.time()
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
    print(f"[scorer] loaded in {time.time()-t0:.1f}s")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    fout = open(args.output, "w", encoding="utf-8")
    print(f"[out] writing predictions to {args.output}")

    for i, e in enumerate(resolved):
        print(f"\n===== [{i+1}/{len(resolved)}] {e.get('example_id')} =====")
        if e.get("_error"):
            print(f"  [skip] {e['_error']}")
            fout.write(json.dumps({
                "example_id":   e.get("example_id"),
                "segment_file": e.get("segment_file"),
                "error":        e["_error"],
            }, ensure_ascii=False) + "\n")
            fout.flush()
            continue

        try:
            audio_path = resolve_audio(e, args.data_root)
        except Exception as exc:
            fout.write(json.dumps({
                "example_id":   e.get("example_id"),
                "segment_file": e.get("segment_file"),
                "error":        f"audio resolve: {exc}",
            }, ensure_ascii=False) + "\n")
            fout.flush()
            continue

        if not os.path.isfile(audio_path):
            print(f"  [skip] audio missing: {audio_path}")
            fout.write(json.dumps({
                "example_id":   e.get("example_id"),
                "segment_file": e.get("segment_file"),
                "error":        f"audio missing: {audio_path}",
            }, ensure_ascii=False) + "\n")
            fout.flush()
            continue

        ideal = e[f"ideal_{args.lang}"]
        chosen_cts = e.get(f"ideal_{args.lang}_chosen_cts")
        print(f"  audio : {audio_path}")
        print(f"  target: {e.get('spoken_content')}")
        if chosen_cts is not None:
            print(f"  ideal : {ideal}   [voted @CTS={chosen_cts}]")
        else:
            print(f"  ideal : {ideal}")

        msgs = build_scorer_messages(audio_path, ideal=ideal, lang=args.lang,
                                     sys_prompt=args.sys_prompt)
        inputs = build_scorer_inputs(processor, msgs)
        text, text_sp = generate_once(model, processor, inputs,
                                      bias_config, gen_kwargs)
        print(f"  [pred ] {text_sp}")

        rec = {
            "example_id":      e.get("example_id"),
            "segment_file":    e.get("segment_file"),
            "audio_path":      audio_path,
            "target_line":     e.get("spoken_content"),
            "ideal":           ideal,
            "prediction": {
                "text":              text,
                "text_with_special": text_sp,
            },
        }
        if chosen_cts is not None:
            rec["ideal_chosen_cts"] = chosen_cts
            if e.get(f"ideal_{args.lang}_per_cts") is not None:
                rec["ideal_per_cts"] = e[f"ideal_{args.lang}_per_cts"]
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()

    fout.close()
    print(f"\n[done] wrote {args.output}")


if __name__ == "__main__":
    main()
