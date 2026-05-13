---
license: cc-by-nc-4.0
---


# CEAEval

<p align="center">
  <img src="logo.png" alt="CEAEval Logo" width="400"/>
</p>


Inference code for **CEAEval**, the framework introduced in our ACL paper
*"Evaluating the Expressive Appropriateness of Speech in Rich Contexts"*.

CEAEval evaluates **context-rich expressive appropriateness** of Mandarin
speech: given a spoken utterance together with its surrounding narrative
context (up to tens of dialogue / narration lines) and the target line,
the system predicts how well the actual expressive realization (emotion,
prosody, recording condition, paralinguistic cues) aligns with the
communicative intent implied by the context, and returns

```
<think>…</think><score>X.X</score>    # X.X ∈ [0.0, 5.0]
```

This repository hosts the **inference pipeline** of the judge model
**CEAEval-M** described in Sec. 3.3 of the paper. Training data
construction, human annotation protocols, and the released dataset live in
the companion data repository; the model weights live in the companion
model repository (see [Related resources](#related-resources)).

## Method overview

CEAEval-M adopts a **planner–judge decoupled** design:

1. **Expressive Planner (text-only).** A frozen Qwen3-8B reads the
   narrative context and target line and predicts an *ideal expressive
   profile* — a four-tuple `{emotion, rhythm, intonation, recording_condition}`.
   Following Appendix C of the paper, the planner is queried with
   **cumulative context windows of size CTS = 1 … 15** preceding lines,
   and the final plan is selected by **joint voting** over the
   4-tuples: most-frequent combination wins; ties are broken by the
   largest CTS at which the combination was observed. Abstracting long
   narrative text into such a voted, structured plan relieves the
   speech judge from heavy long-context textual reasoning and
   stabilises scoring across context sizes.
2. **Speech-LLM Judge.** A LoRA-tuned Qwen2.5-Omni-7B-Thinker, distilled
   from an audio-captioning teacher and further optimized with
   CoT-style supervision, an **adaptive audio attention bias**, and
   GRPO reinforcement learning (Sec. 3.3 of the paper). Given the input
   speech and the planner's ideal plan, it produces a chain-of-thought
   comparison across emotion / rhythm / intonation / recording /
   paralinguistic cues, and a final appropriateness score in `[0, 5]`.


### Special tokens

The judge model adds six special tokens used both in training and
during autoregressive region-aware attention biasing:

```
<think> </think>        # CoT reasoning region
<score> </score>        # final score region
<focus_audio> </focus_audio>   # spans that refer to audio-dependent cues
```

## Install

```bash
conda create -n ceaeval python=3.11 -y
conda activate ceaeval
# ffmpeg is required for .m4a decoding
conda install -n ceaeval -c conda-forge ffmpeg -y
pip install -r requirements.txt
```

`transformers>=4.57.1` and `torch==2.5.1 / torchaudio==2.5.1` are required.
If your cluster already has a system `ffmpeg`, skip the conda install, or
point at it explicitly: `export FFMPEG_EXE=/abs/path/to/ffmpeg`.

## Quick start (3 commands)

```bash
conda activate ceaeval
bash scripts/download_models.sh          # fetches both checkpoints (~20 GB) into model_ckpts/
bash scripts/run_examples.sh             # runs the 10-sample sanity check
```

Run with Chinese prompts:

```bash
bash scripts/run_examples.sh zh
```

Or invoke the Python entry directly on your own JSON:

```bash
python examples/run_examples.py \
    --model_name_or_path /path/to/scorer \
    --qwen3_model_path   /path/to/Qwen3-8B \
    --samples            /path/to/your_samples.json \
    --data_root          /path/to/dir_with_audios \
    --lang en
```

## Checkpoints

Both the scorer weights and the sanity samples live in
**[`TianRW/CEAEval-Model`](https://huggingface.co/TianRW/CEAEval-Model)**.
The stage-1 text-only planner is
**[`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B)**.

`scripts/download_models.sh` places both under `model_ckpts/`:

```
model_ckpts/
├── ceaeval/        # scorer weights (tokenizer, safetensors, …)
│   └── test_datas/ # samples + audios
└── qwen3_8b/       # Qwen3-8B snapshot
```

Once the directory exists, every entry point auto-detects it — no env
vars required. You can still override:

* `CEAEVAL_MODEL` (or `--model_name_or_path`) – local scorer dir or HF repo id.
* `QWEN3_MODEL_PATH` (or `--qwen3_model_path`) – local Qwen3-8B dir or HF repo id.
* `CEAEVAL_HF_REPO` – override the default scorer repo id (`TianRW/CEAEval-Model`).

If `model_ckpts/` is absent, the entry scripts fall back to on-the-fly
download from the Hub on first use.

Other environment knobs (see `scripts/_common.sh`):

| Var | Purpose |
| --- | --- |
| `PYBIN` | Python binary (default `python`) |
| `CUDA_VISIBLE_DEVICES` | GPU selection (default `0`) |
| `FFMPEG_EXE` | custom ffmpeg path |
| `CEAEVAL_FORCE_M4A` | `1` (default) routes every non-m4a input through an in-memory M4A/AAC re-encode → decode so that the PCM matches the training distribution; `0` to decode directly. |


## Input schema

Each evaluation sample needs:

| Field | Required | Notes |
| --- | --- | --- |
| `audio_path` (or `audio`) | ✅ | wav / mp3 / m4a; absolute or relative to `--data_root` |
| `novel_context` (or `novel_context_lines` / `novel_context_text`) | ✅ unless `--skip_qwen3` | string or list of context lines (narration + dialogue) |
| `target_line` (or `spoken_content`) | ✅ unless `--skip_qwen3` | e.g. `"张三说：你走吧。"` |
| `lang` (`"zh"` / `"en"`) | optional | overrides `--lang` |
| `ideal_zh` / `ideal_en` | optional | if present, the planner stage is skipped |

See `model_ckpts/ceaeval/test_datas/infer_samples.json` for a minimal skeleton.

## Output

Each prediction contains:

| Key | Meaning |
| --- | --- |
| `prediction.text` | `<think>…</think><score>X.X</score>`, special tokens stripped |
| `prediction.text_with_special` | same, keeping special tokens (useful for `<score>` parsing) |



## Related resources

This code repository is one of three companion releases for the paper.
**Please use them together:**

| Resource | Link |
| --- | --- |
| 📄 Paper | *Evaluating the Expressive Appropriateness of Speech in Rich Contexts* (ACL) |
| 💻 Code (this repo) | <https://github.com/wangtianrui/CEAEval> |
| 🤖 Model (CEAEval-M) | <https://huggingface.co/TianRW/CEAEval-Model> |
| 📚 Dataset (CEAEval-D) | <https://huggingface.co/datasets/TianRW/CEAEval-Data> |

## Citation

If you use this code, please cite our paper:

<!-- ```bibtex
@inproceedings{wang2026ceaeval,
  title     = {Evaluating the Expressive Appropriateness of Speech in Rich Contexts},
  author    = {Wang, Tianrui and Ma, Ziyang and Peng, Yizhou and others},
  booktitle = {Proceedings of the Association for Computational Linguistics (ACL)},
  year      = {2026}
}
``` -->

## License

Released under **CC BY-NC 4.0** — non-commercial academic research use only.
See the LICENSE file and refer to the Ethical Statement of the paper for
details on responsible use.
