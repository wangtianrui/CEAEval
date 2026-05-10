"""One-click downloader: fetch every artefact CEAeval inference needs.

Targets two Hugging Face repos:
  * TianRW/CEAEval-Model   (scorer weights + the 10-sample ``test_datas/``)
  * Qwen/Qwen3-8B          (stage-1 ideal labeller)

By default everything lands inside ``model_ckpts/`` at the repo root:

    <repo>/model_ckpts/
    ├── ceaeval/      # full snapshot of TianRW/CEAEval-Model
    │   ├── config.json, model-*.safetensors, tokenizer files, …
    │   └── test_datas/
    │       ├── infer_samples.json
    │       └── audios/{0..9}.m4a
    └── qwen3_8b/     # full snapshot of Qwen/Qwen3-8B

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --no_qwen3       # only CEAeval repo
    python scripts/download_models.py --dest /my/path  # custom target dir
"""
from __future__ import annotations

import argparse
import os
import sys

# Allow running this file directly from any cwd.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hf_utils import CEAEVAL_HF_REPO  # noqa: E402


DEFAULT_DEST = os.path.join(_REPO_ROOT, "model_ckpts")
DEFAULT_QWEN3_REPO = "Qwen/Qwen3-8B"


def _format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path, followlinks=False):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except FileNotFoundError:
                pass
    return total


def snapshot(repo_id: str, dest: str) -> str:
    """Download the full snapshot of ``repo_id`` into ``dest`` (created if needed).

    Uses ``local_dir=dest`` so the files end up directly under ``dest`` (not
    inside an HF cache layout).  Existing files are skipped via content-hash
    check, so re-running is cheap.
    """
    from huggingface_hub import snapshot_download

    os.makedirs(dest, exist_ok=True)
    print(f"\n[hf] Downloading {repo_id!s} → {dest}")
    out = snapshot_download(repo_id=repo_id, local_dir=dest)
    print(f"[hf] done: {out}  ({_format_bytes(_dir_size(out))})")
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", default=DEFAULT_DEST,
                   help=f"Parent directory under which checkpoints land.  "
                        f"Default: {DEFAULT_DEST}")
    p.add_argument("--ceaeval_repo", default=CEAEVAL_HF_REPO,
                   help=f"HF repo id for the CEAeval scorer + test_datas.  "
                        f"Default: {CEAEVAL_HF_REPO}")
    p.add_argument("--qwen3_repo", default=DEFAULT_QWEN3_REPO,
                   help=f"HF repo id for the stage-1 ideal labeller.  "
                        f"Default: {DEFAULT_QWEN3_REPO}")
    p.add_argument("--no_qwen3", action="store_true",
                   help="Skip the Qwen3-8B download (~16 GB).")
    p.add_argument("--no_ceaeval", action="store_true",
                   help="Skip the CEAeval scorer / test_datas download.")
    return p


def main():
    args = build_argparser().parse_args()

    dest = os.path.abspath(args.dest)
    ceaeval_dir = os.path.join(dest, "ceaeval")
    qwen3_dir   = os.path.join(dest, "qwen3_8b")
    test_datas  = os.path.join(ceaeval_dir, "test_datas")

    print(f"[cfg] dest={dest}")
    print(f"[cfg] ceaeval  → {ceaeval_dir}  (from {args.ceaeval_repo})")
    print(f"[cfg] qwen3-8b → {qwen3_dir}    (from {args.qwen3_repo})")

    if not args.no_ceaeval:
        snapshot(args.ceaeval_repo, ceaeval_dir)
        if not os.path.isdir(test_datas):
            print(f"[WARN] expected test_datas/ under the CEAeval snapshot "
                  f"but it's missing: {test_datas}")
        else:
            samples = os.path.join(test_datas, "infer_samples.json")
            audios  = os.path.join(test_datas, "audios")
            print(f"[cfg] samples:  {samples}"
                  f"  (exists={os.path.isfile(samples)})")
            print(f"[cfg] audios:   {audios}"
                  f"  (exists={os.path.isdir(audios)})")

    if not args.no_qwen3:
        snapshot(args.qwen3_repo, qwen3_dir)

    print("\n[done] run the sanity check with:")
    print(f"    CEAEVAL_MODEL={ceaeval_dir} QWEN3_MODEL_PATH={qwen3_dir} \\")
    print(f"        bash scripts/run_examples.sh")


if __name__ == "__main__":
    main()
