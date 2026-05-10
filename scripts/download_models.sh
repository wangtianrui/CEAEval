#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# One-click downloader for CEAeval.
#
# Usage:
#   bash scripts/download_models.sh
#   bash scripts/download_models.sh --no_qwen3
#   bash scripts/download_models.sh --dest /my/path
#
# Downloads:
#   <repo>/model_ckpts/ceaeval/     (TianRW/CEAEval-Model, incl. test_datas/)
#   <repo>/model_ckpts/qwen3_8b/    (Qwen/Qwen3-8B)
#
# After this finishes, `scripts/run_examples.sh` will pick up the local
# checkpoints automatically.
# -----------------------------------------------------------------------------
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CEAEVAL_REPO_ROOT="$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/_common.sh"

cd "$REPO_ROOT"
ceaeval_banner

PYTHONPATH="$REPO_ROOT" \
    "$PYBIN" scripts/download_models.py "$@"
