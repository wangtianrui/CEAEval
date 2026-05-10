#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Run the bundled CEAeval example samples end-to-end.
#
# Usage:
#   bash scripts/run_examples.sh [LANG]
#
# If `model_ckpts/ceaeval/` (from `scripts/download_models.sh`) exists the
# local checkpoint is used automatically.  Otherwise the scorer + test data
# are pulled from `TianRW/CEAEval-Model` on the fly.
#
# Env vars (see scripts/_common.sh):
#   PYBIN             — python binary
#   CEAEVAL_MODEL     — override scorer dir / HF repo id
#   QWEN3_MODEL_PATH  — Qwen3-8B local directory / HF repo id
#
# Output: examples/predictions.jsonl
# -----------------------------------------------------------------------------
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CEAEVAL_REPO_ROOT="$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/_common.sh"

LANG="${1:-en}"

# Auto-use the local snapshots produced by scripts/download_models.sh when
# the user hasn't overridden CEAEVAL_MODEL / QWEN3_MODEL_PATH explicitly.
LOCAL_CEAEVAL="$REPO_ROOT/model_ckpts/ceaeval"
LOCAL_QWEN3="$REPO_ROOT/model_ckpts/qwen3_8b"
LOCAL_TEST_DATAS="$LOCAL_CEAEVAL/test_datas"

if [ -z "$CEAEVAL_MODEL" ] && [ -d "$LOCAL_CEAEVAL" ]; then
    export CEAEVAL_MODEL="$LOCAL_CEAEVAL"
fi
if [ -d "$LOCAL_QWEN3" ]; then
    # Only override if the caller didn't already point somewhere that exists.
    if [ ! -d "$QWEN3_MODEL_PATH" ]; then
        export QWEN3_MODEL_PATH="$LOCAL_QWEN3"
    fi
fi

ceaeval_banner
cd "$REPO_ROOT"

MODEL_FLAG=()
if [ -n "$CEAEVAL_MODEL" ]; then
    MODEL_FLAG=(--model_name_or_path "$CEAEVAL_MODEL")
fi

SAMPLES_FLAG=()
if [ -f "$LOCAL_TEST_DATAS/infer_samples.json" ]; then
    SAMPLES_FLAG=(
        --samples   "$LOCAL_TEST_DATAS/infer_samples.json"
        --data_root "$LOCAL_TEST_DATAS"
    )
fi

PYTHONPATH="$REPO_ROOT" \
  "$PYBIN" examples/run_examples.py \
    "${MODEL_FLAG[@]}" \
    "${SAMPLES_FLAG[@]}" \
    --qwen3_model_path   "$QWEN3_MODEL_PATH" \
    --lang               "$LANG"
