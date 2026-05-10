#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Shared header for every CEAeval wrapper script.
#
# Override any of these via env vars before invoking a wrapper.
# -----------------------------------------------------------------------------

CEAEVAL_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CEAEVAL_REPO_ROOT="${CEAEVAL_REPO_ROOT:-$CEAEVAL_COMMON_DIR}"

# Python binary to execute (default: whatever "python" resolves to on $PATH).
export PYBIN="${PYBIN:-python}"

# Qwen3-8B local directory (defaults to the HF repo id — will be downloaded on
# first use if nothing local is provided).
export QWEN3_MODEL_PATH="${QWEN3_MODEL_PATH:-Qwen/Qwen3-8B}"

# CEAeval scorer — local dir or HF repo id.  Empty means "let the Python
# entry point fall back to the default HF repo (TianRW/CEAEval-Model)".
export CEAEVAL_MODEL="${CEAEVAL_MODEL:-}"

# Root dir where relative audio paths in the test JSON are resolved.
export DATA_ROOT="${DATA_ROOT:-}"

# GPU / tokenizer toggles.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

ceaeval_banner() {
    echo "[ceaeval] REPO_ROOT        = $CEAEVAL_REPO_ROOT"
    echo "[ceaeval] PYBIN            = $PYBIN"
    echo "[ceaeval] CEAEVAL_MODEL    = ${CEAEVAL_MODEL:-<default HF repo>}"
    echo "[ceaeval] QWEN3_MODEL_PATH = $QWEN3_MODEL_PATH"
    echo "[ceaeval] DATA_ROOT        = $DATA_ROOT"
    echo "[ceaeval] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
}
