"""Hugging Face helpers shared by CEAeval inference entry points.

CEAeval is released as a single HF repo (default: ``TianRW/CEAEval-Model``)
that carries

* the scorer checkpoint at the repo root (``config.json``,
  ``model-*.safetensors``, tokenizer, …) and
* a ``test_datas/`` folder with sanity-check samples and the
  corresponding audios.

Users can pass a local directory everywhere a model / data path is
expected; when they don't, we fall back to this repo and either
* load the model directly from the hub via ``transformers.from_pretrained``
  (which handles caching on its own), or
* snapshot-download the ``test_datas/`` subtree for the sample runner.
"""
from __future__ import annotations

import os
from typing import Optional

CEAEVAL_HF_REPO = os.environ.get("CEAEVAL_HF_REPO", "TianRW/CEAEval-Model")


def resolve_model_source(name_or_path: Optional[str]) -> str:
    """Resolve what gets passed to ``*.from_pretrained``.

    * ``None`` / empty → the default HF repo id ``TianRW/CEAEval-Model``.
    * A string that exists on disk → that local directory.
    * Otherwise → treat as a HF repo id and let ``from_pretrained``
      download it on first use.
    """
    if not name_or_path:
        return CEAEVAL_HF_REPO
    return name_or_path


def download_test_datas(
    repo_id: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """Pull only the ``test_datas/`` subtree from the CEAeval repo.

    Returns the absolute path to the local ``test_datas/`` directory
    (inside the snapshot).  Nothing is downloaded if an up-to-date
    local copy already exists in the HF cache.
    """
    from huggingface_hub import snapshot_download

    repo_id = repo_id or CEAEVAL_HF_REPO
    print(f"[hf] snapshot_download(repo_id={repo_id}, "
          f"allow_patterns=['test_datas/*'])")
    snap_dir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=["test_datas/*", "test_datas/**/*"],
        cache_dir=cache_dir,
    )
    out = os.path.join(snap_dir, "test_datas")
    if not os.path.isdir(out):
        raise RuntimeError(
            f"Expected `test_datas/` under snapshot {snap_dir} but it is missing."
        )
    return out
