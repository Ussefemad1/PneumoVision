"""Pretrained caches baked into the image, used offline at runtime.

``Text_encoder`` calls ``BertModel.from_pretrained(bert_model_name)`` and the
CXR encoder ``timm.create_model(name, pretrained=True)``. Both fetch weights
that the trained checkpoint then overwrites, so the service only needs them to
*exist*. The Docker image bakes them into ``HF_HOME`` (``scripts/bake_caches.py``)
and runs with ``HF_HUB_OFFLINE=1``; :func:`check_caches` makes a missing cache
a clear start-up error instead of a confusing one from deep inside
transformers.
"""

from __future__ import annotations

import os

#: The pretrained models our trained readers were built from (manifest
#: ``bert_model_name``; ``--cxr_encoder`` of every CXR reader).
BERT_MODEL = "dmis-lab/biobert-v1.1"
CXR_TIMM_MODEL = "vit_small_patch16_384"

_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")


def timm_hub_repo(name: str) -> str:
    """The Hugging Face repo timm downloads ``name``'s pretrained weights from."""
    from timm.models import get_pretrained_cfg  # noqa: PLC0415

    cfg = get_pretrained_cfg(name)
    if cfg is None or not cfg.hf_hub_id:
        raise RuntimeError(f"timm has no Hugging Face weights for {name}")
    return str(cfg.hf_hub_id)


def _cached(repo: str, filename: str) -> bool:
    from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415

    return isinstance(try_to_load_from_cache(repo, filename), str)


def missing_caches(bert_models: list[str], timm_models: list[str]) -> list[str]:
    """Which of the pretrained models are absent from the local HF cache."""
    missing = []
    for repo in bert_models:
        if not (_cached(repo, "config.json") and _cached(repo, "vocab.txt")) or not any(
            _cached(repo, f) for f in _WEIGHT_FILES
        ):
            missing.append(repo)
    for name in timm_models:
        repo = timm_hub_repo(name)
        if not any(_cached(repo, f) for f in _WEIGHT_FILES):
            missing.append(f"{repo} (timm {name})")
    return missing


def check_caches(bert_models: list[str], timm_models: list[str]) -> None:
    """Raise when running offline without a baked cache. Online it is a no-op:
    a local run may still download them, as training did."""
    if not offline():
        return
    missing = missing_caches(bert_models, timm_models)
    if missing:
        raise RuntimeError(
            "HF_HUB_OFFLINE=1 but these pretrained caches are missing under "
            f"HF_HOME={os.environ.get('HF_HOME', '~/.cache/huggingface')}: "
            + ", ".join(missing)
            + ". Rebuild the inference image with INSTALL_ML=true (it bakes them), "
            "or unset HF_HUB_OFFLINE to download."
        )
