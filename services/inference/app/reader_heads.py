"""The Round 1 standalone readers' classifier heads, for fusion mode.

Fusion mode shows, per modality, what the trained Round 1 reader on its own
predicts ("By source"). CMSMAFusion's frozen encoders are those readers'
encoders (F2, bit-identical to the Round 2b parents, which carry the Round 1
weights), so only each reader's small classifier head is missing. This module
extracts those heads from the Round 1 checkpoints once (on a laptop:
``python -m tools.pv.extract_reader_heads``) into

    <FUSION_DIR>/reader_heads.pt    the head tensors + what each reader needs
    <FUSION_DIR>/reader_heads.json  source files + their sha256, and the .pt's sha256

together with a digest of every encoder tensor of each source reader, so the
service can prove at start-up that the fusion model's encoders are the
readers' own before it applies a head to their features. No training, no data.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .readers import READERS, ReaderSpec

FORMAT_VERSION = 1
HEADS_FILE = "reader_heads.pt"
MANIFEST_FILE = "reader_heads.json"

#: The reader's own classifier, inside its unimodal fusion model.
HEAD_PREFIX = {m: f"fusion_model.{m}_classifier." for m in ("ehr", "cxr", "rr", "dn")}
#: Top-level encoder tensors a reader's features depend on (the same names in a
#: Round 1 reader and in a Round 3 fusion model; ``fusion_model.*`` aliases and
#: the fusion's ``*_fixed`` copies are the same tensors).
ENCODER_PREFIXES = {
    "ehr": ("ehr_model.",),
    "cxr": ("cxr_model.",),
    "rr": ("text_model.bert.", "text_model.fc_rr."),
    "dn": ("text_model.bert.", "text_model.fc_dn."),
}


class ReaderHeadsError(RuntimeError):
    """reader_heads.pt is present but cannot be trusted."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(tensor: Any) -> str:
    """sha256 of a tensor's dtype, shape and bytes (bit-exact identity)."""
    t = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{t.dtype}|{tuple(t.shape)}|".encode())
    digest.update(t.numpy().tobytes())
    return digest.hexdigest()


def encoder_digests(state_dict: dict[str, Any], modality: str, cache: dict | None = None) -> dict[str, str]:
    """Digest of every encoder tensor ``modality``'s reader features depend on."""
    out = {}
    for key, tensor in state_dict.items():
        if key.startswith(ENCODER_PREFIXES[modality]):
            if cache is not None and key in cache:
                out[key] = cache[key]
                continue
            out[key] = tensor_digest(tensor)
            if cache is not None:
                cache[key] = out[key]
    return out


def _entry(spec: ReaderSpec, state_dict: dict[str, Any], source: Path) -> dict[str, Any]:
    prefix = HEAD_PREFIX[spec.modality]
    head = {k[len(prefix):]: v.detach().cpu().clone() for k, v in state_dict.items() if k.startswith(prefix)}
    if not head:
        raise ReaderHeadsError(f"{source}: no {prefix}* tensors -- not a {spec.key} reader")
    digests = encoder_digests(state_dict, spec.modality)
    if not digests:
        raise ReaderHeadsError(f"{source}: no encoder tensors under {ENCODER_PREFIXES[spec.modality]}")
    return {
        "key": spec.key,
        "modality": spec.modality,
        "task": spec.task,
        "label": spec.label,
        "args": list(spec.args),
        "output_index": spec.output_index,
        "test_auroc": spec.test_auroc,
        "normalizer": spec.normalizer,
        "source": spec.checkpoint,
        "source_sha256": sha256_file(source),
        "head": head,
        "encoder_digests": digests,
    }


def extract(checkpoint_dir: Path, out_dir: Path, keys: list[str] | None = None) -> dict[str, Any]:
    """Read every Round 1 reader under ``checkpoint_dir`` (the readers layout,
    services/inference/READERS.md) and write reader_heads.pt + .json."""
    import torch  # noqa: PLC0415

    specs = [s for s in READERS if keys is None or s.key in keys]
    missing = [s.checkpoint for s in specs if not (checkpoint_dir / s.checkpoint).is_file()]
    if missing:
        raise ReaderHeadsError(
            f"missing under {checkpoint_dir}:\n  " + "\n  ".join(missing)
        )
    readers = {}
    for spec in specs:
        source = checkpoint_dir / spec.checkpoint
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        readers[spec.key] = _entry(spec, checkpoint["state_dict"], source)
        del checkpoint
    out_dir.mkdir(parents=True, exist_ok=True)
    heads = out_dir / HEADS_FILE
    torch.save({"format": FORMAT_VERSION, "readers": readers}, heads)
    manifest = {
        "format": FORMAT_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "file": HEADS_FILE,
        "sha256": sha256_file(heads),
        "sources": [
            {"reader": r["key"], "checkpoint": r["source"], "sha256": r["source_sha256"]}
            for r in readers.values()
        ],
    }
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load(folder: Path) -> dict[str, Any] | None:
    """The verified heads payload, None when not installed, or raise."""
    import torch  # noqa: PLC0415

    heads, manifest_file = folder / HEADS_FILE, folder / MANIFEST_FILE
    if not heads.is_file():
        return None
    if not manifest_file.is_file():
        raise ReaderHeadsError(f"{heads} has no {MANIFEST_FILE} next to it")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    actual = sha256_file(heads)
    if actual != manifest.get("sha256"):
        raise ReaderHeadsError(
            f"{HEADS_FILE}: sha256 {actual[:12]}... does not match {MANIFEST_FILE} "
            f"({str(manifest.get('sha256'))[:12]}...); refusing it"
        )
    payload = torch.load(heads, map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT_VERSION:
        raise ReaderHeadsError(f"{HEADS_FILE}: format {payload.get('format')}, expected {FORMAT_VERSION}")
    payload["sha256"] = actual
    return payload
