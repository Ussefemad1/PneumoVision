"""Repack an extracted torch checkpoint folder into a loadable file.

``torch.save`` writes a zip archive. If someone unzips it (or a download tool
does), the result is a folder holding ``data.pkl``, ``data/<n>``, ``version``
and ``byteorder``, which ``torch.load`` cannot open. This writes those entries
back into an uncompressed zip under a single top-level folder, which is the
layout ``torch.load`` expects, then checks it loads and has a ``state_dict``.

Usage:
    python services/inference/tools/repack_checkpoint.py <extracted folder> <out.pth.tar>
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path


def _long_path(path: Path) -> Path:
    """Windows paths past 260 characters need the extended-length prefix."""
    if os.name == "nt":
        resolved = str(path.resolve())
        if not resolved.startswith("\\\\?\\"):
            return Path("\\\\?\\" + resolved)
    return path


def repack(src: Path, out: Path) -> None:
    src = _long_path(src)
    if not (src / "data.pkl").is_file():
        raise SystemExit(f"{src} has no data.pkl; not an extracted torch checkpoint")
    out.parent.mkdir(parents=True, exist_ok=True)
    prefix = "archive"
    files = [src / "data.pkl"] + [
        p for p in sorted(src.rglob("*")) if p.is_file() and p.name != "data.pkl"
    ]
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:
        # data.pkl first: torch reads the archive prefix from the first record.
        # A fixed date, because extracted files often carry a 1970 mtime,
        # which zip cannot store.
        for path in files:
            info = zipfile.ZipInfo(f"{prefix}/{path.relative_to(src).as_posix()}", (1980, 1, 1, 0, 0, 0))
            zf.writestr(info, path.read_bytes())

    import torch  # noqa: PLC0415 — only needed for the check

    checkpoint = torch.load(out, map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint:
        raise SystemExit(f"{out} loaded but has no 'state_dict' key")
    print(f"ok: {out} ({len(checkpoint['state_dict'])} tensors)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    repack(Path(sys.argv[1]), Path(sys.argv[2]))
