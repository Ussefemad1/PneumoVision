"""The run manifest: the team's checkpoint Sheet, in code.

One CSV row per checkpoint. Round 2 reads its parent from here and refuses to
run unless the parent is an r1 row for the same reader whose file still hashes
to the recorded sha256 -- that is what makes a teammate's real checkpoint a
drop-in replacement for a virtual one: register it, and the next r2 run picks
it up by path, with no code change.

Locations (both overridable, e.g. to a Google Drive folder on Colab):

  PV_MANIFEST  default <repo>/runs/manifest.csv
  RUNS_ROOT    default <repo>/runs/virtual
"""

from __future__ import annotations

import csv
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

from .medpatch_bridge import REPO_ROOT

COLUMNS = [
    "id",
    "who",
    "date",
    "stage",
    "reader",
    "task",
    "file_path",
    "sha256",
    "parent_id",
    "data",
    "seed",
    "val_auroc",
    "val_auprc",
    "notes",
]
STAGES = ("r1", "r2")
DATA_KINDS = ("virtual", "real")


class ManifestError(SystemExit):
    """Raised (as a clean CLI exit) when lineage or integrity checks fail."""


def manifest_path() -> Path:
    return Path(os.environ.get("PV_MANIFEST", REPO_ROOT / "runs" / "manifest.csv"))


def runs_root() -> Path:
    return Path(os.environ.get("RUNS_ROOT", REPO_ROOT / "runs" / "virtual"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path | None = None) -> list[dict[str, str]]:
    path = path or manifest_path()
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_row(row: dict[str, object], path: Path | None = None) -> dict[str, str]:
    path = path or manifest_path()
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise ValueError(f"unknown manifest columns: {sorted(unknown)}")
    if row.get("stage") not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    if row.get("data") not in DATA_KINDS:
        raise ValueError(f"data must be one of {DATA_KINDS}")
    full = {c: "" if row.get(c) is None else str(row.get(c)) for c in COLUMNS}
    full["date"] = full["date"] or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(full)
    return full


def next_id(stage: str, reader: str, path: Path | None = None) -> str:
    prefix = f"{stage}-{reader}-"
    taken = [r["id"] for r in read_rows(path) if r["id"].startswith(prefix)]
    numbers = [int(i[len(prefix) :]) for i in taken if i[len(prefix) :].isdigit()]
    return f"{prefix}{max(numbers, default=0) + 1:03d}"


def find(row_id: str, path: Path | None = None) -> dict[str, str] | None:
    return next((r for r in read_rows(path) if r["id"] == row_id), None)


def latest(
    stage: str, reader: str, data: str | None = None, path: Path | None = None
) -> dict[str, str] | None:
    rows = [
        r
        for r in read_rows(path)
        if r["stage"] == stage and r["reader"] == reader and (data is None or r["data"] == data)
    ]
    return rows[-1] if rows else None


def verify_parent(parent: dict[str, str] | None, reader: str, data: str) -> Path:
    """The r1 parent's file, or a ManifestError explaining why it is unusable."""
    if parent is None:
        raise ManifestError(
            f"No r1 row for reader '{reader}' ({data} data) in {manifest_path()}. "
            f"Run `python -m tools.pv.run r1 --reader {reader} --data {data}` or register "
            f"a teammate file with `python -m tools.pv.import_checkpoint`."
        )
    if parent["stage"] != "r1":
        raise ManifestError(f"Parent {parent['id']} is stage '{parent['stage']}', not r1.")
    if parent["reader"] != reader:
        raise ManifestError(
            f"Parent {parent['id']} is a '{parent['reader']}' checkpoint; this run is '{reader}'."
        )
    if parent["data"] != data:
        raise ManifestError(
            f"Parent {parent['id']} was trained on {parent['data']} data; this run uses {data}. "
            "Mixing virtual and real lineage is refused."
        )
    file = Path(parent["file_path"])
    if not file.is_file():
        raise ManifestError(f"Parent {parent['id']} file is missing: {file}")
    actual = sha256_file(file)
    if actual != parent["sha256"]:
        raise ManifestError(
            f"Parent {parent['id']} sha256 mismatch: manifest {parent['sha256'][:12]}..., "
            f"file {actual[:12]}... ({file}). The file changed after it was registered."
        )
    return file
