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
    # Added 2026-10-05, last so older manifests keep their column order. Text
    # readers (rr/dn) only: the Hugging Face BERT the checkpoint was trained
    # with. Empty for ehr/cxr, and for rows written before the column existed.
    "bert_model_name",
]
STAGES = ("r1", "r2", "r2b")
#: The stage a run of each stage must load: r2 trains on a frozen r1 reader;
#: r2b (calibration) loads the r2 reader + confidence head and adds temperatures.
PARENT_STAGE = {"r2": "r1", "r2b": "r2"}
DATA_KINDS = ("virtual", "real")

#: Rows written before tasks were tracked are phenotyping (the only task then).
DEFAULT_TASK = "phenotyping"

#: Readers whose checkpoints embed a BERT, so lineage must carry its name.
TEXT_READERS = ("rr", "dn")
#: medpatch's default (--bert_model_name in arguments.py): what every virtual
#: run used before the flag existed.
DEFAULT_BERT = "emilyalsentzer/Bio_ClinicalBERT"


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
    """All rows; columns a row predates (older manifests) read as "".

    An empty ``task`` reads as "phenotyping": every row written before tasks
    were tracked was a phenotyping run.
    """
    path = path or manifest_path()
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = [{c: row.get(c) or "" for c in COLUMNS} for row in csv.DictReader(f)]
    for row in rows:
        row["task"] = row["task"] or DEFAULT_TASK
    return rows


def _upgrade_header(path: Path) -> None:
    """Rewrite an older manifest with the current columns (new ones empty).

    Appending a row with more columns than the header would misalign every
    later read, so the file is upgraded first -- via a temp file and an atomic
    replace, so an interrupted write cannot lose the manifest.
    """
    with open(path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), [])
    if header == COLUMNS:
        return
    unknown = set(header) - set(COLUMNS)
    if unknown:
        raise ManifestError(f"{path} has unknown columns {sorted(unknown)}; refusing to rewrite it")
    rows = read_rows(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def bert_model_name_of(row: dict[str, str]) -> str:
    """The BERT a text-reader row was trained with ("" if not a text reader).

    Virtual rows written before the column existed used medpatch's default.
    Real rows never get a silent default: "" means unknown, and run.py refuses
    to build on it.
    """
    if row.get("reader") not in TEXT_READERS:
        return ""
    if row.get("bert_model_name"):
        return row["bert_model_name"]
    return DEFAULT_BERT if row.get("data") == "virtual" else ""


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
    if not new:
        _upgrade_header(path)
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
    stage: str,
    reader: str,
    data: str | None = None,
    path: Path | None = None,
    task: str | None = None,
) -> dict[str, str] | None:
    """The newest row for a stage and reader, optionally of one data kind / task."""
    rows = [
        r
        for r in read_rows(path)
        if r["stage"] == stage
        and r["reader"] == reader
        and (data is None or r["data"] == data)
        and (task is None or r["task"] == task)
    ]
    return rows[-1] if rows else None


def _missing_parent_hint(expected_stage: str, reader: str, data: str, task: str) -> str:
    task_flag = "" if task == DEFAULT_TASK else f" --task {task}"
    if expected_stage == "r1":
        return (
            f"Run `python -m tools.pv.run r1 --reader {reader} --data {data}{task_flag}` or "
            f"register a teammate file with `python -m tools.pv.import_checkpoint{task_flag}`."
        )
    return (
        f"Run `python -m tools.pv.run {expected_stage} --reader {reader} --data {data}"
        f"{task_flag}` first."
    )


def verify_parent(
    parent: dict[str, str] | None,
    reader: str,
    data: str,
    expected_stage: str = "r1",
    task: str = DEFAULT_TASK,
) -> Path:
    """The parent's file, or a ManifestError explaining why it is unusable.

    ``expected_stage`` is the stage the parent must be: "r1" for an r2 run,
    "r2" for an r2b run (see PARENT_STAGE). ``task`` must match too: a
    phenotyping checkpoint (25 classes) can never feed a mortality run (1 class)
    or the reverse.
    """
    if parent is None:
        raise ManifestError(
            f"No {expected_stage} row for reader '{reader}' ({data} data, task {task}) in "
            f"{manifest_path()}. " + _missing_parent_hint(expected_stage, reader, data, task)
        )
    if parent["stage"] != expected_stage:
        raise ManifestError(
            f"Parent {parent['id']} is stage '{parent['stage']}', not {expected_stage}."
        )
    if parent["reader"] != reader:
        raise ManifestError(
            f"Parent {parent['id']} is a '{parent['reader']}' checkpoint; this run is '{reader}'."
        )
    if parent["data"] != data:
        raise ManifestError(
            f"Parent {parent['id']} was trained on {parent['data']} data; this run uses {data}. "
            "Mixing virtual and real lineage is refused."
        )
    parent_task = parent.get("task") or DEFAULT_TASK
    if parent_task != task:
        raise ManifestError(
            f"Parent {parent['id']} is a {parent_task} checkpoint; this run is {task}. "
            "Mixing tasks is refused (different classes, labels and data)."
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


def verify_r2_parent(
    parent: dict[str, str] | None, reader: str, data: str, task: str = DEFAULT_TASK
) -> Path:
    """An r2b run's parent: an r2 row for the same reader, data kind and task."""
    return verify_parent(parent, reader, data, expected_stage="r2", task=task)
