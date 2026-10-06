"""Rebuild a task's train_listfile.csv for the real runs: cohort filter + excluded stays.

    python -m tools.pv.prepare_lists --task phenotyping --listfile-dir /content/data/ehr
    python -m tools.pv.prepare_lists --task mortality --listfile-dir /content/data/ehr \\
        --exclude-stay 30007216

What it does, in <listfile-dir>/<medpatch task>/ (or <listfile-dir> itself when the
listfiles are directly there):

1. The first time, copies train_listfile.csv aside as train_listfile.full.csv. That
   copy is never overwritten, and every later run rebuilds from it (idempotent).
2. Keeps only train rows whose subject is listed in
   medpatch/mimic4extract/mimic3benchmark/resources/testset_iv.csv (Farida's
   cohort filter, which the text readers were trained on).
3. Drops the excluded stays by stay_id (listfile column 2). Default: 30007216,
   whose chest X-ray is a permanent 404 upstream; any loader that asks for it
   raises KeyError in cxr_dataset.py. At most len(excluded) rows may go.
4. Leaves val_listfile.csv and test_listfile.csv untouched.

It prints counts and md5 sums only -- never rows. Counts are compared with the
ones the team confirmed (EXPECTED); a mismatch is refused before anything is
written unless --allow-count-mismatch is given.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.pv.medpatch_bridge import MEDPATCH  # noqa: E402
from tools.pv.paper_scripts import TASK_ALIASES, canonical_task  # noqa: E402

TESTSET = MEDPATCH / "mimic4extract" / "mimic3benchmark" / "resources" / "testset_iv.csv"
#: Stays no run may load. 30007216: its CXR is a permanent 404 upstream.
DEFAULT_EXCLUDE = ("30007216",)
STAY_ID_COLUMN = 2
FULL_NAME = "train_listfile.full.csv"

#: Confirmed row counts (header excluded). "train_filtered" is after the cohort
#: filter, before the stay drop; "train" after both (None: not confirmed).
EXPECTED: dict[str, dict[str, int | None]] = {
    "phenotyping": {"train_filtered": 42_328, "train": 42_327, "val": 4_756, "test": 11_845},
    "in-hospital-mortality": {"train_filtered": 19_064, "train": None, "val": 2_161, "test": 5_302},
}


@dataclass
class Result:
    folder: Path
    full_rows: int = 0
    train_filtered: int = 0
    train: int = 0
    val: int = 0
    test: int = 0
    dropped_stays: list[str] = field(default_factory=list)
    absent_stays: list[str] = field(default_factory=list)
    excluded_in_val_test: dict[str, int] = field(default_factory=dict)
    copied_full: bool = False
    md5: dict[str, str] = field(default_factory=dict)
    mismatches: list[str] = field(default_factory=list)


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_cohort(testset: Path) -> set[str]:
    """Subject ids listed in testset_iv.csv (lines "subject_id,flag", no header)."""
    subjects = set()
    with open(testset, encoding="utf-8") as f:
        for line in f:
            first = line.split(",", 1)[0].strip()
            if first.isdigit():
                subjects.add(first)
    return subjects


def _rows(path: Path) -> tuple[str, list[str]]:
    """(header line, data lines) with their original text and line endings."""
    with open(path, encoding="utf-8", newline="") as f:
        lines = f.readlines()
    if not lines:
        raise SystemExit(f"{path} is empty (no header).")
    return lines[0], [ln for ln in lines[1:] if ln.strip()]


def _fields(line: str) -> list[str]:
    return next(csv.reader([line]))


def _subject(fields: list[str]) -> str:
    # "stay" is "<subject_id>_episode<n>_timeseries.csv" (DataFusion.py splits on "_").
    return fields[0].split("_", 1)[0]


def _count(path: Path) -> int:
    return len(_rows(path)[1])


def resolve_folder(listfile_dir: Path, task: str) -> Path:
    for folder in (listfile_dir / task, listfile_dir):
        if (folder / "train_listfile.csv").is_file() or (folder / FULL_NAME).is_file():
            return folder
    raise SystemExit(f"No train_listfile.csv in {listfile_dir / task} or {listfile_dir}.")


def prepare(
    task: str,
    listfile_dir: Path,
    exclude: tuple[str, ...] = DEFAULT_EXCLUDE,
    testset: Path = TESTSET,
    expected: dict[str, int | None] | None = None,
    allow_count_mismatch: bool = False,
) -> Result:
    task = canonical_task(task)
    folder = resolve_folder(Path(listfile_dir), task)
    train, full = folder / "train_listfile.csv", folder / FULL_NAME
    res = Result(folder=folder)
    for name in ("val_listfile.csv", "test_listfile.csv"):
        if not (folder / name).is_file():
            raise SystemExit(f"Missing {folder / name}.")

    cohort = read_cohort(testset)
    if not cohort:
        raise SystemExit(f"No subject ids read from {testset}.")
    source = full if full.is_file() else train
    header, lines = _rows(source)
    res.full_rows = len(lines)

    kept = [ln for ln in lines if _subject(_fields(ln)) in cohort]
    res.train_filtered = len(kept)
    excluded = set(exclude)
    final = []
    for ln in kept:
        stay = _fields(ln)[STAY_ID_COLUMN].strip()
        if stay in excluded:
            res.dropped_stays.append(stay)
        else:
            final.append(ln)
    assert len(kept) - len(final) <= len(excluded), (
        f"stay rule dropped {len(kept) - len(final)} rows for {len(excluded)} excluded stay(s)"
    )
    res.train = len(final)
    res.absent_stays = sorted(excluded - set(res.dropped_stays))
    res.val = _count(folder / "val_listfile.csv")
    res.test = _count(folder / "test_listfile.csv")
    for name in ("val_listfile.csv", "test_listfile.csv"):
        n = sum(
            1 for ln in _rows(folder / name)[1] if _fields(ln)[STAY_ID_COLUMN].strip() in excluded
        )
        if n:
            res.excluded_in_val_test[name] = n

    for key, want in (expected or {}).items():
        got = getattr(res, key)
        if want is not None and got != want:
            res.mismatches.append(f"{key}: {got:,} rows, expected {want:,}")
    if res.mismatches and not allow_count_mismatch:
        raise SystemExit(
            "Counts differ from the confirmed ones; nothing written:\n  "
            + "\n  ".join(res.mismatches)
            + "\n(--allow-count-mismatch to write anyway)"
        )

    if not full.is_file():
        shutil.copy2(train, full)  # the original, kept aside once and never overwritten
        res.copied_full = True
    with open(train, "w", encoding="utf-8", newline="") as f:
        f.write(header)
        f.writelines(final)
    for name in (FULL_NAME, "train_listfile.csv", "val_listfile.csv", "test_listfile.csv"):
        res.md5[name] = md5_of(folder / name)
    return res


def describe(res: Result, task: str, exclude: tuple[str, ...]) -> str:
    lines = [
        f"[prepare_lists] {task}  folder {res.folder}",
        f"  {FULL_NAME:<24}: {res.full_rows:,} rows"
        + ("  (copied aside now)" if res.copied_full else "  (kept, not changed)"),
        f"  after cohort filter      : {res.train_filtered:,} rows (testset_iv.csv subjects)",
        f"  train_listfile.csv       : {res.train:,} rows "
        f"({len(res.dropped_stays)} excluded stay(s) dropped)",
        f"  val_listfile.csv         : {res.val:,} rows (unchanged)",
        f"  test_listfile.csv        : {res.test:,} rows (unchanged)",
    ]
    for stay in exclude:
        where = "present in train, dropped" if stay in res.dropped_stays else "absent from train"
        lines.append(f"  excluded stay {stay}   : {where}")
    if res.train_filtered == res.full_rows:
        lines.append(
            f"  WARNING the cohort filter removed nothing: {FULL_NAME} may already be a "
            "filtered list, not the original."
        )
    for name, n in res.excluded_in_val_test.items():
        lines.append(f"  WARNING {n} excluded stay row(s) in {name} (left unchanged)")
    for m in res.mismatches:
        lines.append(f"  WARNING count mismatch, written anyway: {m}")
    lines += [f"  md5 {name:<24}: {digest}" for name, digest in res.md5.items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> Result:
    p = argparse.ArgumentParser(
        prog="python -m tools.pv.prepare_lists", description=__doc__.splitlines()[0]
    )
    p.add_argument("--task", choices=sorted(TASK_ALIASES), required=True)
    p.add_argument("--listfile-dir", dest="listfile_dir", type=Path, required=True)
    p.add_argument(
        "--exclude-stay",
        dest="exclude",
        nargs="+",
        default=list(DEFAULT_EXCLUDE),
        help=f"stay_id(s) to drop from train (default {' '.join(DEFAULT_EXCLUDE)})",
    )
    p.add_argument("--testset", type=Path, default=TESTSET, help=argparse.SUPPRESS)
    p.add_argument("--allow-count-mismatch", dest="allow_count_mismatch", action="store_true")
    args = p.parse_args(argv)
    task = canonical_task(args.task)
    exclude = tuple(str(s) for s in args.exclude)
    res = prepare(
        task,
        args.listfile_dir,
        exclude,
        args.testset,
        EXPECTED[task],
        args.allow_count_mismatch,
    )
    print(describe(res, task, exclude))
    return res


if __name__ == "__main__":
    main()
