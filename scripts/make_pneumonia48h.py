"""Build the `pneumonia-48h` EHR folder from the existing `phenotyping` folder.

Phase 2 of the CXR track (see docs/cxr_progress_notes.md, "DECISION: two-phase
task structure"). The extension pairs MedPatch's 48-hour window, which the
upstream code reaches only under `in-hospital-mortality`, with the phenotype
labels, which it reaches only under `phenotyping`. MedPatch never evaluates
that combination, so this measures early pneumonia detection rather than
whole-stay recognition.

**No new data is downloaded and no raw MIMIC file is read.** Everything comes
from the phenotyping task folder that already exists: the whole-stay timeseries
are truncated to their first 48 hours and the labels are carried over
unchanged. The label is the admission's diagnosis either way — only how much
the model is allowed to see changes, which is exactly what "early prediction"
means.

Stays with less than 48 hours of record are dropped, matching
`create_in_hospital_mortality.py`, which refuses `los < n_hours`.

Usage (Colab, against the copy already on local disk):

    python scripts/make_pneumonia48h.py \
        --src /content/ehr/phenotyping \
        --dst /content/ehr/pneumonia-48h

Then train with:

    --task pneumonia-48h --labels_set pheno --num_classes 25

and report the pneumonia column (index 21 of the 25).
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

WINDOW_HOURS = 48.0

# The listfile's first three columns; everything after them is a label column.
META_COLUMNS = 3


def truncate_timeseries(src: Path, dst: Path, hours: float) -> bool:
    """Copy one timeseries CSV keeping only rows inside the window.

    Returns False if the file has no rows inside it, which would leave the
    model with an empty sequence.
    """
    with src.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # The benchmark writes "Hours" first; fall back to index 0 if renamed.
        try:
            hour_col = header.index("Hours")
        except ValueError:
            hour_col = 0
        kept = []
        for row in reader:
            if not row:
                continue
            try:
                t = float(row[hour_col])
            except (ValueError, IndexError):
                continue
            # Same bound as create_in_hospital_mortality.py.
            if -1e-6 < t < hours + 1e-6:
                kept.append(row)

    if not kept:
        return False

    with dst.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(kept)
    return True


def convert_split(src_dir: Path, dst_dir: Path, listfile: Path, out_listfile: Path) -> dict:
    """Convert one split. Returns counts for the summary."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    with listfile.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if r]

    n_labels = len(header) - META_COLUMNS
    kept, short, empty, missing = [], 0, 0, 0

    for row in rows:
        stay_file, period_length, stay_id = row[0], row[1], row[2]
        labels = row[META_COLUMNS:]

        # Drop stays that never reach the window, as the benchmark does.
        try:
            if float(period_length) < WINDOW_HOURS:
                short += 1
                continue
        except ValueError:
            short += 1
            continue

        src_ts = src_dir / stay_file
        if not src_ts.exists():
            missing += 1
            continue

        if not truncate_timeseries(src_ts, dst_dir / stay_file, WINDOW_HOURS):
            empty += 1
            continue

        # period_length becomes the window; the labels are unchanged.
        kept.append([stay_file, f"{WINDOW_HOURS:.6f}", stay_id, *labels])

    with out_listfile.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(kept)

    return {
        "in": len(rows),
        "kept": len(kept),
        "dropped_short": short,
        "dropped_empty": empty,
        "missing_file": missing,
        "n_labels": n_labels,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path, help="existing phenotyping/ folder")
    ap.add_argument("--dst", required=True, type=Path, help="pneumonia-48h/ folder to create")
    ap.add_argument(
        "--pneumonia-index",
        type=int,
        default=21,
        help="label column to summarise (CCS pneumonia is 21 of the 25)",
    )
    args = ap.parse_args()

    if not args.src.is_dir():
        raise SystemExit(f"source folder not found: {args.src}")
    args.dst.mkdir(parents=True, exist_ok=True)

    # val rows live in train/ upstream, so both read from the same directory.
    splits = [
        ("train_listfile.csv", "train", "train"),
        ("val_listfile.csv", "train", "train"),
        ("test_listfile.csv", "test", "test"),
    ]

    totals = []
    for listname, src_sub, dst_sub in splits:
        listfile = args.src / listname
        if not listfile.exists():
            raise SystemExit(f"missing listfile: {listfile}")
        stats = convert_split(
            args.src / src_sub,
            args.dst / dst_sub,
            listfile,
            args.dst / listname,
        )
        totals.append((listname, stats))
        print(
            f"{listname:20s} {stats['in']:>6,} in -> {stats['kept']:>6,} kept"
            f"   (short {stats['dropped_short']:,}"
            f", empty {stats['dropped_empty']:,}"
            f", missing {stats['missing_file']:,})"
        )

    # all_stays.csv is read from root/ by DataFusion, not per task; leave it be.
    root_src = args.src.parent / "root"
    if root_src.is_dir() and not (args.dst.parent / "root").exists():
        shutil.copytree(root_src, args.dst.parent / "root")
        print("copied root/ alongside the new task folder")

    n_labels = totals[0][1]["n_labels"]
    print(f"\nlabel columns carried over unchanged: {n_labels}")
    if n_labels != 25:
        print("  WARNING: expected 25 phenotype columns -- check the source listfile")
    if not 0 <= args.pneumonia_index < n_labels:
        print(f"  WARNING: --pneumonia-index {args.pneumonia_index} is outside 0..{n_labels - 1}")

    print(f"\nWrote {args.dst}")
    print("Train with:  --task pneumonia-48h --labels_set pheno --num_classes 25")
    print(f"Report label column {args.pneumonia_index} (pneumonia).")


if __name__ == "__main__":
    main()
