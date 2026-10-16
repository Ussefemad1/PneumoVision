"""Export a few real MIMIC test-set notes to try on the LOCAL site.

Run this yourself, on the credentialed machine. It writes radiology-report (RR)
and discharge-note (DN) text files into ``mimic-local-examples/`` (gitignored):

- pneumonia: N admissions with the pneumonia phenotype and N without,
  each with an RR file (all radiology reports of the admission, joined with a
  space, as DataFusion builds it for phenotyping) and a DN file;
- mortality: N stays that died in hospital and N that survived, each with an
  RR file (radiology reports from the first 48 h of the ICU stay, as for
  in-hospital mortality). No DN: it leaks the outcome.

Only test-split stays are used, so none of these were seen in training.

PhysioNet data use agreement: the output is MIMIC data. Keep it on this
machine. Never commit it, upload it to the Railway site, paste it into an
online tool or AI assistant, or show it to anyone without PhysioNet
credentialed access (that includes projecting it at a defense). This script
prints counts and file names only, never note text or identifiers.

Usage (repo root, the repo .venv):
    python tools/demo/export_mimic_examples.py \\
        --ehr-dir data/ehr_full --notes-dir D:/PneumoVision_data --n 3
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import pandas as pd

PNEUMONIA = "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)"
NOTE_COLUMNS = ["subject_id", "hadm_id", "charttime", "text"]
# Column names DataFusion.loadmetadata assigns to the mortality listfile.
IHM_COLUMNS = ["stay", "period_length", "stay_id", "y_true", "intime", "endtime"]


def read_notes(path: Path, hadm_ids: set[int]) -> pd.DataFrame:
    """The rows of a multi-GB note file whose admission is in ``hadm_ids``."""
    kept = []
    for chunk in pd.read_csv(path, usecols=NOTE_COLUMNS, chunksize=200_000):
        chunk = chunk.dropna(subset=["hadm_id", "text"])
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        kept.append(chunk[chunk["hadm_id"].isin(hadm_ids)])
    notes = pd.concat(kept, ignore_index=True)
    notes["charttime"] = pd.to_datetime(notes["charttime"])
    return notes.sort_values("charttime")


def joined(notes: pd.DataFrame) -> str:
    return " ".join(notes["text"].astype(str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ehr-dir", type=Path, required=True,
                        help="Folder with root/all_stays.csv, phenotyping/ and in-hospital-mortality/")
    parser.add_argument("--notes-dir", type=Path, required=True,
                        help="MIMIC-IV-Note folder with radiology.csv and discharge.csv")
    parser.add_argument("--out", type=Path, default=Path("mimic-local-examples"))
    parser.add_argument("--n", type=int, default=3, help="examples per class")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    stays = pd.read_csv(args.ehr_dir / "root" / "all_stays.csv",
                        usecols=["subject_id", "hadm_id", "stay_id", "intime"])
    stays["intime"] = pd.to_datetime(stays["intime"])

    pheno = pd.read_csv(args.ehr_dir / "phenotyping" / "test_listfile.csv")
    if PNEUMONIA not in pheno.columns:
        raise SystemExit("phenotyping/test_listfile.csv has no pneumonia column")
    pheno = pheno[["stay_id", PNEUMONIA]].merge(stays, on="stay_id")

    ihm = pd.read_csv(args.ehr_dir / "in-hospital-mortality" / "test_listfile.csv")
    ihm = ihm.iloc[:, : len(IHM_COLUMNS)]
    ihm.columns = IHM_COLUMNS[: ihm.shape[1]]
    ihm = ihm[["stay_id", "y_true"]].merge(stays, on="stay_id")

    hadm_ids = set(pheno["hadm_id"]) | set(ihm["hadm_id"])
    print(f"test admissions: {len(hadm_ids)}; reading notes (this takes a few minutes)...")
    radiology = read_notes(args.notes_dir / "radiology.csv", hadm_ids)
    discharge = read_notes(args.notes_dir / "discharge.csv", hadm_ids)
    rr_by_hadm = dict(tuple(radiology.groupby("hadm_id")))
    dn_by_hadm = dict(tuple(discharge.groupby("hadm_id")))

    args.out.mkdir(parents=True, exist_ok=True)
    labels: list[tuple[str, str, str, str]] = []

    def write(name: str, text: str) -> None:
        (args.out / name).write_text(text, encoding="utf-8")

    # Pneumonia: admissions with both a radiology report and a discharge note.
    for value, label in ((1, "pneumonia"), (0, "no-pneumonia")):
        rows = pheno[(pheno[PNEUMONIA] == value)
                     & pheno["hadm_id"].isin(rr_by_hadm.keys())
                     & pheno["hadm_id"].isin(dn_by_hadm.keys())]
        hadms = sorted(set(rows["hadm_id"]))
        for i, hadm in enumerate(rng.sample(hadms, min(args.n, len(hadms))), start=1):
            base = f"pneumonia_{label}_{i}"
            write(f"{base}_radiology.txt", joined(rr_by_hadm[hadm]))
            # The latest discharge note of the admission.
            write(f"{base}_discharge.txt", str(dn_by_hadm[hadm]["text"].iloc[-1]))
            labels.append((base, "pneumonia", label, "radiology + discharge"))

    # Mortality: radiology reports from the first 48 h of the ICU stay only.
    for value, label in ((1, "died"), (0, "survived")):
        candidates = []
        for row in ihm[ihm["y_true"] == value].itertuples():
            notes = rr_by_hadm.get(row.hadm_id)
            if notes is None:
                continue
            window = notes[(notes["charttime"] >= row.intime)
                           & (notes["charttime"] <= row.intime + pd.Timedelta(hours=48))]
            if not window.empty:
                candidates.append(window)
        for i, window in enumerate(rng.sample(candidates, min(args.n, len(candidates))), start=1):
            base = f"mortality_{label}_{i}"
            write(f"{base}_radiology.txt", joined(window))
            labels.append((base, "mortality", label, "radiology (first 48 h)"))

    with open(args.out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["example", "task", "true_label", "files"])
        writer.writerows(labels)

    print(f"wrote {len(labels)} examples to {args.out}/ (labels in labels.csv):")
    for base, task, label, _ in labels:
        print(f"  {base}  [{task}: {label}]")


if __name__ == "__main__":
    main()
