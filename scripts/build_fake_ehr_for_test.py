"""Fabricate the EHR side of data/test_run/ so the full training path can be
exercised before Caroll's real listfile arrives.

The reference command (scripts/phenotyping/Unimodal/CXR.sh) runs with
--modalities EHR-CXR, so fusion_main.py loads the EHR side even for the
"unimodal CXR" run. That requires four things beyond the images:

  {ehr_data_dir}/root/all_stays.csv
  {ehr_data_dir}/phenotyping/{train,val,test}_listfile.csv
  {ehr_data_dir}/phenotyping/train/<stay>.csv   (val reads from train/ too)
  {ehr_data_dir}/phenotyping/test/<stay>.csv

Crucially the ICU intime/outtime written here are the REAL values from
MIMIC-IV for those stays, not invented ones, so the 48-hour window filter in
DataFusion.py is genuinely tested against the real StudyDateTime values rather
than passing trivially.

Everything else (vital signs, labels) is fake. Nothing here is a scientific
result.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_ROOT = REPO_ROOT / "data" / "test_run"
CXR_DIR = REPO_ROOT / "data" / "mimic-cxr-jpg" / "2.0.0"
LIST_DIR = REPO_ROOT / "data" / "cxr_download_lists"

# the exact file fusion_main.py hardcodes; kept so the unpatched code also runs
HARDCODED_TS = "14991576_episode3_timeseries.csv"

rng = np.random.default_rng(49297)


def load_channels():
    cfg = json.loads((REPO_ROOT / "medpatch" / "ehr_utils" / "resources"
                      / "discretizer_config.json").read_text())
    return cfg["id_to_channel"], cfg["normal_values"], cfg["is_categorical_channel"]


def make_timeseries(channels, normal_values, is_categorical, n_rows=60):
    """A plausible 48h timeseries: 'Hours' plus the 17 discretizer channels."""
    hours = np.sort(rng.uniform(0, 47.9, size=n_rows))
    rows = []
    for h in hours:
        row = [f"{h:.4f}"]
        for ch in channels:
            # leave most cells blank, as real extracted data is very sparse
            if rng.random() < 0.15:
                if is_categorical[ch]:
                    row.append(str(normal_values[ch]))
                else:
                    base = float(normal_values[ch])
                    row.append(f"{base * rng.uniform(0.85, 1.15):.2f}")
            else:
                row.append("")
        rows.append(",".join(row))
    header = "Hours," + ",".join(channels)
    return header + "\n" + "\n".join(rows) + "\n"


def main():
    channels, normal_values, is_categorical = load_channels()
    print(f"Discretizer channels: {len(channels)}")

    # --- which stays are in the test set -------------------------------------
    listfiles = {
        "train": TEST_ROOT / "phenotyping" / "train_listfile.csv",
        "val": TEST_ROOT / "phenotyping" / "val_listfile.csv",
        "test": TEST_ROOT / "phenotyping" / "test_listfile.csv",
    }
    frames = {k: pd.read_csv(v) for k, v in listfiles.items()}
    all_rows = pd.concat(frames.values(), ignore_index=True)
    stay_ids = set(all_rows.stay_id)
    print(f"Stays in the fake listfiles: {len(stay_ids)}")

    # --- real ICU times for those stays --------------------------------------
    icu = pd.read_csv(REPO_ROOT / "data" / "mimic-iv" / "icu" / "icustays.csv.gz",
                      usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"])
    icu = icu[icu.stay_id.isin(stay_ids)].copy()
    print(f"Matched against real icustays.csv: {len(icu)} / {len(stay_ids)}")

    patients = pd.read_csv(REPO_ROOT / "data" / "mimic-iv" / "hosp" / "patients.csv.gz",
                           usecols=["subject_id", "gender", "anchor_age"])
    stays = icu.merge(patients, how="left", on="subject_id")
    stays["age"] = stays["anchor_age"]
    stays["ethnicity"] = "UNKNOWN"          # DataFusion selects this column by name

    all_stays = stays[["subject_id", "stay_id", "intime", "outtime",
                       "hadm_id", "age", "ethnicity", "gender"]]
    root_dir = TEST_ROOT / "root"
    root_dir.mkdir(parents=True, exist_ok=True)
    all_stays.to_csv(root_dir / "all_stays.csv", index=False)
    print(f"Wrote {root_dir/'all_stays.csv'} ({len(all_stays)} stays, REAL intime/outtime)")

    # --- per-stay timeseries files -------------------------------------------
    pheno = TEST_ROOT / "phenotyping"
    (pheno / "train").mkdir(parents=True, exist_ok=True)
    (pheno / "test").mkdir(parents=True, exist_ok=True)

    written = 0
    for split, frame in frames.items():
        # ehr_dataset.get_datasets reads val from the train/ directory
        target = pheno / ("test" if split == "test" else "train")
        for stay_name in frame["stay"]:
            (target / stay_name).write_text(
                make_timeseries(channels, normal_values, is_categorical))
            written += 1
    print(f"Wrote {written} timeseries files")

    # the filename fusion_main.py hardcodes, so unpatched code still runs
    hard = pheno / "train" / HARDCODED_TS
    hard.write_text(make_timeseries(channels, normal_values, is_categorical))
    print(f"Wrote hardcoded-path stand-in: {HARDCODED_TS}")

    # --- report the filter behaviour on the real selection -------------------
    print()
    print("=" * 62)
    print("FILTER CHECK on the 50 selected images (real metadata, real times)")
    print("=" * 62)

    chosen_ids = set()
    for frame in frames.values():
        chosen_ids |= set(frame.stay_id)

    lst = pd.read_csv(LIST_DIR / "cxr_download_list.csv")
    lst = lst[lst.stay_id.isin(chosen_ids)]

    meta = pd.read_csv(CXR_DIR / "mimic-cxr-2.0.0-metadata.csv")
    subjects = set(all_stays.subject_id)
    cohort_rows = meta[meta.subject_id.isin(subjects)]
    print(f"  CXR rows for these patients (any view) : {len(cohort_rows)}")

    ap_rows = cohort_rows[cohort_rows.ViewPosition == "AP"]
    print(f"  after AP-only filter                   : {len(ap_rows)}")
    print(f"    dropped (PA / lateral / other)       : {len(cohort_rows) - len(ap_rows)}")

    # build StudyDateTime exactly as DataFusion.py does
    ap = ap_rows.copy()
    ap["StudyTime"] = ap["StudyTime"].apply(lambda x: f"{int(float(x)):06}")
    ap["StudyDateTime"] = pd.to_datetime(
        ap["StudyDate"].astype(str) + " " + ap["StudyTime"].astype(str),
        format="%Y%m%d %H%M%S")

    merged = ap.merge(all_stays[["subject_id", "stay_id", "intime", "outtime"]],
                      how="inner", on="subject_id")
    merged["intime"] = pd.to_datetime(merged["intime"])
    merged["outtime"] = pd.to_datetime(merged["outtime"])

    in_window = merged[(merged.StudyDateTime >= merged.intime)
                       & (merged.StudyDateTime <= merged.intime + pd.Timedelta(hours=48))]
    print(f"  after 48h window filter                : {len(in_window)}")

    before = len(in_window)
    one_per_stay = in_window.sort_values("StudyDateTime").groupby("stay_id").last()
    print(f"  before one-image-per-stay              : {before}")
    print(f"  after  one-image-per-stay              : {len(one_per_stay)}")
    print(f"    collapsed                            : {before - len(one_per_stay)}")
    print("=" * 62)


if __name__ == "__main__":
    main()
