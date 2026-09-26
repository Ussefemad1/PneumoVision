"""Norhan / x-ray track: build the list of CXR studies to download.

Does NOT download any images -- it only produces a candidate list and
prints counts so you can sanity check before committing to a 40-100 GB
download.

Deliberately does not wait on Caroll's mimic4extract output. It rebuilds
the same ICU cohort filters directly from the raw MIMIC-IV hosp/icu tables,
mirroring medpatch/mimic4extract/mimic3benchmark/mimic3csv.py exactly:

  - drop ICU stays that transferred between care units mid-stay
    (first_careunit != last_careunit)
  - keep only admissions with EXACTLY one qualifying ICU stay left after
    that (mimic3csv.filter_admissions_on_nb_icustays, min=max=1) --
    this drops multi-stay admissions entirely, it does not pick "the first"
  - age = anchor_age (MIMIC-IV convention), filtered to >= 18

Task A (in-hospital-mortality, 48h window) additionally requires
los >= 48h, matching create_in_hospital_mortality.py's n_hours=48 cutoff.

The CXR side reuses the exact ViewPosition == 'AP' filter and StudyDateTime
construction found in datasets/DataFusion.py (lines ~189-190, ~295, ~361),
so the candidate list is consistent with what the real training pipeline
will compute later.
"""
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MIMIC_IV_DIR = REPO_ROOT / "data" / "mimic-iv"
CXR_DIR = REPO_ROOT / "data" / "mimic-cxr-jpg" / "2.0.0"
OUT_DIR = REPO_ROOT / "data" / "cxr_download_lists"

TASK_A_HOURS = 48


def load_icu_cohort():
    patients_path = MIMIC_IV_DIR / "hosp" / "patients.csv.gz"
    admissions_path = MIMIC_IV_DIR / "hosp" / "admissions.csv.gz"
    icustays_path = MIMIC_IV_DIR / "icu" / "icustays.csv.gz"
    for path in (patients_path, admissions_path, icustays_path):
        if not path.is_file():
            sys.exit(f"Missing {path}")

    patients = pd.read_csv(patients_path, usecols=["subject_id", "anchor_age", "dod"])
    admissions = pd.read_csv(admissions_path, usecols=["subject_id", "hadm_id", "admittime", "dischtime"])
    icustays = pd.read_csv(
        icustays_path,
        usecols=["subject_id", "hadm_id", "stay_id", "first_careunit", "last_careunit", "intime", "outtime", "los"],
    )
    icustays.intime = pd.to_datetime(icustays.intime)
    icustays.outtime = pd.to_datetime(icustays.outtime)

    print(f"[icustays] raw: {len(icustays)} stays, {icustays.subject_id.nunique()} subjects")

    stays = icustays[icustays.first_careunit == icustays.last_careunit].copy()
    print(f"[filter] no mid-stay ICU transfer: {len(stays)} stays")

    stays = stays.merge(admissions, how="inner", on=["subject_id", "hadm_id"])
    stays = stays.merge(patients, how="inner", on="subject_id")
    print(f"[filter] merged with admissions+patients: {len(stays)} stays")

    counts = stays.groupby("hadm_id")["stay_id"].transform("count")
    stays = stays[counts == 1].copy()
    print(f"[filter] admissions with exactly one qualifying ICU stay: {len(stays)} stays")

    stays["age"] = stays["anchor_age"]
    stays.loc[stays["age"] < 0, "age"] = 90
    stays = stays[stays["age"] >= 18]
    print(f"[filter] age >= 18: {len(stays)} stays")

    stays = stays[stays.outtime > stays.intime]
    print(f"[filter] outtime > intime: {len(stays)} stays")

    stays["los_hours"] = (stays.outtime - stays.intime).dt.total_seconds() / 3600
    return stays[["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los_hours"]].reset_index(drop=True)


def load_cxr_catalogue():
    meta_path = CXR_DIR / "mimic-cxr-2.0.0-metadata.csv"
    if not meta_path.is_file():
        sys.exit(f"Missing {meta_path}. Run scripts/download_cxr_metadata.py first.")

    meta = pd.read_csv(meta_path)
    print(f"[cxr] raw metadata: {len(meta)} images")

    meta = meta[meta.ViewPosition == "AP"].copy()
    print(f"[cxr] AP view only (portable films -- ICU patients too ill to stand): {len(meta)} images")

    meta["StudyTime"] = meta["StudyTime"].apply(lambda x: f"{int(float(x)):06}")
    meta["StudyDateTime"] = pd.to_datetime(
        meta["StudyDate"].astype(str) + " " + meta["StudyTime"].astype(str),
        format="%Y%m%d %H%M%S",
    )
    return meta[["dicom_id", "subject_id", "study_id", "StudyDateTime"]]


def build_task_list(cxr, cohort, end_time_col, label):
    merged = cxr.merge(cohort, how="inner", on="subject_id")
    windowed = merged[(merged.StudyDateTime >= merged.intime) & (merged.StudyDateTime <= merged[end_time_col])]
    print(f"[{label}] studies inside window: {len(windowed)}")

    windowed = windowed.sort_values("StudyDateTime")
    latest = windowed.groupby("stay_id", as_index=False).last()
    print(f"[{label}] after one-per-stay (most recent kept): {len(latest)}")
    return latest[["dicom_id", "subject_id", "study_id", "stay_id", "StudyDateTime"]]


def main():
    cohort = load_icu_cohort()
    cxr = load_cxr_catalogue()
    print()

    task_a_cohort = cohort[cohort.los_hours >= TASK_A_HOURS].copy()
    print(f"[cohort] Task A eligible (los >= {TASK_A_HOURS}h): {len(task_a_cohort)} stays")
    task_a_cohort["end_time"] = task_a_cohort.intime + pd.Timedelta(hours=TASK_A_HOURS)

    task_b_cohort = cohort.copy()
    task_b_cohort["end_time"] = task_b_cohort.outtime
    print()

    task_a = build_task_list(cxr, task_a_cohort, "end_time", "Task A (48h)")
    print()
    task_b = build_task_list(cxr, task_b_cohort, "end_time", "Task B (whole stay)")

    union_df = pd.concat([task_a.assign(task="A"), task_b.assign(task="B")], ignore_index=True)
    tasks_by_dicom = union_df.groupby("dicom_id")["task"].apply(lambda s: "+".join(sorted(set(s))))
    download_list = union_df.drop_duplicates("dicom_id").drop(columns=["task"]).copy()
    download_list["needed_for"] = download_list["dicom_id"].map(tasks_by_dicom)

    print()
    print("=" * 60)
    print(f"Task A candidate studies: {len(task_a)}  (expect roughly 4,000-6,000)")
    print(f"Task B candidate studies: {len(task_b)}  (expect roughly 10,000)")
    print(f"Union (to download):      {len(download_list)}")
    print("=" * 60)

    if len(download_list) > 50000:
        print("[WARN] union is far larger than expected -- check the AP filter and one-per-stay rule.")
    elif len(download_list) < 500:
        print("[WARN] union is far smaller than expected -- check StudyDateTime parsing / cohort merges.")
    else:
        print("Counts look in range. Review data/cxr_download_lists/ before downloading images.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    task_a.to_csv(OUT_DIR / "task_a_studies.csv", index=False)
    task_b.to_csv(OUT_DIR / "task_b_studies.csv", index=False)
    download_list.to_csv(OUT_DIR / "cxr_download_list.csv", index=False)

    print()
    print(f"Wrote {OUT_DIR / 'cxr_download_list.csv'} ({len(download_list)} rows)")
    print("No images were downloaded.")


if __name__ == "__main__":
    main()
