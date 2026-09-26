"""Sanity-check cxr_download_list.csv / task_a_studies.csv before downloading
any images. Six checks:

  1. every row is ViewPosition == AP (verified indirectly: the list was
     already built from an AP-only catalogue, so this re-derives it from
     the raw metadata to catch any regression)
  2. StudyDateTime falls after intime and before outtime -- MIMIC-IV
     date-shifts every patient's timeline by a random per-patient offset,
     so absolute years are meaningless and NOT checked; only the
     relationship to that patient's own intime/outtime matters
  3. times of day are spread across the clock, not all sitting at
     00:00:00 (which would indicate silent time-parsing failure)
  4. every Task A study falls within [0, 48] hours of intime -- anything
     outside that range is either pre-admission noise or label leakage
  5. exactly one row per stay_id in the Task A file
  6. the union arithmetic is consistent (|A| + |B| - |A ^ B| == |union|)

Also exports the Task A hours-from-admission values as a CSV -- plot it
in Colab (matplotlib's native extension is blocked by a Windows
Application Control policy on this machine, so rendering happens there
instead, not here).
"""
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
LIST_DIR = REPO_ROOT / "data" / "cxr_download_lists"
CXR_DIR = REPO_ROOT / "data" / "mimic-cxr-jpg" / "2.0.0"


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail else ""))
    return condition


def main():
    task_a = pd.read_csv(LIST_DIR / "task_a_studies.csv", parse_dates=["StudyDateTime"])
    task_b = pd.read_csv(LIST_DIR / "task_b_studies.csv", parse_dates=["StudyDateTime"])
    union = pd.read_csv(LIST_DIR / "cxr_download_list.csv")
    meta = pd.read_csv(CXR_DIR / "mimic-cxr-2.0.0-metadata.csv", usecols=["dicom_id", "ViewPosition"])

    ok = True

    # 1. AP-only
    merged_view = task_a.merge(meta, on="dicom_id", how="left")
    non_ap = merged_view[merged_view.ViewPosition != "AP"]
    ok &= check("Task A: all rows are ViewPosition == AP", len(non_ap) == 0,
                f"{len(non_ap)} non-AP rows found" if len(non_ap) else "")

    # need intime/outtime -- task_a_studies.csv doesn't carry them, so
    # rejoin from the same ICU cohort source used to build the list
    icu = pd.read_csv(
        REPO_ROOT / "data" / "mimic-iv" / "icu" / "icustays.csv.gz",
        usecols=["stay_id", "intime", "outtime"],
        parse_dates=["intime", "outtime"],
    )
    with_intime = task_a.merge(icu, on="stay_id", how="left")
    with_intime["hours_from_admission"] = (
        with_intime.StudyDateTime - with_intime.intime
    ).dt.total_seconds() / 3600

    # 2. StudyDateTime within [intime, outtime] -- absolute year is not
    # checked, since MIMIC-IV shifts each patient's timeline independently
    before_intime = (with_intime.StudyDateTime < with_intime.intime).sum()
    after_outtime = (with_intime.StudyDateTime > with_intime.outtime).sum()
    ok &= check(
        "Task A: StudyDateTime within [intime, outtime]",
        before_intime == 0 and after_outtime == 0,
        f"{before_intime} before intime, {after_outtime} after outtime",
    )

    # 3. times of day are spread out, not a wall of 00:00:00
    midnight_frac = (task_a.StudyDateTime.dt.time == pd.Timestamp("00:00:00").time()).mean()
    ok &= check(
        "Task A: times of day are spread across the clock",
        midnight_frac < 0.5,
        f"{midnight_frac:.1%} of studies sit exactly at 00:00:00",
    )

    # 4. hours-from-admission bounds
    below_zero = (with_intime.hours_from_admission < 0).sum()
    above_48 = (with_intime.hours_from_admission > 48).sum()
    ok &= check(
        "Task A: every study within [0, 48] hours of intime",
        below_zero == 0 and above_48 == 0,
        f"{below_zero} negative, {above_48} above 48h",
    )

    # 5. one row per stay_id
    dup_stays = task_a.groupby("stay_id").size()
    dup_stays = dup_stays[dup_stays > 1]
    ok &= check("Task A: exactly one row per stay_id", len(dup_stays) == 0,
                f"{len(dup_stays)} stay_ids with multiple rows" if len(dup_stays) else "")

    # 6. union arithmetic
    overlap = len(set(task_a.dicom_id) & set(task_b.dicom_id))
    expected_union = len(task_a) + len(task_b) - overlap
    ok &= check(
        "Union arithmetic consistent (|A|+|B|-overlap == |union|)",
        expected_union == len(union),
        f"{len(task_a)} + {len(task_b)} - {overlap} = {expected_union}, union file has {len(union)}",
    )

    print()
    print("=" * 60)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED -- do not download yet")
    print("=" * 60)

    # export hours-from-admission for plotting in Colab -- matplotlib's
    # native extension is blocked by a Windows Application Control policy
    # on this machine, so rendering happens there instead, not here
    out_path = REPO_ROOT / "results" / "task_a_hours_from_admission.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with_intime[["dicom_id", "stay_id", "hours_from_admission"]].to_csv(out_path, index=False)
    print(f"\nExported {out_path} -- plot hours_from_admission as a histogram in Colab.")
    print("Compare against the paper's shape: a spike in the first few hours (admission")
    print("film), then a second rise around 24-40 hours.")

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
