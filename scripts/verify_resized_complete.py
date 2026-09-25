"""Check that resized/ holds exactly the images the download list calls for.

Run this after the Task B download finishes and resize.py has caught up.

The expected count is 15,181 -- the number of rows in cxr_download_list.csv --
NOT 17,920 (6,683 + 11,237), because 2,739 images satisfy both tasks and are
counted once. Getting 17,920 would mean something duplicated images; getting
fewer than 15,181 means resize.py has not caught up with the download, or some
downloads are still missing.
"""
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CXR_DIR = REPO_ROOT / "data" / "mimic-cxr-jpg" / "2.0.0"
LIST_DIR = REPO_ROOT / "data" / "cxr_download_lists"


def main():
    wanted = set(pd.read_csv(LIST_DIR / "cxr_download_list.csv")["dicom_id"])
    task_a = set(pd.read_csv(LIST_DIR / "task_a_studies.csv")["dicom_id"])
    task_b = set(pd.read_csv(LIST_DIR / "task_b_studies.csv")["dicom_id"])

    originals = {p.stem for p in (CXR_DIR / "files").rglob("*.jpg")}
    resized = {p.stem for p in (CXR_DIR / "resized").glob("*.jpg")}

    print("=" * 62)
    print(f"  expected (cxr_download_list.csv) : {len(wanted):>6}")
    print(f"    task A                         : {len(task_a):>6}")
    print(f"    task B                         : {len(task_b):>6}")
    print(f"    serving both (counted once)    : {len(task_a & task_b):>6}")
    print(f"    naive sum, should NOT match    : {len(task_a) + len(task_b):>6}")
    print("-" * 62)
    print(f"  downloaded (files/)              : {len(originals):>6}")
    print(f"  resized    (resized/)            : {len(resized):>6}")
    print("=" * 62)

    problems = []

    not_downloaded = wanted - originals
    if not_downloaded:
        problems.append(f"{len(not_downloaded)} images on the list are not downloaded yet")

    not_resized = originals - resized
    if not_resized:
        problems.append(f"{len(not_resized)} downloaded images have no resized copy "
                        "(re-run medpatch/resize.py)")

    missing_from_resized = wanted - resized
    if missing_from_resized:
        problems.append(f"{len(missing_from_resized)} wanted images are missing from resized/")

    unexpected = resized - wanted
    if unexpected:
        problems.append(f"{len(unexpected)} images in resized/ are not on the download list")

    print()
    print(f"  task A complete in resized/ : {len(task_a & resized)} / {len(task_a)}")
    print(f"  task B complete in resized/ : {len(task_b & resized)} / {len(task_b)}")
    print()

    if problems:
        for p in problems:
            print(f"  [!] {p}")
        print()
        print("NOT COMPLETE")
        return 1

    print("COMPLETE -- resized/ holds exactly the 15,181 expected images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
