"""
Task A (in-hospital mortality) cohort filter.

This is NOT part of your phenotyping/leakage cohort (all_stays.csv stays as-is).
This is a SEPARATE file for whoever builds the mortality/EHR-only baseline
(Task A), matching the exact logic in the real repo's
mimic4extract/mimic3benchmark/scripts/create_in_hospital_mortality.py

What it does, exactly matching that script's rule:
  - keep only ICU stays with length-of-stay >= 48 hours
  - the 48-hour clock starts at ICU admission (intime), which is already
    what "los" is measured from in icustays.csv - nothing extra needed there

What it does NOT do (an honest limitation):
  - The real script also checks that there is at least one real vitals/labs
    measurement within the first 48 hours for each patient (it literally
    reads each patient's timeseries file and drops empty windows). That
    check needs the actual EHR timeseries data, which isn't built yet.
    Once that data exists, re-run the official script instead of this one
    for the final, exact numbers - this script gives a close, correct
    stay-level approximation in the meantime.
"""

import pandas as pd

INPUT_PATH = "data/mimic-iv-hosp/all_stays.csv"           # your existing cohort file
OUTPUT_PATH = "data/mimic-iv-hosp/all_stays_task_a_48h.csv"  # new, separate file

N_HOURS = 48
EPS = 1e-6

stays = pd.read_csv(INPUT_PATH)
print(f"START: {len(stays)} stays")

# 'los' in icustays.csv is in DAYS, measured from ICU intime to ICU outtime,
# i.e. the clock already starts at ICU admission - convert to hours to compare
los_hours = stays["los"] * 24.0

before = len(stays)
stays = stays[los_hours >= (N_HOURS - EPS)]
after = len(stays)

print(f"AFTER REQUIRING ICU LOS >= {N_HOURS}h: {after} stays (removed {before - after})")

stays.to_csv(OUTPUT_PATH, index=False)
print(f"Saved to {OUTPUT_PATH}")