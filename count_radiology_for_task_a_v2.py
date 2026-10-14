import pandas as pd

# your chest-only reports (has note_id)
chest_only = pd.read_csv("data/mimic-iv-note/radiology_sections.csv")
chest_note_ids = set(chest_only["note_id"])

# metadata with charttime (currently includes ALL report types)
metadata = pd.read_csv("data/mimic-iv-note/radiology_metadata.csv")
metadata["charttime"] = pd.to_datetime(metadata["charttime"])

# restrict to chest-only reports first
metadata = metadata[metadata["note_id"].isin(chest_note_ids)]

# Task A cohort
task_a_stays = pd.read_csv("data/mimic-iv-hosp/all_stays_task_a_48h.csv")
task_a_stays["intime"] = pd.to_datetime(task_a_stays["intime"])

merged = metadata.merge(task_a_stays[["hadm_id", "intime"]], on="hadm_id", how="inner")

window_end = merged["intime"] + pd.Timedelta(hours=48)
in_window = merged[(merged["charttime"] >= merged["intime"]) & (merged["charttime"] <= window_end)]

print(f"Chest-only radiology reports for Task A patients (all time): {len(merged):,}")
print(f"Chest-only radiology reports for Task A patients (within 48h): {len(in_window):,}")