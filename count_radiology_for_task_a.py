import pandas as pd

radiology = pd.read_csv("data/mimic-iv-note/radiology_sections.csv")
task_a_stays = pd.read_csv("data/mimic-iv-hosp/all_stays_task_a_48h.csv")

task_a_hadm_ids = set(task_a_stays["hadm_id"])
matching_reports = radiology[radiology["hadm_id"].isin(task_a_hadm_ids)]

print(f"Total radiology reports: {len(radiology):,}")
print(f"Radiology reports for Task A patients: {len(matching_reports):,}")