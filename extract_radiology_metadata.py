import pandas as pd

metadata = pd.read_csv(
    "data/mimic-iv-note/radiology.csv",
    usecols=["note_id", "subject_id", "hadm_id", "charttime"]
)

metadata.to_csv("data/mimic-iv-note/radiology_metadata.csv", index=False)
print(f"Saved {len(metadata):,} rows to radiology_metadata.csv")