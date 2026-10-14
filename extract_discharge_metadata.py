import pandas as pd

# only read the 4 small columns we actually need - fast, low memory
metadata = pd.read_csv(
    "data/mimic-iv-note/discharge.csv",
    usecols=["note_id", "subject_id", "hadm_id", "charttime"]
)

metadata.to_csv("data/mimic-iv-note/discharge_metadata.csv", index=False)
print(f"Saved {len(metadata):,} rows to discharge_metadata.csv")