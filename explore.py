import pandas as pd

# nrows=5 means "just peek", don't load all 3.5GB
discharge_peek = pd.read_csv("data/mimic-iv-note/discharge.csv", nrows=5)
radiology_peek = pd.read_csv("data/mimic-iv-note/radiology.csv", nrows=5)

print("=== DISCHARGE COLUMNS ===")
print(discharge_peek.columns.tolist())
print(discharge_peek.head())

print("\n=== RADIOLOGY COLUMNS ===")
print(radiology_peek.columns.tolist())
print(radiology_peek.head())