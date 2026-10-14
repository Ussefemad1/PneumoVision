import pandas as pd

chunksize = 100_000
chest_rows = []
total_seen = 0

for chunk in pd.read_csv("data/mimic-iv-note/radiology.csv", chunksize=chunksize):
    total_seen += len(chunk)
    chest_chunk = chunk[chunk['text'].str.contains('CHEST', case=False, na=False)]
    chest_rows.append(chest_chunk)
    print(f"Processed {total_seen:,} rows so far, kept {len(chest_chunk)} chest reports in this chunk...")

chest_only = pd.concat(chest_rows, ignore_index=True)
print(f"\nTotal radiology reports: {total_seen:,}")
print(f"Total chest reports kept: {len(chest_only):,}")

chest_only.to_csv("data/mimic-iv-note/radiology_chest_clean.csv", index=False)
print("Saved to data/mimic-iv-note/radiology_chest_clean.csv")