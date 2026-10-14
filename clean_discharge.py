import pandas as pd

# Step A: read just the light ID columns first (fast, avoids loading 3.5GB of text)
id_cols = pd.read_csv("data/mimic-iv-note/discharge.csv", usecols=['note_id', 'hadm_id'])
print(f"Total discharge notes: {len(id_cols):,}")
print(f"Unique hospital stays (hadm_id): {id_cols['hadm_id'].nunique():,}")

# Keep the LAST note per hospital stay (most complete/final version)
keep_ids = set(id_cols.drop_duplicates(subset='hadm_id', keep='last')['note_id'])
print(f"Keeping {len(keep_ids):,} notes after dedup")

# Step B: now read the full file in chunks, keeping only the rows we decided to keep
clean_rows = []
for chunk in pd.read_csv("data/mimic-iv-note/discharge.csv", chunksize=100_000):
    filtered = chunk[chunk['note_id'].isin(keep_ids)]
    clean_rows.append(filtered)

discharge_clean = pd.concat(clean_rows, ignore_index=True)
discharge_clean.to_csv("data/mimic-iv-note/discharge_clean.csv", index=False)
print(f"\nSaved {len(discharge_clean):,} clean discharge notes to data/mimic-iv-note/discharge_clean.csv")