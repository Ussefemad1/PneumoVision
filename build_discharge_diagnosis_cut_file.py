import pandas as pd

# small metadata file (subject_id, hadm_id, charttime, note_id)
metadata = pd.read_csv("data/mimic-iv-note/discharge_metadata.csv")

# your leakage file (note_id, full_text, diagnosis_cut_text)
variants = pd.read_csv("data/mimic-iv-note/discharge_leakage_variants.csv")

# combine them: metadata + diagnosis-cut text, shaped like the original discharge.csv
merged = metadata.merge(variants[["note_id", "diagnosis_cut_text"]], on="note_id", how="inner")
merged = merged.rename(columns={"diagnosis_cut_text": "text"})

merged.to_csv("data/mimic-iv-note/discharge_diagnosis_cut.csv", index=False)
print(f"Saved {len(merged):,} rows to discharge_diagnosis_cut.csv")