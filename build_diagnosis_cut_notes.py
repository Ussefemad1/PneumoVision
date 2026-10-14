import pandas as pd

discharge = pd.read_csv("data/mimic-iv-note/discharge_sections.csv")

# Sections that would leak the diagnosis/answer directly
LEAKAGE_SECTIONS = [
    "Discharge Diagnosis",
    "Discharge Condition",   # sometimes restates diagnosis-adjacent status
]

# Build the "full note" text (all sections concatenated) and the "diagnosis-cut" version
all_section_cols = [c for c in discharge.columns if c != "note_id"]
keep_cols = [c for c in all_section_cols if c not in LEAKAGE_SECTIONS]

def join_sections(row, cols):
    parts = []
    for col in cols:
        val = row[col]
        if isinstance(val, str) and val.strip():
            parts.append(f"{col}: {val}")
    return "\n\n".join(parts)

print("Building full-text and diagnosis-cut versions...")
discharge["full_text"] = discharge.apply(lambda r: join_sections(r, all_section_cols), axis=1)
discharge["diagnosis_cut_text"] = discharge.apply(lambda r: join_sections(r, keep_cols), axis=1)

output = discharge[["note_id", "full_text", "diagnosis_cut_text"]]
output.to_csv("data/mimic-iv-note/discharge_leakage_variants.csv", index=False)

print(f"Saved {len(output):,} notes with both full and diagnosis-cut versions")

# Quick sanity check: confirm the cut version is actually shorter and doesn't contain "Discharge Diagnosis:"
sample = output.sample(3, random_state=1)
for _, row in sample.iterrows():
    print(f"\nNote {row['note_id']}: full={len(row['full_text'])} chars, cut={len(row['diagnosis_cut_text'])} chars")