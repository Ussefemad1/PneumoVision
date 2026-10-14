import pandas as pd
import re

DISCHARGE_HEADERS = [
    "Chief Complaint", "Major Surgical or Invasive Procedure", "History of Present Illness",
    "Past Medical History", "Social History", "Family History", "Physical Exam",
    "Pertinent Results", "Brief Hospital Course", "Medications on Admission",
    "Discharge Medications", "Discharge Disposition", "Discharge Diagnosis",
    "Discharge Condition", "Discharge Instructions", "Followup Instructions"
]

def split_sections(text, headers):
    if not isinstance(text, str):
        return {}
    # FIX: deidentification sometimes redacts "Discharge" itself right before "Diagnosis"
    text = re.sub(r"___\s+Diagnosis:", "Discharge Diagnosis:", text)

    pattern = r"(?:^|\n)\s*(" + "|".join(re.escape(h) for h in headers) + r")\s*:\s*"
    parts = re.split(pattern, text)
    sections = {}
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        content = parts[i+1].strip() if i+1 < len(parts) else ""
        sections[header] = content
    return sections

discharge = pd.read_csv("data/mimic-iv-note/discharge_clean.csv")
results = []
missing_diagnosis = 0

for idx, row in discharge.iterrows():
    sections = split_sections(row['text'], DISCHARGE_HEADERS)
    if "Discharge Diagnosis" not in sections:
        missing_diagnosis += 1
    results.append(sections)

print(f"Total notes: {len(discharge):,}")
print(f"Notes missing 'Discharge Diagnosis' section: {missing_diagnosis:,} ({100*missing_diagnosis/len(discharge):.1f}%)")

discharge_sections_df = pd.DataFrame(results)
discharge_sections_df['note_id'] = discharge['note_id'].values
discharge_sections_df.to_csv("data/mimic-iv-note/discharge_sections.csv", index=False)
print("Saved updated discharge_sections.csv")