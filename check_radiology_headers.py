import pandas as pd
import re

def split_sections(text, headers):
    if not isinstance(text, str):
        return {}
    pattern = r"(?:^|\n)\s*(" + "|".join(re.escape(h) for h in headers) + r")\s*:\s*"
    parts = re.split(pattern, text)
    sections = {}
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        content = parts[i+1].strip() if i+1 < len(parts) else ""
        sections[header] = content
    return sections

RADIOLOGY_HEADERS_STRICT = ["EXAMINATION", "INDICATION", "TECHNIQUE", "COMPARISON", "FINDINGS", "IMPRESSION"]

radiology = pd.read_csv("data/mimic-iv-note/radiology_chest_clean.csv")

findings_list = []
impression_list = []

for idx, row in radiology.iterrows():
    sections = split_sections(row['text'], RADIOLOGY_HEADERS_STRICT)
    findings_list.append(sections.get("FINDINGS", ""))       # empty string if missing
    impression_list.append(sections.get("IMPRESSION", ""))   # empty string if missing

radiology_sections_df = pd.DataFrame({
    "note_id": radiology["note_id"].values,
    "subject_id": radiology["subject_id"].values,
    "hadm_id": radiology["hadm_id"].values,
    "FINDINGS": findings_list,
    "IMPRESSION": impression_list
})

radiology_sections_df.to_csv("data/mimic-iv-note/radiology_sections.csv", index=False)
print(f"Saved {len(radiology_sections_df):,} radiology reports with split sections")
print(f"Reports with empty FINDINGS: {(radiology_sections_df['FINDINGS'] == '').sum():,}")
print(f"Reports with empty IMPRESSION: {(radiology_sections_df['IMPRESSION'] == '').sum():,}")

missing_findings_only = radiology_sections_df[
    (radiology_sections_df['FINDINGS'] == '') & (radiology_sections_df['IMPRESSION'] != '')
]
print(f"Missing FINDINGS but HAS impression: {len(missing_findings_only):,}")
print(missing_findings_only.iloc[0]['IMPRESSION'][:300])