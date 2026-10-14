import pandas as pd
import re
from abbreviations import ABBREVIATIONS

# Build ONE combined regex instead of 40+ separate ones — much faster
pattern = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in ABBREVIATIONS.keys()) + r")\b"
)
cva_tender = re.compile(r"\bCVA\b(?=\s+tenderness)")
cva_other = re.compile(r"\bCVA\b")

def expand_fast(text):
    if not isinstance(text, str):
        return text
    text = cva_tender.sub("costovertebral angle", text)
    text = cva_other.sub("cerebrovascular accident (stroke)", text)
    return pattern.sub(lambda m: ABBREVIATIONS[m.group(0)], text)

discharge = pd.read_csv("data/mimic-iv-note/discharge_sections.csv")
text_cols = [c for c in discharge.columns if c != "note_id"]

print(f"Starting discharge expansion: {len(discharge):,} rows, {len(text_cols)} columns")
for i, col in enumerate(text_cols):
    discharge[col + "_expanded"] = discharge[col].apply(expand_fast)
    print(f"  Column {i+1}/{len(text_cols)} done: {col}")

discharge.to_csv("data/mimic-iv-note/discharge_sections_expanded.csv", index=False)
print(f"Discharge done: {len(discharge):,} rows")