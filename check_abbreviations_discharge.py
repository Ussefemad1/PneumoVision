import pandas as pd
import re
from collections import Counter
from abbreviations import ABBREVIATIONS, AMBIGUOUS_ABBREVIATIONS

discharge = pd.read_csv("data/mimic-iv-note/discharge_sections.csv")

# Combine all the discharge section columns into one text blob per note
text_cols = [c for c in discharge.columns if c not in ("note_id",)]
dn_text = discharge[text_cols].fillna("").agg(" ".join, axis=1)

all_known = list(ABBREVIATIONS.keys()) + list(AMBIGUOUS_ABBREVIATIONS.keys())
counts = Counter()

sample = dn_text.sample(n=50000, random_state=42)
for abbr in all_known:
    pattern = rf"\b{re.escape(abbr)}\b"
    hits = sample.str.contains(pattern, regex=True, na=False).sum()
    if hits > 0:
        counts[abbr] = hits

print("=== Frequency in 50,000 sampled discharge sections ===")
for abbr, count in counts.most_common():
    tag = " (AMBIGUOUS)" if abbr in AMBIGUOUS_ABBREVIATIONS else ""
    print(f"{abbr}: {count}{tag}")