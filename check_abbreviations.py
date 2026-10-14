import pandas as pd
import re
from collections import Counter
from abbreviations import ABBREVIATIONS, AMBIGUOUS_ABBREVIATIONS

radiology = pd.read_csv("data/mimic-iv-note/radiology_sections.csv")
discharge = pd.read_csv("data/mimic-iv-note/discharge_sections.csv")

# Combine all text we have into one series to search
rr_text = radiology["FINDINGS"].fillna("") + " " + radiology["IMPRESSION"].fillna("")

all_known = list(ABBREVIATIONS.keys()) + list(AMBIGUOUS_ABBREVIATIONS.keys())
counts = Counter()

sample = rr_text.sample(n=50000, random_state=42)  # sample for speed, not all 939k
for abbr in all_known:
    pattern = rf"\b{re.escape(abbr)}\b"
    hits = sample.str.contains(pattern, regex=True, na=False).sum()
    if hits > 0:
        counts[abbr] = hits

print("=== Frequency in 50,000 sampled radiology sections ===")
for abbr, count in counts.most_common():
    tag = " (AMBIGUOUS)" if abbr in AMBIGUOUS_ABBREVIATIONS else ""
    print(f"{abbr}: {count}{tag}")