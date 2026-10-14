"""
Answers Norhan's question 3: for chest-filtered vs unfiltered radiology
reports, how many tokens survive BioBERT's 512-token cap?

Run this on your own machine, where you have the full files:
  - radiology.csv (or radiology_sections_expanded.csv / radiology_sections.csv
    - whichever one has columns: subject_id, hadm_id, charttime, text)
  - all_stays_task_a_48h.csv (the 36,648-stay Task A cohort)

Usage:
    pip install pandas transformers --break-system-packages
    python measure_rr_truncation.py

Edit the two paths below to match where your files actually are.
"""

import pandas as pd

RADIOLOGY_PATH = "data/mimic-iv-note/radiology.csv"          # <-- edit
STAYS_PATH = "data/mimic-iv-hosp/all_stays_task_a_48h.csv"    # <-- edit
TEXT_COLUMN = "text"                                          # <-- edit if different (e.g. "findings_impression")

N_HOURS = 48

# ---------------------------------------------------------------------------
# 1. Load and window the radiology reports: keep only reports whose charttime
#    falls in [intime, intime + 48h] for a Task A stay. Matches the repo's
#    DataFusion.py logic (intime-based window, not admittime).
# ---------------------------------------------------------------------------
stays = pd.read_csv(STAYS_PATH, usecols=["subject_id", "hadm_id", "stay_id", "intime"])
stays["intime"] = pd.to_datetime(stays["intime"])
stays["window_end"] = stays["intime"] + pd.DateOffset(hours=N_HOURS)

rad = pd.read_csv(RADIOLOGY_PATH, usecols=["subject_id", "hadm_id", "charttime", TEXT_COLUMN])
rad["charttime"] = pd.to_datetime(rad["charttime"])

merged = rad.merge(stays, on=["subject_id", "hadm_id"], how="inner")
in_window = merged[(merged["charttime"] >= merged["intime"]) & (merged["charttime"] <= merged["window_end"])]

print(f"Reports in 48h window (all modalities, unfiltered): {len(in_window)}")

# ---------------------------------------------------------------------------
# 2. Build the two versions: unfiltered (all modalities) and chest-only
#    (matches your original text-contains-"chest" filter).
# ---------------------------------------------------------------------------
unfiltered = in_window.copy()
chest_only = in_window[in_window[TEXT_COLUMN].str.contains("chest", case=False, na=False)].copy()

print(f"Reports in window, chest-only: {len(chest_only)}")

def concat_per_stay(df):
    # Matches DataFusion.py: groupby + ' '.join, no sort beforehand (arbitrary order).
    return (
        df.groupby("stay_id")[TEXT_COLUMN]
        .apply(lambda texts: " ".join(texts))
        .reset_index(name="radiology_text")
    )

unfiltered_per_stay = concat_per_stay(unfiltered)
chest_per_stay = concat_per_stay(chest_only)

print(f"Stays with >=1 unfiltered report: {len(unfiltered_per_stay)}")
print(f"Stays with >=1 chest report:      {len(chest_per_stay)}")

# ---------------------------------------------------------------------------
# 3. Tokenize with BioBERT and report the distribution vs the 512 cap.
# ---------------------------------------------------------------------------
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")

def token_stats(per_stay_df, label):
    lengths = per_stay_df["radiology_text"].apply(
        lambda t: len(tokenizer.encode(t, add_special_tokens=True))
    )
    median = lengths.median()
    p90 = lengths.quantile(0.90)
    over_512 = (lengths > 512).mean() * 100
    print(f"\n{label}")
    print(f"  stays measured   : {len(lengths)}")
    print(f"  median tokens    : {median:.0f}")
    print(f"  p90 tokens       : {p90:.0f}")
    print(f"  % stays > 512    : {over_512:.1f}%")
    return median, p90, over_512

token_stats(chest_per_stay, "CHEST-FILTERED")
token_stats(unfiltered_per_stay, "UNFILTERED")