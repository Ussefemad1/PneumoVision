"""Aggregate-only audit of Caroll's EHR outputs. Prints counts/rates/columns only."""
import os, re, glob
import pandas as pd

C = r"C:\Users\Asus\PneumoVision\caroll's work"
pid = lambda s: s.str.split('_').str[0].astype(int)

for task in ["phenotyping", "in-hospital-mortality", "pneumonia-48h"]:
    print(f"\n========== {task}")
    sets = {}
    for sp in ["train", "val", "test"]:
        df = pd.read_csv(os.path.join(C, task, f"{sp}_listfile.csv"))
        sets[sp] = df
        if sp == "train":
            print("n_columns:", df.shape[1])
            print("columns:", list(df.columns)[:3], "...", list(df.columns)[-3:] if df.shape[1] > 6 else "")
        print(f"{sp:5s} rows={len(df):6d} unique_subjects={pid(df.iloc[:,0]).nunique()}")
    tot = sum(len(d) for d in sets.values())
    print("total rows:", tot, " proportions:", {k: round(len(v)/tot, 3) for k, v in sets.items()})
    subj = {k: set(pid(v.iloc[:,0])) for k, v in sets.items()}
    print("subject overlaps  train&val:", len(subj['train']&subj['val']),
          " train&test:", len(subj['train']&subj['test']), " val&test:", len(subj['val']&subj['test']))
    lab = [c for c in sets['train'].columns if c not in ('stay', 'period_length', 'stay_id')]
    if task == "phenotyping":
        pn = [c for c in lab if 'neumonia' in c]
        print("n_label_cols:", len(lab), " pneumonia col:", pn)
        for sp, df in sets.items():
            print(f"  pneumonia prevalence {sp}: {df[pn[0]].mean()*100:.2f}%  (n_pos={int(df[pn[0]].sum())})")
        allp = pd.concat(sets.values())
        prev = (allp[lab].mean()*100).round(1).sort_values(ascending=False)
        print("  all-label prevalence %:\n", prev.to_string())
    else:
        for sp, df in sets.items():
            y = df[lab[-1]]
            print(f"  label '{lab[-1]}' rate {sp}: {y.mean()*100:.2f}%")
    # timeseries files present on disk
    for sub in glob.glob(os.path.join(C, task, "**", "train"), recursive=True) + glob.glob(os.path.join(C, task, "**", "test"), recursive=True):
        print("  dir", os.path.relpath(sub, C), "files:", len(os.listdir(sub)))
