"""Check Caroll's EHR splits against the repo's fixed subject lists. Aggregates only."""
import os, pandas as pd
R = r"C:\Users\Asus\PneumoVision\MedPatch\mimic4extract"
C = r"C:\Users\Asus\PneumoVision\caroll's work"
test = pd.read_csv(os.path.join(R, "mimic3benchmark/resources/testset_iv.csv"), header=None, names=["s", "t"])
val = pd.read_csv(os.path.join(R, "mimic3models/resources/valset_iv.csv"), header=None, names=["s", "v"])
print("testset_iv.csv: subjects", len(test), " marked test:", int(test.t.sum()))
print("valset_iv.csv : subjects", len(val), " marked val:", int(val.v.sum()))
known = set(test.s)
T1, V1 = set(test.s[test.t == 1]), set(val.s[val.v == 1])
for task in ["phenotyping", "in-hospital-mortality"]:
    print("\n==", task)
    keep_known = {}
    for sp in ["train", "val", "test"]:
        df = pd.read_csv(os.path.join(C, task, f"{sp}_listfile.csv"))
        s = df.stay.str.split('_').str[0].astype(int)
        in_list = s.isin(known)
        print(f"{sp:5s} rows={len(df):6d}  rows whose subject is NOT in repo testset_iv list: {(~in_list).sum():6d}"
              f"  | in test-list={s.isin(T1).sum()}  in val-list={s.isin(V1).sum()}")
        keep_known[sp] = int(in_list.sum())
    print("rows if restricted to subjects known to the repo's split list:", keep_known, "total", sum(keep_known.values()))
