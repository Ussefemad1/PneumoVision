"""Demographics + stay-table comparison. Aggregates only."""
import os, pandas as pd
R = r"C:\Users\Asus\PneumoVision\MedPatch\mimic4extract"
C = r"C:\Users\Asus\PneumoVision\caroll's work"
M = r"C:\Users\Asus\PneumoVision\data\mimic-iv-hosp"
icu = pd.read_csv(os.path.join(M, "icustays.csv"), usecols=["subject_id", "stay_id", "intime"])
print("your icustays.csv: stays", len(icu), " subjects", icu.subject_id.nunique(),
      " intime year range", icu.intime.str[:4].min(), "-", icu.intime.str[:4].max(), "(shifted years)")
mine = pd.read_csv(os.path.join(M, "all_stays.csv")); car = pd.read_csv(os.path.join(C, "all_stays.csv"))
print("your all_stays   :", mine.shape, " Caroll all_stays:", car.shape)
print("cols only in yours:", sorted(set(mine.columns) - set(car.columns)), " only in Caroll's:", sorted(set(car.columns) - set(mine.columns)))
print("stay_id overlap:", len(set(mine.stay_id) & set(car.stay_id)), " only yours:", len(set(mine.stay_id) - set(car.stay_id)), " only Caroll:", len(set(car.stay_id) - set(mine.stay_id)))

known = set(pd.read_csv(os.path.join(R, "mimic3benchmark/resources/testset_iv.csv"), header=None)[0])
ph = pd.concat([pd.read_csv(os.path.join(C, "phenotyping", f"{s}_listfile.csv"), usecols=["stay", "stay_id"]) for s in ["train", "val", "test"]])
ph["subject_id"] = ph.stay.str.split('_').str[0].astype(int)
agecol = "age" if "age" in car.columns else "anchor_age"
sexcol = "gender" if "gender" in car.columns else None
d = ph.merge(car[["stay_id", agecol, sexcol]], on="stay_id", how="left")
print("phenotyping rows unmatched in all_stays:", d[agecol].isna().sum())
bins = [-1, 20, 40, 60, 80, 200]; labels = ["0-20", "21-40", "41-60", "61-80", "80+"]
for name, sub in [("ALL Caroll rows", d), ("restricted to repo split subjects", d[d.subject_id.isin(known)])]:
    print(f"\n{name}: n={len(sub)}")
    print(" age buckets:", pd.cut(sub[agecol], bins, labels=labels).value_counts().reindex(labels).to_dict())
    print(" sex:", sub[sexcol].value_counts().to_dict())
