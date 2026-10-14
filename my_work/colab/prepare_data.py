"""Build the EHR folder layout fusion_main.py expects, from Caroll's phenotyping files.

Caroll's files are only read, never changed. Output (all new files / links):
  <out>/root/all_stays.csv
  <out>/phenotyping/{train,val,test}_listfile.csv   rows restricted to subjects in the repo's split list
  <out>/phenotyping/train  -> Caroll's timeseries train folder (link)
  <out>/phenotyping/test   -> Caroll's timeseries test folder (link)

Why the restriction: Caroll built from MIMIC-IV v3.x. The repo's split list (testset_iv.csv)
was made for the older version the paper used; subjects missing from it were all put in train.
Keeping only listed subjects gives counts within ~1% of the paper (42,328 / 4,756 / 11,845).

Prints only counts and rates (no patient rows).
"""
import argparse
import os
import shutil

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--all_stays', required=True, help="Caroll's all_stays.csv")
ap.add_argument('--listfile_dir', required=True, help="folder with Caroll's {train,val,test}_listfile.csv")
ap.add_argument('--timeseries_dir', required=True, help='folder that contains the train/ and test/ timeseries folders')
ap.add_argument('--repo', required=True, help='path to the medpatch code folder')
ap.add_argument('--out', required=True, help='new ehr_data_dir to create')
ap.add_argument('--task', default='phenotyping', choices=['phenotyping', 'in-hospital-mortality'])
ap.add_argument('--small', type=int, default=0, help='if >0, keep only this many rows per split (sanity run)')
args = ap.parse_args()

PNEUMONIA = 'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)'

testset = pd.read_csv(os.path.join(args.repo, 'mimic4extract/mimic3benchmark/resources/testset_iv.csv'),
                      header=None, names=['subject_id', 'is_test'])
known_subjects = set(testset.subject_id)

os.makedirs(os.path.join(args.out, 'root'), exist_ok=True)
os.makedirs(os.path.join(args.out, args.task), exist_ok=True)
shutil.copy(args.all_stays, os.path.join(args.out, 'root', 'all_stays.csv'))

for split in ['train', 'val', 'test']:
    df = pd.read_csv(os.path.join(args.listfile_dir, f'{split}_listfile.csv'))
    subj = df.stay.str.split('_').str[0].astype(int)
    kept = df[subj.isin(known_subjects)]
    print(f'{split:5s}: {len(df):6d} rows -> kept {len(kept):6d} (removed {len(df) - len(kept)} new-version rows); '
          + (f'pneumonia {kept[PNEUMONIA].mean() * 100:.2f}%' if args.task == 'phenotyping' else f'mortality {kept.y_true.mean() * 100:.2f}%'))
    if args.small:
        kept = kept.sample(n=min(args.small, len(kept)), random_state=0)
        print(f'       sanity subset: {len(kept)} rows')
    kept.to_csv(os.path.join(args.out, args.task, f'{split}_listfile.csv'), index=False)

for split in ['train', 'test']:
    src = os.path.abspath(os.path.join(args.timeseries_dir, split))
    dst = os.path.abspath(os.path.join(args.out, args.task, split))
    if not os.path.isdir(src):
        raise SystemExit(f'timeseries folder not found: {src}')
    if not os.path.exists(dst):
        if os.name == 'nt':  # Windows: directory junction, no admin rights needed
            import _winapi
            _winapi.CreateJunction(src, dst)
        else:
            os.symlink(src, dst)
    print(f'{split} timeseries: {len(os.listdir(dst))} files linked')

print('Expected EHR counts (paper): ' + ('train 42,628 / val 4,802 / test 11,914' if args.task == 'phenotyping'
      else 'train 18,845 / val 2,138 / test 5,243'))
print('EHR data dir ready:', os.path.abspath(args.out))
