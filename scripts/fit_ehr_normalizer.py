"""Fit the EHR normalizer (per-channel mean/std) on the TRAINING split only.

The repo ships normalizers fitted by the MedPatch authors on their own cohort, and
fusion_main.py loads the phenotyping one for every task. This refits them on our
MIMIC-IV v3.1 train_listfile.csv, using the exact discretizer settings and
time bounds that fusion_main.py / ehr_dataset.py use at training time, so the
statistics match what the model actually sees. Validation and test are never read.

Usage (from repo root):
    python scripts/fit_ehr_normalizer.py --task phenotyping
        --train_listfile handoff_output/train_listfile.csv --train_dir data/root/phenotype_labels/train
    python scripts/fit_ehr_normalizer.py --task in-hospital-mortality
        --train_listfile data/in-hospital-mortality/train_listfile.csv --train_dir data/in-hospital-mortality/train
"""
import argparse
import os
import sys

import numpy as np

MEDPATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'medpatch')
sys.path.insert(0, MEDPATCH)
from ehr_utils.preprocessing import Discretizer, Normalizer  # noqa: E402

PREFIX = {'phenotyping': 'ph', 'in-hospital-mortality': 'ihm'}
IHM_PERIOD_LENGTH = 48.0  # ehr_dataset.EHRdataset default, used when period_length == 0


def read_timeseries(path):
    with open(path) as f:
        header = f.readline().strip().split(',')
        assert header[0] == 'Hours'
        rows = [np.array(line.strip().split(',')) for line in f]
    return np.stack(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--train_listfile', required=True, help='train_listfile.csv (NOT val or test)')
    parser.add_argument('--train_dir', required=True, help='Folder holding the per-stay timeseries CSVs')
    parser.add_argument('--task', required=True, choices=sorted(PREFIX))
    parser.add_argument('--timestep', type=float, default=1.0)
    parser.add_argument('--output_dir', default=os.path.join(MEDPATCH, 'normalizers'))
    args = parser.parse_args()

    with open(args.train_listfile) as f:
        f.readline()
        rows = [line.strip().split(',') for line in f if line.strip()]

    # Same settings as fusion_main.py
    discretizer = Discretizer(timestep=args.timestep, store_masks=True,
                              impute_strategy='previous', start_time='zero',
                              config_path=os.path.join(MEDPATCH, 'ehr_utils/resources/discretizer_config.json'))
    header = discretizer.transform(read_timeseries(os.path.join(args.train_dir, rows[0][0])))[1].split(',')
    normalizer = Normalizer(fields=[i for i, h in enumerate(header) if '->' not in h])

    for i, (stay, period_length) in enumerate((r[0], float(r[1])) for r in rows):
        end = period_length if period_length > 0.0 else IHM_PERIOD_LENGTH
        X = read_timeseries(os.path.join(args.train_dir, stay))
        X = X[X[:, 0].astype(float) <= end + 1e-6]  # ehr_dataset reads up to the time bound only
        if len(X) == 0:
            continue
        normalizer._feed_data(discretizer.transform(X, end=end)[0])
        if i % 2000 == 0:
            print(f'{i} / {len(rows)}', flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir,
                       f'{PREFIX[args.task]}_ts{args.timestep}.input_str_previous.start_time_zero.mimic4v31_train.normalizer')
    normalizer._save_params(out)
    print(f'Fitted on {len(rows)} training stays -> {out}')


if __name__ == '__main__':
    main()
