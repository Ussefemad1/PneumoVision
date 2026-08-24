import pandas as pd

import numpy as np


def create_split(
    ehr_data_dir='/scratch/fs999/shamoutlab/data/mimic-iv-extracted/phenotyping',
    cxr_data_dir='/scratch/fs999/shamoutlab/data/physionet.org/files/mimic-cxr-jpg/2.0.0',
    output_path='.//mimic-cxr-ehr-split.csv',
):
    cxr_splits = pd.read_csv(f'{cxr_data_dir}/mimic-cxr-2.0.0-split.csv')
    print(f'before update {cxr_splits.split.value_counts()}')

    ehr_split_val = pd.read_csv(f'{ehr_data_dir}/val_listfile.csv')
    ehr_split_test = pd.read_csv(f'{ehr_data_dir}/test_listfile.csv')

    val_subject_ids = [stay.split('_')[0] for stay in ehr_split_val.stay.values]
    test_subject_ids = [stay.split('_')[0] for stay in ehr_split_test.stay.values]

    cxr_splits.loc[:, 'split'] = 'train'
    print(cxr_splits.subject_id.dtype)
    cxr_splits.loc[cxr_splits.subject_id.isin(val_subject_ids), 'split'] = 'validate'
    cxr_splits.loc[cxr_splits.subject_id.isin(test_subject_ids), 'split'] = 'test'

    print(f'after update {cxr_splits.split.value_counts()}')

    cxr_splits.to_csv(output_path, index=False)


if __name__ == '__main__':
    create_split()

