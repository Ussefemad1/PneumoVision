from __future__ import absolute_import
from __future__ import print_function

# Task A: pneumonia prediction 48 hours after ICU admission.
# Copy of create_in_hospital_mortality.py with the 48-hour window and the ">= 48h stay" filter
# unchanged; the mortality label is replaced by the pneumonia label, computed exactly as in
# create_phenotyping.py (the stay's diagnoses.csv mapped through the phenotype yaml).

import os
import argparse
import pandas as pd
import yaml
import random
random.seed(49297)
from tqdm import tqdm

PNEUMONIA = 'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)'


def process_partition(args, code_to_group, partition, eps=1e-6, n_hours=48):
    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    xy_pairs = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    for patient in tqdm(patients, desc='Iterating over patients in {}'.format(partition)):
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))

        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))

                # empty label file
                if label_df.shape[0] == 0:
                    continue
                icustay = label_df['Icustay'].iloc[0]

                los = 24.0 * label_df.iloc[0]['Length of Stay']  # in hours
                if pd.isnull(los):
                    print("\n\t(length of stay is missing)", patient, ts_filename)
                    continue

                if los < n_hours - eps:
                    continue

                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]

                ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                            if -eps < t < n_hours + eps]

                # no measurements in ICU
                if len(ts_lines) == 0:
                    print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue

                # pneumonia label (label block from create_phenotyping.py, pneumonia entry only)
                diagnoses_df = pd.read_csv(os.path.join(patient_folder, "diagnoses.csv"),
                                           dtype={"icd_code": str})
                diagnoses_df = diagnoses_df[diagnoses_df.stay_id == icustay]
                pneumonia = int(any(code_to_group.get(code) == PNEUMONIA for code in diagnoses_df.icd_code))

                output_ts_filename = patient + "_" + ts_filename
                with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        outfile.write(line)

                xy_pairs.append((output_ts_filename, icustay, pneumonia))

    print("Number of created samples:", len(xy_pairs))
    if partition == "train":
        random.shuffle(xy_pairs)
    if partition == "test":
        xy_pairs = sorted(xy_pairs)

    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('stay,period_length,stay_id,y_true\n')
        for (x, icustay, y) in xy_pairs:
            listfile.write('{},0,{},{:d}\n'.format(x, icustay, y))


def main():
    parser = argparse.ArgumentParser(description="Create data for 48-hour pneumonia prediction task (Task A).")
    parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    parser.add_argument('--phenotype_definitions', '-p', type=str,
                        default=os.path.join(os.path.dirname(__file__), '../resources/icd_9_10_definitions_2.yaml'),
                        help='YAML file with phenotype definitions.')
    args, _ = parser.parse_known_args()

    with open(args.phenotype_definitions) as definitions_file:
        definitions = yaml.load(definitions_file, Loader=yaml.FullLoader)
    code_to_group = {code: group for group in definitions for code in definitions[group]['codes']}

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_partition(args, code_to_group, "test")
    process_partition(args, code_to_group, "train")


if __name__ == '__main__':
    main()
