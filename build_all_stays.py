import pandas as pd
import numpy as np

# EDIT THESE THREE PATHS to match your unzipped files
PATIENTS_PATH = "data/mimic-iv-hosp/patients.csv"
ADMISSIONS_PATH = "data/mimic-iv-hosp/admissions.csv"
ICUSTAYS_PATH = "data/mimic-iv-hosp/icustays.csv"
OUTPUT_PATH = "data/mimic-iv-hosp/all_stays.csv"

# --- Load tables ---
patients = pd.read_csv(PATIENTS_PATH)[['subject_id', 'gender', 'anchor_age', 'dod']]
patients['dod'] = pd.to_datetime(patients['dod'])

admits = pd.read_csv(ADMISSIONS_PATH)
# MIMIC-IV newer versions call this column "race" instead of "ethnicity" - handle both
ethnicity_col = 'ethnicity' if 'ethnicity' in admits.columns else 'race'
admits = admits[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', ethnicity_col]]
admits = admits.rename(columns={ethnicity_col: 'ethnicity'})
admits['admittime'] = pd.to_datetime(admits['admittime'])
admits['dischtime'] = pd.to_datetime(admits['dischtime'])
admits['deathtime'] = pd.to_datetime(admits['deathtime'])

stays = pd.read_csv(ICUSTAYS_PATH)
stays['intime'] = pd.to_datetime(stays['intime'])
stays['outtime'] = pd.to_datetime(stays['outtime'])

print(f"START: {stays.stay_id.nunique()} stays, {stays.hadm_id.nunique()} admissions, {stays.subject_id.nunique()} patients")

# --- Step 1: remove ICU stays with ward transfers ---
stays = stays[stays.first_careunit == stays.last_careunit]
stays = stays[['subject_id', 'hadm_id', 'stay_id', 'last_careunit', 'intime', 'outtime', 'los']]
print(f"AFTER REMOVING TRANSFERS: {stays.stay_id.nunique()} stays")

# --- Step 2: merge with admissions and patients ---
stays = stays.merge(admits, how='inner', on=['subject_id', 'hadm_id'])
stays = stays.merge(patients, how='inner', on='subject_id')

# --- Step 3: keep only admissions with exactly 1 ICU stay ---
counts = stays.groupby('hadm_id')['stay_id'].count().reset_index()
keep_hadm = counts[counts.stay_id == 1][['hadm_id']]
stays = stays.merge(keep_hadm, how='inner', on='hadm_id')
print(f"AFTER KEEPING SINGLE-ICU-STAY ADMISSIONS: {stays.stay_id.nunique()} stays")

# --- Step 4: age, mortality flags ---
stays['age'] = stays['anchor_age']
stays.loc[stays.age < 0, 'age'] = 90

mortality = stays.dod.notnull() & (stays.admittime <= stays.dod) & (stays.dischtime >= stays.dod)
mortality = mortality | (stays.deathtime.notnull() & (stays.admittime <= stays.deathtime) & (stays.dischtime >= stays.deathtime))
stays['mortality'] = mortality.astype(int)
stays['mortality_inhospital'] = stays['mortality']

mortality_unit = stays.dod.notnull() & (stays.intime <= stays.dod) & (stays.outtime >= stays.dod)
mortality_unit = mortality_unit | (stays.deathtime.notnull() & (stays.intime <= stays.deathtime) & (stays.outtime >= stays.deathtime))
stays['mortality_inunit'] = mortality_unit.astype(int)

# --- Step 5: filter to age 18+ ---
stays = stays[(stays.age >= 18)]
print(f"AFTER FILTERING AGE 18+: {stays.stay_id.nunique()} stays, {stays.hadm_id.nunique()} admissions, {stays.subject_id.nunique()} patients")

# --- Save ---
stays.to_csv(OUTPUT_PATH, index=False)
print(f"Saved to {OUTPUT_PATH}")