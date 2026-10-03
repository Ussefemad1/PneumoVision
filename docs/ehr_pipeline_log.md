# EHR pipeline log (Caroll)

Running record of numbers and decisions for the thesis Data, Patient selection and Label definition chapters.
Aggregate counts only; no patient-level data belongs in this file.

## Data source

- MIMIC-IV v3.1. `patients.csv` and `admissions.csv` copied into `core/` (roadmap B3).
- v3.1 renamed `admissions.ethnicity` to `race` (33 fine-grained values).

## Cohort (extract_subjects_iv.py)

| Stage | ICU stays | Admissions | Patients |
|---|---|---|---|
| START | 94,458 | 85,242 | 65,366 |
| REMOVE ICU TRANSFERS | 94,458 | 85,242 | 65,366 |
| REMOVE MULTIPLE STAYS PER ADMIT | 77,547 | 77,547 | 60,757 |
| REMOVE PATIENTS AGE < 18 | 77,547 | 77,547 | 60,757 |

- Matches Norhan's independent count (94,458 → 77,547). Her stricter 77,534 additionally drops 13 stays
  with a null `outtime` (still in ICU at extraction); these have null length of stay and never reach a task.
- Transfer filter removes nothing: in MIMIC-IV a ward transfer creates a new stay row.
- One-stay rule drops the whole admission (`mimic3csv.filter_admissions_on_nb_icustays`), because diagnosis
  codes belong to the admission, not to a single ICU stay.
- Age filter removes nothing: MIMIC-IV ICU patients are all adults. Ages above 89 are shifted (appear ~91).

## Ethnicity

DataFusion.py only knows the 8 categories of the MIMIC-IV release the paper used. v3.1 `race` was collapsed to them
(`mimic3csv.collapse_race_to_ethnicity`): prefix WHITE / BLACK / HISPANIC / ASIAN / AMERICAN INDIAN → that group;
UNKNOWN and UNABLE TO OBTAIN kept; PATIENT DECLINED TO ANSWER → UNABLE TO OBTAIN; everything else → OTHER.
Without this, 10,045 stays (13.0%) raised a KeyError in DataFusion. Ethnicity is used for the fairness breakdown only.

| Category | Stays |
|---|---|
| WHITE | 51,435 |
| BLACK/AFRICAN AMERICAN | 8,510 |
| UNKNOWN | 6,862 |
| OTHER | 3,148 |
| HISPANIC/LATINO | 3,032 |
| ASIAN | 2,396 |
| UNABLE TO OBTAIN | 2,002 |
| AMERICAN INDIAN/ALASKA NATIVE | 162 |

## Splits

- Train/test from `split_train_and_test.py`; validation from `split_train_val.py` (bundled `valset_iv.csv`).
- No patient in two splits, in either task. Every patient with a chest x-ray is in the same split as in
  `mimic-cxr-ehr-split.csv`.
- The 48-hour clock starts at ICU admission (`intime`) for EHR, x-rays and reports (agreed with Norhan and Farida).

## Task B: phenotyping (whole stay, 25 labels)

| Split | Stays | Pneumonia |
|---|---|---|
| Train | 60,909 | 13.09% |
| Val | 4,756 | 12.83% |
| Test | 11,845 | 12.44% |

77,510 stays; 37 dropped for having no ICU measurements. Paper (older MIMIC-IV): 42,628 / 4,802 / 11,914 stays,
12.7% / 12.4% / 12.3% pneumonia. Norhan's whole-stay x-rays pair with 99.95% of her stays.

## In-hospital mortality (first 48 h, death label)

| Split | Stays | Died |
|---|---|---|
| Train | 29,171 | 13.65% |
| Val | 2,161 | 12.36% |
| Test | 5,302 | 12.58% |

Requires a stay of at least 48 h; 14 eligible stays had no measurements in the window. The label is death at any
point during the hospital admission. Norhan's 48-hour x-rays pair with 99.91% of her stays.

## Pneumonia label definition (roadmap E2/E3)

- Definition: CCS category 122 in `resources/icd_9_10_definitions_2.yaml`. Full list in `pneumonia_ccs122_codes.csv`.
- The yaml lists 120 entries, but only **109 are unique codes (62 ICD-9, 47 ICD-10)**; the rest are duplicates.
- Unmapped codes (E2): every ICD-9 code in the cohort maps. 39.7% of diagnosis rows (all ICD-10; 10,972 of 18,004
  distinct codes) are not in the yaml. Most belong to no benchmark label (e.g. history of nicotine dependence).
- **Open issue:** four ICD-10 pneumonia codes are missing from CCS 122: J13 (Streptococcus pneumoniae), J18.8 (other
  pneumonia), J12.82 (COVID-19 pneumonia, created in 2020, after the paper's data) and J12.3 (metapneumovirus).
  418 stays that carry one of them are currently labelled negative. Adding them would raise pneumonia prevalence
  from 13.09 / 12.83 / 12.44% to 13.71 / 12.95 / 12.73% (train / val / test). Not changed yet: team decision.

## Time-series data quality (open issue)

- Outlier removal is disabled, as in the paper (roadmap D3). Impossible values remain: `999999` placeholders
  (glucose, O2 saturation, pH) and typos (e.g. diastolic BP in the tens of thousands); under 0.1% of readings
  except pH.
- About 6% of pH readings are urine pH: lab items 51491 and 51094 (urine) are mapped to the blood pH channel.
- GCS total, capillary refill rate and height are 100% missing in the extracted time series.
- A normalizer refit on our train split was distorted by these values, so training keeps the paper's shipped
  normalizers (task-matched; `fusion_main.py` previously loaded the phenotyping one for every task).
- Options: blank out-of-range values at load time, and/or remove the urine pH items and re-run D3/D5.
