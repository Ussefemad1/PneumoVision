# Demo examples (synthetic)

Four invented patients for testing the Analyze page. **No real patient data.**
MIMIC records must never be put on the website or shared with online
services (PhysioNet data use agreement), so these are written by hand to look
realistic, using the units and category spellings the models were trained on.

| Example                                   | Task      | Files                                                                                                |
| ----------------------------------------- | --------- | ---------------------------------------------------------------------------------------------------- |
| Pneumonia, sick                           | Pneumonia | `pneumonia_sick_vitals.csv`, `pneumonia_sick_radiology.txt`, `pneumonia_sick_discharge.txt`          |
| Pneumonia, healthy control                | Pneumonia | `pneumonia_control_vitals.csv`, `pneumonia_control_radiology.txt`, `pneumonia_control_discharge.txt` |
| Mortality, high risk (septic shock, ARDS) | Mortality | `mortality_high_risk_vitals.csv`, `mortality_high_risk_radiology.txt`                                |
| Mortality, low risk                       | Mortality | `mortality_low_risk_vitals.csv`, `mortality_low_risk_radiology.txt`                                  |

## How to use one

1. Analyze → **+ New analysis**, choose the task.
2. **EHR → CSV upload** → pick the `_vitals.csv` (48 rows: `hour` -47 … 0, where 0 is now).
3. **Clinical notes → Add note → Load .txt** → pick each `.txt`, and set its type
   (radiology or discharge) to match the file name.
4. **Run analysis.** Optionally add a chest X-ray (see below).

For mortality, leave discharge notes out: they describe the outcome and the
site excludes them anyway.

## What to expect (fusion mode, `INFERENCE_MODE=fusion`, the deployed default)

The site serves the trained Round 3 MedPatch fusion (`r3-fusion-001` for
mortality, `r3-fusion-002` for pneumonia). The headline is the `late` head as
an **uncalibrated** probability; the other heads are under "Model details".

**Scores not measured yet.** The fusion scores for these examples have not been
measured here: the checkpoints are not in this repo, and the development
machine has none. Fill this table on the first AWS deploy by running each
example through the site (Analyze → the files below):

| Example              | Fusion (late) | Notes |
| -------------------- | ------------- | ----- |
| Pneumonia, sick      | —             |       |
| Pneumonia, control   | —             |       |
| Mortality, high risk | —             |       |
| Mortality, low risk  | —             |       |

**PNGs not deployed.** `sick cxr.png` and `not sick cxr.png` have no recorded
source or licence, so they are **not deployed**. The AWS server's sparse clone
leaves them out and its data guard refuses them (`infra/aws/README.md`). Use
the two credited JPGs below.

## What to expect (readers mode, `INFERENCE_MODE=readers`)

Measured on 2026-10-16 with the Round 1 readers. Every per-source score is a
real model output and repeats exactly. The combined score is their plain
average (shown with its formula on the report), not the MedPatch fusion.

| Example              | EHR   | Radiology report | Discharge note | Combined (average) |
| -------------------- | ----- | ---------------- | -------------- | ------------------ |
| Pneumonia, sick      | 30.7% | 72.0%            | 99.9%          | 67.5%              |
| Pneumonia, control   | 7.0%  | 0.5%             | 0.2%           | 2.6%               |
| Mortality, high risk | 89.9% | 68.2%            | —              | 79.1%              |
| Mortality, low risk  | 4.9%  | 3.2%             | —              | 4.1%               |

### Chest X-rays (real, openly licensed — not MIMIC)

| File                         | Shows                                                | Pneumonia score | Mortality score |
| ---------------------------- | ---------------------------------------------------- | --------------- | --------------- |
| `cxr_pneumonia_covid_ap.jpg` | Bilateral COVID-19 pneumonia, portable AP, ICU       | 42.7%           | 23.1%           |
| `cxr_pneumonia_lobar.jpg`    | Right middle lobe pneumonia, standing PA, outpatient | 16.5%           | 2.9%            |

Read the scores against the base rates in the training data (pneumonia about
13%, in-hospital mortality about 12%): the AP image is flagged at about three
times the pneumonia base rate. The PA image scores lower because the reader
learned from portable AP ICU films (supine patient, leads and tubes); a sharp
outpatient PA film is out of its distribution. **Use the AP image to demo.**
The CXR reader is the weakest of the readers (validation AUROC about 0.70).

With the full examples: pneumonia sick + `cxr_pneumonia_lobar.jpg` → average
54.8%; mortality high risk + `cxr_pneumonia_covid_ap.jpg` → average 60.4%.

Never upload a MIMIC-CXR image to the site.

#### Image credits

- `cxr_pneumonia_lobar.jpg` — "X-ray of lobar pneumonia", Mikael Häggström,
  M.D. (written informed consent obtained), **CC0**,
  <https://commons.wikimedia.org/wiki/File:X-ray_of_lobar_pneumonia.jpg>
- `cxr_pneumonia_covid_ap.jpg` — "COVID-19 Pneumonie - 82m Roe Thorax ap - 001",
  Hellerhoff, **CC BY-SA 3.0**,
  <https://commons.wikimedia.org/wiki/File:COVID-19_Pneumonie_-_82m_Roe_Thorax_ap_-_001.jpg>
  (unmodified; redistributed under the same licence).

On the Railway site (mock mode) every number is simulated.

Edit the CSV values directly to try other cases; the site rejects unknown
variables and category values the models never saw.
