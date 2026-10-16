# Demo examples (synthetic)

Four invented patients for testing the Analyze page. **No real patient data.**
MIMIC records must never be put on the website or shared with online
services (PhysioNet data use agreement), so these are written by hand to look
realistic, using the units and category spellings the models were trained on.

| Example | Task | Files |
| --- | --- | --- |
| Pneumonia, sick | Pneumonia | `pneumonia_sick_vitals.csv`, `pneumonia_sick_radiology.txt`, `pneumonia_sick_discharge.txt` |
| Pneumonia, healthy control | Pneumonia | `pneumonia_control_vitals.csv`, `pneumonia_control_radiology.txt`, `pneumonia_control_discharge.txt` |
| Mortality, high risk (septic shock, ARDS) | Mortality | `mortality_high_risk_vitals.csv`, `mortality_high_risk_radiology.txt` |
| Mortality, low risk | Mortality | `mortality_low_risk_vitals.csv`, `mortality_low_risk_radiology.txt` |

## How to use one

1. Analyze → **+ New analysis**, choose the task.
2. **EHR → CSV upload** → pick the `_vitals.csv` (48 rows: `hour` -47 … 0, where 0 is now).
3. **Clinical notes → Add note → Load .txt** → pick each `.txt`, and set its type
   (radiology or discharge) to match the file name.
4. **Run analysis.** Optionally add a chest X-ray (see below).

For mortality, leave discharge notes out: they describe the outcome and the
site excludes them anyway.

## What to expect (readers mode, `INFERENCE_MODE=readers`)

Measured on 2026-10-16 with the Round 1 readers. Every per-source score is a
real model output and repeats exactly. The combined score is their plain
average (shown with its formula on the report), not the MedPatch fusion.

| Example | EHR | Radiology report | Discharge note | Combined (average) |
| --- | --- | --- | --- | --- |
| Pneumonia, sick | 30.7% | 72.0% | 99.9% | 67.5% |
| Pneumonia, control | 7.0% | 0.5% | 0.2% | 2.6% |
| Mortality, high risk | 89.9% | 68.2% | — | 79.1% |
| Mortality, low risk | 4.9% | 3.2% | — | 4.1% |

**Chest X-ray:** no image is included, because a made-up picture is not a
radiograph and the X-ray reader's score on it means nothing. To demo the X-ray
reader, use a chest radiograph you are allowed to share (for example a
public-domain or CC-licensed one from Wikimedia Commons, credited) — never a
MIMIC-CXR image. The reader was trained on frontal (AP) ICU radiographs.

On the Railway site (mock mode) every number is simulated.

Edit the CSV values directly to try other cases; the site rejects unknown
variables and category values the models never saw.
