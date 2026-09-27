"""Generate a fully invented dataset in the exact layout ``medpatch/fusion_main.py``
expects for ``--task phenotyping``.

SYNTHETIC -- not a scientific result. Nothing here is read from, sampled from
or derived from MIMIC. Column names follow the *published* schemas of
MIMIC-IV / MIMIC-CXR-JPG / MIMIC-IV-Note and the repo's own code resources
(discretizer_config.json); every value is drawn from a seeded RNG.

Layout written under ``<out>`` (default ``data/virtual/<preset>``)::

    ehr/root/all_stays.csv
    ehr/phenotyping/{train,val,test}_listfile.csv
    ehr/phenotyping/train/<subject>_episode1_timeseries.csv   (train + val stays)
    ehr/phenotyping/test/<subject>_episode1_timeseries.csv
    ehr/ph_ts1.0.virtual.normalizer        (fit on the virtual train split)
    cxr/resized/p1x/p<subject>/s<study>/<dicom_id>.jpg
    cxr/mimic-cxr-2.0.0-metadata.csv
    cxr/mimic-cxr-2.0.0-chexpert.csv
    cxr/mimic-cxr-ehr-split.csv            (virtual; the real one is never touched)
    notes/radiology.csv
    notes/discharge.csv
    README.md                              (SYNTHETIC stamp)

A weak, learnable signal is planted for pneumonia (phenotype index 21):
higher respiratory rate / temperature / heart rate and lower SpO2, a brighter
blob in a lower lung field, and consolidation language in radiology reports
(plus a pneumonia mention in some discharge summaries). The other 24 labels are
independent noise.

Usage::

    python -m tools.synthetic.make_virtual_dataset --preset smoke
    python -m tools.synthetic.make_virtual_dataset --preset small --seed 7
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
MEDPATCH = REPO_ROOT / "medpatch"
DISCRETIZER_CONFIG = MEDPATCH / "ehr_utils" / "resources" / "discretizer_config.json"

# Label order used by medpatch (datasets/DataFusion.py CLASSES). Index 21 is pneumonia.
PHENOTYPES = [
    "Acute and unspecified renal failure",
    "Acute cerebrovascular disease",
    "Acute myocardial infarction",
    "Cardiac dysrhythmias",
    "Chronic kidney disease",
    "Chronic obstructive pulmonary disease and bronchiectasis",
    "Complications of surgical procedures or medical care",
    "Conduction disorders",
    "Congestive heart failure; nonhypertensive",
    "Coronary atherosclerosis and other heart disease",
    "Diabetes mellitus with complications",
    "Diabetes mellitus without complication",
    "Disorders of lipid metabolism",
    "Essential hypertension",
    "Fluid and electrolyte disorders",
    "Gastrointestinal hemorrhage",
    "Hypertension with complications and secondary hypertension",
    "Other liver diseases",
    "Other lower respiratory disease",
    "Other upper respiratory disease",
    "Pleurisy; pneumothorax; pulmonary collapse",
    "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)",
    "Respiratory failure; insufficiency; arrest (adult)",
    "Septicemia (except in labor)",
    "Shock",
]
PNEUMONIA = 21

CHEXPERT = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]
# Keys of DataFusion.ETHNICITY -- any other value would KeyError in the loader.
ETHNICITIES = [
    "WHITE",
    "UNKNOWN",
    "OTHER",
    "BLACK/AFRICAN AMERICAN",
    "HISPANIC/LATINO",
    "ASIAN",
    "AMERICAN INDIAN/ALASKA NATIVE",
    "UNABLE TO OBTAIN",
]

SYNTHETIC_STAMP = "SYNTHETIC — not a scientific result"


@dataclass(frozen=True)
class Preset:
    stays: int
    cxr_rate: float
    # P(X-ray | pneumonia). Pneumonia work-ups almost always include a chest
    # film, so X-rays are concentrated on pneumonia stays; the other stays fill
    # the remainder so the overall rate is still `cxr_rate`.
    cxr_given_pneumonia: float
    split: tuple[float, float, float]  # train / val / test
    image_px: int


PRESETS = {
    # CPU smoke test. X-ray availability is raised above the realistic 18% on
    # purpose: at 60 stays, 18% would leave ~2 paired validation images, too few
    # for the CXR reader to train or be scored at all. See TESTING_ROUND2.md.
    "smoke": Preset(
        stays=60, cxr_rate=0.5, cxr_given_pneumonia=0.9, split=(0.5, 0.25, 0.25), image_px=224
    ),
    # Realistic proportions (X-ray for ~18% of stays).
    "small": Preset(
        stays=400, cxr_rate=0.18, cxr_given_pneumonia=0.5, split=(0.6, 0.2, 0.2), image_px=256
    ),
}


# ── Vitals ───────────────────────────────────────────────────────────────────

# (mean, sd, per-hour observation probability); pneumonia shift applied below.
CONTINUOUS = {
    "Heart Rate": (86.0, 12.0, 0.9),
    "Respiratory rate": (18.0, 3.0, 0.9),
    "Oxygen saturation": (97.0, 1.5, 0.9),
    "Systolic blood pressure": (120.0, 14.0, 0.8),
    "Diastolic blood pressure": (62.0, 9.0, 0.8),
    "Mean blood pressure": (80.0, 10.0, 0.8),
    "Temperature": (36.9, 0.4, 0.3),
    "Glucose": (130.0, 30.0, 0.15),
    "Fraction inspired oxygen": (0.3, 0.08, 0.2),
    "pH": (7.4, 0.04, 0.1),
    "Weight": (80.0, 15.0, 0.0),  # charted once, at hour 0
    "Height": (170.0, 10.0, 0.0),  # charted once, at hour 0
}
PNEUMONIA_SHIFT = {
    "Respiratory rate": 5.0,
    "Temperature": 0.8,
    "Heart Rate": 9.0,
    "Oxygen saturation": -2.5,
    "Fraction inspired oxygen": 0.08,
}
BOUNDS = {
    "Oxygen saturation": (70.0, 100.0),
    "Fraction inspired oxygen": (0.21, 1.0),
    "pH": (6.9, 7.7),
    "Respiratory rate": (6.0, 50.0),
    "Temperature": (34.0, 41.5),
}


def _vital_value(rng, name: str, stay_offset: float, pneumonia: bool) -> float:
    mean, sd, _ = CONTINUOUS[name]
    value = mean + stay_offset * sd + rng.normal(0.0, 0.6 * sd)
    if pneumonia:
        value += PNEUMONIA_SHIFT.get(name, 0.0)
    lo, hi = BOUNDS.get(name, (0.0, float("inf")))
    return float(np.clip(value, lo, hi))


def _fmt(name: str, value: float) -> str:
    if name in ("Fraction inspired oxygen", "pH"):
        return f"{value:.2f}"
    return f"{value:.1f}"


def make_timeseries(rng, config: dict, hours: float, pneumonia: bool) -> pd.DataFrame:
    """One stay's irregular hourly chart, in discretizer channel order."""
    channels = config["id_to_channel"]
    offsets = {name: rng.normal(0.0, 0.5) for name in CONTINUOUS}
    rows = []
    t = float(rng.uniform(0.05, 0.9))
    while t < hours:
        row = {"Hours": f"{t:.4f}"}
        for name in channels:
            cell = ""
            if name in CONTINUOUS:
                prob = CONTINUOUS[name][2]
                first = len(rows) == 0 and name in ("Weight", "Height")
                if first or rng.random() < prob:
                    cell = _fmt(name, _vital_value(rng, name, offsets[name], pneumonia))
            elif config["is_categorical_channel"].get(name) and rng.random() < 0.2:
                values = config["possible_values"][name]
                # Mostly the normal value, occasionally something else.
                normal = config["normal_values"][name]
                cell = normal if rng.random() < 0.7 else str(rng.choice(values))
            row[name] = cell
        rows.append(row)
        t += float(rng.uniform(0.6, 1.4))
    return pd.DataFrame(rows, columns=["Hours"] + channels)


# ── Images ───────────────────────────────────────────────────────────────────


def make_xray(rng, px: int, pneumonia: bool) -> Image.Image:
    """A crude synthetic frontal film: dark lung fields, bright mediastinum,
    grain, and -- mostly for pneumonia -- a brighter blob low in one lung."""
    yy, xx = np.mgrid[0:px, 0:px] / px
    img = np.full((px, px), 150.0)
    for cx in (0.3, 0.7):
        lung = ((xx - cx) / 0.17) ** 2 + ((yy - 0.5) / 0.32) ** 2 < 1.0
        img[lung] = 55.0
    img[np.abs(xx - 0.5) < 0.07] = 190.0
    blob = rng.random() < (0.8 if pneumonia else 0.1)
    if blob:
        cx = 0.3 if rng.random() < 0.5 else 0.7
        cy = rng.uniform(0.6, 0.72)
        r = rng.uniform(0.07, 0.11)
        img += 110.0 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))
    img += rng.normal(0.0, 12.0, size=img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="L")


# ── Notes ────────────────────────────────────────────────────────────────────

RR_POSITIVE = [
    "Dense right lower lobe consolidation with air bronchograms, consistent with pneumonia.",
    "New left basilar airspace consolidation concerning for pneumonia.",
    "Patchy multifocal opacities with consolidation, compatible with infection.",
]
RR_NEGATIVE = [
    "The lungs are clear. No acute cardiopulmonary process.",
    "Heart size is normal. Lungs are well expanded and clear.",
    "Stable appearance of the chest. No pleural effusion or pneumothorax.",
]
RR_FILLER = [
    "Portable AP view of the chest.",
    "Comparison is made to the prior study.",
    "Endotracheal tube terminates above the carina.",
    "Mediastinal contours are unchanged.",
    "Osseous structures are intact.",
    "Lines and tubes are in standard position.",
]
DN_FILLER = [
    "The patient was admitted to the intensive care unit for monitoring.",
    "Home medications were reconciled prior to discharge.",
    "Follow up with primary care physician in one to two weeks.",
    "Vital signs remained within acceptable limits on the day of discharge.",
    "Diet was advanced as tolerated.",
]


def radiology_text(rng, pneumonia: bool) -> str:
    finding = RR_POSITIVE if rng.random() < (0.75 if pneumonia else 0.08) else RR_NEGATIVE
    parts = list(rng.choice(RR_FILLER, size=2, replace=False)) + [str(rng.choice(finding))]
    rng.shuffle(parts)
    return "FINDINGS: " + " ".join(parts) + " IMPRESSION: " + parts[-1]


def discharge_text(rng, labels: np.ndarray) -> str:
    diagnoses = []
    if labels[PNEUMONIA] and rng.random() < 0.7:
        diagnoses.append("community acquired pneumonia treated with intravenous antibiotics")
    elif not labels[PNEUMONIA] and rng.random() < 0.05:
        diagnoses.append("possible pneumonia, ruled out")
    diagnoses.append(
        str(
            rng.choice(
                [
                    "hypertension",
                    "atrial fibrillation",
                    "diabetes",
                    "acute kidney injury",
                    "COPD exacerbation",
                ]
            )
        )
    )
    body = " ".join(rng.choice(DN_FILLER, size=3, replace=False))
    return f"Discharge Diagnosis: {'; '.join(diagnoses)}. Hospital Course: {body}"


# ── Assembly ─────────────────────────────────────────────────────────────────


def _stratified_split(rng, keys: list[tuple[int, int]], split) -> list[str]:
    """Assign train/val/test per stratum so every split sees positives and X-rays."""
    names = np.array(["train", "val", "test"])
    out = [""] * len(keys)
    for stratum in sorted(set(keys)):
        idx = [i for i, k in enumerate(keys) if k == stratum]
        rng.shuffle(idx)
        n = len(idx)
        n_val = max(1, round(n * split[1])) if n >= 3 else 0
        n_test = max(1, round(n * split[2])) if n >= 3 else 0
        for j, i in enumerate(idx):
            out[i] = names[1] if j < n_val else names[2] if j < n_val + n_test else names[0]
    return out


def generate(out: Path, preset_name: str, seed: int = 0) -> dict:
    preset = PRESETS[preset_name]
    rng = np.random.default_rng(seed)
    config = json.loads(DISCRETIZER_CONFIG.read_text())

    ehr, cxr, notes = out / "ehr", out / "cxr", out / "notes"
    for d in (
        ehr / "root",
        ehr / "phenotyping" / "train",
        ehr / "phenotyping" / "test",
        cxr / "resized",
        notes,
    ):
        d.mkdir(parents=True, exist_ok=True)

    n = preset.stays
    labels = np.zeros((n, len(PHENOTYPES)), dtype=int)
    prevalence = rng.uniform(0.05, 0.3, size=len(PHENOTYPES))
    prevalence[PNEUMONIA] = 0.12
    for c in range(len(PHENOTYPES)):
        labels[:, c] = rng.random(n) < prevalence[c]
    # Exactly ~12% pneumonia, independent of chance at small n.
    pos = rng.choice(n, size=max(2, round(0.12 * n)), replace=False)
    labels[:, PNEUMONIA] = 0
    labels[pos, PNEUMONIA] = 1
    has_cxr = np.zeros(n, dtype=bool)
    n_cxr = max(3, round(preset.cxr_rate * n))
    positives = np.flatnonzero(labels[:, PNEUMONIA] == 1)
    negatives = np.flatnonzero(labels[:, PNEUMONIA] == 0)
    n_pos_cxr = min(len(positives), n_cxr, round(preset.cxr_given_pneumonia * len(positives)))
    has_cxr[rng.choice(positives, size=n_pos_cxr, replace=False)] = True
    has_cxr[rng.choice(negatives, size=n_cxr - n_pos_cxr, replace=False)] = True

    splits = _stratified_split(
        rng, list(zip(labels[:, PNEUMONIA], has_cxr, strict=True)), preset.split
    )

    stays, listfiles = [], {"train": [], "val": [], "test": []}
    metadata, chexpert, cxr_split, rad, dis = [], [], [], [], []
    base = datetime(2150, 1, 1)
    note_id = 0
    for i in range(n):
        subject_id, hadm_id, stay_id = 10_000_000 + i, 20_000_000 + i, 30_000_000 + i
        pneumonia = bool(labels[i, PNEUMONIA])
        hours = float(rng.uniform(24.0, 72.0))
        intime = base + timedelta(
            days=int(rng.integers(0, 3650)), minutes=int(rng.integers(0, 1440))
        )
        outtime = intime + timedelta(hours=hours)
        stays.append(
            {
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "stay_id": stay_id,
                "last_careunit": "Medical Intensive Care Unit (MICU)",
                "intime": intime,
                "outtime": outtime,
                "los": round(hours / 24, 4),
                "admittime": intime - timedelta(hours=2),
                "dischtime": outtime + timedelta(hours=20),
                "deathtime": "",
                "ethnicity": str(rng.choice(ETHNICITIES)),
                "gender": str(rng.choice(["M", "F"])),
                "age": int(rng.integers(18, 91)),
                "mortality_inunit": 0,
                "mortality": 0,
                "mortality_inhospital": 0,
            }
        )

        split = splits[i]
        folder = "test" if split == "test" else "train"
        stay_file = f"{subject_id}_episode1_timeseries.csv"
        make_timeseries(rng, config, hours, pneumonia).to_csv(
            ehr / "phenotyping" / folder / stay_file, index=False
        )
        listfiles[split].append([stay_file, round(hours, 6), stay_id, *labels[i].tolist()])

        if has_cxr[i]:
            study_id = 50_000_000 + i
            dicom_id = "-".join(f"{int(x):08x}" for x in rng.integers(0, 2**32, size=5))
            taken = intime + timedelta(hours=float(rng.uniform(1.0, hours - 1.0)))
            img_dir = (
                cxr / "resized" / f"p{str(subject_id)[:2]}" / f"p{subject_id}" / f"s{study_id}"
            )
            img_dir.mkdir(parents=True, exist_ok=True)
            make_xray(rng, preset.image_px, pneumonia).save(img_dir / f"{dicom_id}.jpg", quality=90)
            metadata.append(
                {
                    "dicom_id": dicom_id,
                    "subject_id": subject_id,
                    "study_id": study_id,
                    "PerformedProcedureStepDescription": "CHEST (PORTABLE AP)",
                    "ViewPosition": "AP",
                    "Rows": preset.image_px,
                    "Columns": preset.image_px,
                    "StudyDate": taken.strftime("%Y%m%d"),
                    "StudyTime": taken.strftime("%H%M%S.000"),
                    "ProcedureCodeSequence_CodeMeaning": "CHEST (PORTABLE AP)",
                    "ViewCodeSequence_CodeMeaning": "antero-posterior",
                    "PatientOrientationCodeSequence_CodeMeaning": "Erect",
                }
            )
            findings = {c: 0.0 for c in CHEXPERT}
            findings["Pneumonia"] = 1.0 if pneumonia else 0.0
            findings["Consolidation"] = 1.0 if pneumonia and rng.random() < 0.6 else 0.0
            findings["No Finding"] = 0.0 if pneumonia else 1.0
            chexpert.append({"subject_id": subject_id, "study_id": study_id, **findings})
            cxr_split.append(
                {
                    "dicom_id": dicom_id,
                    "study_id": study_id,
                    "subject_id": subject_id,
                    "split": {"val": "validate"}.get(split, split),
                }
            )

        for _ in range(int(rng.integers(1, 3))):
            note_id += 1
            charted = intime + timedelta(hours=float(rng.uniform(0.5, hours)))
            rad.append(
                {
                    "note_id": f"{subject_id}-RR-{note_id}",
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "note_type": "RR",
                    "note_seq": note_id,
                    "charttime": charted,
                    "storetime": charted + timedelta(minutes=30),
                    "text": radiology_text(rng, pneumonia),
                }
            )
        dis.append(
            {
                "note_id": f"{subject_id}-DS-1",
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "note_type": "DS",
                "note_seq": 1,
                "charttime": outtime,
                "storetime": outtime + timedelta(hours=4),
                "text": discharge_text(rng, labels[i]),
            }
        )

    fmt = "%Y-%m-%d %H:%M:%S"
    pd.DataFrame(stays).to_csv(ehr / "root" / "all_stays.csv", index=False, date_format=fmt)
    header = ["stay", "period_length", "stay_id", *PHENOTYPES]
    for split, rows in listfiles.items():
        pd.DataFrame(rows, columns=header).to_csv(
            ehr / "phenotyping" / f"{split}_listfile.csv", index=False
        )
    pd.DataFrame(metadata).to_csv(cxr / "mimic-cxr-2.0.0-metadata.csv", index=False)
    pd.DataFrame(chexpert, columns=["subject_id", "study_id", *CHEXPERT]).to_csv(
        cxr / "mimic-cxr-2.0.0-chexpert.csv", index=False
    )
    pd.DataFrame(cxr_split, columns=["dicom_id", "study_id", "subject_id", "split"]).to_csv(
        cxr / "mimic-cxr-ehr-split.csv", index=False
    )
    pd.DataFrame(rad).to_csv(notes / "radiology.csv", index=False, date_format=fmt)
    pd.DataFrame(dis).to_csv(notes / "discharge.csv", index=False, date_format=fmt)

    normalizer_path = fit_normalizer(ehr, listfiles["train"])

    summary = {
        "preset": preset_name,
        "seed": seed,
        "stays": n,
        "split_sizes": {k: len(v) for k, v in listfiles.items()},
        "pneumonia_prevalence": round(float(labels[:, PNEUMONIA].mean()), 3),
        "cxr_available": round(float(has_cxr.mean()), 3),
        "cxr_given_pneumonia": round(float(has_cxr[labels[:, PNEUMONIA] == 1].mean()), 3),
        "cxr_by_split": {
            s: int(sum(has_cxr[i] for i in range(n) if splits[i] == s))
            for s in ("train", "val", "test")
        },
        "normalizer": str(normalizer_path.relative_to(out)),
    }
    (out / "README.md").write_text(readme(summary), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def fit_normalizer(ehr: Path, train_rows: list) -> Path:
    """Fit the repo's own Normalizer on the virtual train split, so nothing
    downstream depends on the MIMIC-derived statistics in medpatch/normalizers."""
    sys.path.insert(0, str(MEDPATCH))
    from ehr_utils.preprocessing import Discretizer, Normalizer  # noqa: PLC0415

    discretizer = Discretizer(
        timestep=1.0,
        store_masks=True,
        impute_strategy="previous",
        start_time="zero",
        config_path=str(DISCRETIZER_CONFIG),
    )
    header = None
    normalizer = None
    for stay_file, period, *_ in train_rows:
        raw = pd.read_csv(ehr / "phenotyping" / "train" / stay_file, dtype=str).fillna("")
        data, header = discretizer.transform(raw.values, end=period)
        if normalizer is None:
            cont = [i for i, x in enumerate(header.split(",")) if x.find("->") == -1]
            normalizer = Normalizer(fields=cont)
        normalizer._feed_data(data)
    path = ehr / "ph_ts1.0.virtual.normalizer"
    normalizer._save_params(str(path))
    return path


def readme(summary: dict) -> str:
    return f"""# {SYNTHETIC_STAMP}

This folder was produced by `tools/synthetic/make_virtual_dataset.py`.
Every value is invented from a seeded random generator. No MIMIC record, row,
time, identifier or image was read, sampled or derived to make it.

Any AUROC/AUPRC computed on this data only proves the training plumbing works.
It says nothing about the model.

```json
{json.dumps(summary, indent=2)}
```
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=None, help="Output folder (default data/virtual/<preset>)"
    )
    args = parser.parse_args(argv)
    out = args.out or REPO_ROOT / "data" / "virtual" / args.preset
    summary = generate(out, args.preset, args.seed)
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"{SYNTHETIC_STAMP}\nwrote {out}\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
