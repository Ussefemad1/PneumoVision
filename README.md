# PneumoVision

Multimodal fusion over MIMIC-IV EHR, MIMIC-CXR chest X-rays, radiology reports (RR),
and discharge notes (DN), for in-hospital-mortality and phenotyping.

---

## 1. Environment Setup

### Requirements

- **Python 3.11.x** (the project is pinned to 3.11 — see [.python-version](.python-version))
- Git
- Windows PowerShell, macOS, or Linux

Verified with Python 3.11.9.

> **Why 3.11 specifically:** parts of `medpatch/mimic4extract/` still import the
> standard-library `imp` module, which was **removed in Python 3.12**. Do not use
> 3.12+ until those files are ported.

### Windows (PowerShell)

```powershell
git clone https://github.com/Ussefemad1/PneumoVision
cd PneumoVision

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python scripts\verify_environment.py
```

### macOS / Linux

```bash
git clone https://github.com/Ussefemad1/PneumoVision
cd PneumoVision

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python scripts/verify_environment.py
```

Both should end with `ENVIRONMENT READY`.

### GPU / CUDA builds

`requirements.txt` pins `torch==2.4.1`, but **plain `pip install torch==2.4.1` does not
give everyone the same build**:

| Platform | What PyPI gives you |
|---|---|
| Windows | CPU-only |
| Linux | CUDA 12.1 build |
| macOS | CPU / MPS |

For a **CPU-only** install on any platform (small, good for development):

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
```

For **CUDA 12.1** (what the cluster jobs use):

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

Confirm what you actually got:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`2.4.1+cpu False` means CPU-only. Training is GPU-bound — CPU is fine for import checks
and tests, not for real runs.

### Verify

```bash
python scripts/verify_environment.py   # pinned versions + project imports
pytest                                 # smoke tests, no data required
```

---

## 2. Data Setup

None of the datasets are in this repo — they are credentialed PhysioNet data and are
gitignored. Each person needs their own copy:

| Data | Source | Passed via |
|---|---|---|
| MIMIC-IV (extracted per-episode timeseries) | built with `medpatch/mimic4extract/` | `--ehr_data_dir` |
| MIMIC-CXR-JPG 2.0.0 | PhysioNet | `--cxr_data_dir` |
| MIMIC-IV-Note 2.2 (`discharge.csv`, `radiology.csv`) | PhysioNet | `--notes_data_dir` |

`--ehr_data_dir` must contain, per task:

```
<ehr_data_dir>/<task>/train_listfile.csv
<ehr_data_dir>/<task>/val_listfile.csv
<ehr_data_dir>/<task>/test_listfile.csv
<ehr_data_dir>/<task>/train/
<ehr_data_dir>/<task>/test/
```

where `<task>` is `in-hospital-mortality` or `phenotyping`.

> The argument defaults still point at the original cluster paths
> (`/scratch/fs999/...`). **Always pass the three data-dir flags explicitly** — the
> defaults will not exist on your machine.

Put local data under `data/` and checkpoints under `checkpoints/`; both are gitignored.

---

## 3. Running

The entrypoint is [medpatch/fusion_main.py](medpatch/fusion_main.py). Its imports are
top-level (`from trainers... import`), so **run it as a script** — either path works:

```bash
python medpatch/fusion_main.py --help      # from the repo root
cd medpatch && python fusion_main.py --help
```

Example — unimodal EHR, in-hospital mortality:

```bash
python medpatch/fusion_main.py \
  --mode train --epochs 100 --batch_size 16 --lr 0.001 \
  --num_classes 1 \
  --modalities EHR --fusion_type unimodal_ehr \
  --ehr_encoder lstm --classifier mlp --loss bce \
  --task in-hospital-mortality --labels_set mortality \
  --output_dim 512 --data_pairs paired \
  --save_dir checkpoints/ehr-mortality \
  --ehr_data_dir  /path/to/mimic-iv-extracted \
  --cxr_data_dir  /path/to/physionet.org/files/mimic-cxr-jpg/2.0.0 \
  --notes_data_dir /path/to/mimic-iv-note/2.2/note
```

The scripts under [medpatch/scripts/](medpatch/scripts/) are **SLURM job scripts**
(`sbatch`), not local runners. They contain `conda activate medpatch` and placeholder
`Your/Directory/...` paths. To use one locally, copy the `python fusion_main.py ...`
block out of it and substitute real paths.

### Weights & Biases

Every trainer calls `wandb.init()` unconditionally. Without setup this **blocks on an
interactive login prompt**. Before your first run, either log in:

```bash
wandb login
```

…or disable it:

```bash
# macOS / Linux
export WANDB_MODE=offline
```
```powershell
# Windows PowerShell
$env:WANDB_MODE = "offline"
```

### Hugging Face weights

The text encoder downloads `emilyalsentzer/Bio_ClinicalBERT` and the CXR encoder pulls
`timm` weights on first use — the first run needs network access. Set `HF_HOME` to share
one cache if you have limited home-directory quota.

---

## 4. Notes for Contributors

- **Windows:** `--num_workers` defaults to `0` on Windows (spawn start method) and `16`
  elsewhere. Override with `--num_workers N` if you have the cores.
- **Line endings:** [.gitattributes](.gitattributes) forces LF on `*.sh`. Do not disable
  it — CRLF in a SLURM script makes bash fail with `$'\r': command not found`.
- **`medpatch/mimic4extract/`** is vendored from the MIMIC-III benchmark repo. It targets
  Python 3.11 and Keras/TensorFlow that is *not* in `requirements.txt`; the Keras paths
  there are not runnable in this environment.
- Before pushing: `python scripts/verify_environment.py && pytest`.

---

## 5. Repository Layout

```
medpatch/
  fusion_main.py       entrypoint
  arguments.py         all CLI flags
  models/              encoders (CXR / EHR / RR / DN) + fusion heads
  trainers/            one trainer per fusion strategy
  datasets/            EHR, CXR and fused dataset/dataloader construction
  ehr_utils/           discretizer + normalizer
  normalizers/         bundled normalizer state files
  scripts/             SLURM job scripts (mortality / phenotyping)
  mimic4extract/       vendored MIMIC benchmark extraction code
scripts/
  verify_environment.py
tests/                 smoke tests (no data required)
data/ checkpoints/ results/    gitignored
```
