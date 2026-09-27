# Testing Round 2 yourself

This guide walks you through running **Round 2 of MedPatch** — training the
token-level _confidence predictors_ — on **virtual data**, and checking that it
worked. It assumes no prior experience with this repo.

> **Everything in this guide runs on SYNTHETIC data — not a scientific result.**
> The virtual dataset is invented from a random number generator. Its AUROC
> numbers only prove the plumbing works. Never quote them as model performance,
> and never report accuracy — we use AUROC and AUPRC.

## What Round 2 is, in one paragraph

MedPatch has four _readers_: EHR (an LSTM over hourly vitals), CXR (a
`vit_small_patch16_384` vision transformer), RR (radiology reports, via
Bio_ClinicalBERT) and DN (discharge notes, same BERT). **Round 1** trains each
reader on its own (teammates own this). **Round 2** (yours) loads a Round 1
checkpoint, **freezes the reader**, and trains only a small
`confidence_predictor` on top: one linear layer that gives every _token_ (an
hour of EHR, an image patch, a word piece) its own prediction. Later stages use
how confident each token is — `max(σ(l), 1 − σ(l))` — to split evidence into
high- and low-confidence groups.

Because Round 1 isn't ready, we train quick **virtual stand-ins** for Round 1
in the real checkpoint format. When the real files arrive you register them,
and nothing else changes (see the last section).

## Where things live

| Path                                      | What                                            |
| ----------------------------------------- | ----------------------------------------------- |
| `tools/synthetic/make_virtual_dataset.py` | makes the virtual dataset                       |
| `tools/pv/run.py`                         | runs one stage (r1 or r2) for one reader        |
| `tools/pv/import_checkpoint.py`           | registers a teammate's real Round 1 file        |
| `tools/pv/smoke_round2.py`                | the whole chain in one command                  |
| `tests/round2/test_round2.py`             | the checks a)–e)                                |
| `data/virtual/<preset>/`                  | generated data (gitignored)                     |
| `runs/virtual/`                           | checkpoints and logs (gitignored; `RUNS_ROOT`)  |
| `runs/manifest.csv`                       | the checkpoint list (gitignored; `PV_MANIFEST`) |
| `docs/model_track_notes.md`               | every change made to `medpatch/`, and why       |

---

## 1. Setup

### Windows (PowerShell)

Run these from the repo folder (`C:\Users\<you>\PneumoVision`):

```powershell
py -3.11 -m venv .venv                      # once
.\.venv\Scripts\Activate.ps1                # every new terminal
python -m pip install -r requirements.txt   # once (pinned; CPU torch is fine)
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`False` for CUDA is normal on a laptop: the smoke test is designed for CPU.

If PowerShell refuses to run `Activate.ps1`, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

### Google Colab

In a new notebook, set **Runtime → Change runtime type → GPU** first. Then:

```python
# Cell 1 — code and packages
!git clone https://github.com/Ussefemad1/PneumoVision.git
%cd PneumoVision
!git checkout model/round2-virtual
!pip install -q -r requirements-colab.txt
import torch; print(torch.__version__, torch.cuda.is_available())   # must print True
```

```python
# Cell 2 — keep checkpoints and the manifest on Drive, so they survive disconnects
from google.colab import drive
drive.mount('/content/drive')
import os
os.environ["RUNS_ROOT"] = "/content/drive/MyDrive/pneumovision/runs/virtual"
os.environ["PV_MANIFEST"] = "/content/drive/MyDrive/pneumovision/runs/manifest.csv"
os.environ["PV_WHO"] = "Youssef"          # your name, recorded in the manifest
```

Always run commands in Colab with `!` at the start of the line, from the
`PneumoVision` folder. Environment variables set with `os.environ` (Cell 2)
are inherited by `!` commands in the same session. After a runtime restart, run
Cell 2 again.

**Never** `pip install -r requirements.txt` on Colab: it replaces Colab's CUDA
torch with a CPU build and you lose the GPU.

### First-run downloads

The first run downloads model weights from Hugging Face:
Bio_ClinicalBERT (~440 MB) and the ViT (~90 MB). On our test machine that took
about 5 minutes. They are cached (`~/.cache/huggingface`), so it happens once
per machine or Colab session.

---

## 2. Generate the virtual data

```powershell
python -m tools.synthetic.make_virtual_dataset --preset smoke      # ~60 stays, seconds
python -m tools.synthetic.make_virtual_dataset --preset small      # ~400 stays
python -m tools.synthetic.make_virtual_dataset --preset smoke --seed 7   # different draw
```

Output goes to `data/virtual/smoke/` (or `small/`). Open its `README.md`: it
starts with **SYNTHETIC — not a scientific result** and lists the split sizes,
pneumonia prevalence (~12%) and X-ray availability.

What's planted, so learning is possible: pneumonia stays get a higher
respiratory rate, temperature and heart rate and lower SpO₂; a bright blob low
in one lung on the X-ray; and "consolidation" language in radiology reports
(and often "pneumonia" in the discharge note).

> The smoke preset gives an X-ray to **50%** of stays, not the realistic 18%,
> and gives them mostly to pneumonia stays (as real pneumonia work-ups include a
> chest film). At 60 stays, 18% would leave about two validation X-rays: too few
> to train or score the CXR reader at all. The `small` preset uses 18%.
>
> Smoke has only ~2 pneumonia stays per validation split, so check e) is a
> plumbing check on the default seed (0), not a stable number. Another `--seed`
> can legitimately land below 0.55.

The files follow exactly the folder layout `fusion_main.py` expects (EHR root,
listfiles, per-stay timeseries, resized JPGs + metadata, radiology.csv and
discharge.csv, and a _virtual_ `mimic-cxr-ehr-split.csv`). The real split file
in `medpatch/ehr_utils/` is never read or written.

Check it loads through the repo's own dataset classes:

```powershell
python -m pytest tests/test_virtual_dataset.py
```

Pass looks like `..........  [100%]` (10 dots, no `F`).

---

## 3. The one-command smoke test

```powershell
python -m tools.pv.smoke_round2
```

On Colab: `!python -m tools.pv.smoke_round2`

It generates the smoke data (if missing), trains the 4 Round 1 stand-ins, runs
Round 2 for all 4 readers, then runs the checks. Expect roughly **SMOKE_TIME**
on a laptop CPU after the one-time downloads (it is dominated by BERT and the
ViT running on CPU). It ends with a summary like:

```
SMOKE_SUMMARY
```

`checks PASSED` is what you want. The AUROC values will differ slightly on your
machine — anything above 0.55 passes.

Useful variations:

```powershell
python -m tools.pv.smoke_round2 --skip-r1          # reuse r1 stand-ins, redo r2 + checks
python -m tools.pv.smoke_round2 --readers ehr cxr  # only some readers
python -m tools.pv.smoke_round2 --regenerate       # rebuild the virtual data first
```

### Running one reader by hand

```powershell
python -m tools.pv.run r1 --reader cxr --data virtual     # a Round 1 stand-in
python -m tools.pv.run r2 --reader cxr --data virtual     # Round 2 for CXR
python -m tools.pv.run r2 --reader cxr --data virtual --dry-run   # show the command only
```

`--dry-run` prints the exact `fusion_main.py` arguments. They come straight
from `medpatch/scripts/phenotyping/Confidence/Confidence-CXR.sh`; only paths,
`--resume`, and the recorded smoke overrides differ (virtual data only:
`batch_size 4`; r2 epochs 20 for EHR/CXR and 5 for RR/DN; r1 epochs 2 and
`bootstrap_iters 20`).
For DN, `--load_dn` is used instead of the script's `--load_rr` (see
`docs/model_track_notes.md`).

If a run is interrupted (e.g. Colab disconnects), rerun the same command with
`--run-id <the id it printed>` and it continues from the last finished epoch.

`run.py` **refuses** to start Round 2 if the parent isn't a Round 1 row for the
same reader, or if the parent file no longer matches the sha256 in the
manifest. That's deliberate: it stops you training on the wrong or a
modified checkpoint.

---

## 4. Running the checks one by one

All checks read the latest r2 row per reader from the manifest. `-s` shows
their printed tables; `-k` picks one check.

```powershell
python -m pytest -m round2 -s                     # all of a)–e)
python -m pytest -m round2 -s -k test_a           # a) frozen reader
python -m pytest -m round2 -s -k test_b           # b) confidence values
python -m pytest -m round2 -s -k test_c           # c) checkpoint round-trip
python -m pytest -m round2 -s -k test_d           # d) lineage
python -m pytest -m round2 -s -k test_e           # e) signal sanity
python -m pytest -m round2 -s -k "test_a and cxr" # one check, one reader
```

(Plain `python -m pytest` skips these on purpose — they need saved runs.)

| Check                    | What it proves                                                                                                              | Pass looks like                                                                                                                                                                                                                              |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **a) frozen reader**     | Every reader weight in the Round 2 file is bit-identical to the Round 1 parent; only `confidence_predictor` weights changed | A list of trainable parameters that contains only `…_confidence_predictor.confidence_layer.weight/bias` and the unused `…_classifier…`, then `N reader tensors bit-identical to r1 parent r1-cxr-001; changed: [...confidence_predictor...]` |
| **b) confidence values** | `γ = max(σ(l), 1−σ(l))` is always in [0.5, 1]                                                                               | `min 0.5… mean 0.9… max 0.99… fraction >= 0.75: 0.9…`                                                                                                                                                                                        |
| **c) round-trip**        | The saved file reloads and gives _identical_ outputs                                                                        | `reload x2 -> identical outputs, shape (4, 577, 25)`                                                                                                                                                                                         |
| **d) lineage**           | Every r2 row's parent is an r1 row of the same reader whose file still matches its sha256                                   | `r2-cxr-001 <- r1-cxr-001 (cxr, sha256 ok)` per row                                                                                                                                                                                          |
| **e) signal sanity**     | The confidence heads pick up the planted pneumonia signal: val AUROC > 0.55                                                 | `SYNTHETIC -- not a scientific result: val pneumonia AUROC 0.8… AUPRC 0.5…`                                                                                                                                                                  |

The trainable list in a) includes `…_classifier…`. That's expected: medpatch
leaves the Round 1 classifier with `requires_grad=True`, but Round 2 never
uses it, so it gets no gradient and never changes — which a) verifies.

Each pytest line ends in `PASSED`, `FAILED` or `SKIPPED`. `SKIPPED ... run
python -m tools.pv.smoke_round2 first` means there are no Round 2 runs yet.

---

## 5. Reading `runs/manifest.csv`

This is the team's checkpoint Sheet in code: one row per checkpoint. Open it in
Excel / Google Sheets, or:

```powershell
python -c "import pandas as pd; print(pd.read_csv('runs/manifest.csv')[['id','stage','reader','parent_id','data','val_auroc','val_auprc']].to_string())"
```

| Column                   | Meaning                                                                                                                          |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| `id`                     | `r1-cxr-001`, `r2-cxr-001`, …                                                                                                    |
| `who`, `date`            | who produced it, when (UTC)                                                                                                      |
| `stage`                  | `r1` (reader alone) or `r2` (confidence predictor)                                                                               |
| `reader`                 | `ehr`, `cxr`, `rr`, `dn`                                                                                                         |
| `task`                   | `phenotyping`                                                                                                                    |
| `file_path`              | the checkpoint file                                                                                                              |
| `sha256`                 | fingerprint of that file; r2 refuses a parent whose file no longer matches                                                       |
| `parent_id`              | for r2: which r1 row it loaded                                                                                                   |
| `data`                   | `virtual` or `real` — never mixed in one lineage                                                                                 |
| `seed`                   | virtual dataset seed (trainer seeds are fixed inside medpatch)                                                                   |
| `val_auroc`, `val_auprc` | pneumonia (class 21) on the validation split. r1: the classifier; r2: mean token confidence-head probability                     |
| `notes`                  | "VIRTUAL stand-in, SYNTHETIC…", the paper script used, and any setting that differs from the paper (e.g. `epochs=5 (paper 100)`) |

On Colab, the manifest is wherever `PV_MANIFEST` points (Drive, in the setup
above).

---

## 6. Inspecting one checkpoint

```python
import torch
path = r"runs/virtual/r2/cxr/r2-cxr-001/phenotyping/c-unimodal_cxr/best_checkpoint_0.001_phenotyping_c-unimodal_cxr_EHR-CXR_paired.pth.tar"
ckpt = torch.load(path, map_location="cpu", weights_only=True)
print("keys:", list(ckpt))                       # epoch, state_dict, best_auroc, optimizer, patience
print("epoch:", ckpt["epoch"], " best (r2: val loss):", ckpt["best_auroc"])
sd = ckpt["state_dict"]
print(len(sd), "tensors;", sum(v.numel() for v in sd.values()), "numbers")
for name, value in sd.items():
    if "confidence_predictor" in name or "_classifier" in name:   # the non-reader parts
        print(f"{name:70s} {tuple(value.shape)}")
```

The file only stores weights, not which were trainable. Check a) is what
proves only the `confidence_predictor` changed; the loop above just shows
where those weights live.

---

## 7. Common failures and fixes

| Symptom                                                                             | Cause                                                                                   | Fix                                                                                                                                                           |
| ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OSError: We couldn't connect to 'https://huggingface.co'` or a timm download error | no internet, or HF rate-limited                                                         | retry; or on a machine with internet run the one-liner below once, then copy `~/.cache/huggingface` over                                                      |
| Download is very slow, warning about `hf_xet`                                       | optional faster downloader not installed                                                | harmless; `pip install hf_xet` to speed it up                                                                                                                 |
| `torch.cuda.is_available()` is `False` on Colab                                     | CPU runtime, or torch was replaced                                                      | Runtime → Change runtime type → GPU; if it was `requirements.txt`, Runtime → Disconnect and delete runtime, then reinstall with `requirements-colab.txt` only |
| `No virtual dataset at …`                                                           | data not generated                                                                      | `python -m tools.synthetic.make_virtual_dataset --preset smoke`                                                                                               |
| `No r1 row for reader 'cxr'`                                                        | no parent registered                                                                    | run the r1 stand-in, or `import_checkpoint` a real file                                                                                                       |
| `sha256 mismatch`                                                                   | the parent file changed after it was registered (re-saved, re-downloaded, overwritten)  | re-register it with `import_checkpoint` (it gets a new id) and pass `--parent <new id>`                                                                       |
| `Mixing virtual and real lineage is refused`                                        | r2 `--data real` with a virtual parent, or the reverse                                  | register the right kind of parent                                                                                                                             |
| `FileNotFoundError` / `does not exist` for a data dir                               | wrong path or not in the repo folder                                                    | run from the repo root; on Colab `%cd /content/PneumoVision`                                                                                                  |
| `ModuleNotFoundError: No module named 'tools'`                                      | run from the wrong folder                                                               | same as above: commands must run from the repo root                                                                                                           |
| `Activate.ps1 cannot be loaded`                                                     | PowerShell execution policy                                                             | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`                                                                                                         |
| A run dies half-way                                                                 | Colab disconnect / sleep                                                                | rerun with `--run-id <id>`; it resumes from the last epoch                                                                                                    |
| Check a) says `nothing changed -- the best checkpoint is the untrained start`       | val loss never improved after epoch 0 (medpatch validates _before_ training each epoch) | train more epochs: `python -m tools.pv.run r2 --reader X --data virtual --epochs 10`                                                                          |

Pre-download the weights (run once with internet):

```powershell
python -c "from transformers import BertModel, BertTokenizerFast as T; BertModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT'); T.from_pretrained('emilyalsentzer/Bio_ClinicalBERT'); import timm; timm.create_model('vit_small_patch16_384', pretrained=True)"
```

---

## 8. When real Round 1 files arrive

A teammate gives you a Round 1 file, e.g.
`best_checkpoint_0.001_phenotyping_unimodal_cxr_EHR-CXR_paired.pth.tar`.
Put it somewhere stable (on Colab: your Drive), then register it. The importer
opens the file and refuses anything that isn't a Round 1 checkpoint of that
reader (e.g. an RR file offered as DN, or a Round 2 file).

**1. Register each file** (replace names and paths; `--val-auroc/--val-auprc`
are the numbers your teammate reports in the team Sheet):

```powershell
python -m tools.pv.import_checkpoint --reader ehr --who "<teammate>" --file "D:\pv\r1\best_checkpoint_0.001_phenotyping_unimodal_ehr_EHR_paired.pth.tar" --val-auroc <x> --val-auprc <y>
python -m tools.pv.import_checkpoint --reader cxr --who "<teammate>" --file "D:\pv\r1\best_checkpoint_0.001_phenotyping_unimodal_cxr_EHR-CXR_paired.pth.tar" --val-auroc <x> --val-auprc <y>
python -m tools.pv.import_checkpoint --reader rr  --who "<teammate>" --file "D:\pv\r1\best_checkpoint_0.001_phenotyping_unimodal_rr_EHR-RR_paired.pth.tar"  --val-auroc <x> --val-auprc <y>
python -m tools.pv.import_checkpoint --reader dn  --who "<teammate>" --file "D:\pv\r1\best_checkpoint_0.001_phenotyping_unimodal_dn_EHR-DN_paired.pth.tar"  --val-auroc <x> --val-auprc <y>
```

On Colab, the same with `!` and Drive paths, e.g.
`--file "/content/drive/MyDrive/pneumovision/r1/best_checkpoint_…_unimodal_cxr_EHR-CXR_paired.pth.tar"`.
If it says the file can't be loaded with `weights_only=True` and the file is
from your teammate, add `--trust-pickle`.

**2. Run Round 2 on real data** (on Colab with a GPU; paper settings, no
smoke overrides). Point at the real data folders:

```python
!python -m tools.pv.run r2 --reader ehr --data real --ehr-data-dir "<ehr>" --cxr-data-dir "<cxr>" --notes-data-dir "<notes>"
!python -m tools.pv.run r2 --reader cxr --data real --ehr-data-dir "<ehr>" --cxr-data-dir "<cxr>" --notes-data-dir "<notes>"
!python -m tools.pv.run r2 --reader rr  --data real --ehr-data-dir "<ehr>" --cxr-data-dir "<cxr>" --notes-data-dir "<notes>"
!python -m tools.pv.run r2 --reader dn  --data real --ehr-data-dir "<ehr>" --cxr-data-dir "<cxr>" --notes-data-dir "<notes>"
```

Each picks the latest **real** r1 row for its reader automatically
(`--parent r1-cxr-00N` to choose a specific one). Real data isn't virtual, so
no smoke overrides apply: epochs 100, batch 16, bootstrap 1000 — the paper's
settings. Expect hours on a GPU, not minutes.

**3. Check it** with `python -m pytest -m round2 -s -k "test_a or test_b or test_c or test_d"`.
Check e) is skipped for real data (it only tests the virtual planted signal).

**4. Log every file in the team Sheet** — the imported r1 files and each new
r2 file — copying `id`, `file_path`, `sha256`, `parent_id`, `val_auroc` and
`val_auprc` from `runs/manifest.csv`. The CSV is your local record; the Sheet
is the team's.
