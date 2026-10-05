# Real-data Round 2 / 2b on Colab (phenotyping and mortality)

How to run Round 2 (confidence heads) and Round 2b (calibration) on the **real**
data, starting from the team's real Round 1 checkpoints. It uses the same
tooling as the virtual runs ([TESTING_ROUND2.md](TESTING_ROUND2.md),
[TESTING_ROUND2B.md](TESTING_ROUND2B.md)); only `--data real` and the data paths
change.

Scope: sections 0–8 are **phenotyping** (pneumonia = class 21), the default
task. **In-hospital mortality** uses the same flow with `--task mortality`; see
[Mortality](#mortality-in-hospital) at the end.

> **Data rules (PhysioNet DUA):** the MIMIC data stays on your Drive and the
> Colab VM. Never copy it into the repo, never commit it, never paste rows or
> notes into chats or issues. These steps only move files between Drive and
> `/content`.

---

## 0. Before you start

- A Colab **GPU** runtime (Runtime → Change runtime type → GPU).
- On Drive: the extracted data and the four real Round 1 files from the team.
- Who trained which file, and with which BERT. Farida's note readers (RR, DN)
  were trained with **`dmis-lab/biobert-v1.1`**, not the default
  Bio_ClinicalBERT. You must register them with that name (step 4).

## 1. Code and packages

```python
# Cell 1
!git clone https://github.com/Ussefemad1/PneumoVision.git
%cd /content/PneumoVision
!git checkout model/round2-virtual
!pip install -q -r requirements-colab.txt
import torch; print(torch.__version__, torch.cuda.is_available())   # must print True
```

Never `pip install -r requirements.txt` on Colab (it replaces the CUDA torch).

## 2. Mount Drive; keep outputs and the manifest there

```python
# Cell 2 — rerun after every runtime restart
from google.colab import drive
drive.mount('/content/drive')
import os
os.environ["RUNS_ROOT"]   = "/content/drive/MyDrive/pneumovision/runs/real"
os.environ["PV_MANIFEST"] = "/content/drive/MyDrive/pneumovision/runs/manifest.csv"
os.environ["PV_WHO"]      = "Youssef"
```

Everything a run writes then lives on Drive and survives a disconnect:

```
/content/drive/MyDrive/pneumovision/runs/
  manifest.csv                      <- every checkpoint, with sha256 and lineage
  real/r2/<reader>/<id>/
    run.json                        <- exact argv, parent, BERT, deviations
    train.log
    phenotyping/c-unimodal_<reader>/best_checkpoint_….pth.tar   (and last_…)
  real/r2b/<reader>/<id>/
    …/temp_c-unimodal_<reader>/best_checkpoint_….pth.tar
    medpatch_outputs/               <- ECE tables, per-token CSVs, calibration plots
```

## 3. Copy the data to the VM's local disk, then unzip

Training reads millions of small files. Reading them straight from the Drive
mount is very slow, so copy the archives to `/content` first. Replace the
`<…>` parts with your own Drive paths.

```python
# Cell 3
!mkdir -p /content/data
!cp "/content/drive/MyDrive/<your folder>/<ehr archive>.zip"   /content/data/
!cp "/content/drive/MyDrive/<your folder>/<cxr archive>.zip"   /content/data/
!cp "/content/drive/MyDrive/<your folder>/<notes archive>.zip" /content/data/
!cd /content/data && unzip -q "<ehr archive>.zip" && unzip -q "<cxr archive>.zip" && unzip -q "<notes archive>.zip"
!ls /content/data
```

The three folders must have the layout `fusion_main.py` expects (the virtual
dataset mirrors it, see `tools/synthetic/make_virtual_dataset.py`):

| Folder (`--…-data-dir`)    | Must contain                                                                                                  |
| -------------------------- | ------------------------------------------------------------------------------------------------------------- |
| EHR (`--ehr-data-dir`)     | `root/all_stays.csv`, `phenotyping/{train,val,test}_listfile.csv`, `phenotyping/train/`, `phenotyping/test/`  |
| CXR (`--cxr-data-dir`)     | `resized/**/*.jpg`, `mimic-cxr-2.0.0-metadata.csv`, `mimic-cxr-2.0.0-chexpert.csv`, `mimic-cxr-ehr-split.csv` |
| Notes (`--notes-data-dir`) | `discharge.csv`, `radiology.csv`                                                                              |

Below, `EHR`, `CXR` and `NOTES` stand for those three paths, e.g.
`/content/data/mimic-iv-extracted`. Set them once:

```python
DATA = '--ehr-data-dir /content/data/<ehr folder> --cxr-data-dir /content/data/<cxr folder> --notes-data-dir /content/data/<notes folder>'
```

## 4. Register the real Round 1 files

Copy them to Drive first if they aren't there, so the paths stay valid. Each
command opens the file, checks it really is that reader's Round 1 checkpoint,
records its sha256, and adds an `r1` row.

```python
!python -m tools.pv.import_checkpoint --reader ehr --who "<who>" --file "/content/drive/MyDrive/<…>/best_checkpoint_0.001_phenotyping_unimodal_ehr_EHR_paired.pth.tar" --val-auroc <x> --val-auprc <y>
!python -m tools.pv.import_checkpoint --reader cxr --who "<who>" --file "/content/drive/MyDrive/<…>/best_checkpoint_0.001_phenotyping_unimodal_cxr_EHR-CXR_paired.pth.tar" --val-auroc <x> --val-auprc <y>
!python -m tools.pv.import_checkpoint --reader rr  --who "Farida" --bert-model-name dmis-lab/biobert-v1.1 --file "/content/drive/MyDrive/<…>/best_checkpoint_0.001_phenotyping_unimodal_rr_EHR-RR_paired.pth.tar" --val-auroc <x> --val-auprc <y>
!python -m tools.pv.import_checkpoint --reader dn  --who "Farida" --bert-model-name dmis-lab/biobert-v1.1 --file "/content/drive/MyDrive/<…>/best_checkpoint_0.001_phenotyping_unimodal_dn_EHR-DN_paired.pth.tar" --val-auroc <x> --val-auprc <y>
```

`--bert-model-name` is **required** for real rr/dn files; there is no default,
because a wrong BERT loads without error and silently tokenizes with the wrong
vocabulary. Round 2 and 2b then inherit it from these rows. If you later pass
a different `--bert-model-name` to a run, it is refused.

If a file "could not be loaded with weights_only=True" and it is from your
teammate, add `--trust-pickle`.

## 5. Dry run first

`--dry-run` verifies the parent and prints the exact `fusion_main.py` command.
It trains nothing and creates no folder.

```python
!python -m tools.pv.run r2 --reader rr --data real {DATA} --dry-run
```

Check four things in its output:

- `parent` is the r1 row you just registered;
- `parent sha256 … matches the file: OK`;
- `BERT dmis-lab/biobert-v1.1 (inherited from parent)` for rr/dn;
- the three data dirs and `--save_dir` point where you expect.

Do the same for `ehr`, `cxr` and `dn`.

**EHR normalizer.** Real runs pass no `--normalizer_state`, and nothing is
fitted on the fly. `fusion_main.py` (lines 105–109) then loads the bundled
file `medpatch/normalizers/ph_ts1.0.input_str_previous.start_time_zero.normalizer`
(the `1.0` is `--timestep`, default 1.0; no phenotyping script changes it). That
is the same file a Round 1 run with `normalizer_state None` loaded, and it has
not changed in git since it was added, so Round 2/2b EHR inputs are standardised
exactly as Round 1's were. Because the file is fixed, excluding a stay (such as 30007216) does not change it. The normalizer is **not** stored in the
checkpoint (only `epoch`, `state_dict`, `best_auroc`, `optimizer`,
`patience`). It is recorded in `args.txt` in each run's save folder, so check
there: Round 1's `args.txt` and your Round 2's must both show
`normalizer_state: None` and `timestep: 1.0`. If Round 1's `args.txt` shows
anything else, do not run Round 2: `tools/pv/run.py` has no option for a custom
normalizer on real data yet, so raise it with the model track first.

## 6. Round 2

```python
!python -m tools.pv.run r2 --reader ehr --data real {DATA}
!python -m tools.pv.run r2 --reader cxr --data real {DATA}
!python -m tools.pv.run r2 --reader rr  --data real {DATA}
!python -m tools.pv.run r2 --reader dn  --data real {DATA}
```

No smoke overrides apply on real data. These are the paper's settings (100
epochs, batch 16), so expect hours per reader on a GPU. Each run prints its id
(e.g. `r2-rr-001`) at the start.

**If Colab disconnects, resume Round 2.** Rerun the same command with that id:

```python
!python -m tools.pv.run r2 --reader rr --data real {DATA} --run-id r2-rr-001
```

`run.py` always passes `--resume`, so medpatch continues from the last finished
epoch (`last_checkpoint_…`) with the optimizer state. Checkpoints are written
via a temp file and an atomic rename, so a disconnect mid-save cannot leave a
corrupt file.

## 7. Round 2b — run it in one session

```python
!python -m tools.pv.run r2b --reader ehr --data real {DATA} --dry-run   # check first
!python -m tools.pv.run r2b --reader ehr --data real {DATA}
# … then cxr, rr, dn
```

Each picks the latest real **r2** row for its reader (BERT inherited again).

**Round 2b cannot resume.** `trainers/Calibration.py` has no resume support:
`--resume` is accepted and ignored, so an interrupted run starts again from
epoch 0. Start each r2b run when you can keep the session alive until it
finishes. It trains only on the validation split, so it is much shorter than
Round 2.

## 8. Check and record

```python
!python -m pytest -m round2 -s
!python -m pytest -m round2b -s
```

The checks read the latest rows in `PV_MANIFEST`. The planted-signal check e)
of Round 2 is skipped on real data. Round 2b's ECE check runs, but it remains
a sanity check, not a result. Copy each new row (`id`, `file_path`, `sha256`,
`parent_id`, `bert_model_name`, `val_auroc`, `val_auprc`, `notes`) into the
team Sheet.

---

## BERT must match everywhere downstream

The RR/DN confidence heads trained here sit on top of **BioBERT**
(`dmis-lab/biobert-v1.1`) features. Anything that later loads these
checkpoints must build the same BERT, or it silently tokenizes with the wrong
vocabulary:

- **Round 3** (`c-msma` / `c-e-msma`): built by `MSMA_Trainer` →
  `Text_encoder`, so pass `--bert_model_name dmis-lab/biobert-v1.1`, taken from
  the manifest row.
- **The website's inference service** currently **hardcodes Bio_ClinicalBERT**
  (`services/inference/app/main.py`, `TEXT_MODEL`). It is not changed here. When
  real weights are wired in, it must load the BERT recorded with the
  checkpoints.
- Other medpatch trainers (the ensembles, DHF, staged, etc.) and
  `models/rr_encoder.py` also hardcode Bio_ClinicalBERT. They are not on the
  Round 2/2b/3 path; check before using any of them with these checkpoints.

---

## Mortality (in-hospital)

Same Colab flow as sections 0–8 (Drive, copy/unzip, import, dry-run, r2, r2b,
checks), with `--task mortality` on every `import_checkpoint` and `run`
command. `mortality` is short for medpatch's `in-hospital-mortality`; the
manifest records the long name.

**Readers: EHR, CXR and RR only.** There is no DN reader for mortality:
discharge notes leak the outcome (they are written after death or discharge).
`import_checkpoint`, `run` and the smoke scripts all refuse `--reader dn` with
`--task mortality`.

**What changes.** The commands are built from
`medpatch/scripts/mortality/{Unimodal,Confidence,Calibrate}/*.sh`, which set
`--task in-hospital-mortality --labels_set mortality --num_classes 1` (epochs
100, batch 16, lr 0.001, `--data_pairs paired`, read from those files). The
EHR folder must contain `in-hospital-mortality/{train,val,test}_listfile.csv`
and `in-hospital-mortality/{train,test}/` from the extraction. Runs land in
`RUNS_ROOT/in-hospital-mortality/<stage>/<reader>/<id>/`, separate from
phenotyping. The metric recorded is mortality AUROC/AUPRC on the val split.

**Lineage is per task.** A mortality run only ever loads a mortality parent.
Pointing `--parent` at a phenotyping row (or the reverse) is refused, and
without `--parent` only rows of the same task are considered.

```python
!python -m tools.pv.import_checkpoint --task mortality --reader ehr --who "<who>" --file "/content/drive/MyDrive/<…>/best_checkpoint_0.001_in-hospital-mortality_unimodal_ehr_EHR_paired.pth.tar" --val-auroc <x> --val-auprc <y>
!python -m tools.pv.import_checkpoint --task mortality --reader cxr --who "Norhan" --file "/content/drive/MyDrive/<…>/best_checkpoint_<lr>_in-hospital-mortality_unimodal_cxr_EHR-CXR_paired.pth.tar" --val-auroc <x> --val-auprc <y>
!python -m tools.pv.import_checkpoint --task mortality --reader rr  --who "Farida" --bert-model-name dmis-lab/biobert-v1.1 --file "/content/drive/MyDrive/<…>/best_checkpoint_0.001_in-hospital-mortality_unimodal_rr_EHR-RR_paired.pth.tar" --val-auroc <x> --val-auprc <y>

!python -m tools.pv.run r2  --task mortality --reader rr --data real {DATA} --dry-run   # check first
!python -m tools.pv.run r2  --task mortality --reader ehr --data real {DATA}
!python -m tools.pv.run r2  --task mortality --reader cxr --data real {DATA}
!python -m tools.pv.run r2  --task mortality --reader rr  --data real {DATA}
!python -m tools.pv.run r2b --task mortality --reader ehr --data real {DATA}          # one session each
!python -m tools.pv.run r2b --task mortality --reader cxr --data real {DATA}
!python -m tools.pv.run r2b --task mortality --reader rr  --data real {DATA}
```

Resume Round 2 with `--run-id` exactly as for phenotyping; Round 2b still
cannot resume.

**Norhan's CXR mortality checkpoint was trained at lr 3e-5.** That only
describes how her Round 1 file was made. It does not change the Round 2 / 2b
settings, which come from the mortality Confidence/Calibrate scripts
(lr 0.001). Her file's name contains its own lr, so match `<lr>` above to it.

**EHR normalizer (check this one).** `fusion_main.py` defaults to the
_phenotyping_ normalizer file for every task (see "EHR normalizer" in
section 5). A mortality Round 1 run with `normalizer_state None` therefore used
the phenotyping file, and so will your Round 2 if you pass nothing, which keeps
them consistent. medpatch also ships a mortality file,
`medpatch/normalizers/ihm_ts1.0.input_str_previous.start_time_zero.normalizer`.
If the Round 1 EHR (or any reader's) `args.txt` shows that file, pass the same
one: `--normalizer-state /content/PneumoVision/medpatch/normalizers/ihm_ts1.0.input_str_previous.start_time_zero.normalizer`.
Round 2/2b must use whatever Round 1 used.

**Checks:** `PV_CHECK_TASK=in-hospital-mortality python -m pytest -m round2 -s`
(and `-m round2b`) checks the mortality rows only.

### Virtual mortality smoke (Colab or any CPU)

```python
!python -m tools.pv.smoke_round2  --task mortality              # generate -> r1 x3 -> r2 x3 -> checks
!python -m tools.pv.smoke_round2b --task mortality --skip-r1 --skip-r2   # r2b x3 -> checks
```

The first command creates `data/virtual/smoke-mortality` (SYNTHETIC — not a
scientific result: invented stays with a planted signal; the discharge-note
file is empty on purpose) and runs the readers EHR, CXR and RR.
