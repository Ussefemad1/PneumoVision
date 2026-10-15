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

## Confirmed facts (teammates, 2026-10-05/06)

**Note readers (Farida): phenotyping RR, phenotyping DN, mortality RR.**

- All use `dmis-lab/biobert-v1.1`, frozen, with mean pooling; the head is
  `Linear(768→512) + LayerNorm + Linear(512→C)`, trained at lr 1.2225e-4.
- That head is medpatch's own layout: `text_model.fc_rr`/`fc_dn` is the
  768→512 layer, and the `*_classifier` (`layernorm` + `fc`) is the rest.
- They were trained on cached BioBERT outputs, so **without BERT dropout**, then
  converted to medpatch's checkpoint format.
- Both RR files reproduce her numbers with `fusion_main.py --mode eval`, with
  no "Not Loaded"/"Not Found" keys. DN was checked on synthetic text only
  (0 missing, 0 extra keys), so run `check_weights` on it (step 4).

**Cohort and train lists.**

- Only subjects in
  `medpatch/mimic4extract/mimic3benchmark/resources/testset_iv.csv` are kept in
  **train**. Val and test are the shared files.
- Counts (train / val / test):

  | Task        | Train  | Val   | Test   |
  | ----------- | ------ | ----- | ------ |
  | phenotyping | 42,328 | 4,756 | 11,845 |
  | mortality   | 19,064 | 2,161 | 5,302  |

- Every reader's Round 2 / 2b / 3 uses these **filtered** train lists.
  `python -m tools.pv.prepare_lists` builds them (section 3b), not the notebook.
- Stay **30007216** is dropped from train as well: its chest X-ray is a
  permanent 404 upstream, so any loader that asks for it raises `KeyError` in
  `cxr_dataset.py`. Phenotyping train is then 42,327.
- Caroll's EHR and Norhan's CXR Round 1 were trained on the **unfiltered**
  train lists. Their weights are fine to load; just remember that their
  Round 1 numbers come from a larger train set than everything after.

**Notes folder for mortality:** the real `radiology.csv` plus a **header-only**
`discharge.csv`. The loader opens `discharge.csv` whenever notes are
requested, but mortality never uses discharge notes (they leak the outcome).
The header must include at least `subject_id,hadm_id,charttime,text`.

**Caroll's EHR runs:** `normalizer_state None`, `timestep 1.0`, stay 30007216
excluded. That matches what Round 2 uses by default (see "EHR normalizer" in
section 5).

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

## 3b. Build the filtered train lists

Run once per task after unzipping (and again after any re-unzip, which
restores the original list):

```python
!python -m tools.pv.prepare_lists --task phenotyping --listfile-dir /content/data/<ehr folder>
!python -m tools.pv.prepare_lists --task mortality   --listfile-dir /content/data/<ehr folder>
```

In `<ehr folder>/<task>/` it:

1. copies `train_listfile.csv` aside as `train_listfile.full.csv` the first
   time. That copy is never overwritten, and every later run rebuilds from it,
   so rerunning is safe;
2. keeps only train rows whose subject is in `testset_iv.csv` (Farida's filter);
3. drops the excluded stays by `stay_id` (default `--exclude-stay 30007216`).
   At most one row per excluded stay may go, or it stops;
4. leaves `val_listfile.csv` and `test_listfile.csv` unchanged.

It prints counts and md5 sums only, never rows, and refuses to write if the
counts differ from the confirmed ones:

| Task        | Train after filter | Train written                     | Val   | Test   |
| ----------- | ------------------ | --------------------------------- | ----- | ------ |
| phenotyping | 42,328             | 42,327                            | 4,756 | 11,845 |
| mortality   | 19,064             | 19,064 − 1 if 30007216 is present | 2,161 | 5,302  |

For mortality it reports whether 30007216 was _present in train, dropped_ or
_absent from train_; copy that line into the team Sheet. It also warns if an
excluded stay appears in val or test (those files are left as they are).
`--allow-count-mismatch` writes anyway, for a deliberate change only.

## 4. Check the weights, then register the real Round 1 files

Copy the files to Drive first if they aren't there, so the paths stay valid.

**4a. Check each file against the model Round 2 will build.** `--load_<reader>`
(medpatch's `Trainer.load_state`) is not strict. It copies matching names,
prints the rest as "Not Loaded"/"Not Found" and carries on, and its `copy_`
silently _broadcasts_ some wrong shapes. `check_weights` builds the reader's
Round 2 model (downloads BioBERT / the ViT on first use) and applies the file
the same way. It fails if any encoder or classifier tensor is missing or
mis-shaped, or if a text file's BERT weights are not the pretrained weights of
the named model. The confidence head is new in Round 2, so it is listed as
"expected new". The seven real Round 1 files:

```python
F = "/content/drive/MyDrive/<…>"   # where the Round 1 files are
!python -m tools.pv.check_weights --task phenotyping --reader ehr --file "{F}/<phenotyping EHR file>.pth.tar"
!python -m tools.pv.check_weights --task phenotyping --reader cxr --file "{F}/<phenotyping CXR file>.pth.tar"
!python -m tools.pv.check_weights --task phenotyping --reader rr  --bert-model-name dmis-lab/biobert-v1.1 --file "{F}/<phenotyping RR file>.pth.tar"
!python -m tools.pv.check_weights --task phenotyping --reader dn  --bert-model-name dmis-lab/biobert-v1.1 --file "{F}/<phenotyping DN file>.pth.tar"
!python -m tools.pv.check_weights --task mortality   --reader ehr --file "{F}/<mortality EHR file>.pth.tar"
!python -m tools.pv.check_weights --task mortality   --reader cxr --file "{F}/<mortality CXR file>.pth.tar"
!python -m tools.pv.check_weights --task mortality   --reader rr  --bert-model-name dmis-lab/biobert-v1.1 --file "{F}/<mortality RR file>.pth.tar"
```

Each ends with `PASS` or `FAIL (n)` and a list. Add `--trust-pickle` if a file
from a teammate can't be loaded with `weights_only=True`. Do not register a
file that fails: tell the model track which keys failed.

**4b. Register.** Each command checks the file is that reader's Round 1
checkpoint, records its sha256 and adds an `r1` row. `--check-weights` repeats
4a and writes e.g. `weights: 100% covered (…)` into the manifest notes.

```python
!python -m tools.pv.import_checkpoint --check-weights --reader ehr --who "Caroll" --file "/content/drive/MyDrive/<…>/best_checkpoint_0.001_phenotyping_unimodal_ehr_EHR_paired.pth.tar" --val-auroc <x> --val-auprc <y>
!python -m tools.pv.import_checkpoint --check-weights --reader cxr --who "Norhan" --file "/content/drive/MyDrive/<…>/best_checkpoint_0.001_phenotyping_unimodal_cxr_EHR-CXR_paired.pth.tar" --val-auroc <x> --val-auprc <y>
!python -m tools.pv.import_checkpoint --check-weights --reader rr  --who "Farida" --bert-model-name dmis-lab/biobert-v1.1 --file "/content/drive/MyDrive/<…>/<phenotyping RR file>.pth.tar" --val-auroc <x> --val-auprc <y>
!python -m tools.pv.import_checkpoint --check-weights --reader dn  --who "Farida" --bert-model-name dmis-lab/biobert-v1.1 --file "/content/drive/MyDrive/<…>/<phenotyping DN file>.pth.tar" --val-auroc <x> --val-auprc <y>
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

**Measured on an L4:** a note reader (RR/DN) takes about **70–85 min per
Round 2 epoch**.

**Loop order** (`MSMA_trainer.train`), every epoch:

1. **validate**. The first validation is of the untrained head. The best
   checkpoint is written when the validation loss improves;
2. **train** one pass over the train split;
3. **save `last`**, which is what `--run-id` resumes from.

So `--epochs N` gives N validations and N training passes, and the last pass
is **never validated**. Its weights are only in `last_checkpoint_…`, never in
`best_…`.

**Progress lines.** Each training pass prints
`progress HH:MM:SS epoch E train step i/n` at step 0 and every 200 steps, and
`run.py` shows them. Before this, a 70-minute pass printed nothing between
"starting train epoch" and the next validation, which looked like a hang.

`run.py` adds `--frozen_readers_eval` to every Round 2 and 2b command. The
released code puts the whole model in train mode for each training epoch, which
turns the frozen BERT's dropout (p = 0.1) on while the head trains. Farida's
readers were trained and evaluated without it, so the frozen reader now stays
in eval mode. It makes no difference for EHR (LSTM, no dropout) or CXR (the
ViT has no active dropout), and it never affects Round 3. It shows in the
`--dry-run` output and in the manifest notes.

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

**Loop order** (`Calibration.train`):

1. one inference pass over val, which gives the "before" ECE and tables;
2. then, every epoch, one **training pass over val**. The ECE is computed from
   the probabilities that pass produced, while the temperature moves batch by
   batch. The best checkpoint is written when the mean ECE improves.

There is no separate validation, no `last` checkpoint and no early stop, so
`--epochs N` is N training passes plus the one inference pass before them.

**Cached frozen logits (`--cache_frozen_logits`, on for rr/dn).** In Round 2b
only the temperature trains. With `--frozen_readers_eval`, the reader,
classifier and confidence head are deterministic, so the pre-temperature
confidence logits never change.

- **Without the cache,** the released loop re-ran BioBERT over every val note
  each epoch. That is about 6–10 min per epoch on an L4, so 10–17 h for 100
  epochs.
- **With the cache,** those logits are captured once during the inference pass
  (under `no_grad`, kept on CPU). Every epoch then replays them through the
  same temperature, loss, optimizer step, ECE tables and checkpoint code,
  batch for batch in `val_dl` order (`shuffle=False`).
- **Output:** the log prints
  `cached frozen logits: <batches> batches, <samples> samples, <MB> MB on CPU, single pass <s>`.
- **Refusals:** the trainer refuses the flag without `--frozen_readers_eval`,
  if any parameter other than the temperature would train, or if any dropout
  or batch-norm layer stays in train mode.
- **Defaults:** `run.py` passes it for rr and dn. For ehr and cxr it is off by
  default, so the finished EHR/CXR runs stay comparable; turn it on with
  `--cache-frozen-logits`, or off with `--no-cache-frozen-logits`. It is
  recorded as "cached frozen logits (numerically equivalent)".
- **Checking equivalence on Colab:** `python -m pytest -m round2b_cache -s`
  runs both versions for 2 epochs on virtual data and compares them. Set
  `PV_CHECK_TASK=in-hospital-mortality` for mortality.

**A run that finished training but crashed before the manifest.** If
`best_checkpoint_…` is in the run's folder but no row was written (for example
the old cross-device crash below), record it without training. Use the same
arguments plus the run's id:

```python
!python -m tools.pv.run r2b --reader ehr --data real {DATA} --run-id r2b-ehr-001 --record-only
```

It refuses if the id is already in the manifest, or if the best checkpoint is
missing (wrong id or arguments). It moves the run's leftover calibration files
from `medpatch/` into its folder, scores the checkpoint on val and writes the
row, exactly as a normal finish does. It works for `r2` too.

The cross-device crash itself (`OSError [Errno 18] Invalid cross-device link`
while moving `calibration_curve_class_*.png` from `/content/PneumoVision/medpatch`
to Drive) is fixed: those files are now moved with `shutil.move` (copy then
delete across filesystems), and a file that still cannot be moved is only a
warning. It can no longer change a run's outcome.

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

## BERT and train lists must match everywhere downstream

The RR/DN confidence heads trained here sit on top of **BioBERT**
(`dmis-lab/biobert-v1.1`) features, and every Round 2 / 2b / 3 run uses the
**filtered** train lists (see "Confirmed facts"). Anything that later loads
these checkpoints must build the same BERT and use the same lists. A different
BERT silently tokenizes with the wrong vocabulary:

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

### Round 3: use `c-msma`, not `c-e-msma`

`scripts/phenotyping/MedPatch/Confidence-Patching.sh` passes
`--fusion_type c-e-msma`, the same as `Entropy-Patching.sh`; the mortality
`Confidence-Patching.sh` passes `c-msma`.

- `c-e-msma` (`EMSMAFusion`) scores a token by binary **entropy**
  `−[p ln p + (1−p) ln(1−p)]`, which lies in [0, ln 2 ≈ 0.693], and marks it
  high-confidence when entropy **≥ θ** (`fusion.py:2463`, `:2469`). With the
  scripts' θ = 0.75, no token can ever qualify, so the high-confidence branch
  is always empty. With a lower θ it would select the _least_ confident
  tokens.
- `c-msma` (`CMSMAFusion`) is the paper's `max(p, 1−p)` in [0.5, 1], with
  high when ≥ θ (`fusion.py:1461`, `:1467`).

Both read the Round 2 confidence heads and Round 2b temperatures through the
same keys, frozen. **Run phenotyping Round 3 with `--fusion_type c-msma`**, and
treat `c-e-msma` as an entropy ablation that needs its threshold rule fixed
first. (CLAUDE.md's task table, which lists `c-e-msma` for pneumonia, came from
that script.)

### Round 3 with `--data_pairs partial` (missing notes or X-ray)

A stay with no radiology report (or, for phenotyping, no discharge note)
loads without crashing:

- the loader left-merges notes, so the text is NaN, and collate turns it into
  `""`;
- a missing X-ray becomes a zero image with `pairs=False`;
- `CMSMAFusion.detect_missingness_batch` (`fusion.py:1366-1378`) marks empty
  notes and all-zero images as missing, and the late fusion masks those
  modalities' predictions and weights (`fusion.py:1796-1812`).

**Risk:** the high/low token pools (`fusion.py:1478-1690`) ignore that mask. An
empty note is still run through BERT (`""` tokenizes to `[CLS][SEP]` plus
padding) and a missing X-ray through the ViT, and those tokens are scored by
the confidence heads and can enter the joint high/low predictions. That is
upstream behaviour, and the main run keeps it. `--mask-missing-tokens`
(Round 3 below) is the ablation that excludes an absent modality's tokens.

---

## 9. Round 3 (MedPatch fusion, `c-msma`)

Round 3 trains `CMSMAFusion` on top of the frozen Round 2b readers, with
`--data_pairs partial`, which keeps every EHR stay and treats a missing X-ray
or note as a missing modality. `python -m tools.pv.run r3` builds the command
from `MedPatch/Confidence-Patching.sh`, with these settings:

- `--fusion_type c-msma`. `c-e-msma` is refused unless `--fusion-type c-e-msma
--allow-entropy`, and then runs live, without the cache.
- `--data_pairs partial`, and every `--*_confidence_threshold 0.75`.
- `--bert_model_name` taken from the text parents in the manifest.
- `--load_*` set to the latest real **r2b** row of each reader; `--parents`
  overrides this.
- 100 epochs, batch 16.
- `--save_dir <RUNS_ROOT>/<task>/r3/<id>`.

Mortality has no DN parent (discharge notes leak the outcome), and every entry
point refuses one. Use the same `--normalizer-state` as the EHR parent's runs
(see "EHR normalizer" above).

**Why a cache.** Every reader, confidence head and temperature is frozen in
Round 3. The released loop still runs the ViT and BioBERT on every stay in
every epoch, twice for each text encoder, which comes to several hours per
phenotyping epoch on an L4. `--step cache` runs the readers **once per stay**,
in eval mode, and stores what the trainable part needs (about 0.5 MB per
phenotyping stay, about 0.02 MB for mortality). Training then reads only the
cache. The cached forward gives the same `high_conf` / `low` / `late` outputs
as the live one: unit-tested to ≤ 1e-5, and checked on real data by
`--step equivalence`.

**Steps:** `cache` → `equivalence` → `train` → `record`. Without `--step`, the
command runs all four.

- **cache:** builds shards on the VM disk (`/content/r3_cache/real/<task>`, not
  Drive). It is **resumable**: rerunning the same command skips finished shards,
  so a disconnect loses at most one. It prints each shard's size, the cache
  total and the free space. It copies the cache to Drive only with
  `--copy-cache-to <dir>`; the phenotyping cache is about 30 GB.
- **equivalence:** compares cached and live forwards on real validation
  batches and prints only the maximum differences.
- **train:**
  - trains the main run, then the same with `--ablation
without_missingness_module`, both from the same cache;
  - `--extra-ablations` adds without_joint_module and without_late_module;
  - every training run picks its head on validation (F3).
- **record:**
  - scores the chosen head on val and test and writes `r3_results.csv` (main
    and ablation, macro and target AUROC/AUPRC with 95% CIs);
  - adds the manifest row: stage `r3`, reader `fusion`, `parent_id` = the r2b
    ids joined with `;`;
  - `val_auroc` / `val_auprc` are pneumonia (class 21) or mortality, from the
    chosen head;
  - the notes also record presence counts and the share ≥ 0.75 per split, the
    equivalence result and every deviation.

**Before training,** `train` checks the cache's fingerprint against the run:
parents' sha256, thresholds, task, the lists' md5, the capture code commit and
the flags. It refuses a cache that differs.

**If the session dies after training,** finish with `--run-id <id>
--record-only`.

**Deviations recorded with each run:**

- readers in eval mode (BERT dropout off);
- train X-rays without augmentation;
- train EHR padding fixed at capture;
- F5.

**Options, all off by default:**

| Option                      | Effect                                                                   |
| --------------------------- | ------------------------------------------------------------------------ |
| `--late-weights-per-class`  | late-fusion weights sum to 1 per class (upstream: 1/25 per class)        |
| `--mask-missing-tokens`     | absent modalities add no high/low tokens (phantom-token ablation)        |
| `--missing-image-as-absent` | a selected X-ray missing on disk is treated as absent (counted, noted)   |
| `--extra-ablations`         | also without_joint_module and without_late_module                        |
| `--no-ablation`             | skip without_missingness_module                                          |
| `--preload`                 | keep the cache in RAM while training (needs the RAM: ~21 GB phenotyping) |

Each option is a medpatch flag, so a cached run can be repeated with an option
in minutes from the same cache.

### Notebook 1 — pre-flight and smoke (≈ 1 h)

Run Cells 1–3 above first, with `!git checkout model/round3` in Cell 1.

```python
# 1. which reader keys each Round 2b parent carries (counts only)
!python -m tools.pv.parent_keys --task phenotyping --trust-pickle
!python -m tools.pv.parent_keys --task mortality   --trust-pickle
# 2. train lists (once per runtime, after unzipping)
!python -m tools.pv.prepare_lists --task phenotyping --listfile-dir /content/data/<ehr folder>
!python -m tools.pv.prepare_lists --task mortality   --listfile-dir /content/data/<ehr folder>
# 3. pre-flight: counts, missing X-ray files, F2 load, share >= 0.75, cache size
!python -m tools.pv.r3_preflight --task phenotyping --data real {DATA}
!python -m tools.pv.r3_preflight --task mortality   --data real {DATA}
# 4. virtual smoke (SYNTHETIC): every step plus pytest -m round3, < 15 min each
!python -m tools.pv.smoke_round3 --task mortality
!python -m tools.pv.smoke_round3 --task phenotyping
# 5. Rounds 2 / 2b on test: 25-label macro and pneumonia / mortality, 95% CIs
!python -m tools.pv.score_rounds --task phenotyping --data real --out /content/drive/MyDrive/PneumoVision_shared/round3/phenotyping/r2_r2b_test.csv
!python -m tools.pv.score_rounds --task mortality   --data real --out /content/drive/MyDrive/PneumoVision_shared/round3/mortality/r2_r2b_test.csv
```

If a pre-flight prints `STOP … not on disk`, add `--missing-image-as-absent` to
the pre-flight **and** every `run r3` command of that task.

### Notebook 2 — mortality Round 3

```python
!python -m tools.pv.run r3 --task mortality --data real {DATA} --dry-run   # prints the next id
RID = 'r3-fusion-001'   # <- the id the dry run printed; every step below uses it
!python -m tools.pv.run r3 --task mortality --data real {DATA} --run-id {RID} --step cache   # rerun to resume
!python -m tools.pv.run r3 --task mortality --data real {DATA} --run-id {RID} --step equivalence --equivalence-batches 2
!python -m tools.pv.run r3 --task mortality --data real {DATA} --run-id {RID} --step train
!python -m tools.pv.run r3 --task mortality --data real {DATA} --run-id {RID} --step record
!PV_CHECK_TASK=in-hospital-mortality python -m pytest -m round3 -s
!mkdir -p /content/drive/MyDrive/PneumoVision_shared/round3/mortality
!cp -r "$RUNS_ROOT/in-hospital-mortality/r3/{RID}" /content/drive/MyDrive/PneumoVision_shared/round3/mortality/
```

If the session dies during `train`, rerun the same `--step train` command: it
resumes each variant from its `last_checkpoint_…`. If it dies after training
but before the row is written, use `--step record`, or `--record-only`.

### Notebook 3 — phenotyping Round 3

Same as Notebook 2 with `--task phenotyping`, `PV_CHECK_TASK=phenotyping`,
the folder `$RUNS_ROOT/phenotyping/r3/<id>` and the shared folder
`PneumoVision_shared/round3/phenotyping/`. The cache is about 30 GB; check that
the pre-flight's free-space line allows it.

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
!python -m tools.pv.import_checkpoint --check-weights --task mortality --reader ehr --who "<who>" --file "/content/drive/MyDrive/<…>/best_checkpoint_0.001_in-hospital-mortality_unimodal_ehr_EHR_paired.pth.tar" --val-auroc <x> --val-auprc <y>
!python -m tools.pv.import_checkpoint --check-weights --task mortality --reader cxr --who "Norhan" --file "/content/drive/MyDrive/<…>/best_checkpoint_<lr>_in-hospital-mortality_unimodal_cxr_EHR-CXR_paired.pth.tar" --val-auroc <x> --val-auprc <y>
!python -m tools.pv.import_checkpoint --check-weights --task mortality --reader rr  --who "Farida" --bert-model-name dmis-lab/biobert-v1.1 --file "/content/drive/MyDrive/<…>/best_checkpoint_0.001_in-hospital-mortality_unimodal_rr_EHR-RR_paired.pth.tar" --val-auroc <x> --val-auprc <y>

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
