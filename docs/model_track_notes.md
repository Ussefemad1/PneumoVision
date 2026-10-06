# Model track notes

A log of every change made to `medpatch/` for the model track, and of known
deviations between the paper's scripts and how we run them. Rule: changes to
`medpatch/` are minimal and either flag-gated or fix a path that could not run
at all; with default flags, behaviour is the paper's.

Newest first.

---

## 2026-10-06 — Round 2b cached frozen logits; progress lines

Real Round 2b on Colab re-ran BioBERT over ~4,500 val notes every epoch (about
6–10 min per epoch on an L4, so 10–17 h per 100-epoch run), although only the
temperature trains.

11. **`--cache_frozen_logits`** (`arguments.py`, default off = released
    behaviour; Round 2b only).
    - **Capture** (`trainers/Calibration.py`): during the existing pre-training
      inference pass, a forward hook on `<reader>_confidence_predictor` stores
      its output under `torch.no_grad()` on CPU. That output is the tensor the
      temperature divides in `TempCUnimodal*.forward` (`fusion.py:1009`,
      `:1061`, `:1110`, `:1168`). The labels for each batch are stored with it.
    - **Replay:** each epoch applies
      `raw / temperature[:L].clamp_min(1e-9).unsqueeze(0)`, the same
      expression as `forward`, batch by batch in `val_dl` order. `val_dl` is
      `shuffle=False` (`DataFusion.py:513`) and val uses the deterministic CXR
      transforms.
    - **Shared code:** everything after the model call (`pad_to_length`, loss,
      optimizer step, ECE, tables, checkpoints and their names) is the same
      code for both paths. `train_epoch` now takes `(pred, y)` from
      `confidence_batches()`.
    - **Refusal** (`check_logit_cache`): it raises `SystemExit` unless
      `--frozen_readers_eval` is on, the temperature is the only parameter
      with `requires_grad`, and no Dropout/BatchNorm stays in train mode.
    - **Tests:**
      - `tests/test_logit_cache.py` runs the real `calibration.train()` for
        3 epochs on a stub reader, both tasks. Cached and uncached give
        temperatures within 1e-6, the same best epoch, the same checkpoint
        tensors and the same ECE tables. The reader runs once instead of once
        per epoch.
      - A 0.1 % change to the replayed temperature fails it.
      - Refusals are tested too.
      - `tests/round2b/test_round2b_cache.py` (marker `round2b_cache`, trains,
        Colab only) does the same through `tools.pv.run` on virtual data.
    - **Plans:** `tools/pv/run.py` passes the flag on r2b plans for rr/dn
      (`--cache-frozen-logits` / `--no-cache-frozen-logits` override; off by
      default for ehr/cxr), with the deviation "cached frozen logits
      (numerically equivalent)". **Plan diff:** 6 of 42 plans change (r2b rr
      and dn phenotyping, r2b rr mortality, virtual and real), each gaining
      only `--cache_frozen_logits` and that note.
12. **Progress lines** (output only). `Trainer.progress()` prints
    `progress HH:MM:SS epoch E <train|inference> step i/n` at step 0 and every
    200 steps. It is called from `MSMA_Trainer.train_epoch` and
    `calibration.train_epoch`, and `tools/pv/run.py` lets these lines (and
    "starting train epoch") through its output filter.

---

## 2026-10-06 — Real-data pre-flight: frozen-reader dropout, strict weight check, Round 3 findings

**Confirmed facts (teammates, 2026-10-05/06).**

- **Note readers (Farida):** phenotyping RR, phenotyping DN, mortality RR, all
  `dmis-lab/biobert-v1.1`.
  - BERT frozen, mean pooling; head `Linear(768→512) + LayerNorm +
Linear(512→C)` (= `text_model.fc_*` + `*_classifier`), lr 1.2225e-4.
  - Trained on cached BioBERT outputs, i.e. without BERT dropout, then
    converted to medpatch format.
  - Both RR files reproduce her numbers with `--mode eval` and no "Not
    Loaded"/"Not Found" keys; DN was checked on synthetic text only.
- **Train lists:** filtered to the subjects in
  `mimic3benchmark/resources/testset_iv.csv` (train only; val/test shared).
  - phenotyping 42,328 / 4,756 / 11,845; mortality 19,064 / 2,161 / 5,302.
  - Round 2 / 2b / 3 of every reader use these filtered lists. Caroll's EHR
    and Norhan's CXR Round 1 saw the unfiltered lists.
- **Mortality notes:** real `radiology.csv` plus a header-only `discharge.csv`.
- **Caroll's EHR:** `normalizer_state None`, `timestep 1.0`, stay 30007216
  excluded.

**A1 — frozen readers and dropout (changed, flag-gated).**

- `MSMA_trainer.py:472` (`self.model.train()` before each training epoch) and
  `Calibration.py:103` (`self.model.train(not inference)`) put the whole model
  in train mode, frozen reader included.
- Active dropout in the readers, checked on randomly initialised modules:
  BERT-base has 37 `Dropout(p=0.1)`; `vit_small_patch16_384` (timm defaults)
  and the 1-layer LSTM (`dropout=0.0`) have none.
- So in Round 2 / 2b, RR/DN heads trained on dropout-noised BERT features,
  unlike how Farida's readers were trained and evaluated.

10. **`--frozen_readers_eval`** (`arguments.py`, default off = released
    behaviour). `Trainer.keep_frozen_readers_in_eval()` (`trainers/trainer.py`)
    runs right after `model.train()` in `MSMA_Trainer.train` and
    `calibration.train_epoch`.
    - It puts every subtree that has parameters and none trainable into eval.
      The confidence head and temperature keep train mode.
    - Only for `c-unimodal_*` / `temp_c-unimodal_*`: Round 3
      (`c-msma`/`c-e-msma`) is never touched.
    - `tools/pv/run.py` passes it on every r2/r2b plan. **This is the only
      run-plan change:** 28 of 42 plans (all r2/r2b, both tasks, virtual and
      real) gain `--frozen_readers_eval` and a deviation note. The 14 r1 plans
      are identical.
    - Effect on the verified virtual runs: none for EHR/CXR (no active
      dropout). RR/DN r2 and r2b results change; their smoke runs need
      re-running.
    - Tests: `tests/test_frozen_eval_and_ece.py`.

**A2 — one-class ECE (no change needed).**

- Released `train()` passed `[N, L]` mortality probs to `compute_ece`, whose
  loop over `num_classes = 1` read `probs[:, 0]`: **token 0 only**, as Farida
  reports.
- Since 9c5d50c, `train()` uses `flat_ece`, which reshapes to `[N·L, 1]`, so
  every token is measured. For mortality today:
  - EHR: all 48 bins;
  - CXR: its 1 token (the CLS default);
  - RR: all 512 positions, padding included, as the head is trained on all of
    them.
- The per-token tables use `probs[:, t]` for every t. Best-epoch selection
  (`post_ece.mean()`) is therefore over all tokens; it changed in 9c5d50c and
  not now.
- Pinned by `test_one_class_ece_is_over_all_tokens_not_token_0`.

**A3 — Round 3 fusion type (no code change; recommendation).**

- `CMSMAFusion` uses `max(p, 1−p)` ∈ [0.5, 1] and high when ≥ θ
  (`fusion.py:1461/1467`, `1526/1545`, `1592/1611`, `1654/1672`).
- `EMSMAFusion` uses binary entropy ∈ [0, ln 2 ≈ 0.693] and high when ≥ θ
  (`fusion.py:2463/2469`, `2528/2547`, `2594/2612`, `2655/2672`). With the
  scripts' θ = 0.75 no token is ever high; with a lower θ it would pick the
  least confident tokens.
- Both consume the Round 2 heads and Round 2b temperatures through the same
  keys (`*_confidence_predictor`, `*_temperature`), frozen.
- `scripts/phenotyping/MedPatch/Confidence-Patching.sh` passes `c-e-msma`
  (same as `Entropy-Patching.sh`); `scripts/mortality/MedPatch/Confidence-Patching.sh`
  passes `c-msma`.
- **Recommendation:** run Round 3 with `c-msma` for both tasks; treat
  `c-e-msma` as an entropy ablation needing its threshold rule fixed.
  CLAUDE.md's task table (`c-e-msma` for pneumonia) came from that script and
  should be revisited.

**A4 — `--data_pairs partial` without notes (no change).**

- A stay with no RR (or DN) report loads without crashing: left merges at
  `DataFusion.py:334`, `:388`, `:428` give NaN text, and collate turns it into
  `""`. A missing X-ray becomes a zero image with `pairs=False`.
- `CMSMAFusion.detect_missingness_batch` (`fusion.py:1366-1378`) marks those as
  missing, and the late fusion masks their predictions and weights
  (`fusion.py:1796-1812`).
- **Risk:** the high/low token pools (`fusion.py:1478-1690`) ignore the mask.
  `""` is BERT-encoded as `[CLS][SEP]` plus padding (`text_models.py:97` only
  zeroes when there are no chunks) and a zero image is ViT-encoded, so phantom
  tokens can enter the joint high/low predictions. Upstream behaviour.

**How a parent is loaded today.** `--load_<reader>` → `Trainer.load_state`.

- It is not strict and filters nothing by reader: every checkpoint key whose
  name exists in the model is copied with `own_state[name].copy_(param)`.
- "Not Loaded" lists checkpoint keys absent from the model; "Not Found" lists
  model keys absent from the checkpoint (left at their initial values). Both
  are printed, never fatal.
- A shape mismatch is not reported: `copy_` broadcasts (a `[512]` or
  `[1, 512]` tensor fills a `[25, 512]` parameter silently) and only raises for
  non-broadcastable shapes.

**New: `tools/pv/check_weights.py`** (+ `import_checkpoint --check-weights`).

- Builds the reader's Round 2 (or 2b) model from the same paper-script argv,
  compares names and shapes, applies the file via `Trainer.load_state` and
  verifies the values.
- Fails on any missing or mis-shaped encoder/classifier tensor, and for rr/dn
  if the file's frozen BERT is not the pretrained `--bert-model-name`. Shapes
  cannot catch that: BioBERT and Bio_ClinicalBERT share a 28,996 vocabulary.
- Confidence head / temperature are "expected new".
- Tests: `tests/test_check_weights.py` (encoders stubbed, both tasks).

**Downstream:** Round 3 and the inference service must use the same BERT
(`dmis-lab/biobert-v1.1` for these text readers) and the same filtered train
lists.

---

## 2026-10-05 — In-hospital mortality in `tools/pv` (Round 2 / 2b)

**Scope:** task `in-hospital-mortality` (CLI `--task mortality`), readers EHR,
CXR, RR. **DN is refused for mortality everywhere** (script lookup,
`import_checkpoint`, `run`, smoke): discharge notes leak the outcome.

**`medpatch/` (one change):**

9. **`Trainer.confidence_logits` with one class** (`trainers/trainer.py`).
   With `num_classes 1`, `ConfidencePredictor` already squeezes the class axis,
   so a mortality head outputs `[B, tokens]`. For CXR's single default token
   that is `[B, 1]`, and `confidence_logits` dropped that axis to `[B]`. The
   next line (`y.unsqueeze(1).repeat(1, pred.shape[1])`) then raised in both
   `MSMA_Trainer` (r2) and `calibration` (r2b). Now only a **3-D** output loses a
   size-1 last axis. Phenotyping outputs are always 3-D `[B, L, 25]`, so that
   path is unchanged. Covered by `test_confidence_logits_handles_one_class`.

Checked and left alone (they already work with one class): MSMA's mortality
branches (`repeat(1, pred.shape[1])`, the `num_classes > 1` max);
`Calibration.train_epoch` (EHR padding is phenotyping-only, mortality EHR is a
fixed 48 bins), `flat_ece`/`compute_ece` with `[N, L]` → `[N·L, 1]`, the
per-token tables (`num_classes = 1` branch); `TempCUnimodal*` 1-D temperature
`[max_seq_len]` against `[B, L]` logits.

**`tools/pv`:**

- `paper_scripts`: task-aware (`script_path(stage, reader, task)`), reading
  `medpatch/scripts/mortality/...`; `SCRIPT_FOR`, `build_argv` and
  `script_settings` take an optional task (default phenotyping, so existing
  callers are unchanged).
- `manifest`: the `task` column already existed; empty values read as
  `phenotyping`; `latest(..., task=)`; `verify_parent(..., task=)` refuses a
  parent of another task.
- `import_checkpoint --task`; `run --task` (folder
  `RUNS_ROOT/in-hospital-mortality/...` for mortality; phenotyping layout
  unchanged), task in `run.json`, the manifest row and `--dry-run`.
- `run --normalizer-state` (real data, opt-in): `fusion_main.py` defaults to
  the _phenotyping_ normalizer for every task, so a mortality Round 2 must pass
  whatever its Round 1 used (see `docs/REAL_RUNS.md`).
- `evaluate`: target class per task (pneumonia 21 / mortality 0); handles
  `[B]` labels and `[B, L]` one-class token outputs.
- Smoke: `smoke_round2 / smoke_round2b --task mortality`; the checks filter
  rows by `PV_CHECK_TASK` (set by the smoke scripts; unset = previous
  behaviour) and lineage now also requires the same task.
- `tools/synthetic/make_virtual_dataset.py --task mortality`:
  `generate_mortality()` writes the benchmark's in-hospital-mortality layout
  (`stay,period_length,stay_id,y_true`, first 48 h only, CXR and RR inside the
  48 h, header-only `discharge.csv`) to `data/virtual/<preset>-mortality`.
  `generate()` is untouched.

**Phenotyping parity (verified, not assumed):** the phenotyping `run.plan()`
output (argv, save_dir, script, parent, deviations, BERT) for r1/r2/r2b ×
4 readers × virtual/real was dumped before and after this change and is
identical, and the phenotyping smoke dataset is byte-identical (102 files).

**Not verified here (no training on this machine):** an actual mortality
r2/r2b run. `tests/test_mortality_tools.py` (38) covers the non-training paths,
including the mortality virtual data loading through medpatch's loaders.

---

## 2026-10-05 — `--bert_model_name`, BERT lineage, real-data prep (phenotyping)

**Why:** Farida trained the real RR/DN Round 1 readers with
`dmis-lab/biobert-v1.1`. medpatch's text encoder defaulted to
`emilyalsentzer/Bio_ClinicalBERT` with no way to change it from the command
line. Loading her checkpoints that way raises no error (same BERT-base shapes)
but tokenizes with the wrong vocabulary and runs the wrong pretrained weights.

**`medpatch/`:**

8. **`--bert_model_name`** (`arguments.py`, default
   `emilyalsentzer/Bio_ClinicalBERT`, so every verified virtual run is
   unchanged). `models/text_models.py` already read `args.bert_model_name` for
   both `BertModel` and `BertTokenizerFast`; it just had no flag. That
   `Text_encoder` is what Round 2 (`MSMA_Trainer`), Round 2b (`calibration`)
   and Round 3 (`c-msma`/`c-e-msma`, also `MSMA_Trainer`) build. Not changed:
   ~20 other trainers (ensembles, DHF, staged, …) and `models/rr_encoder.py`
   hardcode Bio_ClinicalBERT; none is on the r2/r2b/r3 path.

**`tools/pv`:**

- Manifest column `bert_model_name` (last, so older files keep their order). An
  older manifest is upgraded in place on the next append (temp file + atomic
  replace); older rows read as empty. `bert_model_name_of(row)`: virtual rr/dn
  rows without a value used the default; real rows without one are "unknown".
- `import_checkpoint --bert-model-name`: required for real rr/dn (no silent
  default), refused for ehr/cxr.
- `run.py`: r2/r2b inherit the parent's BERT and pass `--bert_model_name`. An
  explicit `--bert-model-name` that differs from the parent's is refused, and a
  real rr/dn parent with no recorded BERT is refused. Recorded in `run.json`
  and the new manifest row.
- `run.py --dry-run`: verifies the parent (stage, reader, data kind, file,
  sha256) and prints the parent, the BERT and the exact `fusion_main.py`
  argv, then exits. It creates no folder.
- Tests: `tests/test_bert_model_name.py` (18, non-training; `from_pretrained`
  stubbed).

**Docs:** `docs/REAL_RUNS.md` — Colab steps for real data, including the
downstream-BERT warning. Round 3 and the website's inference service (which
hardcodes Bio_ClinicalBERT in `services/inference/app/main.py`, not changed
here) must use the checkpoints' BERT.

**Not in this change:** mortality support in `tools/pv`.

---

## 2026-09-28 — transformers 4.44.2 → 4.45.2 (Colab on Python 3.13)

**Why:** Colab's runtime moved to Python 3.13. `transformers==4.44.2` requires
`tokenizers<0.20`, and tokenizers 0.19.x publishes no cp313 wheel (the newest is
cp312). pip therefore falls back to building tokenizers from source, which fails
on a fresh Colab runtime even with Rust installed. So `requirements-colab.txt`
no longer installed. `transformers==4.45.2` requires `tokenizers>=0.20,<0.21`,
which has cp313 wheels; it is the smallest bump that installs.

**What changed:** `transformers` pinned to 4.45.2 in both `requirements.txt` and
`requirements-colab.txt`, so local and Colab stay matched. No code change. The
local environment resolves `tokenizers` 0.20.3.

**Verified (locally, non-training only):** the tooling imports and
`pytest -m "not round2 and not round2b"` (100 passed). `test_environment.py`
imports `models.text_models`, which imports medpatch's
`BertModel`/`BertTokenizerFast`, so those imports load on 4.45.2.

**Not verified:** Bio_ClinicalBERT outputs on 4.45.2 vs 4.44.2. No text model
was run (training and inference happen on Colab). A minor-version bump should
not change `BertModel` numerics, but the first Colab run of RR/DN should be
compared against earlier virtual results if exact reproducibility matters.
Also not verified: that requirements-colab.txt now installs on a fresh Colab
runtime (the cp313 wheel availability was checked by the user on PyPI).

---

## 2026-09-28 — Round 2b (calibration) on virtual data, code only

**Status: written and checked without training.** Imports and the non-training
test suite pass. `pytest -m round2b` and `tools/pv/smoke_round2b.py` have
**not** been run yet; that run happens on Colab. Everything below about
runtime behaviour is from reading the code.

### Round 2 → Round 2b map (verified by reading the code)

|                    | Round 2                        | Round 2b                                                                                                                    |
| ------------------ | ------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| Script             | `Confidence/Confidence-<R>.sh` | `Calibrate/Calibrate-<R>.sh`                                                                                                |
| `--fusion_type`    | `c-unimodal_<r>`               | `temp_c-unimodal_<r>`                                                                                                       |
| Trainer            | `MSMA_Trainer`                 | `trainers/Calibration.py` `calibration` (routed by `fusion_main.py`: `'temp_c-unimodal' in fusion_type`) — no wiring needed |
| Model              | `Unimodal<R>Confidence`        | `TempCUnimodal<R>`: same submodule names plus `<r>_temperature`                                                             |
| Loads              | r1 file                        | **r2** file, via the same `--load_<r>` flag (`Calibrate-DN.sh` already uses `--load_dn`)                                    |
| Trains             | the confidence head, on train  | only `<r>_temperature`, on **val only**                                                                                     |
| Best checkpoint by | lowest val loss                | lowest val ECE (saved only when it improves)                                                                                |

Key match: `TempCUnimodal<R>` registers `<r>_model` / `text_model`,
`<r>_classifier` and `<r>_confidence_predictor` under the same names as Round 2,
so `load_state` restores all of them from the r2 file. Only `<r>_temperature`
(init 1.0) is new. `tests/test_round2b_tools.py::test_r2_keys_cover_every_r2b_weight_but_temperature`
checks this for all four readers.

Temperature shape: `[max_seq_len, classes]`: EHR 2646×25, CXR 578×25,
RR/DN 512×25. With the default CXR input (one CLS token), only row 0 of CXR's
578 rows is ever used.

### Changes to `medpatch/`

6. **`confidence_logits` moved to the base `Trainer`** (`trainers/trainer.py`;
   removed from `MSMA_trainer.py`, behaviour unchanged). Round 2 and Round 2b
   now share it without importing each other.

7. **Round 2b trainer fixes** (`trainers/Calibration.py`). As released it could
   not run on any reader with the default CXR input, nor on EHR:
   - **squeeze:** `train_epoch` used `output[...].squeeze()`, dropping CXR's
     single token (`[B,1,25]` → `[B,25]`), so the target became `[B,25,25]`.
     Now uses `self.confidence_logits(output)`.
   - **EHR length:** EHR batches differ in token length (one per hour). Only
     `probs` was padded (`pad_to_length(probs, 48)`), and that crashed for any
     batch longer than 48; the labels were never padded, so `torch.cat(outGT)`
     failed. `pad_to_length` now truncates to 48 as well as padding, and the
     labels are padded the same way. The loss still uses the full length; only
     the ECE bookkeeping is cut to 48 tokens (the length upstream chose).
     Padded positions (prob 0, label 0) count as confident and correct in the
     trainer's own ECE, as they did in upstream's intent. `tools/pv/evaluate.py`
     reports ECE over real tokens only.
   - **ECE input shape:** `compute_ece` documents `[N, classes]`, but `train()`
     passed `[N, tokens, classes]`, so `probs[:, c]` selected token c: the
     "per-class" ECE (also the checkpoint-selection criterion) was really per
     token index 0–24, and with CXR's single token it raised `IndexError`. New
     `flat_ece()` flattens the token axis first; `train()` and the
     calibration-curve plot use it. **This changes which epoch is selected as
     best for RR/DN/EHR** compared with the released code (which selected on
     the per-token-index quantity). The per-token ECE table is unchanged.

### Tooling

- `manifest.STAGES` has `r2b`; `PARENT_STAGE = {"r2": "r1", "r2b": "r2"}`;
  `verify_parent(..., expected_stage=)` and `verify_r2_parent()`.
- `run.py r2b`: parent = the latest **r2** row for the reader/data kind (or
  `--parent`), verified like r1 parents. Virtual defaults: batch 4, epochs
  EHR/CXR 30, RR/DN 10 (see `VIRTUAL_R2B_EPOCHS` for the reasoning: at the
  paper's lr of 0.001 the temperature moves ≤ ~0.001 per step).
- `run.py` now moves any file a trainer writes into `medpatch/` (its working
  directory) into `<run>/medpatch_outputs/`. Calibration writes ECE tables,
  per-token probability CSVs and 25 PNGs there.
- Manifest `notes` for r2b rows record ECE before → after on val (the fit split)
  and test (held out). "Before" = the r2 parent in the r2b model with
  temperature 1.0.

### Known limits (not changed)

- `Calibration.py` has no resume; `--resume` is accepted and ignored.
- It writes a per-token probability CSV with one row per (sample, token), and
  25 PNGs per ECE pass. That is fine on virtual data; on real data (thousands of
  stays × 512–2646 tokens) it will be large and slow.
- If val ECE never improves, no best checkpoint is written and `run.py` stops
  with a clear error.

---

## 2026-09-27 — Round 2 on virtual data (branch `model/round2-virtual`)

### Round 1 → Round 2 map (verified by reading the code)

|                    | Round 1                                                                                                                                    | Round 2                                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| Script             | `scripts/phenotyping/Unimodal/{EHR,CXR,RR,DN}.sh`                                                                                          | `scripts/phenotyping/Confidence/Confidence-{EHR,CXR,RR,DN}.sh`                                            |
| `--fusion_type`    | `unimodal_{ehr,cxr,rr,dn}`                                                                                                                 | `c-unimodal_{ehr,cxr,rr,dn}`                                                                              |
| Trainer            | `MSMA_Trainer` (fallback branch of `fusion_main.py`)                                                                                       | same                                                                                                      |
| Model              | `Unimodal{EHR,CXR,RR,DN}`: reader + `*_classifier`                                                                                         | `Unimodal*Confidence`: reader + `*_classifier` + `*_confidence_predictor` (`Linear(dim → 25)` per token)  |
| Loads              | nothing                                                                                                                                    | `--load_{ehr,cxr,rr}` (DN: see below) via `Trainer.load_state`, which copies every key whose name matches |
| Best checkpoint by | highest val mean AUROC                                                                                                                     | lowest val loss (no AUROC is computed)                                                                    |
| File               | `{save_dir}/phenotyping/{fusion_type}/best_checkpoint_{lr}_phenotyping_{fusion_type}_{modalities}_paired.pth.tar` (+ `last_…` every epoch) | same pattern                                                                                              |
| Contents           | `epoch, state_dict, best_auroc, optimizer, patience`                                                                                       | same (`best_auroc` holds the val loss)                                                                    |

What trains in Round 2: every reader parameter is set `requires_grad=False`
unconditionally in `Unimodal*Confidence.__init__`. The optimizer receives all
model parameters, but only the confidence predictor receives gradients:
`*_classifier` still has `requires_grad=True` yet is unused in the Round 2
forward, so it gets no gradient and Adam leaves it unchanged. Checked by
`pytest -m round2` (check a).

Text readers: Bio_ClinicalBERT is frozen in _both_ rounds, and the Round 2
head reads raw BERT hidden states (not Round 1's `fc_rr`/`fc_dn`). A Round 1
text checkpoint therefore does not change Round 2's text output; it matters
for lineage only.

Each epoch validates **before** training, so an `--epochs 1` run's best
checkpoint is the untrained model. Stand-ins use at least 2 epochs.

### Changes to `medpatch/`

1. **`c-unimodal_cxr` could not be constructed** (`models/cxr_encoder.py`).
   `UnimodalCXRConfidence` sizes its head from `cxr_model.full_feats_dim`, which
   `CXRTransformer` did not define → `AttributeError`. Added
   `self.full_feats_dim = self.feats_dim` (patch tokens and pooled feature have
   the same width, 384). Additive; no other code reads it.

2. **CXR confidence input: one helper for Round 2, 2b and 3** (`models/fusion.py`,
   `cxr_confidence_input`; flag `--cxr_token_confidence` in `arguments.py`).
   The four sites that feed `cxr_confidence_predictor` — `UnimodalCXRConfidence`
   (Round 2), `TempCUnimodalCXR` (Round 2b), `CMSMAFusion` and `EMSMAFusion`
   (Round 3) — now all go through one function, so a head can never be trained
   on one representation and used on another (CLS and patch tokens are both
   384 wide, so a mismatch would not crash).
   - **Default (flag off) = the released code's choice**: the second element of
     the encoder output, i.e. the **CLS vector, one confidence per image**, as
     the reproduction gate requires.
   - It is passed as a length-1 sequence `[B, 1, D]`, not a bare `[B, D]`. The
     released code cannot run with a bare CLS vector anywhere: Round 2's trainer
     repeats the target over `pred.shape[1]` (giving `[B, 25, 25]` against a
     `[B, 25]` prediction), Round 2b slices its temperature by `shape[1]`
     (the class axis), and Round 3 unpacks `B, L, D = feats.shape`. `[B, 1, D]`
     keeps "one confidence per image" and is the smallest change that runs.
   - In Round 3 the high/low projections read the same features as the
     confidence (their token masks are built from it).
   - `tests/test_cxr_confidence_input.py` hooks the predictor in all four
     classes and asserts one shape per flag value: `(B, 1, 384)` off,
     `(B, 577, 384)` on. On the previous commit it fails (Round 2 tokens vs
     Round 2b/3 CLS).
   - An earlier commit on this branch had switched Round 2 alone to patch
     tokens; that is now behind the flag.

3. **Checkpoints saved on GPU could not load on CPU** (`trainers/trainer.py`,
   `load_state`). Added `map_location=self.device`. On the device a file was
   saved from this is identical to before; trainers without a `device`
   attribute keep torch's default.

4. **`--bootstrap_iters` (flag-gated, default 1000 = paper)**
   (`arguments.py`, `trainers/trainer.py` `computeAUROC`). Round 1 bootstraps
   1000 resamples × 25 classes for every train and val AUROC, which was ~90% of
   a 7.6-minute CPU smoke run. Virtual r1 stand-ins pass `--bootstrap_iters 20`
   (recorded in the manifest). The point estimates are unaffected; only the CI
   resolution changes.

5. **Round 2 trainer kept the token axis only when there were several tokens**
   (`trainers/MSMA_trainer.py`, new `confidence_logits`). `train_epoch` and
   `validate` did `pred.squeeze()` on the `c-unimodal` output, which drops
   _every_ size-1 axis. With the default CXR input (CLS as one token,
   `[B, 1, 25]`) that removed the token axis, the target was repeated over the
   class axis (`[B, 25, 25]`) and the loss failed on shape at the first
   validation. Now only the trailing class axis is dropped, and only when it is
   1 (mortality). For every case the old code handled (batch and tokens > 1)
   the result is identical; it also no longer drops the batch axis when a
   batch has one sample. Covered by `test_round2_trainer_keeps_the_token_axis`.
   **Resolved for Round 2b (2026-09-28):** `trainers/Calibration.py` had the
   same bare `.squeeze()`; `confidence_logits` moved to the base `Trainer` and
   both trainers use it. See the Round 2b section above.

### Planned ablations

- **`--cxr_token_confidence` — per-patch CXR confidence.** Off by default.
  Switches Round 2, 2b and 3 together from one CLS confidence per image to one
  confidence per ViT patch (577 tokens), which is what "token-level
  confidence" suggests for the other readers. Run after the reproduction gate
  (week 7, AUROC 0.88–0.92) on the released behaviour. Not yet evaluated on
  real data. On the virtual smoke set the per-patch head scored val AUROC 0.417
  (8 images, 2 positives): meaningless as a result; recorded only to show the
  flag path runs end to end.

### Deviations in how we run the paper's scripts (no code change)

- **DN load flag.** `Confidence-DN.sh` passes `--load_rr` for the DN reader —
  a copy-paste slip (`--load_dn` exists in `arguments.py`). It happens to be
  harmless for training (the frozen BERT loads identically; `fc_rr` and
  `rr_classifier` don't exist in an EHR-DN model and are skipped; `fc_dn` and
  `dn_classifier` are unused in Round 2), but it records the wrong parent.
  `tools/pv/run.py` passes `--load_dn`. The `.sh` file is left untouched.
- **`--notes_data_dir`** is required for RR/DN (`DataFusion.load_cxr_ehr_rr_dn`)
  but no paper script passes it; `run.py` supplies it.
- **wandb**: `MSMA_Trainer` always calls `wandb.init`; tooling sets
  `WANDB_MODE=disabled`.
- **Smoke overrides** (virtual data only, recorded per row in the manifest
  `notes`): r1 `--epochs 2 --batch_size 4 --bootstrap_iters 20`; r2
  `--batch_size 4` with `--epochs 20` for EHR and CXR and `--epochs 5` for RR
  and DN (a one-layer head on a 30-stay training set needs more passes than 5;
  BERT epochs are the expensive ones on CPU). Everything else is taken verbatim
  from the scripts by `tools/pv/paper_scripts.py`.

### Virtual data design choices

- The smoke preset gives an X-ray to 50% of stays (not the realistic 18%), and
  X-rays go preferentially to pneumonia stays (P(X-ray | pneumonia) ≈ 0.86 in
  smoke, 0.5 in `small`), as pneumonia work-ups include a chest film. At 60
  stays and 12% prevalence there are ~7 pneumonia stays in total; without this,
  the CXR validation split had a single positive and its AUROC was a coin flip
  (we saw 0.0 on the first run). The `small` preset keeps 18% overall.
- Check e) (AUROC > 0.55) on the smoke preset rests on ~2 validation positives
  per reader. It proves the plumbing learns on seed 0; it is not a stable
  estimate and other seeds may fall below the floor.
- **Confidence "sure" pile on virtual data** (check b, γ = max(σ, 1−σ),
  fraction of token-class values ≥ 0.75, θ): EHR 0.713, CXR 0.800 (one token
  per image), RR 0.782, DN 0.781 — smoke run of 2026-09-27. Most of it is the
  heads learning the 24 random labels' base rates, not pneumonia. An earlier
  EHR run with 5 Round 2 epochs had γ in [0.50, 0.65], i.e. **0% ≥ 0.75**: an
  empty high-confidence group. Before reading any Round 3 result on virtual
  data, check these fractions; an empty or near-empty pile makes the high/low
  split meaningless and is an artefact of the stand-ins.
- **Normalizer**: virtual runs pass `--normalizer_state` pointing at a
  normalizer fit on the virtual train split, so no MIMIC-derived statistics are
  involved. Real runs keep medpatch's default.
