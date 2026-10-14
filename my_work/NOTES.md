# PneumoVision — running notes

All numbers below were computed by code in `my_work/audit/` (aggregate counts only, no patient rows).
Repo commit at audit time: `ed255d2`. Environment: Python 3.11.9, pandas 2.2.2, torch 2.4.1+cpu.

## 2026-10-01 — Audit of Caroll's data (`caroll's work/`)

### Where things actually are (differs from the brief)
- Caroll's folder is `caroll's work/` (not `carol/`). It contains: `all_stays.csv`, EHR listfiles +
  timeseries for `phenotyping/`, `in-hospital-mortality/`, `pneumonia-48h/`. **No scripts, logs, CXR split
  file or index files are in it.**
- `paper/` folder does not exist — the paper PDF and Handbook are not in the workspace, so Table B1/B3/B4
  values below are taken from the brief, not re-read from the PDF.
- Notes dir `data/mimic-iv-note -> D:\PneumoVision_data` has `radiology.csv` but **no `discharge.csv`**
  (only processed variants). The loader reads raw `discharge.csv` + `radiology.csv` on every run.
- No MIMIC-CXR-JPG data locally (images, metadata, chexpert, split CSVs).

### Data-checks table
| Check | Expected | Caroll (as delivered) | Restricted to repo split subjects* | Status |
|---|---|---|---|---|
| Pneumonia prevalence tr/val/te | 12.7/12.4/12.3% | 13.09 / 12.83 / 12.44% | (not recomputed) | ✅ within gate |
| # phenotype labels | 25 | 25 | — | ✅ |
| Pneumonia ICD codes | ~120 | 120 (62 ICD-9 + 58 ICD-10) in `icd_9_10_definitions_2.yaml` | — | ✅ |
| Pheno EHR tr/val/te | 42,628 / 4,802 / 11,914 (59,344) | **60,909** / 4,756 / 11,845 (77,510) | 42,328 / 4,756 / 11,845 (58,929) | ❌ as delivered → ✅ restricted (−0.7%) |
| Mortality EHR tr/val/te | 18,845 / 2,138 / 5,243 | **29,171** / 2,161 / 5,302 | 19,064 / 2,161 / 5,302 | ❌ → ✅ restricted (+1.1%) |
| Split proportions | ~70/10/20 | 78.6 / 6.1 / 15.3 | 71.8 / 8.1 / 20.1 | ❌ → ✅ |
| Subject overlap between splits | 0 | 0 / 0 / 0 (pheno and mortality) | 0 | ✅ |
| Age 0–20/21–40/41–60/61–80/80+ | 561/6,096/17,336/25,385/9,994 | 700/8,069/22,668/33,709/12,364 | 561/6,050/17,365/25,247/9,706 | ❌ → ✅ (≤3%) |
| Sex M / F | 33,113 / 26,259 | 43,094 / 34,416 | 32,856 / 26,073 | ❌ → ✅ (−0.8%) |
| 17 EHR variables, 5 categorical | per Table B1 | 17 channels, 5 categorical (CRR, GCS eye/motor/verbal/total) in `discretizer_config.json`; `--timestep 1.0` | — | ✅ names/types; impute values need PDF check |
| CXR / RR / DN counts | Fig 1a | cannot compute — no CXR data, no discharge.csv | — | ⏳ blocked |
| CXR split file (377,110 rows) | from repo | **not shipped in the repo**; not in Caroll's folder | — | ❌ missing |
| ~20% dropped by validate_events | ~20% | no logs from Caroll | — | ⏳ unknown |

\* "Restricted" = keep only rows whose `subject_id` appears in the repo's `testset_iv.csv` (47,046 subjects).

Impute ("normal") values in the repo's discretizer config, to compare against Table B1:
CRR 0.0 · DBP 59.0 · FiO2 0.21 · GCS eye "4 Spontaneously" · GCS motor "6 Obeys Commands" · GCS total 15 ·
GCS verbal "5 Oriented" · glucose 128.0 · HR 86 · height 170.0 · MBP 77.0 · SpO2 98.0 · RR 19 · SBP 118.0 ·
temp 36.6 · weight 81.0 · pH 7.4.

### Root cause of the count mismatch
- `icustays.csv` used has **94,458 stays** → this is **MIMIC-IV v3.x**. The repo's split lists
  (`testset_iv.csv`, `valset_iv.csv`) were made for the older version the paper used.
- `split_train_and_test.py` / `split_train_val.py` put **any subject not in those lists into train**.
  18,581 phenotyping rows (10,107 mortality rows) belong to such subjects → train is +43%.
- Val/test sets are exactly the repo's subjects (all val rows in `valset_iv`, all test rows in `testset_iv`).
- Your `data/mimic-iv-hosp/all_stays.csv` and Caroll's contain the identical 77,547 stay_ids (same 18 columns).

**Decision needed:** for reproduction, use listfiles restricted to repo-known subjects (counts within ~1% of paper).
This filters rows; it does not change the split assignment of any subject.

### Task A (`pneumonia-48h/`) — Caroll already built it
- Same 36,634 stays as mortality; label = pneumonia; rate 18.65 / 18.23 / 17.90%. Same new-subject train
  inflation. Not reviewed further — Task A is on hold until Task B gates pass.

## Code vs paper / scripts — mismatches found
1. **Seed hard-coded** `seed = 1002` in `fusion_main.py`; no `--seed` flag. Needed for 5-seed runs.
2. **Eval reports max over heads chosen on TEST**: `MSMA_trainer.eval()` writes `max(high, low, late, combined,
   reduced)` by test AUROC to `results_*_test.txt`. Never quote that line; report each head from logs.
   `--stat_test` does the same and saves the "best" predictions.
3. Validation model selection also uses max over the 5 heads (val, so legitimate, but not "combined" only).
4. **Text encoder is `emilyalsentzer/Bio_ClinicalBERT`**, not BioBERT; `--text_encoder biobert` does not change it.
   Frozen ✅, 512-token non-overlapping chunks (stride 0), mean-pooled ✅.
5. Adam `betas=(0.9, args.beta_1)` with `beta_1` default 0.9 → **β2 = 0.9**, not 0.999. ReduceLROnPlateau
   (factor 0.5, patience 10) is used — not in the paper.
6. Repo scripts: `Confidence-DN.sh` passes `--load_rr` instead of `--load_dn`; `Evaluate.sh` has no
   `--dn_confidence_threshold` (falls back to 0.3) and no `--load_dn`. `Confidence-Patching.sh` passes 0.75 for all four ✅.
7. Unimodal/confidence/calibration scripts use `--data_pairs paired`; Round 3 uses `partial` ✅.
8. `ehr_utils/create_split.py` compares int `subject_id` to string IDs → `isin` matches nothing (verified on
   pandas 2.2.2 with synthetic data): **regenerating the CXR split puts every image in train.** Don't regenerate.
9. CXR transforms hard-coded 384×384 (large setting) ✅; missing-CXR placeholder also 384.
10. `fusion_main.py` reads one hard-coded timeseries file `train/14991576_episode3_timeseries.csv` (exists in Caroll's data).
11. Path layout the code expects vs Caroll's:
    - `<ehr_data_dir>/root/all_stays.csv` — Caroll's is at top level.
    - `<ehr_data_dir>/phenotyping/train/` — Caroll's is `phenotyping/phenotype_labels/train/`.
    - `<ehr_data_dir>/in-hospital-mortality/train/` — Caroll's is nested `in-hospital-mortality/in-hospital-mortality/train/`.
12. Even an EHR-only run loads CXR metadata/chexpert/split CSVs and `discharge.csv`/`radiology.csv` → all must exist.
13. Eval checkpoint path is built from `save_dir/task/fusion_type/best_checkpoint_{lr}_...` — must match the Sheet.

## Data-governance flags
- `caroll's work/` is **not gitignored** (holds per-patient timeseries). Add it to `.gitignore` before any commit.
- `hand_check_discharge_v2.txt`, `hand_check_radiology.txt` (repo root, untracked, not ignored) — likely contain
  note text (not opened). Move under `data/` or gitignore.
- `explore.py` prints `.head()` of note files — violates the no-display rule.

## Hardware
- Laptop: RTX 3050 Laptop, 4 GB VRAM; installed torch is the **CPU build** (`2.4.1+cpu`).

## 2026-10-01 — Farida's scope = text only (RR + DN), training on Colab
What a text-only run (`--modalities EHR-RR` / `EHR-DN`, `--data_pairs paired`) still reads, per the code:
- X-ray: no images needed, but `get_cxr_datasets` + `loadmetadata` always open
  `mimic-cxr-2.0.0-metadata.csv`, `mimic-cxr-2.0.0-chexpert.csv`, `mimic-cxr-ehr-split.csv` (in `--cxr_data_dir`). Missing locally.
- EHR: Caroll's listfiles (labels) + timeseries + `root/all_stays.csv` (labels come from the listfiles; timeseries are loaded per sample).
- Notes: files must be named exactly `radiology.csv` and `discharge.csv` with columns subject_id, hadm_id, charttime, text.
  `radiology.csv` OK. Raw `discharge.csv` missing locally. `discharge_diagnosis_cut.csv` has the right columns → usable for Stage 3 (B).
- Repo code does NOT apply a chest-only filter: it uses every radiology report for the admission. Stage 1 reproduction should use raw `radiology.csv`.

## 2026-10-01 — Colab setup for text runs (RR, DN), smoke-tested locally
Code changes in `medpatch/` (all tested):
- `datasets/cxr_dataset.py`, `datasets/DataFusion.py`: skip X-ray files when CXR not in --modalities; read only the needed notes file.
- `trainers/MSMA_trainer.py`: `last_checkpoint_*.pth.tar` after every epoch (atomic write) + `--resume` continues from it.
- `arguments.py`, `fusion_main.py`, `MSMA_trainer.py`: `--seed` (default keeps 1002 / 379647).
- `trainers/utils.py`: bootstrap-CI crash with newer sklearn (float - list) — original repo bug.
- Must be launched from inside `medpatch/` (relative path to discharge config).
Smoke test (64 subjects/split, CPU, `data/smoke_ehr`): RR datasets 58/61/55; train/val/test + checkpoints OK;
resume test continued at epoch 2. Scores meaningless at this size.
Notes: CI loop treats NaN as valid (`auprc != np.nan` is always True) — only matters on tiny sets.
Checkpoints are ~440 MB each (include frozen BERT).
Deliverables: `my_work/colab/medpatch_code.zip`, `my_work/colab/PneumoVision_text_training.ipynb`, `prepare_data.py`.
- 2026-10-01 Colab sanity (64/split) passed end to end on T4 (RR 58/61/55, test mean AUROC 0.508 — meaningless at this size).
  Repo never writes results_*.txt (write is commented out) → results live in runs/<run>/train.log (tee'd to Drive).
  Colab used a newer transformers than pinned 4.44.2 (v5-style "LOAD REPORT") — record exact versions for the thesis.

## 2026-10-02 — Text encoder switched to BioBERT (paper) — decision by Farida
- First official RR run (`runs/taskB_unimodal_rr_seed1002`) used the repo's hard-coded `emilyalsentzer/Bio_ClinicalBERT`.
  Val AUROC by epoch: 0.489 (untrained), 0.693, 0.700, 0.706, 0.707 — plateauing ~0.07 below paper test 0.776.
- Paper states BioBERT → added `--bert_model_name` (default unchanged = Bio_ClinicalBERT); notebook now passes
  `dmis-lab/biobert-v1.1`. Run folders named `..._biobert_seed<seed>` so they never resume a ClinicalBERT checkpoint.
  Startup prints `TEXT ENCODER MODEL: <name>`. Verified locally that biobert-v1.1 loads (hidden 768) and chunking works.
- Other open suspects for the gap: LR (paper did 10-sweep search), mean-pooling includes [PAD] positions,
  all radiology reports (not chest-only), BERT dropout active during training.

## 2026-10-02 — Fast path (option C) for text, BioBERT, laptop RTX 3050
- Paper §4.4 (arXiv 2508.09182 html): "batch size 16, epochs 100, 10 sweeps ... learning rates randomly sampled
  between 1e-5 to 1e-3", early stopping patience 15, Adam, A100s; f_rr = BioBert, frozen; 512-token non-overlapping
  chunks, mean-pooling over the 512 tokens. **Final LR per model is not reported.**
- Laptop venv now torch 2.4.1+cu121 (was +cpu).
- `my_work/fast_text/cache_embeddings.py` (BioBERT once, eval mode) + `train_head.py` (fc+LayerNorm+Linear head,
  repo optimizer/scheduler/early-stopping rules, 10-LR log-uniform search, val selection, single test).
- Equivalence check (synthetic texts, 1/4/1/2 chunks incl. empty): repo `process_notes(...).mean(1)` vs fc(cached)
  max abs diff 1.2e-7 → identical up to float rounding. Only difference: repo trains with BERT dropout on.
- Timing: 320 stays ≈ 0.7 min (~4.6 chunks/stay) → full RR (54,328 stays) ≈ 2 h.

## 2026-10-02 — RESULT: unimodal RR, BioBERT, Task B — matches paper
Run: `my_work/runs/2026-10-02_taskB_rr_biobert_fast_lrsearch/` (config.json, train.log, metrics.json, best_head.pth)
- Cache: dmis-lab/biobert-v1.1, eval mode; train/val/test 39,028 / 4,356 / 10,944 stays (paper RR 38,497 / 4,300 / 10,779).
  Train+val cached in first pass; job hit the 2 h background limit during test → test recomputed with `--splits test`
  (train/val arrays checked: correct shapes, no zero/NaN rows; meta.json rebuilt from the log).
- 10 LRs log-uniform [1e-5, 1e-3] (sweep_seed 0), training seed 1002. Best val AUROC per LR: 0.7776–0.7793 (very flat).
- Chosen on validation: lr 1.22e-4 (val 0.7793). TEST (once): mean AUROC **0.7783**, mean AUPRC **0.4735**
  vs paper 0.776 (0.764–0.788) / 0.474. Pneumonia RR test AUROC 0.8591 [0.8502, 0.8680], AUPRC 0.4869.
- Best epochs cluster at 54/79 across LRs: val AUROC plateaus ~0.7782–0.7789 after epoch ~40; same shuffling seed
  across sweeps aligns the tiny fluctuations. Not a bug.
- Caveats: 1 seed (5 planned); fast path (no BERT dropout in training); LR sampling distribution unspecified in paper;
  CI for the 25-label mean not computed (only per-label CIs).
- Earlier Colab run with repo-default Bio_ClinicalBERT @ lr 1e-3 plateaued at val 0.708 (epochs 1–5: .693/.700/.706/.707/.708).
- 2026-10-02 RR 5 seeds (lr 1.2225e-4 fixed, seeds 1002/2002/3002/4002/5002), `my_work/runs/2026-10-02_taskB_rr_biobert_fast_5seeds/summary.json`:
  test AUROC mean 0.7783 (0.7780–0.7787); AUPRC 0.4740 (0.4721–0.4756); pneumonia AUROC 0.8593 (0.8580–0.8602), AUPRC 0.4869 (0.4841–0.4890).
  Seeds vary init + shuffling only (cached BioBERT features are deterministic).

## 2026-10-02 — In-hospital mortality: investigation (no training yet)
- Repo scripts: scripts/mortality/{Unimodal,Confidence,Calibrate}/{EHR,CXR,RR}.sh, MedPatch/{Confidence,Entropy}-Patching.sh, Evaluate.sh.
- `c-msma` = confidence max(p,1-p) (paper eq.; verified in arXiv text). `c-e-msma` = binary entropy (max ln2≈0.693 < θ=0.75 → no
  token can ever be "high"). Mortality Confidence-Patching.sh uses c-msma; **phenotyping Confidence-Patching.sh uses c-e-msma**.
- Paper (arXiv html): mortality EHR 0.861/0.523, CXR 0.723/0.351, RR 0.742/0.287; MedPatch EHR+CXR+RR 0.876/0.558 (Table 3),
  headline = LATE (Table C1: multimodal mortality late 0.876; bimodal late .868 / combined .850 / reduced .861).
  Calibration: 5 epochs, lowest val ECE (repo scripts pass --epochs 100). θ = 0.75. Rates 12.5/11.9/12.4%.
- Caroll's mortality data, restricted to repo split subjects: 19,064 / 2,161 / 5,302 (paper 18,845 / 2,138 / 5,243, +1.1%);
  mortality 12.66 / 12.36 / 12.58%. Listfile period_length=0 → ehr_dataset uses 48 h (line 161). Timeseries already built
  (in-hospital-mortality/in-hospital-mortality/{train,test}); no regeneration needed.
- 48 h filters: DataFusion.loadmetadata — paired branch ~L249-273, partial branch ~L353-394 (CXR StudyDateTime in
  [intime, intime+48h], AP only, latest per stay; RR charttime in [intime, intime+48h], joined per stay; discharge_text=None).
- Normalizer: fusion_main default `ph_ts1.0...` for every task; `ihm_ts1.0...` exists; repo scripts don't pass --normalizer_state.
- Calibration (trainers/Calibration.py) trains temperature on val_dl only; writes ECE/per-token-probability CSVs into the CWD
  (patient-derived — must not land in a tracked folder). No resume in Calibration (short run).
- Eval of c-msma prints max(high,low,late,combined,reduced) chosen on TEST — per-head numbers only go to wandb → need explicit LATE print.
- Blocker: no MIMIC-CXR-JPG data (images/resized, metadata, chexpert, mimic-cxr-ehr-split.csv) anywhere in the workspace.

## 2026-10-02 — RESULT: in-hospital mortality, unimodal RR (48 h), BioBERT, fast path (laptop)
- Scripts gained `--task in-hospital-mortality` (cache_embeddings, prepare_data; train_head names the single label).
  Cache uses the repo's paired mortality branch: RR charttime in [ICU intime, intime+48h], joined per stay. DN never read.
- Mortality EHR dir: data/ehr_full/in-hospital-mortality (restricted 19,064 / 2,161 / 5,302; mortality 12.66/12.36/12.58%).
- RR stays with ≥1 report in 48 h: 16,796 / 1,904 / 4,650 (paper 16,279 / 1,847 / 4,500, ≈+3%); ~2.2 chunks/stay; cache 26.5 min.
- LR search (same 10 LRs): best val 0.7601–0.7628; chosen 1.2225e-4 (val 0.7628). Runs stop early (25–58 epochs).
- 5 seeds (`my_work/runs/2026-10-02_mortality_rr_biobert_fast_5seeds/summary.json`): test AUROC 0.7297 (0.7275–0.7322),
  AUPRC 0.2726 (0.2650–0.2766); seed-1002 bootstrap CI AUROC [0.709, 0.749], AUPRC [0.246, 0.310].
  Paper: 0.742 (0.723–0.762) / 0.287 (0.258–0.322) → inside the paper's CI, ~0.012 below its point estimate.
- Val→test drop (0.762 → 0.730) is larger than for phenotyping; val is small (1,904 stays, ~12% positive) → noisy selection.

## 2026-10-02 — Repo-format checkpoints (Round 1, RR) for the team
- `my_work/fast_text/to_repo_checkpoint.py`: builds the repo's Fusion(unimodal_<m>) with frozen pretrained BioBERT and copies
  head fc→text_model.fc_rr, layernorm/out→fusion_model.rr_classifier.classifier.{layernorm,fc}; saves at the exact
  MSMA_Trainer path. Files (~435 MB, include BioBERT), gitignored under checkpoints/:
  - checkpoints/mortality/r1/in-hospital-mortality/unimodal_rr/best_checkpoint_0.00012225031390095305_in-hospital-mortality_unimodal_rr_EHR-RR_paired.pth.tar
  - checkpoints/phenotyping/r1/phenotyping/unimodal_rr/best_checkpoint_0.00012225031390095305_phenotyping_unimodal_rr_EHR-RR_paired.pth.tar
- Verified mortality with the repo's own `fusion_main.py --mode eval`: test AUROC 0.729 (0.709, 0.749), AUPRC 0.275
  (0.246, 0.310) = fast path exactly; no Not Loaded / Not Found keys. Must pass `--bert_model_name dmis-lab/biobert-v1.1`
  and `--lr 0.00012225031390095305` (lr is part of the file name).
- Round 2 (c-unimodal_rr): ConfidencePredictor = Linear(768→C) on each of the 512 token positions of the (chunk-averaged)
  BERT output; loss = BCE per token vs stay label; selection by val loss. Needs token-level features (no mean shortcut).
  Token cache fp16 ≈ 0.79 MB/stay → mortality train+val ≈ 14.7 GB; phenotyping train+val ≈ 34 GB.
  Free disk: C 9.3 GB, D 24 GB. Round 2b (temp_c-unimodal_rr) trains a 512-vector temperature on VAL only (cheap).
- Verified phenotyping RR repo checkpoint with repo eval: test mean AUROC 0.778, CI (0.766, 0.790); AUPRC 0.473, CI (0.449, 0.499);
  pneumonia 0.859 (0.850, 0.868) — equals fast path (0.7783 / 0.4735). No Not Loaded / Not Found keys.
  (This also gives the CI of the 25-label mean, which the fast path did not compute; paper 0.776 (0.764, 0.788).)
- 2026-10-03 DN (Colab fast run) copied to my_work/runs/fast_taskB_unimodal_dn_biobert_seed1002/ (cache: 40,464/4,529/11,333 stays,
  ~8 chunks/stay; chosen lr 1.2225e-4; test 0.859/0.6068). Converted → checkpoints/phenotyping/r1/phenotyping/unimodal_dn/
  best_checkpoint_0.00012225031390095305_phenotyping_unimodal_dn_EHR-DN_paired.pth.tar. Synthetic-text check: 0 missing/extra
  keys, repo forward = head (max diff 2.6e-6). Full repo eval not run (no raw discharge.csv on laptop).
- Handoff folder for Youssef: my_work/handoff_youssef/ (README + aggregate metrics only).
- 2026-10-05 Handoff complete (per Farida): 3 Round-1 checkpoints (mortality RR, phenotyping RR, phenotyping DN) + handoff_youssef/
  uploaded to shared Drive; Sheet rows added; Youssef messaged. Rounds 2/2b/3 are Youssef's (handbook). The local Round-2 RR
  mortality fast run stopped mid-sweep 2/10 and is not needed. Open (Farida): DN 5 seeds; leakage experiment (DN without
  Discharge Diagnosis).
