# Model track notes

A log of every change made to `medpatch/` for the model track, and of known
deviations between the paper's scripts and how we run them. Rule: changes to
`medpatch/` are minimal and either flag-gated or fix a path that could not run
at all; with default flags, behaviour is the paper's.

Newest first.

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

2. **`c-unimodal_cxr` scored the CLS vector, not the tokens** (`models/fusion.py`,
   `UnimodalCXRConfidence.forward`). `CXRTransformer` returns
   `(tokens [B,577,D], cls [B,D])`; `_, full = …` took the CLS vector, giving one
   "token" per image. Now `cxr_features(...)` (the existing helper) selects the
   token sequence, which is the paper's per-patch confidence. This path
   previously crashed (item 1), so no working behaviour changed.

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
- **Normalizer**: virtual runs pass `--normalizer_state` pointing at a
  normalizer fit on the virtual train split, so no MIMIC-derived statistics are
  involved. Real runs keep medpatch's default.
