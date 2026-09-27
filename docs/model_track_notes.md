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
   **Open for Round 2b:** `trainers/Calibration.py:115` has the same bare
   `.squeeze()`, so Round 2b on CXR will hit the same shape failure with the
   default input. Not changed here (Round 2b is out of scope for this branch);
   apply the same `confidence_logits` treatment before running it.

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
