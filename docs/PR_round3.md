# Round 3 (MedPatch fusion, c-msma): fixes, cached training, threshold sweep

> **Do not merge until phenotyping Round 3 has run.** Mortality Round 3
> (`r3-fusion-001`) has run on this branch. The phenotyping run is the last
> real-data check of the cached path before `main` (which auto-deploys).

Branch `model/round3` turns the paper's Round 3 into something we can train
on Colab within budget, and fixes the defects that stopped it from building
or that biased its numbers. Every `medpatch/` change is minimal, commented
and covered by a test; see `docs/model_track_notes.md`, items 13–38.

## Fixes to the Round 3 model and trainer (F1–F7)

- **F1:** CMSMAFusion / EMSMAFusion now build with our ViT CXR encoder,
  which has no `projection_layer`.
- **F2:** each Round 2b parent is loaded strictly, with only its own reader's
  keys. The rr and dn parents must carry the same BERT, and every reader
  tensor is checked bit-identical to its file after loading.
- **F3:** the reported head is chosen on validation and stored in the
  checkpoint. It was the best of five on test. All five test heads go to a
  separate reference table.
- **F4:** late fusion no longer swaps the RR and DN missingness masks
  (phenotyping).
- **F5:** phenotyping EHR confidence uses the Round 2b temperature.
- **F6:** the tools force `c-msma`; `c-e-msma` needs `--allow-entropy`.
- **F7:** a last batch of one stay no longer crashes (c-msma outputs, the
  LSTM's `ht.squeeze()`, Round 1 outputs).
- **Kept from upstream and documented:**
  - EHR pooling over padded hours;
  - the missingness head's constant output on all-present rows;
  - late weights normalised over columns × classes. `--late_weights_per_class`
    normalises per class instead (off by default).

## Cached Round 3 (`tools/pv/r3_cache.py`)

Every reader, confidence head and temperature is frozen in Round 3, so the
readers run **once per stay**.

- **What is stored:** for the main threshold, the masked means of the
  unprojected token features per class and group, plus counts, confidences
  and missingness. `CMSMAFusion.forward_cached` rebuilds high_conf / low /
  late from them; the trainable code after the readers is shared with
  `forward()`.
- **Equivalence:** cached equals live (outputs, gradients, one optimizer step)
  to ≤ 1e-5 in unit tests for both tasks, every ablation and both fusers, and
  on real mortality validation (1.2e-07).
- **Storage:** resumable shards, a fingerprint check (parents' sha256, lists'
  md5, thresholds, capture format) and a disk check.
- **Reads:** memory-mapped, so a variant reads only the keys it needs.
- **Per-epoch cost:** bootstrap CIs are no longer recomputed every epoch,
  which saves ~1 h of CPU per phenotyping epoch; selection never used them.

## `python -m tools.pv.run r3` steps

- **cache:** the reader pass, resumable; `--sweep-thresholds` also captures
  swept thresholds, into its own directory.
- **equivalence:** cached vs live at 0.75, and at one swept threshold when
  sweeping.
- **train:** the main run plus `without_missingness_module`.
- **record:** the manifest row (stage `r3`, reader `fusion`, the r2b parents)
  and `r3_results.csv`. Rerunning is a no-op.
- **sweep:** one variant per swept threshold → `r3_threshold_sweep.csv`.
  Ablations only: never the manifest row, the reported result or the shared
  model.
- **ablations:** without_joint_module, without_late_module,
  late_weights_per_class and mask_missing_tokens → `r3_ablations.csv`.
- **share:** copies only the main θ = 0.75 run (checkpoint sha256-checked,
  tables, logs) to the shared folder; never a variant.

Every step exits 0 / non-zero, prints `[r3] <step> OK (<duration>)`, and
writes its whole output to `<run>/<step>.log`. Also new:

- `r3_preflight` (counts, missing X-ray files, F2 load, share ≥ 0.75, cache
  size);
- `parent_keys`;
- `score_rounds` (Rounds 2/2b on test with the 25-label macro);
- `smoke_round3`, a virtual end-to-end run with `pytest -m round3` checks.

## Tests

- **Local** (`python -m pytest`): the default suite, on stub readers and
  medpatch's real LSTM.
- **Colab:** `-m round3` and `-m round2b_cache`, against real readers.
- **Unchanged:** the 42 r1/r2/r2b run plans are byte-identical to `main`'s.

## Not in this PR

The inference service still runs in MOCK_MODE; serving the Round 3
checkpoints is planned in `docs/REAL_INFERENCE_PLAN.md`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
