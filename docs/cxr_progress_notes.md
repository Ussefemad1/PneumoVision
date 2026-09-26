# CXR pipeline — progress notes

Owner: Norhan · chest x-ray track

Running log of decisions, numbers and gotchas for the image side of the
project. Kept in version control so the rest of the team can see it.

---

## Cohort rule — correction worth knowing

The repo does **not** "keep the first ICU stay per admission".
`mimic4extract/mimic3benchmark/mimic3csv.py` calls
`filter_admissions_on_nb_icustays(stays, min_nb_stays=1, max_nb_stays=1)`,
which **drops the entire admission** unless exactly one qualifying ICU stay
remains after transfers are removed.

    94,458 ICU stays  ->  77,547 after the rule

**Caroll:** your pipeline must apply the same rule, or our patient lists will
not reconcile — I would have images for patients absent from your data.

Related: in MIMIC-IV, `first_careunit` always equals `last_careunit`
(0 rows differ), so `remove_icustays_with_transfers` is a no-op here. That is
expected — a ward transfer creates a *separate* icustay row in MIMIC-IV,
unlike MIMIC-III which the benchmark code was originally written for.

---

## Colab cannot install requirements.txt as pinned

`requirements.txt` pins `torch==2.4.1` / `torchvision==0.19.1`, and **Colab no
longer offers those builds**. Installing them there either fails outright or
pulls a CPU-only wheel over Colab's CUDA build — which silently costs you the
GPU, so training still "works" but is unusably slow.

All three of us train on Colab, so nobody could install as it stood.

**Fix:** use `requirements-colab.txt`, which leaves Colab's own
torch/torchvision/numpy/pandas/scipy alone and installs only what is missing or
version-sensitive (`transformers`, `timm`, `einops`, `wandb`, `sentencepiece`,
`openpyxl`). Local development still uses the fully pinned `requirements.txt`.

Always check after installing, in a Colab cell:

    import torch; print(torch.__version__, torch.cuda.is_available())

If that prints `False`, Colab's torch has been replaced — fix it before training.

**Not yet verified on Colab.** `numpy` in particular is left unpinned because
Colab ships 2.x while we pin 1.26.4; if anything in `mimic4extract` or the
discretizer misbehaves there, that is the first thing to check.

---

## PhysioNet access — Basic auth is rejected, session cookie required

HTTP Basic auth (`wget --user/--password`, the method PhysioNet's own docs
describe) returns **403 on every file** for our accounts — including files that
download fine through the browser.

The 403 is useless as a diagnostic: correct credentials, deliberately wrong
credentials and no credentials at all produce byte-identical responses.
PhysioNet also never sends a 401 challenge, so Python's `HTTPBasicAuthHandler`
never attaches the credentials at all; sending the header preemptively does not
help either.

**What works:** a browser session cookie. F12 → Application → Cookies →
physionet.org → copy `sessionid` into `.env` as `PHYSIONET_COOKIE`, and send it
as `Cookie: sessionid=...`. Cookies last roughly two weeks; refresh when
downloads start 403ing again.

The failed Basic-auth attempt is preserved at
`docs/dead_ends/download_cxr_metadata_BASIC_AUTH_FAILS.py` rather than deleted,
so Farida and Caroll do not rediscover it the hard way.

---

## PhysioNet access — DUA is signed for v2.1.0, not v2.0.0

Every request to `/files/mimic-cxr-jpg/2.0.0/...` returns **403**.
The same paths under **2.1.0** return 200.

Also: **HTTP Basic auth does not work** for this account — `curl -u` gets 403
even on 2.1.0. Downloads require a browser `sessionid` cookie.

v2.1.0 contains the same 377,110 images as 2.0.0 and still ships the metadata
files under their original `mimic-cxr-2.0.0-*` names, so nothing downstream
changes. The local folder stays named `2.0.0` because that is what the
MedPatch code expects as `cxr_data_dir`.

**Team:** if you hit 403s on MIMIC-CXR-JPG, check the version in your URL
before assuming your credentials are wrong. This cost a day.

---

## Image selection

Filters, in order (all sourced from the repo, not the paper text):

1. patient in the ICU cohort above
2. `ViewPosition == 'AP'` only — hardcoded in `datasets/DataFusion.py`
   in three places. ICU patients are too ill to stand for a PA film.
3. study inside the task's time window
   (Task A: intime → intime+48h · Task B: intime → outtime)
4. one image per `stay_id`, the most recent in the window

`StudyDateTime` is built with the same expression as `DataFusion.py`
(lines ~189-190), not reimplemented.

| | Mine | Paper (Fig 1a / Table 1) | Diff |
|---|---|---|---|
| Task A (48h) | 6,683 | 6,215 | +7.5% |
| Task B (full stay) | 11,237 | 10,804 | +4.0% |
| Overlap | 2,739 | — | — |
| **Union to download** | **15,181** | — | — |

---

## Task A split vs the paper

    Mine        : 4,962 train / 490 validate / 1,231 test
    Paper Fig 1a: 4,485 train / 488 validate / 1,242 test

Validate and test match closely. Train is ~10% higher, most likely because
this count is taken **before** intersecting with the EHR cohort — my numbers
are the image side alone.

**Re-check this after Caroll's listfile arrives.** If train does not fall
toward ~4,485 once paired, something in the cohort logic differs from hers.

---

## Date shifting — matters beyond the x-rays

MIMIC shifts **each patient's entire timeline** forward by a random
per-patient offset. Study dates land in the 2100s-2200s; this is correct, not
corruption.

Consequences for everyone:

- Time gaps *within* one patient are real and usable.
- Absolute dates are fiction. **Never** compare timestamps *between* patients,
  sort patients by admission date, split train/test by time, or count
  "patients admitted in 2015". All meaningless.

---

## Verification before download

All six checks passed on the 15,181-image list:

- every row `ViewPosition == AP`
- `StudyDateTime` within `[intime, outtime]`
- times spread across the clock (0% sitting at 00:00:00 — rules out silent
  time-parsing failure)
- **every Task A study within [0, 48] hours of intime — 0 negative, 0 over**
  (this is the leakage check)
- exactly one row per `stay_id`
- union arithmetic: 6,683 + 11,237 − 2,739 = 15,181 ✓

After download, all 6,683 Task A images were decoded individually:
**0 corrupt, 0 missing**, sizes 901 KB – 4.9 MB.

---

## Known PhysioNet data gap — one image is unavailable

One study on our download list returns **HTTP 404** from PhysioNet while being
**listed in `IMAGE_FILENAMES.txt`**. Their manifest and their served files
disagree; this is upstream, not a fault in our pipeline, and it is **not
retriable** (confirmed a genuine 404 rather than a transient failure).

    dicom_id   09e5d1be-9c17d28b-ca19988b-47e76494-ac4eb1e7
    subject_id 18860233
    study_id   58466825
    stay_id    30007216
    needed_for Task B only

**Task A is unaffected and complete at 6,683.**

**For the thesis:** Task B's true available count is **11,236 of 11,237**
listed studies (0.009% loss), and the combined download is **15,180 of 15,181**.
Use those figures rather than the listed ones so the counts reconcile.

`scripts/verify_resized_complete.py` carries this dicom_id in `KNOWN_MISSING`,
so it reports complete rather than flagging a permanent one-image gap on every
run.

---

## Resize

`medpatch/resize.py` had the authors' cluster paths hardcoded
(`/scratch/fs999/shamoutlab/...`); fixed to resolve relative to the repo.

Output folder **must** be named exactly `resized/` — `datasets/cxr_dataset.py`
globs `{cxr_data_dir}/resized/**/*.jpg` and silently finds zero images
otherwise.

    11.70 GB (originals)  ->  0.15 GB (512px wide, aspect preserved)

That is 1%, not the ~25% the handbook predicted, because the originals are
~2500px wide and 512px is a 5x linear shrink. Archive for Drive is ~155 MB,
not ~3 GB.

Originals are **kept** for now. Deleting saves 11.7 GB out of 93 GB free,
against an 18-hour re-download if anything turns out wrong. Revisit after the
first successful training run.

---

## Repo defects — running list

### 1. `ehr_utils/create_split.py` produces an all-train split — NOT FIXED, do not run

Compares string `subject_id`s against int64, so nothing ever matches and every
row ends up `train`. The bundled `mimic-cxr-ehr-split.csv` is correct
(325,200 / 15,282 / 36,628). **Use it as-is; never regenerate.**

### 2. `fusion_main.py` hardcoded one patient's filename — FIXED

`read_timeseries()` opened
`{ehr_data_dir}/{task}/train/14991576_episode3_timeseries.csv` by name. The
file is read only to discover the discretizer's column layout, which is
identical in every timeseries — but if that subject was absent from the
extraction, the run crashed before training started.

Now prefers that file when present (byte-identical on the authors' data) and
otherwise falls back to the first `*_timeseries.csv` found, with a clear error
if the directory is empty. **Caroll / Youssef: this would have hit the real
extraction too** unless subject 14991576 happened to survive the cohort filters.

### 3. Checkpoint resume is broken — FIXED

Four separate problems, which together meant a resumed run silently restarted
from scratch:

- `load_state()` restores **`state_dict` only**. The checkpoint also stores
  `epoch`, `best_auroc` and `optimizer`, and all three were ignored.
- `self.start_epoch = 0` hardcoded in `MSMA_trainer.py`, with the loop
  `for self.epoch in range(self.start_epoch, self.args.epochs)`.
- `save_checkpoint()` only called on improvement, so a session dying during a
  flat stretch lost back to the last *improving* epoch.
- `--resume` existed in `arguments.py` but **nothing read it**.

Consequence at 384px across multiple Colab sessions: every reconnect restarted
at epoch 0 with a fresh optimizer while printing "Loaded model checkpoint" —
it looked like it had worked.

**The fix:**

- `checkpoint_path(prefix)` — `save_checkpoint(prefix=...)` previously accepted
  a prefix but **ignored it in the filename**, always writing
  `best_checkpoint_...`. The prefix now selects the file. The `'best'` filename
  is byte-identical to before, so existing checkpoints are still found.
- `resume_from_checkpoint()` — restores weights, optimizer, `best_auroc`,
  `patience` and `start_epoch = epoch + 1`. Prefers the `last` checkpoint,
  falls back to `best`. `load_state()` is left alone: it is also used to
  warm-start from someone else's checkpoint, where carrying over the epoch
  counter would be wrong.
- `save_checkpoint(prefix='last')` now runs **every epoch**, alongside the
  existing best-only save. Both files are kept.
- `--resume` is wired up. Without it, behaviour is unchanged, so the
  reproduction of 0.692 stays faithful.
- Startup is loud either way: it prints the checkpoint file, the epoch it
  resumes from and the restored `best_auroc`, or
  `"--resume not set: starting fresh from epoch 0"`.

**Side effect worth knowing:** `frozen_trainer.py:426` already called
`save_checkpoint(prefix='last')` and inherits the base method, so under the old
code it was overwriting its own *best* checkpoint every epoch. That trainer is
now correct too.

### 4. `DataFusion.py` loaded notes unconditionally — FIXED

`load_cxr_ehr_rr_dn()` read `discharge.csv` and `radiology.csv` before
checking whether any note modality was requested, and `--notes_data_dir`
defaulted to the authors' cluster path
(`/scratch/baj321/MIMIC-Note/...`). Any EHR-CXR run therefore died with a
confusing `FileNotFoundError` on a path nobody recognises.

Now the notes are only read when `--modalities` contains RR, DN or CXRR;
otherwise correctly-shaped empty frames are used. The default for
`--notes_data_dir` is `None`, and requesting notes without setting it raises a
message naming the flag.

Note this had to cover both branches of `loadmetadata`: the `paired` path
guards note usage by modality, but the **`partial` path merges notes
unconditionally** — and `partial` is what Round 3 uses.

### 5. `cxr_dataset.py` path parsing broke on Windows — FIXED

`MIMICCXR.__init__` built its filename lookup with:

    {path.split('/')[-1].split('.')[0]: path for path in paths}

`glob` returns **backslash**-separated paths on Windows, so `split('/')`
returned the whole path unchanged and every key was a full path rather than a
`dicom_id`. Every lookup then missed, and the CXR loader could not run locally
at all. Verified directly: for the same file, the old expression finds the
`dicom_id` → `False`, the new one → `True`.

Now uses `os.path.basename()`. Behaviour on Linux/Colab is identical; this only
makes local runs possible, which is useful for quick checks without spinning up
Colab.

---

### 6. `--cxr_encoder` did not select the architecture — FIXED (was the most serious)

**This one does not crash. It silently trains the wrong model and reports a
real-looking AUROC that means nothing.** Everything else on this list fails
loudly; this one succeeds wrongly.

`--cxr_encoder vit_small_patch16_384` is read as a **boolean gate**, never as an
architecture:

    if 'CXR' in args.modalities and args.cxr_encoder is not None:
        self.cxr_encoder = CXR_encoder(args)      # always CXRModels

The string is discarded. `CXR_encoder` is an alias for `CXRModels`, which builds
`getattr(torchvision.models, args.vision_backbone)` — and `--vision-backbone`
defaults to **densenet121**, a flag the reference script never passes.

| Intended | Actually built |
|---|---|
| timm `vit_small_patch16_384` | torchvision `densenet121` |
| `[B, 577, D]` patch tokens | `[B, 1024]` pooled vector |
| `CXRTransformer` → `(v_cxr, cls)` | `CXRModels` → `(preds, loss, visual_feats)` |

Consequences:

- `--use_cls_token cls` is meaningless for DenseNet: there is no CLS token, and
  its output has no token dimension at all.
- `max_seq_len = 578` in `fusion.py` (24² + 1 CLS, plus one spare) is ViT
  arithmetic sitting next to an encoder that produces no tokens.
- **Any AUROC produced this way is not a reproduction of the paper's 0.692.**

Same trainers affected: `MSMA_trainer`, `Calibration`, `retired_trainer` — so
every CXR fusion type shares this path, not just `unimodal_cxr`.

**Fix, part 1 — the crash.** Six call sites in `fusion.py` did
`cxr_feats = self.cxr_model(img)` then `cxr_feats[:, 0, :]`, raising
`TypeError: tuple indices must be integers or slices, not tuple`. They now go
through `cxr_pool()`, which normalises both encoder signatures and skips token
pooling for an already-pooled `[B, D]` output.

**Fix, part 2 — the actual selection bug.** `CXR_encoder` is no longer a bare
alias for `CXRModels`; it is a factory that reads `--cxr_encoder`:

- a **token-based timm name** → `CXRTransformer`, `[B, N, D]` patch tokens
- anything else → `CXRModels` with `--vision-backbone`, `[B, D]` pooled

Construction mirrors `DHF_trainer.py`, which already did this correctly. All
four call sites (`MSMA_trainer`, `Calibration`, `retired_trainer`, `fusion.py`)
go through the factory, so every CXR fusion type is fixed at once.

Two things this required:

- `CXRTransformer` had **no `feats_dim`**, which fusion heads read to size the
  classifier (`Classifier(self.cxr_model.feats_dim, args)`). It could not be
  swapped in without an `AttributeError`. Now set from
  `feature_extractor.num_features` (384 for `vit_small_patch16_384`).
- `timm.is_model()` alone is **not** a sufficient test: it returns `True` for
  `densenet121`, whose `forward_features` gives a 4-D feature map, while
  `CXRTransformer.forward` does `b, n, _ = x.shape` and needs 3-D tokens.
  Restricted to token families (`vit_`, `deit_`, `beit_`, `eva_`, ...), with an
  explicit fallback message otherwise.

**Verification — the parameter count settles it:**

    Building CXR encoder: vit_small_patch16_384
      class          : CXRTransformer
      parameters     : 27.6M total, 27.6M trainable
      feats_dim      : 384
      backbone       : 21.8M
      tokens/image   : 577 (384/16 = 24, 24^2 + 1 CLS)
    forward -> v_cxr (2, 577, 384), cls (2, 384)

**27.6M matches the paper's Table C4 exactly**, and 577 tokens matches
`max_seq_len = 578` in `fusion.py`. Compare what was actually being trained
before: `CXRModels`, **7.0M** parameters, `[B, 1024]` pooled, no tokens.

The 21.8M/27.6M split also answers whether the unused internal
`transformer`/`pos_embedding` should be removed: the backbone alone is 21.8M, so
the paper's reported 27.6M **includes** those dead parameters. Keeping them is
the faithful choice.

A startup banner now prints the class, parameter count, `feats_dim` and token
count on every run, so the architecture can be confirmed visually before
trusting any result.

**Consequence for the confidence paths:** `fusion.py` 876, 1031, 1482, 2029 and
2479 unpack a 2-tuple and were previously broken against `CXRModels`' 3-tuple.
With a ViT selected they now receive the 2-tuple they expect, so Round 2's
`c-unimodal_cxr` should work without further changes.

**Related, deliberately not patched:** five other call sites (`fusion.py` 876,
1031, 1482, 2029, 2479) unpack a **2**-tuple — `_, full_cxr_feats =
self.cxr_model(img)`. Those are correct for `CXRTransformer` but raise
`ValueError: too many values to unpack` against `CXRModels`' 3-tuple. They are
the confidence-predictor paths used by Round 2's `c-unimodal_cxr`. Left as-is on
purpose: they start working once the encoder selection is fixed, and patching
them now would hide the real problem. **Youssef: this is why Round 2 will fail
until defect 6 is fully fixed.**

---

### 7. Off-by-one in loss averaging — FIXED

Both training and validation loops used `enumerate(dl)`, which starts at **0**,
then divided the accumulated loss by `i` — the last *index* rather than the
batch *count*. Six locations in `MSMA_trainer.py`: 186, 212, 218, 319, 327, 364.

| Batches | Divided by | Should be | Effect |
|---|---|---|---|
| 1 | 0 | 1 | **ZeroDivisionError** |
| 2 | 1 | 2 | loss 100% too high |
| 31 (our real val set: 490 imgs @ bs 16) | 30 | 31 | loss ~3% too high |

Found when the 50-image dry run gave 10 validation images at `batch_size 16` —
exactly one batch, so `i` stayed 0 and validation crashed.

Fixed by `enumerate(dl, 1)` in both loops, so `i` counts batches.

**What was and was not affected — this distinction matters:**

- **AUROC and AUPRC were never affected.** They are computed by `computeAUROC`
  from `outPRED` / `outGT`, tensors accumulated across batches, and never touch
  `i`. **No number in the results table was ever wrong.**
- **Printed and wandb-logged loss values were inflated** by `n/(n-1)`. Loss
  curves from before this fix are slightly too high.
- **Model selection and early stopping were _not_ misled**, for a reason worth
  stating precisely. Two paths exist:
  - most fusion types (including `unimodal_cxr`) select on
    `ret['auroc_mean']` — untouched by this bug;
  - `c-unimodal_*` (Round 2 confidence training) selects on
    `avg_loss = epoch_loss/i`.

  Even in the second case the inflation is a **constant multiplicative factor**
  within a run — the validation set and batch size do not change between epochs
  — and scaling all values by a positive constant preserves their ordering. So
  `if self.best_auroc > avg_loss` picked the same epoch either way, and the
  patience counter behaved identically. **No previously selected checkpoint was
  the wrong epoch.**
- One genuine inconsistency it caused: the LR scheduler at line 316 used
  `epoch_loss/len(self.val_dl)` — the *correct* divisor — while selection used
  `epoch_loss/i`. Both are now the same.

**Note on loss comparability:** post-fix loss values are slightly lower than any
the original authors reported, because theirs carry the off-by-one. That is a
correction, not a regression, but worth a line in the thesis if loss curves are
ever compared directly. AUROC/AUPRC remain directly comparable.

**Two things deliberately left alone:**

- `get_eta()` in `trainer.py` does `iter += 1`, which was compensating for the
  0-indexed `i`. It now double-counts by one iteration out of tens of thousands
  (negligible). Left as-is because `get_eta` lives in the base class and the
  other trainers (`DHF`, `daft`, `ensemble_*`, ...) are **still 0-indexed** —
  removing the compensation would give them a division by zero on their first
  batch.
- Line 184 reads `if i % 100 == 9:`, not `== 0`. That offset appears to be the
  original author's workaround for the very same bug: printing at `i == 0` would
  have divided by zero. Evidence the off-by-one was known and dodged rather than
  fixed.

**Only `MSMA_trainer` is fixed.** The same pattern exists in `DHF_trainer`,
`daft_trainer`, `ensemble_*` and others. They will crash identically on any
validation set that yields a single batch.

---

### 8. Metrics broke on phenotypes with no positives — FIXED

Surfaced as a crash after three clean epochs:

    trainers/utils.py:32  delta = (true_value - list_)
    TypeError: unsupported operand type(s) for -: 'float' and 'list'

**The causal chain matters more than the fix**, because two of the three links
fail silently and the last one is the damaging one:

**(a) The crash.** `evaluate_new()` returns `np.nan` when a label column has
fewer than two classes. `computing_confidence_intervals()` then evaluates
`true_value - list_` with `list_` a plain Python list. That only ever worked
because `roc_auc_score` returns a **numpy** scalar, which broadcasts over a
list; `np.nan` is a **Python** float, which does not. The code silently depended
on the type sklearn happened to return.

**(b) The filter that never filtered.**

    if auprc != np.nan and auroc != np.nan:     # ALWAYS True

`nan != nan` is True under IEEE 754, so NaNs were appended to the bootstrap
lists regardless. `np.percentile` over a NaN-containing array returns NaN, so
rare phenotypes got **confidence intervals that looked computed but were not**,
and would have reached the results table unnoticed. More dangerous than the
crash.

**(c) NaN poisoned model selection — the serious one.** The per-class point
estimate is appended to `auc_scores`, and the aggregate was `np.mean(...)`,
which propagates NaN. So **one** phenotype with zero positives made
`auroc_mean` NaN for all 25. That value drives selection:

    if self.best_auroc < ret['auroc_mean']:    # False for nan, always

Any comparison against NaN is False, so **no checkpoint would ever be saved and
patience would increment every epoch** — training would run to
`--patience 15`, save nothing, and print `nan` throughout. Fixing only (a)
would have converted a loud crash into exactly that silent failure.

**The fixes:**

- `computing_confidence_intervals` converts to `np.asarray` and returns
  `(nan, nan)` when the point estimate is undefined or the bootstrap list is
  empty.
- The filter uses `not np.isnan(...)`; discarded draws are counted and reported
  per column, e.g. `22/200 bootstrap samples discarded (resample had only one
  class); CI computed from 178 samples`.
- Undefined columns print a named warning, e.g.
  `[metrics] Rare condition A: AUROC/AUPRC undefined (0 positives out of 60)`.
  Column names come from the listfile header via `Trainer.set_class_names()`,
  falling back to indices.
- AUPRC is treated as undefined too when there are no positives. sklearn returns
  0.0 there, which would otherwise drag the reported mean down with a
  meaningless value.
- **`auroc_mean` uses `np.nanmean`**, averaging the computable classes, and
  every epoch prints the contributing count:

      [metrics] auroc_mean 0.4633 over 2/3 classes  <-- PARTIAL

  **Decision and reasoning:** leaving the mean as NaN would be "honest" but
  makes training impossible via chain (c). `nanmean` keeps checkpointing
  working while `n_classes_scored` / `n_classes_total` make the gap impossible
  to miss. The count is printed **every epoch**, not once, so drift is visible
  during the run — a different phenotype can go empty in a different epoch.
  Nothing is fabricated; a partial mean can never be mistaken for the full
  25-class figure.

**Also removed: five dead sklearn calls.** `computeAUROC` computed
`roc_auc_score` (average=None/micro/macro/weighted) and
`average_precision_score`, then overwrote `auc_scores`/`auprc_scores` with empty
lists two lines later. Every result was discarded. They were the main source of
the `UndefinedMetricWarning` wall on any batch containing an empty phenotype.

#### Which phenotypes are exposed, and what you will see in the results table

Evaluation loops over **all 25 phenotype columns** and computes a CI for each,
so this is not about pneumonia's 12.7%. With **490 validation images**, a
condition needs only a handful of positives to be at risk:

- **Zero positives in the split** → that row shows `(nan, nan)` and the epoch
  line reads `over 24/25 classes`. **Expected, not a bug.**
- **A few positives** (roughly single digits) → the CI is computed, but from
  fewer than 1000 bootstrap samples; the discard count is printed.
- Pneumonia itself (~62 positives at 12.7%) is comfortable.

The benchmark's rarer conditions are the candidates — things like
*Pleurisy; pneumothorax; pulmonary collapse*, *Other liver diseases*,
*Conduction disorders* and *Shock* sit well below the common cardiac and renal
categories. **Check the actual per-column positive counts once Caroll's
listfile arrives** rather than trusting this ordering.

So a `(nan, nan)` in the results table means "this condition had too few
positives to measure here", not "something broke". Worth a footnote in the
thesis rather than an apology.

---

### Still carrying the authors' cluster paths

`--ehr_data_dir` and `--cxr_data_dir` still default to
`/scratch/fs999/shamoutlab/...`. Same failure mode as defect 4, but the README
already says to always pass them explicitly, so they are less dangerous. Worth
changing to `None` with a clear error for consistency.

---

## Where the CXR loader's safety actually comes from — read before debugging it alone

`mimic-cxr-2.0.0-metadata.csv` lists **all 377,110 images**, and it stays that
way even when `resized/` holds only our cohort. So what stops the loader asking
for an image we never downloaded?

**The listfile join, not the metadata.** In the fused path the images are
indexed **by `dicom_id` string**, and those ids come from `metadata_with_labels`
*after* it has been inner-joined with the EHR listfile on `stay_id`. That join
is what restricts requests to our cohort.

The practical consequence:

- With `--data_pairs paired` (and `partial`), the join protects you. This is
  the normal path and it is fine.
- **Running the CXR loader standalone**, without that join, against a full
  `metadata.csv` and a partial `resized/`, will raise `KeyError` on the first
  image present in the metadata but absent from disk.

This bit us on the 50-image dry run: the test folder had 50 images but
cohort-free metadata and split files, so the standalone loader happily
requested images that did not exist. Fixed there by filtering all three CSVs in
`data/test_run/` down to the 50 images actually present.

It will **not** recur on the real data, because the listfile join does the
restricting — but only as long as you go through the fused path. If you ever
debug the CXR encoder in isolation, filter the metadata to your cohort first.

---

## Other repo gotchas

| Finding | Consequence |
|---|---|
| `--resize` and `--crop` are dead arguments | `get_transforms()` hardcodes `(384, 384)` and ignores them. |
| Training augmentation uses `RandomAffine(degrees=45)` | Far more rotation than real chest x-rays vary. Reproduce as-is first, test 10-15° later as an ablation. |
| Labels do **not** come from the images | `--data_pairs paired` routes through `DataFusion.py`, which pulls phenotype labels from the **EHR listfile**. `chexpert.csv` must exist for the loader but its Pneumonia column is not the training label. |

---

## Dry run on fake data (before Caroll's listfile)

`data/test_run/` holds 50 real resized images plus a fabricated EHR side, so
the whole training path can be exercised without real labels. Built by
`scripts/build_fake_ehr_for_test.py`.

Note the reference command runs `--modalities EHR-CXR`, **not CXR alone**, so
even the "unimodal CXR" run loads the EHR side. It needs
`root/all_stays.csv`, the three listfiles, and one timeseries file per
listfile row under `phenotyping/train/` (val reads from `train/` too) and
`phenotyping/test/`.

The fake `all_stays.csv` carries the **real** `intime`/`outtime` from MIMIC-IV
for those 50 stays, so the 48-hour filter is genuinely tested rather than
passing trivially. Running MedPatch's own filter chain against the real
metadata reproduced exactly the 50 images selected independently:

    CXR rows for these 50 patients (any view) : 1,014
      after AP-only filter                    :   802   (212 dropped)
      after 48h window filter                 :   105   (697 dropped)
      before one-image-per-stay               :   105
      after  one-image-per-stay               :    50   (55 collapsed)

Both filters did real work, and the pipeline's selection agrees with mine.

Labels in the fake listfiles are random, with the pneumonia column sampled at
the paper's 12.7% prevalence rather than 50/50 so that class imbalance is
present. **The AUROC from this run is meaningless** — it only proves images
are found, nothing crashes, and checkpoints reach Drive.

---

## Image size decision

**384px**, matching the paper (`vit_small_patch16_384`). No code change needed
— `--image_size` already defaults to 384 and the transforms are hardcoded to
it. 384÷16 = 24, so 24² + 1 CLS = 577 tokens; `fusion.py` sets
`max_seq_len = 578`.

Cost: 20-30 GPU hours per run, across multiple Colab sessions.
**This makes checkpoint resume mandatory, not optional.**

---

## Open items

- Waiting on Caroll's phenotype listfile — blocks all training
- Task B images still downloading (8,461 remaining, ~15 h)
- After Task B + resize: `resized/` must contain **15,181** files, not 17,920
  (2,739 images serve both tasks). Run `scripts/verify_resized_complete.py`.
- Ask Dr. Gouda about a department GPU
- RAD-DINO: check its published `training_images.csv` against our test set for
  overlap before reporting results
