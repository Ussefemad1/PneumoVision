# Testing Round 2b (calibration) yourself

This continues [TESTING_ROUND2.md](TESTING_ROUND2.md): same setup, same virtual
data, same manifest. Do Round 2 first (or at least read it) — Round 2b loads
Round 2's checkpoints.

> **Everything in this guide runs on SYNTHETIC data — not a scientific result.**
> Every AUROC, AUPRC and ECE number printed here is watermarked that way. Never
> report accuracy; we use AUROC, AUPRC and, for calibration, ECE.

## What Round 2b is, in one paragraph

Round 2 taught each reader's confidence head to _rank_ evidence: a token with
γ = 0.9 is surer than one with γ = 0.6. Round 2b makes those numbers
_honest_: 0.9 should mean "right about 90% of the time". It loads the Round 2
checkpoint, freezes **everything** (reader, classifier, Round 2 confidence
head) and trains only a **temperature**: one number per token position and
class, dividing the confidence logits. It trains on the **validation split
only**, never train or test. The check is ECE, the expected calibration error:
the average gap between stated confidence and actual correctness (lower is
better; 0 is perfect).

In medpatch this is `--fusion_type temp_c-unimodal_<reader>`, which
`fusion_main.py` routes to `trainers/Calibration.py`, driven by
`scripts/phenotyping/Calibrate/Calibrate-<READER>.sh`.

**Hardware:** like Round 2, this is CPU-scale on virtual data. A laptop CPU is
enough; Colab works the same way (commands below) but no GPU is needed.

## Where things live

| Path                                               | What                                                                                        |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `tools/pv/run.py r2b`                              | runs Round 2b for one reader                                                                |
| `tools/pv/smoke_round2b.py`                        | the whole chain in one command (r1 → r2 → r2b → checks)                                     |
| `tests/round2b/test_round2b.py`                    | the checks a)–f)                                                                            |
| `tests/test_round2b_tools.py`                      | non-training checks of the tooling and trainer fixes                                        |
| `runs/virtual/r2b/<reader>/<id>/medpatch_outputs/` | the ECE tables, per-token probability CSVs and 25 calibration-curve PNGs the trainer writes |
| `docs/model_track_notes.md`                        | the Round 2b trainer fixes, and why                                                         |

---

## 1. Setup

Exactly as in [TESTING_ROUND2.md § 1](TESTING_ROUND2.md#1-setup). On Colab,
keep `RUNS_ROOT` and `PV_MANIFEST` on Drive (Cell 2 there): Round 2b finds its
Round 2 parent through the manifest.

---

## 2. The one-command chain

From the repo root:

```powershell
python -m tools.pv.smoke_round2b                      # r1 -> r2 -> r2b -> checks
python -m tools.pv.smoke_round2b --skip-r1 --skip-r2  # reuse r1 and r2, do r2b + checks
python tools/pv/smoke_round2b.py --skip-r1 --skip-r2  # the same, as a file path
python -m tools.pv.smoke_round2b --readers cxr        # one reader
```

On Colab: `!python -m tools.pv.smoke_round2b --skip-r1 --skip-r2`

If you already ran the Round 2 smoke test, `--skip-r1 --skip-r2` reuses those
rows and only adds the calibration step.

**Expected output — not yet measured.** Round 2b has not been run end to end
yet (it was written and checked without training). The summary it ends with
has this shape; the numbers are to be filled in from the first real run:

```
=== summary (SYNTHETIC -- not a scientific result) ===
  r2b-ehr-001   parent r2-ehr-001   val AUROC ...  AUPRC ...  ECE val a->b (fit split)  ECE test c->d (held out)
  r2b-cxr-001   parent r2-cxr-00N   ...
  r2b-rr-001    parent r2-rr-001    ...
  r2b-dn-001    parent r2-dn-001    ...
  ...timings per step...
  total time ...   checks PASSED   manifest: ...
```

How to read the two ECE numbers per reader:

- **ECE val a → b** is on the split the temperature was fitted to. It should
  not get worse (check e allows +0.02).
- **ECE test c → d** is held out. It is the honest one, and on ~15 synthetic
  stays with ~2 pneumonia cases it is noise. Don't read anything into it.

With the paper's learning rate (0.001) each temperature moves by at most about
0.001 per step. On virtual data that is ~4 steps per epoch, so 30 epochs (EHR,
CXR) or 10 (RR, DN) move it by at most ~0.1. **Expect small ECE changes.**
That is enough to show the plumbing works, not a calibrated model.

### One reader by hand

```powershell
python -m tools.pv.run r2b --reader cxr --data virtual            # needs an r2 cxr row
python -m tools.pv.run r2b --reader cxr --data virtual --dry-run  # show the command only
```

`run.py` **refuses** Round 2b unless its parent is an **r2** row (not r1) for
the same reader and data kind, whose file still matches the manifest sha256.
With no r2 row it tells you to run
`python -m tools.pv.run r2 --reader cxr --data virtual` first.

---

## 3. The checks one by one

```powershell
python -m pytest -m round2b -s                # all of a)–f)
python -m pytest -m round2b -s -k test_e      # one check
python -m pytest -m round2b -s -k "test_a and cxr"
```

**Check f) trains** (one calibration epoch, to prove only the validation split
is read). The others only load and score saved checkpoints.

| Check                               | What it proves                                                                                                           | Pass looks like                                                                                                                                                                           |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **a) only the temperature trained** | Every weight except `<reader>_temperature` is bit-identical to the Round 2 parent, and only the temperature is trainable | trainable list shows only `fusion_model.<reader>_temperature`; `N tensors (reader, classifier, Round 2 confidence head) bit-identical to r2 parent r2-cxr-00N; changed: [...temperature]` |
| **b) temperature positive**         | The learned temperature stays above the forward's `clamp_min(1e-9)` floor                                                | `temperature: ... min 0.9…  mean …  max 1.0…` and no failure                                                                                                                              |
| **c) round-trip**                   | The saved file reloads and gives identical outputs                                                                       | `reload x2 -> identical outputs, shape (4, 1, 25)` for CXR                                                                                                                                |
| **d) lineage**                      | r2b → r2 → r1, every file still matches its sha256                                                                       | `r2b-cxr-001 <- r2-cxr-00N <- r1-cxr-001  (cxr, sha256 ok)`                                                                                                                               |
| **e) ECE sanity**                   | ECE on the calibration split is not obviously worse after training                                                       | `SYNTHETIC -- not a scientific result: ECE ... val x -> y (fitted here)  test ... (held out)`                                                                                             |
| **f) val only**                     | Calibration never iterates the train or test loader                                                                      | `calibration read val twice; train and test never iterated`                                                                                                                               |

Plain `python -m pytest` skips all of these on purpose. It still runs
`tests/test_round2b_tools.py` (no training), which checks the parts that can be
checked without a model: script parsing, the r2-parent rule, the planned
command, the trainer fixes on synthetic tensors, and that every Round 2
checkpoint key restores into the Round 2b model.

---

## 4. Reading the manifest

Same file and columns as [TESTING_ROUND2.md § 5](TESTING_ROUND2.md#5-reading-runsmanifestcsv).
Round 2b rows have `stage` = `r2b` and `parent_id` = an `r2-…` id. Their
`notes` column adds the two ECE pairs, e.g.
`ECE val 0.1234->0.1200 (fit split); ECE test ...` (numbers illustrative).

---

## 5. Common failures and fixes

Everything in [TESTING_ROUND2.md § 7](TESTING_ROUND2.md#7-common-failures-and-fixes)
applies. Round 2b adds:

| Symptom                                              | Cause                                                                                | Fix                                                                             |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| `No r2 row for reader 'cxr'`                         | Round 2b needs a Round 2 parent                                                      | `python -m tools.pv.run r2 --reader cxr --data virtual`                         |
| `Parent r1-cxr-001 is stage 'r1', not r2`            | `--parent` pointed at a Round 1 row                                                  | pass an `r2-…` id, or omit `--parent`                                           |
| `run finished but …best_checkpoint… was not written` | Calibration only saves when ECE improves on the val split; it never did              | more epochs: `python -m tools.pv.run r2b --reader X --data virtual --epochs 60` |
| An interrupted r2b run starts again from epoch 0     | `trainers/Calibration.py` has no resume support (`--resume` is accepted but ignored) | rerun; r2b is short on virtual data                                             |
| Check a) says `temperature unchanged`                | same cause as the missing checkpoint                                                 | as above                                                                        |

---

## 6. When real Round 2 files exist

Round 2b needs **r2** parents. If Round 2 was run on real data with
`tools/pv/run.py r2 --data real`, the r2 rows are already in the manifest:

```python
!python -m tools.pv.run r2b --reader ehr --data real --ehr-data-dir "<ehr>" --cxr-data-dir "<cxr>" --notes-data-dir "<notes>"
!python -m tools.pv.run r2b --reader cxr --data real --ehr-data-dir "<ehr>" --cxr-data-dir "<cxr>" --notes-data-dir "<notes>"
!python -m tools.pv.run r2b --reader rr  --data real --ehr-data-dir "<ehr>" --cxr-data-dir "<cxr>" --notes-data-dir "<notes>"
!python -m tools.pv.run r2b --reader dn  --data real --ehr-data-dir "<ehr>" --cxr-data-dir "<cxr>" --notes-data-dir "<notes>"
```

Paper settings apply on real data (epochs 100, batch 16). Then run the checks
with `python -m pytest -m round2b -s`. Check e) runs on real data too, but it
remains a sanity check, not a result. Log each r2b file in the team Sheet, as
for Round 2.
