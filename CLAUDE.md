# CLAUDE.md — PneumoVision

Guidance for Claude Code and for anyone new to this repo. Read the **Hard rules**
and **Ground truth** sections before writing code that touches inference.

PneumoVision is a multimodal clinical decision-support prototype for early
respiratory failure detection (AASTMT graduation project). It fuses ICU EHR
time-series, chest X-rays and clinical notes using the **MedPatch**
confidence-guided multi-stage fusion design (Al Jorf & Shamout, MLHC 2025,
[arXiv:2508.09182](https://arxiv.org/abs/2508.09182)).

---

## Hard rules

1. **Never open, read, print, grep or summarise raw MIMIC-IV / MIMIC-CXR /
   MIMIC-IV-Note records.** The PhysioNet credentialed-access DUA forbids
   sharing that data with third-party online services. Work from code, schemas
   and column names only. Where sample data is needed, generate **synthetic**
   fixtures matching the schema.
   - Off limits: anything under `data/`, `*_listfile.csv`, `mimic-cxr-*.csv`,
     `discharge.csv`, `radiology.csv`, the `*.pkl` test resources under
     `medpatch/mimic4extract/`, and `medpatch/ehr_utils/mimic-cxr-ehr-split.csv`.
   - Fine to read: `discretizer_config.json`, `channel_info.json`, model and
     training code, SLURM scripts, `variable_ranges.csv`.
2. **Never commit datasets, model weights, `.env` files or secrets.** They are
   gitignored; keep it that way.
3. **Do not restructure `medpatch/`.** It is the training codebase and is read
   from, not reorganised. Inference must import its preprocessing rather than
   reimplement it.
4. TypeScript everywhere on the JS side with `strict: true`. Python 3.11 with
   type hints on the ML side.
5. Build in the phases below. At the end of each: typecheck, lint, test, then
   summarise before continuing.
6. Align to what the training code **actually does**, never to an assumed
   architecture. See Ground truth.

---

## Ground truth from the training code

Verified by reading `medpatch/` and by running the real discretizer on
synthetic input. Several of these contradict the original project brief — the
code wins.

| Thing          | Reality                                                          | Not                  |
| -------------- | ---------------------------------------------------------------- | -------------------- |
| EHR tensor     | `[B, 48, 76]`                                                    | not 48×17            |
| EHR encoder    | `LSTM`, 1 layer, hidden 512, `input_dim=76`                      | —                    |
| CXR encoder    | timm `vit_small_patch16_384`, **384×384**                        | not 224×224          |
| CXR patch grid | **24×24** (+1 CLS token, excluded from heatmaps)                 | not 14×14            |
| Text encoder   | `emilyalsentzer/Bio_ClinicalBERT`, frozen                        | not dmis-lab BioBERT |
| Modalities     | **four**: EHR, CXR, RR (radiology reports), DN (discharge notes) | not three            |
| Fusion model   | `CMSMAFusion` (`--fusion_type c-msma`)                           | —                    |
| Output         | `{'high_conf', 'low', 'late'}`, all **logits**                   | not probabilities    |
| θ              | `0.75` (`--*_confidence_threshold`)                              | —                    |

### The 76-column EHR layout

The 17 variables in `discretizer_config.json → id_to_channel` expand to 76
columns per hourly bin:

- **12** continuous scalars (at column indices 2, 3, 49–58 — the ones the
  `Normalizer` standardises)
- **47** one-hot columns for the 5 categorical variables (capillary refill 2,
  GCS eye 8, GCS motor 12, GCS total 13, GCS verbal 12)
- **17** `mask->` columns flagging which variables were actually charted

`packages/shared/src/generated/ehr-variables.ts` is generated from that config
by `npm run gen:ehr` — never hand-edit it. A test in
`packages/shared/src/constants.test.ts` fails if it drifts from the Python.

### Tasks

| Platform task | medpatch task           | classes          | modalities    | fusion     |
| ------------- | ----------------------- | ---------------- | ------------- | ---------- |
| `mortality`   | `in-hospital-mortality` | 1                | EHR-CXR-RR    | `c-msma`   |
| `pneumonia`   | `phenotyping`           | 25, **index 21** | EHR-CXR-RR-DN | `c-e-msma` |

Pneumonia is the CCS **phenotype** label
`'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)'`,
not the CheXpert CXR finding. Class 22 is `'Respiratory failure; insufficiency;
arrest (adult)'` — relevant to the project's framing and available if needed.

**F8 leakage control** is already the upstream design: mortality trains on
`EHR-CXR-RR` with discharge notes excluded. The API enforces the same rule
server-side (`NOTE_TYPES_BY_TASK` in `packages/shared`), unit-tested.

### Checkpoints

Saved by `Trainer.save_checkpoint()` as:

```
{save_dir}/{task}/{fusion_type}/best_checkpoint_{lr}_{task}_{fusion_type}_{modalities}_{data_pairs}.pth.tar
```

A dict with `epoch`, `state_dict`, `best_auroc`, `optimizer`, plus `weights`
(the learnable α tensor) for `c-msma` / `c-e-msma`. Normalizer state lives in
`medpatch/normalizers/*.normalizer` (pickled `{means, stds}`, 76-wide).

### Model status — read before wiring real inference

- **`checkpoints/` is empty. No trained weights exist in this repo.** MOCK_MODE
  is the only mode that runs today; `MOCK_MODE=false` raises at startup by
  design rather than failing obscurely later.
- **Known upstream defect:** the CXR branch of `CMSMAFusion` references
  `self.cxr_model.cxr_encoder.projection_layer`
  ([`fusion.py:1158`](medpatch/models/fusion.py)), which no class in this repo
  provides — `cxr_models.py` defines a torchvision DenseNet and never reads
  `--cxr_encoder`. Trimodal `c-msma` therefore cannot be constructed as-is.
  EHR-only `c-msma` **does** build and run. Fixing this belongs to the ML
  track; the web platform does not patch `medpatch/`.
- `fusion_main.py` defaults the normalizer to the **phenotyping** file
  (`ph_ts{timestep}...`) regardless of task. Pass `--normalizer_state`
  explicitly for mortality runs.

---

## Repository layout

```
medpatch/            EXISTING training code — read only, do not restructure
  models/            encoders + fusion heads (CMSMAFusion is MedPatch)
  trainers/          MSMA_Trainer drives c-msma
  ehr_utils/         Discretizer + Normalizer (+ resources/)
  normalizers/       bundled normalizer state
  mimic4extract/     vendored MIMIC benchmark extraction code
services/inference/  FastAPI inference service (imports preprocessing from medpatch/)
apps/api/            Express + TypeScript gateway
apps/web/            React 18 + Vite SPA
packages/shared/     zod schemas, types and constants shared by api + web
infra/nginx/         edge reverse proxy (TLS) and SPA static config
infra/scripts/       gen-certs.sh, gen-ehr-constants.mjs
checkpoints/ data/ results/   gitignored
```

## Commands

| Task                     | Command                                           |
| ------------------------ | ------------------------------------------------- |
| Install (JS)             | `npm install`                                     |
| Typecheck all            | `npm run typecheck`                               |
| Lint / fix               | `npm run lint` · `npm run lint:fix`               |
| Format                   | `npm run format`                                  |
| Test all (JS)            | `npm test`                                        |
| Test inference           | `cd services/inference && python -m pytest`       |
| Test ML env              | `python -m pytest` (repo root)                    |
| Regenerate EHR constants | `npm run gen:ehr`                                 |
| Dev certs                | `bash infra/scripts/gen-certs.sh`                 |
| Full stack               | `docker compose up --build` → <https://localhost> |
| Audit deps               | `npm run audit:js` · `npm run audit:py`           |

The ML side uses the repo-root `.venv` (Python 3.11): activate it before
running `pytest` or any `fusion_main.py` command.

## Conventions

- **Responses**: `{ data, meta? }` on success, `{ error: { code, message, requestId } }`
  on failure. Pagination via `?page&limit`, limit ≤ 100.
- **Validation**: zod on every input, schemas imported from `packages/shared`.
  Config is validated at boot and the process crashes on anything missing.
- **Probabilities** render as a percentage with one decimal, everywhere.
- **Logging**: pino with redaction. Never log note text, demographics, cookies
  or auth headers. The inference service logs only request id, task,
  availability vector and latency — never input content.
- **Colour**: the risk palette is colour-blind-safe (Okabe–Ito) and colour is
  never the only signal — always pair it with an icon and a text label.
- **Imports**: `.js` extensions in relative TS imports (NodeNext resolution).
- `packages/shared/src/generated/` is generated; edit the generator instead.

## Build phases

1. ✅ **Scaffold** — workspaces, TS/ESLint/Prettier, Husky, Docker stack, env,
   docs, shared constants generated from the training config.
2. **Shared contracts** — zod schemas for every entity and the prediction
   payload; export JSON Schema; validate the pydantic models against it.
3. **Auth, RBAC, audit** — argon2id, TOTP MFA, rotating refresh tokens with
   reuse detection, hash-chained audit log.
4. **Clinical data** — patients, stays, vitals (time-series collection), images
   (MinIO), encrypted notes, synthetic seed (≥ 20 stays).
5. **Inference service** — real pipeline behind a flag, preprocessing parity
   test against `medpatch`, HMAC verification.
6. **Prediction flow** — BullMQ job, F8 leakage enforcement, Socket.IO status,
   prediction report (F1–F6).
7. **Explainability UI** — CXR heatmap, note highlighting, EHR confidence band.
8. **Alerts, ward dashboard, ICU replay** (P1–P4).
9. **Model performance page** (F10) — ROC/PR with CIs, ablations, reliability
   diagram, MedPatch reference column.
10. **Hardening** — security pass, Playwright e2e, production compose,
    `SECURITY.md`.

Definition of done per phase: typecheck, lint and tests pass, and the stack
still builds from a clean clone.
