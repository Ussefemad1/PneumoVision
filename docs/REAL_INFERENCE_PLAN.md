# Plan: serving the real Round 3 checkpoints (replacing MOCK_MODE)

Status: **plan only, nothing built.** The service (`services/inference`) runs in
MOCK_MODE today, and `MOCK_MODE=false` raises at startup by design. This page
lists what it needs to serve:

- mortality: `r3-fusion-001`, available now;
- phenotyping / pneumonia: the next `r3` run.

Facts below are from `medpatch/`, `tools/pv/round3.py` and the run folders.

## 1. What a Round 3 checkpoint is

`best_checkpoint_<lr>_<task>_c-msma_<modalities>_partial.pth.tar`, saved by
`Trainer.save_checkpoint`. It is a dict with:

- `state_dict`: the **whole** model. That includes the frozen readers (LSTM,
  ViT-small, BERT), their confidence heads and temperatures, and the fusion.
- `best_head`: the head chosen on validation (F3). `r3-fusion-001` uses
  `late`.
- `weights`, `epoch`, `best_auroc`, `optimizer`. The service ignores the
  optimizer.

The shared folder (`PneumoVision_shared/round3/<task>/`, written by
`--step share`) holds this checkpoint and `run.json`. `run.json` has the exact
`argv` (fusion type, modalities, thresholds, `--bert_model_name`, normalizer,
`--load_*` parents) and `manifest_row` (id, sha256, `bert_model_name`).

## 2. Loading one checkpoint

1. Read `run.json`. Rebuild the args with medpatch's own parser
   (`arguments.args_parser().parse_args(argv)`), dropping `--load_*`,
   `--resume` and `--save_dir`. The model is then built exactly as it was
   trained: modalities, `patch_output_dim`, thresholds, `cxr_token_confidence`
   and the flags.
2. **Verify the file.** Its sha256 must equal `run.json["manifest_row"]["sha256"]`
   (= the team manifest's row). Refuse to start otherwise, as
   `tools/pv/check_weights` and `--step share` do.
3. Build the encoders as `MSMA_Trainer.__init__` does: `EHR_encoder(args)`,
   `CXR_encoder(args)`, `Text_encoder(args, device)`. Then build
   `Fusion(args, …)`.
4. Load the checkpoint `state_dict` **strictly**: every model key present, no
   extra key, equal shapes. Do not use `Trainer.load_state`; it is
   non-strict, and its `copy_` broadcasts. Then call `model.eval()` and keep
   `best_head`.
5. Run one warm-up forward pass on a synthetic input. Report `models_loaded`
   and `warmed_up` from `/v1/health`.

**Pretrained downloads.** `Text_encoder` calls
`BertModel.from_pretrained(args.bert_model_name)`, and `CXR_encoder` calls
`timm.create_model(..., pretrained=True)`. Both download weights that the
checkpoint then overwrites. On Railway, either bake the two Hugging Face
caches into the image or volume (`HF_HUB_OFFLINE=1`), or build without
pretrained weights. The second needs a flag in medpatch (open question 3).

## 3. BERT from the manifest, not a constant

- The text reader must be `run.json["manifest_row"]["bert_model_name"]`,
  i.e. **`dmis-lab/biobert-v1.1`** for our runs, for both the weights and the
  **tokenizer**.
- `services/inference/app/main.py` hardcodes
  `TEXT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"`, and `/v1/model-info`
  reports it. Both must come from the loaded model.
- **Why the name matters:** a wrong name loads without error (both BERTs have
  a 28,996-token vocabulary) but tokenizes differently.
- **What to check at load:** that the name in `run.json` matches the
  tokenizer actually loaded, and refuse a mismatch.

## 4. Same preprocessing as training, θ = 0.75

Import medpatch's code; reimplement nothing (CLAUDE.md rule 3).

- **EHR:**
  - `Discretizer(timestep=1.0, store_masks=True, impute_strategy="previous",
start_time="zero")` → 76 columns per hour.
  - Then `Normalizer.load_params(<the run's normalizer file>)`: the
    `--normalizer_state` in `run.json`, or fusion_main's default
    (`ph_ts1.0…normalizer`) when absent.
  - Then `pad_zeros`, as in `DataFusion.my_collate`: mortality pads to the
    batch maximum (48 h window); phenotyping pads to 2,646 rows. The LSTM
    packs by `seq_lengths`.
- **CXR:** `cxr_dataset.get_transforms` test transform: Resize 384,
  CenterCrop 384, ToTensor, ImageNet Normalize. A **missing** X-ray is a
  raw all-zero `[3, 384, 384]` tensor, not normalized. That is what
  `detect_missingness_batch` recognises.
- **Notes:**
  - `Text_encoder.process_notes`: tokenizer padding to 512, overflowing
    chunks averaged.
  - A missing note is `""`. It is still encoded, because the main run keeps
    upstream's phantom tokens.
  - Radiology reports for one stay were **joined with a space** in training
    (`DataFusion` `groupby(...).agg(' '.join)`); see open question 4.
- **θ:** the four `--*_confidence_threshold` values in `run.json` (0.75).
  `Settings.default_theta` must equal them; refuse to start otherwise.
- **Output:**
  - `model(...)` returns `high_conf`, `low` and `late` logits. Use
    `best_head`'s logits (for `combined` / `less_combined`, the same averages
    as `MSMA_Trainer`).
  - Probability = sigmoid. Pneumonia is phenotype class **21**; mortality is
    class 0.
  - These are **not calibrated** probabilities: Round 2b calibrated the
    confidence heads, not the fused prediction.

## 5. Two tasks; DN never for mortality

- **One model per task.** `mortality` → `r3-fusion-001` (EHR-CXR-RR), and
  `pneumonia` → the phenotyping `r3` run (EHR-CXR-RR-DN).
- **Mortality never receives discharge notes.**
  - The API already filters (`NOTE_TYPES_BY_TASK` in `packages/shared`).
  - The service must refuse too: a 400 if a `discharge` note arrives with
    `task=mortality`, as `Trainer.load_reader_parents` and the tools refuse
    DN for mortality.
- **Note types → readers:** `radiology` → RR; `discharge` → DN (pneumonia
  only). `progress` / `nursing` have no reader (open question 4).

## 6. CPU vs GPU, memory on Railway

- **Weights per model:**
  - BERT-base ≈ 110M parameters (~440 MB);
  - ViT-small/16-384 ≈ 22M (~90 MB);
  - LSTM ≈ 1.2M; fusion ≈ 1.2M (mortality) to 4.8M (phenotyping).

  That is about **0.55 GB per model in fp32, ~1.1 GB for both**, plus
  ~1–1.5 GB for the torch/transformers runtime.

- **Sharing the readers:** both tasks' readers are frozen copies of the same
  BioBERT, and the same ViT backbone if the CXR parents agree. Sharing one
  copy would save ~0.5 GB, but only after a bit-equality check at load (open
  question 6).
- **CPU latency (no GPU on Railway):** one request runs BERT on each note
  chunk (~0.3–1 s per 512-token chunk on a few vCPUs; a long discharge note
  is several chunks), the ViT (~0.3–0.5 s) and the LSTM (fast).
  - Expect **~1–2 s for mortality** and **~2–6 s for pneumonia** with a
    discharge note.
  - Use `torch.set_num_threads`, `torch.inference_mode()` and one worker
    process (models are not fork-safe to copy).
- **Image:** `INSTALL_ML=true` with CPU torch (not the CUDA wheel),
  `transformers==4.45.2` and `timm` as in `requirements.txt`. That is
  roughly 1.5–2.5 GB.

## 7. Checkpoint integrity

- **sha256 check:** at startup, compare each checkpoint's sha256 with
  `run.json["manifest_row"]["sha256"]`.
- **Optionally:** also compare with the team `runs/manifest.csv` row,
  carried as a small signed "model card" JSON next to the file.
- **On mismatch:** refuse to start (`/v1/health` degraded, no predictions).
  Report the run id and the first 12 characters of the hash in
  `/v1/model-info` (`version`).

## Files to change

| File                                                | Change                                                                                                                                                                                           |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `services/inference/app/config.py`                  | per-task model folders (`MORTALITY_MODEL_DIR`, `PNEUMONIA_MODEL_DIR` holding checkpoint + `run.json`), `HF_HOME` / offline flag; θ check                                                         |
| `services/inference/app/models/` (new `loader.py`)  | build from `run.json` argv, sha256 check, strict load, `best_head`, warm-up; one model per task                                                                                                  |
| `services/inference/app/preprocessing/` (new)       | `ehr.py` (medpatch Discretizer/Normalizer, the run's normalizer, `pad_zeros`), `cxr.py` (medpatch test transform, zero image when absent), `notes.py` (RR/DN assembly, DN refused for mortality) |
| `services/inference/app/main.py`                    | lifespan loads both models when `MOCK_MODE=false`; `/v1/predict` real path; `TEXT_MODEL` and `version` from the loaded model                                                                     |
| `services/inference/app/text.py`                    | token counts from the loaded tokenizer when available (the estimate stays for mock mode)                                                                                                         |
| `services/inference/app/schemas.py`                 | model version / run id in `ModelInfo`; chosen head in the result                                                                                                                                 |
| `services/inference/Dockerfile`, `requirements.txt` | `INSTALL_ML=true` build with CPU torch; HF caches baked or mounted                                                                                                                               |
| `services/inference/tests/`                         | parity test: service output == `MSMA_Trainer` forward on the same synthetic stay; sha mismatch refused; DN for mortality refused; missing modalities                                             |
| infra (Railway)                                     | volume or download step for the two model folders (~0.6 GB each); memory limit                                                                                                                   |
| `CLAUDE.md`                                         | Model status once real inference is live                                                                                                                                                         |

## Open questions

1. **Where the checkpoints live for Railway:** a volume, a download at
   startup (from where?), or baked into the image? They are ~0.55 GB each and
   must stay out of git (CLAUDE.md rule 2).
2. **Railway resources:** the plan's RAM and vCPUs. Is 1–6 s per request on
   CPU acceptable for the website?
3. **Pretrained downloads:** bake the BioBERT and timm caches (simplest), or
   add a medpatch flag to build the encoders without pretrained weights?
4. **Notes:**
   - Several radiology reports per stay: training joined them with a space,
     in the merged table's row order. What order does the website send?
   - Do `progress` / `nursing` notes get dropped, or should they be refused?
5. **EHR window:** mortality uses the first 48 h. Does the website send the
   whole stay for pneumonia (phenotyping saw the whole stay)?
6. **Memory:** share the frozen readers between the two task models (after a
   bit-equality check), or keep them separate for simplicity?
7. **Display:** show only `best_head`'s probability, or the other heads as
   well (as the reference table does)?
8. **Explainability** (CXR heatmap, note spans, EHR confidence band): wire
   the real confidence maps now or in Phase 7?
9. **Missing modalities:** the main run keeps phantom tokens (empty notes and
   blank X-rays still pass the readers), and inference must do the same. Is
   that the behaviour we want to show clinicians?
10. **Calibration:** do we report raw sigmoid(logit) or add a post-hoc
    calibration on validation? Round 3 outputs are uncalibrated.
