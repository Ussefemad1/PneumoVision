# Readers mode — real single-modality models, simulated fusion

> **The deployed mode is now `fusion`** (the trained Round 3 MedPatch
> models; see "Fusion mode" at the end). Everything below describes
> `INFERENCE_MODE=readers`, which still works. In particular, "combined = plain
> average" is **readers mode only**: in fusion mode the headline number is the
> trained fusion's chosen head.

`INFERENCE_MODE=readers` runs the team's trained Round 1 readers. Each one
scores its own modality; the MedPatch fusion (Rounds 2, 2b, 3) is not trained
yet, so the fused output stays simulated, centred on the readers' scores. Every
number in a result says which it is (`result.sources`), and the report labels
them "Trained model" or "Simulated".

| Reader                                     | Task output              | Test AUROC                           | Trained by |
| ------------------------------------------ | ------------------------ | ------------------------------------ | ---------- |
| EHR LSTM (`unimodal_ehr`)                  | mortality                | 0.861                                | Caroll     |
| EHR LSTM (`unimodal_ehr`)                  | pneumonia (phenotype 21) | 0.812                                | Caroll     |
| Radiology reports, BioBERT (`unimodal_rr`) | mortality                | 0.730                                | Farida     |
| Radiology reports, BioBERT (`unimodal_rr`) | pneumonia (phenotype 21) | 0.859                                | Farida     |
| Discharge notes, BioBERT (`unimodal_dn`)   | pneumonia (phenotype 21) | 0.897                                | Farida     |
| Chest X-ray, ViT-Small (`unimodal_cxr`)    | mortality                | not reported (val 0.723)             | Norhan     |
| Chest X-ray, ViT-Small (`unimodal_cxr`)    | pneumonia (phenotype 21) | not reported (val 0.702, mean of 25) | Norhan     |

For the CXR readers the lr 3e-5 checkpoints were chosen on validation AUROC
(mortality 0.723 vs 0.664 at 1e-4 and 0.558 at 1e-3). No test log came with
them; add the test AUROCs to `READERS` in `app/readers.py` when Norhan reports
them. There is no mortality DN reader: discharge notes leak the outcome and are
never sent for mortality.

The combined score is the plain **average** of the trained readers' scores
(`sources.fusion = "average"`), shown with its formula on the report. It is not
the MedPatch fusion.

## Confidence heads (Youssef, Rounds 2 and 2b)

Each reader also loads its **Round 2 confidence head** (a per-token confidence
predictor on the same frozen reader) and, where the file exists, its **Round
2b temperatures** (calibration). They give the real token confidence
gamma = max(p, 1 - p) for this task's output, used for the report's
"above threshold" shares and the stay page's per-hour EHR band.

| Reader                    | Confidence file under `checkpoints/`                      |            |
| ------------------------- | --------------------------------------------------------- | ---------- |
| EHR mortality / pneumonia | `{mortality,phenotyping}/r2b/temp_c-unimodal_ehr.pth.tar` | calibrated |
| CXR mortality / pneumonia | `{mortality,phenotyping}/r2b/temp_c-unimodal_cxr.pth.tar` | calibrated |
| RR mortality / pneumonia  | `{mortality,phenotyping}/r2b/temp_c-unimodal_rr.pth.tar`  | calibrated |
| DN pneumonia              | `phenotyping/r2b/temp_c-unimodal_dn.pth.tar`              | calibrated |

All seven Round 2 files carry exactly the Round 1 reader weights (checked
tensor by tensor); start-up re-checks each head's classifier against its
reader and refuses a mismatch. The CXR head reads one CLS token per image
(`cxr_token_confidence` off, as trained), so there is no patch heatmap.
The high/low joint predictions, the missingness branch and the fusion
weights need Round 3 and stay simulated; the combined score stays a plain
average.

**CXR pooling:** the CXR readers run with `--use_cls_token cls`, as every
released CXR script and Youssef's args do. (Before 2026-10-16 the site used
the argparse default `mean`, which gave wrong CXR scores from the same weights.)

## Run it locally

1. The repo `.venv` (Python 3.11) with the training stack and the service:
   `pip install -r requirements.txt -r services/inference/requirements.txt`
2. Checkpoints under `checkpoints/` (gitignored; never commit them):

   ```
   mortality/r1/in-hospital-mortality/unimodal_ehr/best_checkpoint_0.001_in-hospital-mortality_unimodal_ehr_EHR_paired.pth.tar
   phenotyping/r1/phenotyping/unimodal_ehr/best_checkpoint_0.001_phenotyping_unimodal_ehr_EHR_paired.pth.tar
   mortality/r1/in-hospital-mortality/unimodal_rr/best_checkpoint_0.00012225031390095305_in-hospital-mortality_unimodal_rr_EHR-RR_paired.pth.tar
   phenotyping/r1/phenotyping/unimodal_rr/best_checkpoint_0.00012225031390095305_phenotyping_unimodal_rr_EHR-RR_paired.pth.tar
   phenotyping/r1/phenotyping/unimodal_dn/best_checkpoint_0.00012225031390095305_phenotyping_unimodal_dn_EHR-DN_paired.pth.tar
   mortality/r1/in-hospital-mortality/unimodal_cxr/best_checkpoint_3e-05_in-hospital-mortality_unimodal_cxr_EHR-CXR_paired.pth.tar
   phenotyping/r1/phenotyping/unimodal_cxr/best_checkpoint_3e-05_phenotyping_unimodal_cxr_EHR-CXR_paired.pth.tar
   ```

   A checkpoint that arrives as an unzipped folder (`data.pkl`, `data/`,
   `version`) can be turned back into a file with
   `python services/inference/tools/repack_checkpoint.py <folder> <out.pth.tar>`.

3. In `.env.demo`: `INFERENCE_MODE=readers`, then `npm run dev:demo`.
   BioBERT (`dmis-lab/biobert-v1.1`) downloads from Hugging Face on first use.

The service refuses to start if any checkpoint is missing, instead of quietly
simulating that modality. Startup takes about 10 s; a prediction about 50 ms
on a GPU.

## How inputs are built (as in training)

- Models are built with medpatch's own `Fusion` / `EHR_encoder` /
  `Text_encoder` from the arguments each reader was trained with, and loaded
  with `strict=True` (verified: 0 missing / 0 extra keys for all five).
- The three text readers share one frozen BioBERT (their copies are checked
  to be identical at load), so memory is about 0.5 GB rather than 1.3 GB.
- EHR: hourly grid → medpatch `Discretizer` (timestep 1.0, `previous`
  imputation, start time zero) → the normalizer the reader was trained with
  (`ihm_ts1.0…` for mortality, `ph_ts1.0…` for phenotyping — Caroll's runs
  printed which they loaded) → `[1, 48, 76]`.
- RR: radiology-type notes joined with a space. Progress and nursing notes are
  not fed to the RR reader, which was trained on radiology reports only.
- DN: discharge notes joined with a space, pneumonia only.
- CXR: the two steps of `medpatch/resize.py` (width 512, aspect kept, re-saved
  as JPEG; it runs as a script, so they are repeated rather than imported),
  `.convert('RGB')`, then medpatch's own `get_transforms(args)[1]`
  (384x384, ImageNet normalisation) into the ViT-Small built by `CXR_encoder`.

## Caveats to state on the report or in the defense

- The phenotyping readers saw the whole ICU stay in training; the platform
  sends a 48-hour window, so the EHR pneumonia score is out of its training
  distribution.
- A discharge note usually states the diagnosis, which is why the DN reader is
  the strongest and why it is excluded from mortality.
- Evidence strength, modality contribution and the confidence heatmaps are
  simulated until the fusion checkpoints exist; the combined score is a plain
  average, not MedPatch.
- The CXR readers were trained on MIMIC-CXR AP radiographs. Photos, scans of
  printed films or other views are out of distribution.

## Fusion mode (`INFERENCE_MODE=fusion`, Round 3)

The trained MedPatch fusion (`CMSMAFusion`, `--fusion_type c-msma`), one model
per task, from `FUSION_DIR/{mortality,phenotyping}/` (checkpoint + `run.json`,
`infra/aws/copy-checkpoints.md`). Code: `app/fusion.py`.

| Task      | Run             | Modalities    | Output       | Head   | Test AUROC |
| --------- | --------------- | ------------- | ------------ | ------ | ---------- |
| mortality | `r3-fusion-001` | EHR-CXR-RR    | class 0      | `late` | 0.865      |
| pneumonia | `r3-fusion-002` | EHR-CXR-RR-DN | phenotype 21 | `late` | 0.900      |

- **Built as trained.** Args come from the run's own `argv` (medpatch's parser,
  minus `--load_*`, `--resume`, `--save_dir`); encoders as `MSMA_Trainer`;
  strict `load_state_dict`.
- **Refused at load** (`/v1/health` 503 "degraded", no predictions): sha256 ≠
  `run.json` manifest row; BERT in `run.json` ≠ argv ≠ loaded tokenizer;
  any `--*_confidence_threshold` ≠ `DEFAULT_THETA` (0.75); not `c-msma`; DN in
  a mortality run.
- **Inputs.** EHR through the Discretizer and the run's normalizer (its
  `--normalizer_state`, else fusion_main's phenotyping default — both runs used
  the default), `pad_zeros` (phenotyping: 2,646 rows); mortality reads the
  first 48 h, pneumonia whatever is sent (trained on the whole stay — the
  report says so). CXR through the dataset's test transform; missing = an
  all-zero `[3, 384, 384]` tensor. Radiology reports joined with a space in the
  order sent (the API sends them chronologically); discharge notes for
  pneumonia only — the service answers **400** to a discharge note for
  mortality. Progress/nursing notes are read by no model; the report says so.
- **Outputs.** `high_conf`, `low`, `late` logits, plus `combined` = mean of the
  three and `less_combined` = mean of high and late (as `MSMA_Trainer`). The
  headline is the checkpoint's `best_head` (`late`), as a **raw sigmoid —
  uncalibrated** (Round 2b calibrated the confidence heads, not the fusion).
  The other heads, run id, sha256, BERT and θ are under "Model details". The
  late-fusion α ("Contribution") are the model's own. The token-confidence
  _maps_ drawn on the report stay simulated and are labelled so.
- **"By source" = the Round 1 standalone readers.** CMSMAFusion's own
  `{ehr,cxr,rr,dn}_classifier` outputs are late-fusion _votes_: `MSMA_Trainer`
  puts loss only on `high`, `low` and `late`, so the votes are trained only
  through the weighted late sum, their scale is unconstrained and they
  saturate (0 % / 100 %). They are **not** standalone probabilities. "By source"
  instead shows each Round 1 reader's own prediction, computed as readers mode
  computes it:
  - CXR, RR, DN: the reader's own classifier head (from `reader_heads.pt`,
    `tools/pv/extract_reader_heads.py`) applied to the fusion's pooled feature,
    i.e. the input of the fusion's `{m}_classifier` (CLS token for the ViT,
    mean of the projected BERT tokens for the notes). That is exactly the
    reader's feature because the encoders are the reader's (F2).
  - EHR: the shared LSTM re-run on the reader's own input (48 h, and the
    normalizer the reader was trained with: `ihm_…` for mortality, while the
    fusion used the phenotyping default), then the reader's head.
  - At start-up `reader_heads.pt` must match the sha256 in
    `reader_heads.json` (else `/v1/health` is degraded), and every fusion
    encoder tensor must equal the Round 1 reader's bit for bit (digests
    recorded by the extract tool). A reader whose encoder differs runs on its
    own full path if its Round 1 checkpoint is under `CHECKPOINT_DIR`, else it
    is reported unavailable (`/v1/model-info` → `standaloneDetail`). Without
    `reader_heads.pt` fusion still serves; "By source" says "Standalone
    readers not installed".
  - Parity: `tests/test_fusion.py` checks these values equal readers mode's
    within 1e-5 on fake runs.
- **Model details → "Inside the MedPatch fusion".** The per-modality votes,
  the joint high/low heads and the missingness head, each as raw logit and
  sigmoid, labelled "not probabilities". The missingness head sees only which
  inputs are present.
- **Speed.** CMSMAFusion runs each frozen reader twice per forward (its
  `*_fixed` copy is the same module); the service answers the second, identical
  call from the first, halving CPU time without changing a number.
