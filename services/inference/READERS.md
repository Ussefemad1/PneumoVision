# Readers mode — real single-modality models, simulated fusion

`INFERENCE_MODE=readers` runs the team's trained Round 1 readers. Each one
scores its own modality; the MedPatch fusion (Rounds 2, 2b, 3) is not trained
yet, so the fused output stays simulated, centred on the readers' scores. Every
number in a result says which it is (`result.sources`), and the report labels
them "Trained model" or "Simulated".

| Reader | Task output | Test AUROC | Trained by |
| --- | --- | --- | --- |
| EHR LSTM (`unimodal_ehr`) | mortality | 0.861 | Caroll |
| EHR LSTM (`unimodal_ehr`) | pneumonia (phenotype 21) | 0.812 | Caroll |
| Radiology reports, BioBERT (`unimodal_rr`) | mortality | 0.730 | Farida |
| Radiology reports, BioBERT (`unimodal_rr`) | pneumonia (phenotype 21) | 0.859 | Farida |
| Discharge notes, BioBERT (`unimodal_dn`) | pneumonia (phenotype 21) | 0.897 | Farida |

No CXR reader yet (Norhan's is pending). There is no mortality DN reader:
discharge notes leak the outcome and are never sent for mortality.

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

## Caveats to state on the report or in the defense

- The phenotyping readers saw the whole ICU stay in training; the platform
  sends a 48-hour window, so the EHR pneumonia score is out of its training
  distribution.
- A discharge note usually states the diagnosis, which is why the DN reader is
  the strongest and why it is excluded from mortality.
- Fused score, evidence strength, modality contribution and the confidence
  heatmaps are simulated until the fusion checkpoints exist.
