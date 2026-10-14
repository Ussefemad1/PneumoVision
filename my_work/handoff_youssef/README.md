# Text readers (Round 1) from Farida, for Rounds 2 / 2b / 3

All checkpoints are in the repo's own format. Each one includes the frozen BioBERT, so it is ~435 MB.
Take the paths from the team checkpoint Sheet, not from memory.

## Checkpoints

| Model | Task | File | Test AUROC | Test AUPRC | Paper |
|---|---|---|---|---|---|
| RR reader | in-hospital mortality (reports from the first 48 h) | `best_checkpoint_0.00012225031390095305_in-hospital-mortality_unimodal_rr_EHR-RR_paired.pth.tar` | 0.729 (0.709–0.749); 5 seeds 0.730 (0.728–0.732) | 0.275; 5 seeds 0.273 | 0.742 / 0.287 |
| RR reader | phenotyping (25 labels) | `best_checkpoint_0.00012225031390095305_phenotyping_unimodal_rr_EHR-RR_paired.pth.tar` | 0.778 (0.766–0.790); 5 seeds 0.778 (0.778–0.779) | 0.473; 5 seeds 0.474 | 0.776 / 0.474 |
| DN reader | phenotyping (25 labels) | `best_checkpoint_0.00012225031390095305_phenotyping_unimodal_dn_EHR-DN_paired.pth.tar` | 0.859 (1 seed) | 0.607 | 0.856 / 0.601 |

Pneumonia (phenotyping): RR 0.859 AUROC / 0.487 AUPRC. DN 0.897 / 0.557.

Both RR checkpoints were checked with the repo's own `fusion_main.py --mode eval`. It reproduced the numbers above exactly and reported no "Not Loaded" or "Not Found" keys.

The DN checkpoint was trained on Colab. A full repo eval was not possible on the laptop because it has no raw `discharge.csv`. Instead it was checked on synthetic texts: 0 missing keys, 0 extra keys, and the repo forward equals the trained head (max diff 2.6e-6). DN settings: `--modalities EHR-DN --fusion_type unimodal_dn --output_dim_dn 512 --load_dn <path>`.

## Settings needed to load them

Without these settings the checkpoint will not be found or will not load correctly.

```
--text_encoder biobert --bert_model_name dmis-lab/biobert-v1.1
--output_dim_rr 512 --classifier mlp --loss bce        (mean pooling, the default --use_cls_token)
--num_classes 1  --task in-hospital-mortality --labels_set mortality
--num_classes 25 --task phenotyping           --labels_set pheno
--lr 0.00012225031390095305      (the lr is part of the checkpoint file name)
--load_rr <path from the Sheet>  (for Round 2)
```

- Run from inside `medpatch/`, because `fusion_main.py` opens a config file by a relative path.
- The repo's default text model is `emilyalsentzer/Bio_ClinicalBERT`. That is not the paper's BioBERT. With it, phenotyping RR plateaued at about 0.71.

## How they were trained

- BioBERT is frozen. Notes are split into non-overlapping 512-token chunks, mean-pooled, then passed through Linear(768→512), LayerNorm and Linear(512→C).
- The paper's protocol was followed:
  - 10 learning rates sampled log-uniformly in [1e-5, 1e-3]
  - batch size 16, up to 100 epochs, early stopping with patience 15
  - Adam with the repo's betas (0.9, 0.9)
  - model selection on validation AUROC; the test set was used once
  - the chosen learning rate was 1.2225e-4 for all three readers
- For speed, the BioBERT outputs were computed once and cached. This is numerically identical to the repo's forward pass (max difference 1.2e-7). One difference: the repo leaves BERT dropout on during training, and the cached run does not.

## Data

- Caroll's MIMIC-IV extraction. Her data is v3.x, so only subjects in the repo's split list (`testset_iv.csv`) are kept. The split files were not changed.
- Phenotyping EHR stays: 42,328 / 4,756 / 11,845. Mortality: 19,064 / 2,161 / 5,302.
- Stays with RR: phenotyping 39,028 / 4,356 / 10,944; mortality 16,796 / 1,904 / 4,650.
- RR text uses every radiology report for the stay, as the repo code does (not chest-only).
- Mortality uses only reports with charttime in [ICU intime, intime + 48 h]. Discharge notes are never used for mortality.

## Please check

1. With `--data_pairs partial`, stays that have no radiology report should be treated as "no RR" and not crash.
2. Rounds 2 and 2b are yours (handbook). I started a fast Round 2 run for mortality RR but stopped it partway. There are no Round 2/2b files from me.
   - Repo quirk: in `Calibration.compute_ece`, with `--num_classes 1` the ECE is measured at token position 0 only.
3. The repo's phenotyping `MedPatch/Confidence-Patching.sh` uses `c-e-msma`, which is entropy, not the paper's max(p, 1−p).
   - Entropy can never reach θ = 0.75, so no token would ever count as "high confidence".
   - Use `c-msma`.

## Answers to Youssef's questions (2026-10-05)

### 1. Which BERT was used? `dmis-lab/biobert-v1.1` for all three readers.
- Each converted checkpoint stores the model name in its `bert_model_name` field. All three say `dmis-lab/biobert-v1.1`.
- `args/train_config_*.json` are the real training records. `cache_meta.model` is `dmis-lab/biobert-v1.1` for phenotyping RR, phenotyping DN and mortality RR.
- `args/args_*_repo_eval.txt` come from the repo's `fusion_main.py --mode eval` runs that verified the converted RR checkpoints. These are evaluation runs, not training runs.
  - Training used the fast path (cached BioBERT, then `train_head.py`). Its `config.json` takes the place of `args.txt`.
  - There is no repo `args.txt` for DN, because DN was never run through `fusion_main.py`. Its training config is `args/train_config_phenotyping_dn.json`.

### 2. Is `data/ehr_full` the same as the shared `data/ehr/in-hospital-mortality`?
`ehr_full` holds Caroll's mortality listfiles **restricted to subjects in the repo's split list**. The timeseries files are Caroll's own, unchanged.

| Split | Rows in ehr_full | md5 | Rows in Caroll's original | md5 |
|---|---|---|---|---|
| train | 19,064 | 8705392cda55d8b29f71bf6c29ffaac4 | 29,171 | 347c30866046f35d434d983e304c0016 |
| val | 2,161 | da8417a7b0a2e202517c60b58d322599 | 2,161 | da8417a7b0a2e202517c60b58d322599 (identical) |
| test | 5,302 | deb7786c0ba5a8f50def306902f0a910 | 5,302 | deb7786c0ba5a8f50def306902f0a910 (identical) |

- Val and test are byte-identical to Caroll's files, so evaluation is comparable.
- Only train differs. 10,107 train rows belong to subjects that exist only in the newer MIMIC-IV (v3.x) and are missing from `testset_iv.csv`. The split script put all of them into train by default. Removing them brings the counts close to the paper: 19,064 / 2,161 / 5,302 vs 18,845 / 2,138 / 5,243.
- To check the Drive copy, compare its md5s with the table:
  - If train is 29,171 rows, it is Caroll's unfiltered version.
  - To build the same filtered version, run `prepare_data.py` (in this folder):
    ```
    python prepare_data.py --task in-hospital-mortality --all_stays <all_stays.csv> --listfile_dir <folder with the 3 listfiles> --timeseries_dir <folder with train/ and test/> --repo <medpatch> --out <new ehr dir>
    ```
    It keeps rows whose subject_id (the part of `stay` before the first "_") is in `medpatch/mimic4extract/mimic3benchmark/resources/testset_iv.csv`.
  - Phenotyping used the same filter: 42,328 / 4,756 / 11,845.
- Recommendation: use the same filtered listfiles for Rounds 2, 2b and 3, so every stage trains on the same patients as Round 1.


## Answers to Youssef's questions (2026-10-06)

### 1. Phenotyping RR and DN: shared listfiles as they are, or filtered?
Filtered, with the same rule as mortality: keep only subjects in `testset_iv.csv`. Only the train list changes. Val and test are identical to Caroll's / the shared files.

| Split | Rows used (filtered) | md5 | Shared / Caroll original | md5 |
|---|---|---|---|---|
| train | 42,328 | 4c51ea81f6a797808f03b2d23f405ff1 | 60,909 | 058832fc3a886947fc0886cd78768f30 |
| val | 4,756 | 6da775f884344c372862585e877c47f8 | 4,756 | identical |
| test | 11,845 | 3337b4b965fe00d9e7d4c8b0e4c1359a | 11,845 | identical |

- RR was trained on the laptop with these filtered lists. Stays that have RR: 39,028 / 4,356 / 10,944.
- DN was trained on Colab. The notebook read the shared `data/ehr/phenotyping` listfiles and applied the same filter (`prepare_data.py`). Stays that have DN: 40,464 / 4,529 / 11,333.
  - This confirms the filter was applied: an unfiltered train list would give far more than 42,328 rows.

### 2. Are the files in `checkpoints/pheno-notes` and `checkpoints/mortality-notes` the final ones?
Yes. Those are the final converted checkpoints. This handoff folder contains no checkpoints, only README, args, results and `prepare_data.py`.

Check your copies against these fingerprints:

| File | Bytes | md5 |
|---|---|---|
| `best_checkpoint_0.00012225031390095305_phenotyping_unimodal_rr_EHR-RR_paired.pth.tar` | 435,034,801 | da727f940c1d9abebde004fb18fbc9ea |
| `best_checkpoint_0.00012225031390095305_phenotyping_unimodal_dn_EHR-DN_paired.pth.tar` | 435,034,801 | cf199e9484af7c8bb478558a87b2a151 |
| `best_checkpoint_0.00012225031390095305_in-hospital-mortality_unimodal_rr_EHR-RR_paired.pth.tar` | 434,987,739 | d5bdd46e5d9e5a1c211fa889ebe60ac5 |
