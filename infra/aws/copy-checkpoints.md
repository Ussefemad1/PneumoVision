# Copying the Round 3 checkpoints to the server

The fusion models are **never** in git or in a Docker image. They are copied
once to the server's disk and mounted read-only into the inference container
(`./checkpoints:/app/checkpoints:ro`). `deploy.sh` refuses to start if a file is
missing, or if its sha256 differs from the `manifest_row.sha256` in its
`run.json`.

## The four files

Only copy these four files. The shared Drive folders also hold `r3_*.csv`
result tables, `test_all_heads_*.csv` and `*.log` files. **Do not copy those**:
the deploy's data-safety guard refuses any CSV or log under `checkpoints/`.

| Server path (under `~/PneumoVision/`)                                                                        | Size                    | sha256                                                             | From Drive                                                    |
| ------------------------------------------------------------------------------------------------------------ | ----------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------- |
| `checkpoints/round3/mortality/run.json`                                                                      | 10,788 B                | (not checked)                                                      | `PneumoVision_shared/round3/mortality/r3-fusion-001/run.json` |
| `checkpoints/round3/mortality/best_checkpoint_0.001_in-hospital-mortality_c-msma_EHR-CXR-RR_partial.pth.tar` | 564,713,990 B (539 MiB) | `a241114adabb6e78f29f4c1f276dececbb86d91384a5c631aee9011f5c4e3b11` | `…/r3-fusion-001/in-hospital-mortality/c-msma/`               |
| `checkpoints/round3/phenotyping/run.json`                                                                    | 13,691 B                | (not checked)                                                      | `PneumoVision_shared/round3/phenotyping/run.json`             |
| `checkpoints/round3/phenotyping/best_checkpoint_0.001_phenotyping_c-msma_EHR-CXR-RR-DN_partial.pth.tar`      | 610,045,154 B (582 MiB) | `294ad302c8eee40035fee2c33872a6470a0174433ff07c27a74c62b6b3665838` | `PneumoVision_shared/round3/phenotyping/`                     |

- **Mortality: take the right file.** The mortality checkpoint has the same
  name in two folders:
  - `r3-fusion-001/in-hospital-mortality/c-msma/`: the **main** model (`r3-fusion-001`). Copy this one.
  - `r3-fusion-001/without_missingness_module/in-hospital-mortality/c-msma/`:
    an ablation, 564,709,078 B. Do not copy it. If you copy it by mistake, its
    sha256 will not match and the deploy stops.
- **Run ids.** The ids come from each `run.json` (`id` and `manifest_row.id`):
  `r3-fusion-001` (mortality) and `r3-fusion-002` (phenotyping).
- **Sizes.** Sizes are from Drive. The sha256 is the one recorded in each
  `run.json` manifest row. It is the only check that counts.

## 1. Download from Drive and check on your PC (PowerShell)

Download the four files into one folder per task:

```
%USERPROFILE%\Downloads\round3\mortality\
%USERPROFILE%\Downloads\round3\phenotyping\
```

Then check the two hashes:

```powershell
Get-FileHash -Algorithm SHA256 "$HOME\Downloads\round3\mortality\*.pth.tar"   | Format-List Hash, Path
Get-FileHash -Algorithm SHA256 "$HOME\Downloads\round3\phenotyping\*.pth.tar" | Format-List Hash, Path
```

Compare each `Hash` with the table above. PowerShell prints them in upper case,
which is the same value.

## 2. Copy them to the server

`$Key` and `$Ip` are as in [README.md](README.md). Run `bootstrap.sh` first: it
creates the folders.

```powershell
scp -i $Key "$HOME\Downloads\round3\mortality\run.json" "$HOME\Downloads\round3\mortality\best_checkpoint_0.001_in-hospital-mortality_c-msma_EHR-CXR-RR_partial.pth.tar" "ubuntu@${Ip}:~/PneumoVision/checkpoints/round3/mortality/"
scp -i $Key "$HOME\Downloads\round3\phenotyping\run.json" "$HOME\Downloads\round3\phenotyping\best_checkpoint_0.001_phenotyping_c-msma_EHR-CXR-RR-DN_partial.pth.tar" "ubuntu@${Ip}:~/PneumoVision/checkpoints/round3/phenotyping/"
```

The upload is about 1.1 GiB. Inbound traffic to EC2 is free.

## 3. Check on the server

```powershell
ssh -i $Key ubuntu@$Ip "cd ~/PneumoVision && sha256sum checkpoints/round3/*/*.pth.tar && ls -la checkpoints/round3/*"
```

`deploy.sh` repeats this check against `run.json` on every deploy.

## Standalone readers for "By source" (reader_heads.pt)

The report's "By source" panel shows each modality's **Round 1 standalone
reader** score. The fusion models already contain those readers' encoders, so
the server only needs each reader's small classifier head: two files, a few
MB, made once on your laptop. You do **not** copy the seven ~0.5 GB Round 1
checkpoints to the server. Without these two files the site still works and
"By source" says "Standalone readers not installed".

### 1. The seven Round 1 files, on your laptop

They are the files `INFERENCE_MODE=readers` uses (services/inference/READERS.md),
under the repo's `checkpoints/` folder with these names. The Round 3 runs
descend from the manifest rows on the right (the parents of their Round 2b
parents); copy each Drive file to its local name if you don't have it yet.

| Local path (under `checkpoints/`)                                                                                                               | Manifest row | Drive file (`PneumoVision_shared/checkpoints/…` or Farida's `MyDrive/pneumovision/r1/…`)                                                                            | sha256 (manifest) |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| `phenotyping/r1/phenotyping/unimodal_ehr/best_checkpoint_0.001_phenotyping_unimodal_ehr_EHR_paired.pth.tar`                                     | r1-ehr-001   | `ehr-phenotyping/best_checkpoint_0.001_phenotyping_unimodal_ehr_EHR_paired.pth.tar`                                                                                 | `1dc8b962f3e7…`   |
| `phenotyping/r1/phenotyping/unimodal_cxr/best_checkpoint_3e-05_phenotyping_unimodal_cxr_EHR-CXR_paired.pth.tar`                                 | r1-cxr-001   | `pheno-cxr/phenotyping/unimodal_cxr/best_checkpoint_3e-05_phenotyping_unimodal_cxr_EHR-CXR_paired.pth.tar`                                                          | `065ecd77330b…`   |
| `phenotyping/r1/phenotyping/unimodal_rr/best_checkpoint_0.00012225031390095305_phenotyping_unimodal_rr_EHR-RR_paired.pth.tar`                   | r1-rr-001    | `MyDrive/pneumovision/r1/phenotyping/best_checkpoint_phenotyping_unimodal_rr.pth.tar`                                                                               | `ad4037e57564…`   |
| `phenotyping/r1/phenotyping/unimodal_dn/best_checkpoint_0.00012225031390095305_phenotyping_unimodal_dn_EHR-DN_paired.pth.tar`                   | r1-dn-001    | `MyDrive/pneumovision/r1/phenotyping/best_checkpoint_phenotyping_unimodal_dn.pth.tar`                                                                               | `9bd950e3cb71…`   |
| `mortality/r1/in-hospital-mortality/unimodal_ehr/best_checkpoint_0.001_in-hospital-mortality_unimodal_ehr_EHR_paired.pth.tar`                   | r1-ehr-002   | `mortality-ehr/best_checkpoint_0.001_in-hospital-mortality_unimodal_ehr_EHR_paired.pth.tar`                                                                         | `1e9cb87cc1f2…`   |
| `mortality/r1/in-hospital-mortality/unimodal_cxr/best_checkpoint_3e-05_in-hospital-mortality_unimodal_cxr_EHR-CXR_paired.pth.tar`               | r1-cxr-002   | `mortality-cxr/BASELINE_mortality_vitS_lr3e-05_ep9.pth.tar`                                                                                                         | `4b272fcba450…`   |
| `mortality/r1/in-hospital-mortality/unimodal_rr/best_checkpoint_0.00012225031390095305_in-hospital-mortality_unimodal_rr_EHR-RR_paired.pth.tar` | r1-rr-002    | `mortality-notes(RR)/mortality/r1/in-hospital-mortality/unimodal_rr/best_checkpoint_0.00012225031390095305_in-hospital-mortality_unimodal_rr_EHR-RR_paired.pth.tar` | `1092a832942d…`   |

### 2. Extract the heads (laptop, repo `.venv`, CPU, about a minute)

```powershell
cd $HOME\PneumoVision
.\.venv\Scripts\python.exe -m tools.pv.extract_reader_heads --checkpoint-dir checkpoints --out checkpoints\round3
```

It lists the seven files (`ok` / `MISSING`), writes
`checkpoints\round3\reader_heads.pt` and `reader_heads.json`, and prints each
source's sha256 next to its manifest row (`= manifest r1-…`). A `WARNING` line
means the local file is not the one the Round 3 run descends from: the service
then finds that the encoders differ and reports that reader as unavailable
instead of guessing.

### 3. Copy them to the server

```powershell
scp -i $Key "$HOME\PneumoVision\checkpoints\round3\reader_heads.pt" "$HOME\PneumoVision\checkpoints\round3\reader_heads.json" "ubuntu@${Ip}:~/PneumoVision/checkpoints/round3/"
ssh -i $Key ubuntu@$Ip "cd ~/PneumoVision && git pull --ff-only && infra/aws/deploy.sh --no-pull"
```

`deploy.sh` checks `reader_heads.pt` against the sha256 in `reader_heads.json`;
at start-up the service checks it again, then checks that every fusion
encoder equals the Round 1 reader's bit for bit. `/v1/model-info` says, per
task and modality, how each "By source" score is computed.

## Readers mode (optional)

`INFERENCE_MODE=readers` serves the Round 1 + 2b single-modality readers
instead of the fusion. It needs the 14 files listed in
`services/inference/READERS.md` under `checkpoints/{mortality,phenotyping}/...`
(several GB). `deploy.sh` lists any that are missing. Fusion is the default and
needs only the four files above.
