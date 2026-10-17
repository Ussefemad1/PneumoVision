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

## Readers mode (optional)

`INFERENCE_MODE=readers` serves the Round 1 + 2b single-modality readers
instead of the fusion. It needs the 14 files listed in
`services/inference/READERS.md` under `checkpoints/{mortality,phenotyping}/...`
(several GB). `deploy.sh` lists any that are missing. Fusion is the default and
needs only the four files above.
