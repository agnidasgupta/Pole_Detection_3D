# VEGASS Nebius training and reporting package

## Local Mac location

Copy/extract this package into:

`/Users/agni/Documents/VEG_Data/DUKE_FLORIDA_150/vegass-main`

Expected structure:

```text
vegass-main/
  Dockerfile.vegass
  scripts/
    train.py
    check_coordinates.py
    check_coordinates_nebius.sh
    build_vegass_image.sh
    smoke_test_nebius.sh
    run_train_nebius.sh
    upload_to_nebius.sh
    download_training_outputs.sh
    run_remote_training_and_download.sh
  vegass/
    __init__.py
    config.py
    dataset.py
    loss.py
    metrics.py
    model.py
    reporting.py
    trainer.py
```

## Generated training artifacts

Each run writes to its own directory under `/outputs`, for example
`/outputs/vegass_full`. The directory includes:

- `training.log`
- `run_config.json`
- `environment.txt`
- `run_summary.json`
- `metrics_phaseN.csv`
- `loss_history_phaseN.png`
- `loss_history_all_phases.png`
- `evaluation_phaseN.txt`
- `evaluation_phaseN.json`
- `dataset_splits_phaseN.txt`
- model checkpoints (`best_phaseN.pth`, `last_phaseN.pth`)

The download script excludes model/checkpoint files.

## Upload from Mac

```bash
cd /Users/agni/Documents/VEG_Data/DUKE_FLORIDA_150/vegass-main
chmod +x scripts/*.sh
bash scripts/upload_to_nebius.sh <VM_PUBLIC_IP> ~/.ssh/nebius_gpu_ed25519
```

## Build the reporting image once on Nebius

The derived image adds matplotlib without changing the working base image:

```bash
bash /workspace/scripts/build_vegass_image.sh
```

This creates:

`cv3d-spconv-vegass:stable`

## Validate coordinates

```bash
bash /workspace/scripts/check_coordinates_nebius.sh
```

## One-epoch smoke test

```bash
bash /workspace/scripts/smoke_test_nebius.sh
```

## Full curriculum

```bash
PHASES="1 2 3" EPOCHS=30 BASE_CH=16 BATCH_SIZE=1 \
VAL_FRAC=0.15 TEST_FRAC=0.15 \
OUT=/outputs/vegass_full \
bash /workspace/scripts/run_train_nebius.sh
```

## Download reports after a completed run

Run on the Mac:

```bash
bash scripts/download_training_outputs.sh \
  <VM_PUBLIC_IP> \
  /outputs/vegass_full \
  ~/.ssh/nebius_gpu_ed25519
```

Downloaded artifacts go to:

`training_outputs/vegass_full/`

## Start remote training and automatically download reports

Run on the Mac:

```bash
PHASES="1 2 3" EPOCHS=30 BASE_CH=16 BATCH_SIZE=1 \
bash scripts/run_remote_training_and_download.sh \
  <VM_PUBLIC_IP> \
  /outputs/vegass_full \
  ~/.ssh/nebius_gpu_ed25519
```

This waits for the remote training process to finish and then downloads every
artifact except model/checkpoint files.
