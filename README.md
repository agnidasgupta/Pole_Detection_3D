# Voxel Pole/Powerline Pipeline v3 — corrected Stages 4–6

This package fixes the failure after the successful MAE and initial segmentation stages.
It is optimized for the Nebius VM described by the user:

- 1 × NVIDIA H100 SXM GPU
- 16 vCPUs
- 200 GiB RAM
- nested session-grouped voxel CSV data

The existing successful outputs are reused:

```text
/outputs/poleline_voxel_run_session_groups/dataset
/outputs/poleline_voxel_run_session_groups/mae
/outputs/poleline_voxel_run_session_groups/seg_initial/seg_best.pt
```

## What changed

### Stage 4

- Uses only `dataset/manifests/train.json`; validation and test data are never mined.
- Does not create thousands of full prediction CSVs.
- Samples geometry-informed patches containing pole-like vertical background, high background clutter, random background and true-asset context.
- Runs batched BF16 inference on the H100.
- Evaluates only central patch cores to reduce boundary artifacts.
- Accumulates confusion matrices incrementally.
- Writes compact per-file `.csv.gz` shards.
- Saves progress after every file and resumes safely.

### Stage 5

- Applies compact shards directly to existing prepared NPZ files.
- Does not re-read 3,744 source CSV files.
- Rewrites only training NPZ files that receive hard negatives.
- Hard-links unchanged files to save time and disk.
- Preserves the exact original train/validation/test split.

### Stage 6

- Starts from the complete Stage-3 `seg_initial/seg_best.pt` model.
- Uses H100 BF16 automatic mixed precision and channels-last 3D tensors.
- Uses 10 data-loader workers by default on the 16-vCPU VM.
- Uses streaming evaluation to prevent RAM exhaustion.
- Uses moderate hard-negative weighting and recall-oriented F2 checkpoint selection.
- Produces loss curves, learning curves, confusion-matrix PNGs, metrics and comparisons.

## Upload from a local Mac

Download `voxel_poleline_pipeline_v3.zip`, then run:

```bash
cd ~/Downloads
unzip -o voxel_poleline_pipeline_v3.zip
cd voxel_poleline_pipeline_v3
chmod +x upload_to_nebius_from_mac.sh
SSH_KEY="$HOME/.ssh/nebius_cpu_ed25519" ./upload_to_nebius_from_mac.sh
```

The passphrase-protected private key is supported. Do not use the `.pub` file.

## Build the v3 image on Nebius

```bash
ssh -i "$HOME/.ssh/nebius_cpu_ed25519" -o IdentitiesOnly=yes agni@89.169.103.7
cd /workspace/voxel_poleline
unzip -o voxel_poleline_pipeline_v3.zip
cd voxel_poleline_pipeline_v3
docker build -t va-voxel-poleline:v3 .
```

## Run Stages 4–6 using existing Stage 1–3 results

From the Nebius host:

```bash
docker run --rm -it --gpus all \
  --mount type=bind,source=/workspace/voxel_poleline/outputs,target=/outputs \
  --workdir /workspace/voxel_poleline \
  va-voxel-poleline:v3 bash
```

Inside the container:

```bash
WORK_DIR=/outputs/poleline_voxel_run_session_groups \
PATCH=64 \
CORE=32 \
MINE_PATCHES_PER_FILE=16 \
MINE_BATCH=8 \
MINE_WORKERS=8 \
MINE_POLE_THRESHOLD=0.40 \
MINE_LINE_THRESHOLD=0.40 \
MINE_MIN_SCORE=0.25 \
MINE_MAX_PER_FILE=5000 \
EPOCHS_HARDNEG=30 \
BATCH_HARDNEG=6 \
SAMPLES_PER_EPOCH=8000 \
EVAL_SAMPLES=1536 \
NUM_WORKERS=10 \
HARDNEG_WEIGHT=4.0 \
HARDNEG_PROB=0.35 \
LR_HARDNEG=0.0002 \
./run_stages_4_6.sh
```

Stage 4 is resumable. If the connection or process stops, execute the same command again. Completed file shards are skipped.

### H100 memory fallback

If Stage 6 reports CUDA out-of-memory, use:

```bash
BATCH_HARDNEG=4 GRAD_ACCUM=2 ./run_stages_4_6.sh
```

If Stage 4 reports CUDA out-of-memory, use:

```bash
MINE_BATCH=4 ./run_stages_4_6.sh
```

## Smoke test before the full run

Run Stage 4 on 10 training files and Stage 6 for 2 epochs:

```bash
WORK_DIR=/outputs/poleline_voxel_run_session_groups \
MINE_DIR=/outputs/poleline_voxel_run_session_groups/hardneg_mining_smoke \
HARDNEG_DATASET=/outputs/poleline_voxel_run_session_groups/dataset_hardneg_smoke \
FINAL_DIR=/outputs/poleline_voxel_run_session_groups/seg_hardneg_smoke \
MINE_MAX_FILES=10 \
MINE_PATCHES_PER_FILE=8 \
EPOCHS_HARDNEG=2 \
SAMPLES_PER_EPOCH=200 \
EVAL_SAMPLES=100 \
BATCH_HARDNEG=2 \
NUM_WORKERS=4 \
SKIP_DIAGNOSTICS=1 \
./run_stages_4_6.sh
```

The full run should use the default Stage-4/5/6 directories, `MINE_MAX_FILES=0`, and `SKIP_DIAGNOSTICS=0`.

## Important outputs

```text
hardneg_mining/hardneg_mining.log
hardneg_mining/hardneg_mining_metrics.txt
hardneg_mining/hardneg_mining_metrics.json
hardneg_mining/hardneg_file_summary.csv

dataset_hardneg/hard_negative_application_summary.txt
dataset_hardneg/hard_negative_application_summary.csv

seg_hardneg/seg_history.csv
seg_hardneg/loss_curve.png
seg_hardneg/learning_curve.png
seg_hardneg/confusion_matrix_validation.png
seg_hardneg/confusion_matrix_test.png
seg_hardneg/evaluation_metrics.txt
seg_hardneg/evaluation_metrics.json

stage3_vs_stage6_metrics.png
stages_4_6_diagnostics.txt
all_stage_learning_curves.png
training_diagnostics.txt
stages_4_6_run_<timestamp>.log
```

## Download diagnostics to the local Mac

From the extracted v3 folder on the Mac:

```bash
chmod +x download_results_from_nebius.sh
SSH_KEY="$HOME/.ssh/nebius_cpu_ed25519" ./download_results_from_nebius.sh
```

Results are stored under:

```text
/Users/agni/Documents/VEG_Data/POLE_Voxel/
```

The download excludes NPZ data, model checkpoints, hard-negative shards and full prediction files.
