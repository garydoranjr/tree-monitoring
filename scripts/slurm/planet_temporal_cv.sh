#!/bin/bash
#SBATCH --job-name=temporal_cv
#SBATCH --output=logs/temporal_cv_%A_%a.out
#SBATCH --error=logs/temporal_cv_%A_%a.err
#SBATCH --array=0-5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#
# One temporal CV fold per array task: train on the chips of every other fold
# (both windows) and test on fold $SLURM_ARRAY_TASK_ID, saving checkpoints on
# the epoch grid. Settings come from config/temporal_cv_20261012.sh. Pass the
# account on the sbatch command line if needed. $TEMPORAL_TRAIN_SCRIPT picks
# the trainer (default: the Mask R-CNN; the DINOv3 configs set
# scripts/train_planet_image_dinov3.py, which takes the same fold options).

set -euo pipefail

: "${TEMPORAL_CHIPS:?}"
: "${TEMPORAL_LOCAL:?}"
: "${TEMPORAL_FOLDS:?}"
: "${TEMPORAL_RUNS:?}"
: "${TEMPORAL_EPOCH_GRID:?}"
: "${TEMPORAL_TRAIN_ARGS:?}"

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flower
set -u
source scripts/slurm/stage_chips.sh

FOLD=$SLURM_ARRAY_TASK_ID
MAX_EPOCH=${TEMPORAL_EPOCH_GRID##*,}
OUT="$TEMPORAL_RUNS/fold$FOLD"

echo "SLURM job:   ${SLURM_JOB_ID} task ${FOLD} on $(hostname)"
CHIPS=$(stage_chips "$TEMPORAL_CHIPS" "$TEMPORAL_LOCAL")
echo "Chips:       $CHIPS"
echo "Output:      $OUT"

# shellcheck disable=SC2086
python "${TEMPORAL_TRAIN_SCRIPT:-scripts/train_planet_image_maskrcnn.py}" "$CHIPS" "$OUT" \
    --folds "$TEMPORAL_FOLDS" --fold "$FOLD" \
    --num-epochs "$MAX_EPOCH" --save-epochs "$TEMPORAL_EPOCH_GRID" \
    $TEMPORAL_TRAIN_ARGS
