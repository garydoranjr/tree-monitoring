#!/bin/bash
#SBATCH --job-name=spatial_train
#SBATCH --output=logs/spatial_train_%j.out
#SBATCH --error=logs/spatial_train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#
# Train one Mask R-CNN on the east/west split (left windows train, right
# windows validate) of every chip in $SPATIAL_CHIPS, saving checkpoints at
# $SPATIAL_SAVE_EPOCHS. Settings come from config/spatial_20261012_cal.sh.

set -euo pipefail

: "${SPATIAL_CHIPS:?}"
: "${SPATIAL_LOCAL:?}"
: "${SPATIAL_RUN:?}"
: "${SPATIAL_NUM_EPOCHS:?}"
: "${SPATIAL_SAVE_EPOCHS:?}"
: "${SPATIAL_TRAIN_ARGS:?}"

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flower
set -u
source scripts/slurm/stage_chips.sh

echo "SLURM job: ${SLURM_JOB_ID} on $(hostname)"
CHIPS=$(stage_chips "$SPATIAL_CHIPS" "$SPATIAL_LOCAL")
echo "Chips:     $CHIPS"
echo "Output:    $SPATIAL_RUN"

# shellcheck disable=SC2086
python scripts/train_planet_image_maskrcnn.py "$CHIPS" "$SPATIAL_RUN" \
    --num-epochs "$SPATIAL_NUM_EPOCHS" --save-epochs "$SPATIAL_SAVE_EPOCHS" \
    $SPATIAL_TRAIN_ARGS
