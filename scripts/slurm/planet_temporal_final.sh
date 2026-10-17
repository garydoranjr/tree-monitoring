#!/bin/bash
#SBATCH --job-name=temporal_final
#SBATCH --output=logs/temporal_final_%j.out
#SBATCH --error=logs/temporal_final_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#
# Train the full-record model on every chip (both windows) for the number of
# epochs the temporal CV selected: set TEMPORAL_FINAL_EPOCHS to best_epoch
# from $TEMPORAL_RUNS/temporal_cv.json.

set -euo pipefail

: "${TEMPORAL_CHIPS:?}"
: "${TEMPORAL_LOCAL:?}"
: "${TEMPORAL_FOLDS:?}"
: "${TEMPORAL_RUNS:?}"
: "${TEMPORAL_TRAIN_ARGS:?}"
: "${TEMPORAL_FINAL_EPOCHS:?set to best_epoch from temporal_cv.json}"

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flower
set -u
source scripts/slurm/stage_chips.sh

CHIPS=$(stage_chips "$TEMPORAL_CHIPS" "$TEMPORAL_LOCAL")
OUT="$TEMPORAL_RUNS/final_e$TEMPORAL_FINAL_EPOCHS"
echo "Output: $OUT"

# shellcheck disable=SC2086
python scripts/train_planet_image_maskrcnn.py "$CHIPS" "$OUT" \
    --folds "$TEMPORAL_FOLDS" --fold all \
    --num-epochs "$TEMPORAL_FINAL_EPOCHS" \
    --save-epochs "$TEMPORAL_FINAL_EPOCHS" \
    $TEMPORAL_TRAIN_ARGS
