#!/bin/bash
#SBATCH --job-name=temporal_eval
#SBATCH --output=logs/temporal_eval_%j.out
#SBATCH --error=logs/temporal_eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#
# Score every fold run at every grid epoch and pick the training length for
# the final model. Submit with --dependency=afterok:<cv job id>.

set -euo pipefail

: "${TEMPORAL_CHIPS:?}"
: "${TEMPORAL_LOCAL:?}"
: "${TEMPORAL_FOLDS:?}"
: "${TEMPORAL_RUNS:?}"
: "${TEMPORAL_EPOCH_GRID:?}"

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flower
set -u
source scripts/slurm/stage_chips.sh

CHIPS=$(stage_chips "$TEMPORAL_CHIPS" "$TEMPORAL_LOCAL")
python scripts/eval_temporal_cv.py \
    --chip-dir "$CHIPS" --folds "$TEMPORAL_FOLDS" \
    --run-template "$TEMPORAL_RUNS/fold{fold}" \
    --epochs "$TEMPORAL_EPOCH_GRID" \
    --json-out "$TEMPORAL_RUNS/temporal_cv.json" \
    --chip-csv "$TEMPORAL_RUNS/temporal_cv_windows.csv"
