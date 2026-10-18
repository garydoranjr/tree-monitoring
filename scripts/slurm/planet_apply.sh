#!/bin/bash
#SBATCH --job-name=planet_apply
#SBATCH --output=logs/planet_apply_%A_%a.out
#SBATCH --error=logs/planet_apply_%A_%a.err
#SBATCH --array=0-3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#
# Apply the full-record model to the coregistered non-drone-date scenes, one
# shard per array task (every N-th scene, N = array size). Finished scenes are
# skipped, so a timed-out array can be resubmitted. Afterwards run
# `python scripts/apply_planet_maskrcnn.py concat $APPLY_OUT`.

set -euo pipefail

: "${TEMPORAL_CHIPS:?}"
: "${TEMPORAL_RUNS:?}"
: "${TEMPORAL_FINAL_EPOCHS:?}"
: "${APPLY_PLANETDIR:?}"
: "${APPLY_OCMDIR:?}"
: "${APPLY_COREG_JSON:?}"
: "${APPLY_CROWNS:?}"
: "${APPLY_OUT:?}"

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flower
set -u

MODEL="$TEMPORAL_RUNS/final_e$TEMPORAL_FINAL_EPOCHS/last_model.pth"
# Any training chip defines the output grid; they all share one footprint.
# (A glob, not `ls | head -1`: under pipefail the SIGPIPE to ls fails the job.)
CHIP_TIFS=("$TEMPORAL_CHIPS"/*_4band.tif)
GRID_LIKE=${CHIP_TIFS[0]}
SHARD="$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT"
echo "SLURM job: ${SLURM_JOB_ID} shard ${SHARD} on $(hostname)"
echo "Model:     $MODEL"

python scripts/apply_planet_maskrcnn.py run "$MODEL" \
    "$APPLY_PLANETDIR" "$APPLY_OCMDIR" "$APPLY_COREG_JSON" "$APPLY_OUT" \
    --grid-like "$GRID_LIKE" --crowns "$APPLY_CROWNS" --shard "$SHARD"
