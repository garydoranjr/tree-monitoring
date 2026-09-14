#!/bin/bash
#SBATCH --output=logs/classify_%A_%a.out
#SBATCH --error=logs/classify_%A_%a.err
#
# Stage 1: per-crown classification for one (image, key) pair.
# Submitted by run_crown_pipeline.sh; settings arrive via PIPELINE_* env vars.
# Array index maps to (image, key) as: image = idx / n_keys, key = idx % n_keys.

set -euo pipefail

: "${PIPELINE_INDIR:?}"
: "${PIPELINE_OUTDIR:?}"
: "${PIPELINE_SHAPE:?}"
: "${PIPELINE_KEYS:?}"
: "${PIPELINE_MODEL_TEMPLATE:?}"
SCALING_CONFIG="${PIPELINE_SCALING_CONFIG:-}"

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flower
set -u

mapfile -t IMAGES < <(ls "$PIPELINE_INDIR"/*.tif)
read -ra KEY_ARR <<< "$PIPELINE_KEYS"
N_KEYS=${#KEY_ARR[@]}

IMG="${IMAGES[$((SLURM_ARRAY_TASK_ID / N_KEYS))]}"
KEY="${KEY_ARR[$((SLURM_ARRAY_TASK_ID % N_KEYS))]}"

STEM=$(basename "$IMG" .tif)
OUTSUB="$PIPELINE_OUTDIR/$STEM/$KEY"
mkdir -p "$OUTSUB"

# MODEL_TEMPLATE contains a literal {key} placeholder, e.g.
# "drone_{key}_geo_out/epoch_020.pth".
MODEL="${PIPELINE_MODEL_TEMPLATE//\{key\}/$KEY}"

echo "SLURM job:   ${SLURM_JOB_ID}"
echo "Array task:  ${SLURM_ARRAY_TASK_ID}"
echo "Key:         $KEY"
echo "Model:       $MODEL"
echo "Input file:  $IMG"
echo "Crown map:   $PIPELINE_SHAPE"
echo "Output dir:  $OUTSUB"
echo "Scaling:     ${SCALING_CONFIG:-<none>}"
echo "CPUs:        ${SLURM_CPUS_PER_TASK}"

if [ ! -f "$MODEL" ]; then
    echo "Model checkpoint not found: $MODEL" >&2
    exit 1
fi

# Keep the math libraries inside the CPU allocation; the model runs on CPU and
# will otherwise try to use every core on the node.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

ARGS=("$MODEL" "$IMG" "$PIPELINE_SHAPE" "$OUTSUB")
if [ -n "$SCALING_CONFIG" ]; then
    ARGS+=(--scaling-config "$SCALING_CONFIG")
fi

python scripts/crown_classification.py "${ARGS[@]}"
