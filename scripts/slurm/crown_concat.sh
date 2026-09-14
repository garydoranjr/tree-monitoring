#!/bin/bash
#SBATCH --output=logs/concat_%A_%a.out
#SBATCH --error=logs/concat_%A_%a.err
#
# Stage 3: combine one image's per-key mosaics into a 2-band COG
# (band 1 flowering probability, band 2 deciduous probability).
# One task per image, so the array index is the image index directly.

set -euo pipefail

: "${PIPELINE_INDIR:?}"
: "${PIPELINE_OUTDIR:?}"

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flower
set -u

mapfile -t IMAGES < <(ls "$PIPELINE_INDIR"/*.tif)
IMG="${IMAGES[$SLURM_ARRAY_TASK_ID]}"
STEM=$(basename "$IMG" .tif)
RESDIR="$PIPELINE_OUTDIR/$STEM"

echo "SLURM job:   ${SLURM_JOB_ID}"
echo "Array task:  ${SLURM_ARRAY_TASK_ID}"
echo "Result dir:  $RESDIR"
echo "CPUs:        ${SLURM_CPUS_PER_TASK}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# concat_classifications.py exits 1 when the output already exists, which would
# otherwise mark a resubmitted task as failed.
if [ -f "$RESDIR/${STEM}_classifications.tif" ]; then
    echo "Output already exists; nothing to do."
    exit 0
fi

python scripts/concat_classifications.py "$RESDIR"
