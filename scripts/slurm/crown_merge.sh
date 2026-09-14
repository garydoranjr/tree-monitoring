#!/bin/bash
#SBATCH --output=logs/merge_%A_%a.out
#SBATCH --error=logs/merge_%A_%a.err
#
# Stage 2: mosaic one image/key's crown tiles into a single-band raster.
# Uses the same (image, key) array indexing as crown_classify.sh so that
# --dependency=aftercorr pairs each merge task with its classify task.

set -euo pipefail

: "${PIPELINE_INDIR:?}"
: "${PIPELINE_OUTDIR:?}"
: "${PIPELINE_KEYS:?}"

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
CLSDIR="$PIPELINE_OUTDIR/$STEM/$KEY"
OUTFILE="$PIPELINE_OUTDIR/$STEM/${STEM}_${KEY}.tif"

echo "SLURM job:   ${SLURM_JOB_ID}"
echo "Array task:  ${SLURM_ARRAY_TASK_ID}"
echo "Key:         $KEY"
echo "Source file: $IMG"
echo "Input dir:   $CLSDIR"
echo "Output file: $OUTFILE"
echo "CPUs:        ${SLURM_CPUS_PER_TASK}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

N_TILES=$(find "$CLSDIR" -name '*.tif' | wc -l)
echo "Tiles found: $N_TILES"
if [ "$N_TILES" -eq 0 ]; then
    echo "No classification tiles in $CLSDIR" >&2
    exit 1
fi

python scripts/merge_classifications.py "$IMG" "$CLSDIR" "$OUTFILE"
