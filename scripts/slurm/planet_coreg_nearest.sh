#!/bin/bash
#SBATCH --job-name=planet_coreg
#SBATCH --output=logs/planet_coreg_%j.out
#SBATCH --error=logs/planet_coreg_%j.err
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#
# Coregister the Planet scenes away from drone dates to the nearest drone
# ortho, for the apply step. Each worker holds one ortho (up to ~4 GB for
# mavic). The JSON is rewritten after every ortho group and finished scenes
# are skipped, so a timed-out job can be resubmitted.

set -euo pipefail

: "${TEMPORAL_DRONE_DIRS:?}"
: "${APPLY_PLANETDIR:?}"
: "${APPLY_OCMDIR:?}"
: "${APPLY_COREG_JSON:?}"

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flower
set -u

DRONE_ARGS=()
for d in $TEMPORAL_DRONE_DIRS; do
    DRONE_ARGS+=(--drone-dir "$d")
done
mkdir -p "$(dirname "$APPLY_COREG_JSON")"
echo "SLURM job: ${SLURM_JOB_ID} on $(hostname)"

python scripts/coreg_planet_to_nearest_drone.py \
    "$APPLY_PLANETDIR" "$APPLY_OCMDIR" "$APPLY_COREG_JSON" \
    "${DRONE_ARGS[@]}" \
    --exclude-within-days 2 --min-clear-fraction 0.1 \
    --workers "${SLURM_CPUS_PER_TASK:-8}"
