#!/bin/bash
#SBATCH --job-name=spatial_compare
#SBATCH --output=logs/spatial_compare_%j.out
#SBATCH --error=logs/spatial_compare_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#
# Score the baseline runs ($SPATIAL_BASELINES, TOA chips) and the new run
# ($SPATIAL_RUN as $SPATIAL_LABEL, its own chip set) at every epoch of
# $SPATIAL_SAVE_EPOCHS on the right halves of the chips in $SPATIAL_TEST_LIST.
# Submit with --dependency=afterok:<spatial_train job id>.

set -euo pipefail

: "${SPATIAL_CHIPS:?}"
: "${SPATIAL_LOCAL:?}"
: "${SPATIAL_RUN:?}"
: "${SPATIAL_SAVE_EPOCHS:?}"
: "${SPATIAL_TEST_TOA:?}"
: "${SPATIAL_TEST_LIST:?}"
: "${SPATIAL_BASELINES:?}"
: "${SPATIAL_LABEL:?}"
: "${SPATIAL_SWEEP_JSON:?}"

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flower
set -u
source scripts/slurm/stage_chips.sh

TOA=$(stage_chips "$SPATIAL_TEST_TOA" "$SPATIAL_LOCAL")
NEW=$(stage_chips "$SPATIAL_CHIPS" "$SPATIAL_LOCAL")
echo "SLURM job: ${SLURM_JOB_ID} on $(hostname)"
echo "TOA chips: $TOA"
echo "New chips: $NEW ($SPATIAL_LABEL)"

CK=()
IFS=, read -ra EPOCHS <<< "$SPATIAL_SAVE_EPOCHS"
for e in "${EPOCHS[@]}"; do
    e3=$(printf '%03d' "$e")
    for spec in $SPATIAL_BASELINES; do
        CK+=(--checkpoint "${spec%%=*}_e$e3=${spec#*=}/epoch_$e3.pth")
    done
    CK+=(--checkpoint "${SPATIAL_LABEL}_e$e3=$SPATIAL_RUN/epoch_$e3.pth")
done

mkdir -p "$(dirname "$SPATIAL_SWEEP_JSON")"
python scripts/compare_maskrcnn_runs.py \
    --chip-dir "$TOA" --chip-dir-for "$SPATIAL_LABEL=$NEW" \
    --chip-list "$SPATIAL_TEST_LIST" \
    "${CK[@]}" --ocm-masks --min-instance-size 64 \
    --json-out "$SPATIAL_SWEEP_JSON"
