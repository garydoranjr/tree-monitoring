#!/bin/bash
#SBATCH --job-name=radsens
#SBATCH --output=logs/radsens_%j.out
#SBATCH --error=logs/radsens_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#
# Test-time offset sensitivity (radiometric_sensitivity.py) of the fold runs of
# one temporal CV config. Source the config, then submit, after its eval
# (--dependency=afterok:<eval job id>) unless SENS_EPOCH is set. Units follow
# the config: reflectance with --fixed-stretch in $TEMPORAL_TRAIN_ARGS, else
# TOA, which needs the calibrated copy of the chips in $SENS_CAL_CHIPS.
#
#   SENS_EPOCH  checkpoint epoch (default: best_epoch of $TEMPORAL_RUNS/temporal_cv.json)
#   SENS_K      offset multiples (default -2,-1,-0.5,0,0.5,1,2)
#   SENS_OUT    output CSV (default $TEMPORAL_RUNS/sensitivity_e<epoch>.csv)

set -euo pipefail

: "${TEMPORAL_CHIPS:?}"
: "${TEMPORAL_LOCAL:?}"
: "${TEMPORAL_FOLDS:?}"
: "${TEMPORAL_RUNS:?}"
: "${TEMPORAL_TRAIN_ARGS:?}"

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flower
set -u
source scripts/slurm/stage_chips.sh

EPOCH=${SENS_EPOCH:-$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['best_epoch'])" \
    "$TEMPORAL_RUNS/temporal_cv.json")}
K=${SENS_K:--2,-1,-0.5,0,0.5,1,2}
OUT=${SENS_OUT:-$TEMPORAL_RUNS/sensitivity_e${EPOCH}.csv}

CHIPS=$(stage_chips "$TEMPORAL_CHIPS" "$TEMPORAL_LOCAL")
if [[ " $TEMPORAL_TRAIN_ARGS " == *" --fixed-stretch "* ]]; then
    UNITS=(--units reflectance)
else
    : "${SENS_CAL_CHIPS:?TOA runs need the calibrated chips in SENS_CAL_CHIPS}"
    UNITS=(--units toa --cal-chip-dir "$(stage_chips "$SENS_CAL_CHIPS" "$TEMPORAL_LOCAL")")
fi

echo "$TEMPORAL_RUNS epoch $EPOCH k $K ${UNITS[*]} -> $OUT"
python scripts/radiometric_sensitivity.py \
    --chip-dir "$CHIPS" --folds "$TEMPORAL_FOLDS" \
    --run-template "$TEMPORAL_RUNS/fold{fold}" --epoch "$EPOCH" \
    "${UNITS[@]}" --device cuda --k="$K" --out-csv "$OUT"
