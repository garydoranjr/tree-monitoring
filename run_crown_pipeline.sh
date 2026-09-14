#!/bin/bash
#
# Crown classification pipeline driver.
#
# Submits the three stages as chained Slurm array jobs:
#
#   1. classify  one task per (image, key) -> per-crown confidence tiles
#   2. merge     one task per (image, key) -> single-band mosaic for that key
#   3. concat    one task per image        -> 2-band *_classifications.tif
#
# Stage 2 waits on stage 1 and stage 3 waits on stage 2, so the whole run
# proceeds unattended. Every stage skips work whose output already exists, so a
# failed or timed-out array can be resubmitted without redoing finished crowns.
#
# Usage:
#   ./run_crown_pipeline.sh <config.sh>          # submit
#   ./run_crown_pipeline.sh <config.sh> --dry-run
#
# See config/pipeline_globus.sh for a worked example.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.sh> [--dry-run]" >&2
    exit 1
fi

CONFIG="$1"
shift
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

if [ ! -f "$CONFIG" ]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG"

: "${INDIR:?must be set in config}"
: "${OUTDIR:?must be set in config}"
: "${SHAPE:?must be set in config}"
: "${KEYS:?must be set in config}"
: "${MODEL_TEMPLATE:?must be set in config}"

SCALING_CONFIG="${SCALING_CONFIG:-}"
CLASSIFY_TIME="${CLASSIFY_TIME:-08:00:00}"
CLASSIFY_MEM="${CLASSIFY_MEM:-16G}"
MERGE_TIME="${MERGE_TIME:-04:00:00}"
MERGE_MEM="${MERGE_MEM:-64G}"
CONCAT_TIME="${CONCAT_TIME:-04:00:00}"
CONCAT_MEM="${CONCAT_MEM:-32G}"
CPUS="${CPUS:-4}"
MAX_CONCURRENT="${MAX_CONCURRENT:-20}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
mkdir -p logs

# ---------------------------------------------------------------------------
# Enumerate work
# ---------------------------------------------------------------------------

mapfile -t IMAGES < <(ls "$INDIR"/*.tif)
if [ "${#IMAGES[@]}" -eq 0 ]; then
    echo "No .tif files in $INDIR" >&2
    exit 1
fi

read -ra KEY_ARR <<< "$KEYS"

N_IMAGES=${#IMAGES[@]}
N_KEYS=${#KEY_ARR[@]}
N_CLASSIFY=$((N_IMAGES * N_KEYS))

echo "Repo:        $REPO_DIR"
echo "Config:      $CONFIG"
echo "Input dir:   $INDIR"
echo "Output dir:  $OUTDIR"
echo "Crown map:   $SHAPE"
echo "Keys:        ${KEY_ARR[*]}"
echo "Scaling:     ${SCALING_CONFIG:-<none, imagery assumed uint8>}"
echo "Images:      $N_IMAGES"
echo "Array size:  $N_CLASSIFY classify / $N_CLASSIFY merge / $N_IMAGES concat"
echo

if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run; nothing submitted. First few classify tasks:"
    for ((i = 0; i < N_CLASSIFY && i < 4; i++)); do
        img="${IMAGES[$((i / N_KEYS))]}"
        key="${KEY_ARR[$((i % N_KEYS))]}"
        echo "  task $i: key=$key image=$(basename "$img")"
    done
    exit 0
fi

mkdir -p "$OUTDIR"

# Export the settings each stage script reads out of the environment.
export PIPELINE_INDIR="$INDIR"
export PIPELINE_OUTDIR="$OUTDIR"
export PIPELINE_SHAPE="$SHAPE"
export PIPELINE_KEYS="$KEYS"
export PIPELINE_MODEL_TEMPLATE="$MODEL_TEMPLATE"
export PIPELINE_SCALING_CONFIG="$SCALING_CONFIG"

SB_EXPORT="ALL"

# ---------------------------------------------------------------------------
# Stage 1: classify
# ---------------------------------------------------------------------------

CLASSIFY_ID=$(sbatch --parsable \
    --job-name=crown_classify \
    --array="0-$((N_CLASSIFY - 1))%${MAX_CONCURRENT}" \
    --time="$CLASSIFY_TIME" \
    --cpus-per-task="$CPUS" \
    --mem="$CLASSIFY_MEM" \
    --export="$SB_EXPORT" \
    scripts/slurm/crown_classify.sh)
echo "Submitted classify: job $CLASSIFY_ID ($N_CLASSIFY tasks)"

# ---------------------------------------------------------------------------
# Stage 2: merge (one per image/key, same indexing as classify)
# ---------------------------------------------------------------------------
#
# aftercorr starts merge task N as soon as classify task N succeeds, so merges
# overlap with the remaining classifications instead of waiting for all of them.

MERGE_ID=$(sbatch --parsable \
    --job-name=crown_merge \
    --array="0-$((N_CLASSIFY - 1))%${MAX_CONCURRENT}" \
    --time="$MERGE_TIME" \
    --cpus-per-task="$CPUS" \
    --mem="$MERGE_MEM" \
    --dependency="aftercorr:$CLASSIFY_ID" \
    --kill-on-invalid-dep=yes \
    --export="$SB_EXPORT" \
    scripts/slurm/crown_merge.sh)
echo "Submitted merge:    job $MERGE_ID (aftercorr:$CLASSIFY_ID)"

# ---------------------------------------------------------------------------
# Stage 3: concat (one per image; needs every key merged for that image)
# ---------------------------------------------------------------------------
#
# Task indices do not line up with stage 2 (one task per image, not per
# image/key), so this waits on the merge array as a whole.

CONCAT_ID=$(sbatch --parsable \
    --job-name=crown_concat \
    --array="0-$((N_IMAGES - 1))%${MAX_CONCURRENT}" \
    --time="$CONCAT_TIME" \
    --cpus-per-task="$CPUS" \
    --mem="$CONCAT_MEM" \
    --dependency="afterok:$MERGE_ID" \
    --kill-on-invalid-dep=yes \
    --export="$SB_EXPORT" \
    scripts/slurm/crown_concat.sh)
echo "Submitted concat:   job $CONCAT_ID (afterok:$MERGE_ID)"

echo
echo "Monitor with:  squeue -u $USER"
echo "Logs in:       $REPO_DIR/logs/"
echo "Cancel all:    scancel $CLASSIFY_ID $MERGE_ID $CONCAT_ID"
