# Repeat of the TOA temporal CV of config/temporal_cv_20261012.sh with a fixed
# seed (--seed 1) and otherwise identical settings, to measure the
# run-to-run spread that the TOA/calibrated and jitter comparisons are judged
# against. See docs/planet_temporal_generalization.md ("Why calibration did
# not help"). Source this, then sbatch planet_temporal_cv.sh and
# planet_temporal_eval.sh from the repo root. CV only: the apply settings are
# unset so that an apply job cannot overwrite the parent run's output.

source config/temporal_cv_20261012.sh
export TEMPORAL_RUNS=$HOME/Documents/tree-flowering/202610_temporal_seed1
export TEMPORAL_TRAIN_ARGS="$TEMPORAL_TRAIN_ARGS --seed 1"
unset APPLY_PLANETDIR APPLY_OCMDIR APPLY_COREG_JSON APPLY_CROWNS APPLY_OUT
