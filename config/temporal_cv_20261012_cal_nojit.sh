# Augmentation ablation for the temporal CV on gattaca2: the MODIS-calibrated
# run of config/temporal_cv_20261012_cal.sh with the training ColorJitter
# turned off (--no-color-jitter). Same chips, folds, epochs and settings
# otherwise. See docs/planet_temporal_generalization.md ("Why calibration did
# not help"). Source this, then sbatch planet_temporal_cv.sh and
# planet_temporal_eval.sh from the repo root. CV only: the apply settings are
# unset so that an apply job cannot overwrite the parent run's output.

source config/temporal_cv_20261012_cal.sh
export TEMPORAL_RUNS=$HOME/Documents/tree-flowering/202610_temporal_cal_nojit
export TEMPORAL_TRAIN_ARGS="$TEMPORAL_TRAIN_ARGS --no-color-jitter"
unset APPLY_PLANETDIR APPLY_OCMDIR APPLY_COREG_JSON APPLY_CROWNS APPLY_OUT
