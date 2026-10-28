# Offset augmentation for the temporal CV on gattaca2: the MODIS-calibrated
# run of config/temporal_cv_20261012_cal.sh with --offset-jitter 0.5, which adds
# k times the across-scene SD of the fitted Planet-MODIS offsets to every band
# of each training sample before the fixed stretch, k ~ N(0, 0.5). Tests whether
# the calibrated model's sensitivity to offset errors (0.12 pooled IoU at
# -1 SD) can be trained away. Same chips, folds, epochs and settings otherwise.
# See docs/planet_temporal_generalization.md ("Test-time offset
# sensitivity"). Source this, then sbatch planet_temporal_cv.sh and
# planet_temporal_eval.sh from the repo root. CV only: the apply settings are
# unset so that an apply job cannot overwrite the parent run's output.

source config/temporal_cv_20261012_cal.sh
export TEMPORAL_RUNS=$HOME/Documents/tree-flowering/202610_temporal_cal_offjit05
export TEMPORAL_TRAIN_ARGS="$TEMPORAL_TRAIN_ARGS --offset-jitter 0.5"
unset APPLY_PLANETDIR APPLY_OCMDIR APPLY_COREG_JSON APPLY_CROWNS APPLY_OUT
