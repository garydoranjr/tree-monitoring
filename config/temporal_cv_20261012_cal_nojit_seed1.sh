# Second seed of the augmentation ablation: config/temporal_cv_20261012_cal_nojit.sh
# with --seed 1 and otherwise identical settings, so that H1 (the training
# ColorJitter stands in for calibration) rests on two runs per arm. See
# docs/planet_temporal_generalization.md ("Augmentation ablation and repeat
# run"). Source this, then sbatch planet_temporal_cv.sh and
# planet_temporal_eval.sh from the repo root. CV only: the apply settings are
# unset so that an apply job cannot overwrite the parent run's output.

source config/temporal_cv_20261012_cal_nojit.sh
export TEMPORAL_RUNS=$HOME/Documents/tree-flowering/202610_temporal_cal_nojit_seed1
export TEMPORAL_TRAIN_ARGS="$TEMPORAL_TRAIN_ARGS --seed 1"
unset APPLY_PLANETDIR APPLY_OCMDIR APPLY_COREG_JSON APPLY_CROWNS APPLY_OUT
