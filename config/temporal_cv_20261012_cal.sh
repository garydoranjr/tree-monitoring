# Settings for the temporal-generalization jobs on gattaca2 with MODIS-
# calibrated chips and the fixed input stretch (see
# docs/planet_radiometric_calibration.md, "Classifier input stretch"). Same
# chips, labels, folds and coregistration as config/temporal_cv_20261012.sh;
# only the pixel values (calibrated reflectance instead of TOA) and the input
# stretch differ, so runs compare one-to-one. Source this, then sbatch from
# the repo root; see docs/planet_temporal_generalization.md.
#
# Data: the chip set and planet_clipped_calfill (rsync -L, its calibrated
# scenes are symlinks) must be copied from /Volumes/Earth03/flower to $F; see
# docs/gattaca2_cluster.md (storage).

F=/scratch/tree-monitoring

# Chip set, staged per node onto local disk (GPFS reads are ~40x slower).
export TEMPORAL_CHIPS=$F/20261012_full_label_extended2_mkl_x4_coreg_4band_cal_curated
export TEMPORAL_LOCAL=/local/gdoran_flower
export TEMPORAL_FOLDS=config/temporal_folds_20261012.csv

# Fold runs go to $TEMPORAL_RUNS/fold<K>, the final model to
# $TEMPORAL_RUNS/final_e<E>.
export TEMPORAL_RUNS=$HOME/Documents/tree-flowering/202610_temporal_cal
export TEMPORAL_EPOCH_GRID=1,2,3,5,8,12,17,25,40
export TEMPORAL_TRAIN_ARGS="--ocm-masks --min-instance-size 64 --lr 1e-4 --no-wandb --fixed-stretch config/planet_fixed_stretch.yml"

# Coregistration and apply steps. Drone ortho dirs are space-separated; where
# two hold the same date, the first wins. The coregistration (computed on the
# TOA clips) is reused: calibration only offsets each band. Do not rerun
# planet_coreg_nearest.sh with this config. 6 of its 798 scenes are Dove
# Classic scenes without SR, absent from calfill; apply skips them.
export TEMPORAL_DRONE_DIRS="$F/stri/globus/RGB $F/stri/C3KW2X/BCI_50ha_timeseries_local_alignment $F/stri/24782016/BCI_50ha_timeseries_local_alignment"
export APPLY_PLANETDIR=$F/planet_clipped_calfill/4band
export APPLY_OCMDIR=$F/planet_clipped/ocm
export APPLY_COREG_JSON=$F/202610_temporal_apply/coreg_nearest_drone.json
export APPLY_CROWNS=$F/stri/24784053/BCI_50ha_2022_09_29_crownmap_improved/BCI_50ha_2022_09_29_crownmap_improved.shp
export APPLY_OUT=$F/202610_temporal_apply_cal
