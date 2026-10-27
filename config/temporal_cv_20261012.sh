# Settings for the temporal-generalization jobs on gattaca2
# (scripts/slurm/planet_temporal_{cv,eval,final}.sh, planet_coreg_nearest.sh
# and planet_apply.sh). Source this, then sbatch from the repo root; see
# docs/planet_temporal_generalization.md.

F=/scratch/tree-monitoring

# Chip set, staged per node onto local disk (GPFS reads are ~40x slower).
export TEMPORAL_CHIPS=$F/20261012_full_label_extended2_mkl_x4_coreg_4band_stretch_stats_curated
export TEMPORAL_LOCAL=/local/gdoran_flower
export TEMPORAL_FOLDS=config/temporal_folds_20261012.csv

# Fold runs go to $TEMPORAL_RUNS/fold<K>, the final model to
# $TEMPORAL_RUNS/final_e<E>.
export TEMPORAL_RUNS=$HOME/Documents/tree-flowering/202610_temporal
export TEMPORAL_EPOCH_GRID=1,2,3,5,8,12,17,25,40
# Set explicitly so that sourcing this after a DINOv3 config (which overrides
# it) does not leave the DINOv3 trainer in the shell.
export TEMPORAL_TRAIN_SCRIPT=scripts/train_planet_image_maskrcnn.py
export TEMPORAL_TRAIN_ARGS="--ocm-masks --min-instance-size 64 --lr 1e-4 --no-wandb"

# Coregistration and apply steps. Drone ortho dirs are space-separated; where
# two hold the same date, the first wins.
export TEMPORAL_DRONE_DIRS="$F/stri/globus/RGB $F/stri/C3KW2X/BCI_50ha_timeseries_local_alignment $F/stri/24782016/BCI_50ha_timeseries_local_alignment"
export APPLY_PLANETDIR=$F/planet_clipped/4band
export APPLY_OCMDIR=$F/planet_clipped/ocm
export APPLY_COREG_JSON=$F/202610_temporal_apply/coreg_nearest_drone.json
export APPLY_CROWNS=$F/stri/24784053/BCI_50ha_2022_09_29_crownmap_improved/BCI_50ha_2022_09_29_crownmap_improved.shp
export APPLY_OUT=$F/202610_temporal_apply
