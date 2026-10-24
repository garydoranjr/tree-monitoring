# Settings for the spatial (east/west) Mask R-CNN experiment on gattaca2 with
# MODIS-calibrated chips and the fixed input stretch: the `full_mkl` run of
# docs/planet_training_chips_50ha.md ("Colour-transfer mavic masks") repeated
# on the calibrated copy of the same 142 chips, then scored with the earlier
# runs on the 56-chip benchmark (right halves of the phantom 2020-23 chips).
# Source this, then sbatch from the repo root (scripts/slurm/planet_spatial_*.sh).
# See docs/handoff_calibrated_training.md.

F=/scratch/tree-monitoring
O=$HOME/Documents/tree-flowering

# Training: same settings as full_mkl (flat lr 1e-4, batch 4, 200 epochs,
# OCM-aware loss, min instance 64 px), plus the fixed stretch. Checkpoints
# only at the benchmark epochs (all 200 would be ~35 GB).
export SPATIAL_CHIPS=$F/20261012_full_label_extended2_mkl_x4_coreg_4band_cal_curated
export SPATIAL_LOCAL=/local/gdoran_flower
export SPATIAL_RUN=$O/202610_full2_mkl_cal_curated_maskrcnn_out_min064
export SPATIAL_NUM_EPOCHS=200
export SPATIAL_SAVE_EPOCHS=1,2,3,5,8,12,17,25,40,60,100,200
export SPATIAL_TRAIN_ARGS="--ocm-masks --min-instance-size 64 --lr 1e-4 --no-wandb --fixed-stretch config/planet_fixed_stretch.yml"

# Benchmark: the 56 test chips are byte-identical files inside the 142-chip
# TOA set (tif, mask and OCM sidecar), so the baselines are scored there and
# the new run on the same names in the calibrated set.
export SPATIAL_TEST_TOA=$F/20261012_full_label_extended2_mkl_x4_coreg_4band_stretch_stats_curated
export SPATIAL_TEST_LIST=config/spatial_test_chips_56.txt
export SPATIAL_BASELINES="base=$O/20260824_maskrcnn_out_min064 phantom=$O/202610_phantom_curated_maskrcnn_out_min064 full_mkl=$O/202610_full2_mkl_curated_maskrcnn_out_min064"
export SPATIAL_LABEL=full_mkl_cal
export SPATIAL_SWEEP_JSON=logs/headtohead_sweep_202610_mkl_cal.json
