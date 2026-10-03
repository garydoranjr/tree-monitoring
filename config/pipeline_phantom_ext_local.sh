# Crown classification pipeline settings for the C3KW2X locally-aligned 50ha
# orthomosaics.
#
# Submit with:
#   ./run_crown_pipeline.sh config/pipeline_phantom_ext_local.sh
#
# Sourced by run_crown_pipeline.sh, which exports these to the stage scripts.
#
# Identical to config/pipeline_phantom_ext_global.sh except for INDIR. The
# locally-aligned products carry a warp that varies by up to ~2 m across the
# plot, where the global ones are a rigid shift; the Planet training-chip build
# in docs/planet_training_chips_50ha.md consumes the local classifications.

# The 16 aligned dates after 2023-10-24 (2023-10-31 .. 2024-03-18) from the
# 2024 Smithsonian release, which extend the 90-date 24782016 timeseries.
#
# Note the files here are filed by suffix, not by the server's directory name:
# on the DataONE share orthomosaic_aligned_global/ actually holds *_local.tif
# and vice versa. See C3KW2X/README_local.md.
INDIR=/scratch/tree-monitoring/stri/C3KW2X/BCI_50ha_timeseries_local_alignment
OUTDIR=/scratch/tree-monitoring/results/phantom_ext

# Both alignments write into the same OUTDIR. The output directory for each
# task is named from the image stem, and the stems already carry the
# _global/_local suffix, so the two runs cannot collide.

# Static crown map: 2280 polygons, no 'date' column, so the same geometry is
# applied to every flight date. crown_classification.py detects the missing
# 'date' column and skips the per-date filtering it applies to the ava
# timeseries crown map. The C3KW2X grid matches 24782016 to within 1 cm, so
# this 2022-09-29 geometry lands correctly on the extension dates.
SHAPE=/scratch/tree-monitoring/stri/24784053/BCI_50ha_2022_09_29_crownmap_improved/BCI_50ha_2022_09_29_crownmap_improved.shp

# concat_classifications.py expects both, in this order.
KEYS="flower decid"

# {key} is substituted per task; epoch_020 matches the globus and ava runs.
MODEL_TEMPLATE="drone_{key}_geo_out/epoch_020.pth"

# No SCALING_CONFIG. These rasters are 4-band uint8 (23425x12697, EPSG:32617),
# the same dtype as the 24782016 mosaics the models were trained on, so they
# are passed through untouched. config/crown_classification_globus.yml exists
# only because the globus mosaics are uint16 and must not be used here.

# 297 Mpx per image against 558 Mpx for globus, but the same 2280 crowns, so
# classify time is comparable (~17 min plus ~1.5 min to load the model) while
# merge peak RSS is roughly half the 7.3 GB measured on a globus mosaic.
CLASSIFY_TIME=04:00:00
CLASSIFY_MEM=16G
MERGE_TIME=02:00:00
MERGE_MEM=32G
CONCAT_TIME=02:00:00
CONCAT_MEM=32G

CPUS=4
# 10 rather than the globus 20, because the global and local runs are
# submitted together and share the queue.
MAX_CONCURRENT=10
