# Crown classification pipeline settings for the mavic 50ha RGB mosaics, with
# the 3x3 colour transfer to phantom (crown_classification_mavic_mkl.yml)
# instead of the per-band uint16 -> uint8 map used by pipeline_mavic.sh.
# See docs/drone_radiometry.md.
#
# Submit with:
#   ./run_crown_pipeline.sh config/pipeline_mavic_mkl.sh
#
# Sourced by run_crown_pipeline.sh, which exports these to the stage scripts.

# 96 single-date 50ha orthomosaics, 2024-03-06 through 2026-01-20.
INDIR=/scratch/tree-monitoring/stri/globus/RGB
# Separate from the per-band run in results/globus: every stage skips outputs
# that already exist, so sharing that directory would keep the old results.
OUTDIR=/scratch/tree-monitoring/results/globus_mkl

# Static crown map: 2280 polygons, no 'date' column, so the same geometry is
# applied to every flight date. crown_classification.py detects the missing
# 'date' column and skips the per-date filtering it applies to the ava
# timeseries crown map.
SHAPE=/scratch/tree-monitoring/stri/24784053/BCI_50ha_2022_09_29_crownmap_improved/BCI_50ha_2022_09_29_crownmap_improved.shp

# concat_classifications.py expects both, in this order.
KEYS="flower decid"

# {key} is substituted per task; epoch_020 matches the existing ava/global run.
MODEL_TEMPLATE="drone_{key}_geo_out/epoch_020.pth"

# These mosaics are uint16 while the models were trained on 8-bit data. The
# colour transfer also corrects mavic's greener, more saturated colour, which
# made the flowering model over-call on mavic.
SCALING_CONFIG=config/crown_classification_mavic_mkl.yml

# Timings measured on one 25595x21815 mosaic (558 Mpx, 2280 crowns):
#   classify ~2.2 crowns/s -> ~17 min, plus ~1.5 min to load the model
#   merge    ~7.3 GB peak RSS, dominated by allocating the full output grid
CLASSIFY_TIME=04:00:00
CLASSIFY_MEM=16G
MERGE_TIME=02:00:00
MERGE_MEM=32G
CONCAT_TIME=02:00:00
CONCAT_MEM=32G

CPUS=4
MAX_CONCURRENT=20
