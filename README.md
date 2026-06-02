Tree Phenology Monitoring
=========================

_Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology._

---

Research project from Caltech/JPL for analyzing tree flowering and deciduousness patterns using high-resolution drone imagery from Barro Colorado Island (BCI, Panama 50ha forest plot), Planet satellite imagery (4-band and 8-band AnalyticMS), machine learning models (SegFormer, Mask R-CNN) for crown segmentation and classification, and image coregistration with time-series analysis.

## Environment

```bash
conda env create -f environment.yml
conda activate flower
```

Key dependencies: `rasterio`, `xarray`, `geopandas`, `transformers`, `torch`, `arosics`, `planet` SDK, `omnicloudmask`.

Large external data volumes are typically mounted at `/Volumes/Earth03/flower` or `/Volumes/Luna01/flower`.

## Key Workflows

### 1. Planet Imagery Management

```bash
# Download scenes
python scripts/fetch_planet.py <configfile> <year> <month> <outputdir>

# Clip to study area
python scripts/clip_all_planet_images.py <planet_dir> <clipconfig> <outputdir>

# Cloud/shadow masks (OmniCloudMask)
python scripts/cloud_mask_planet.py   # defaults to planet_clipped/4band -> planet_clipped/ocm

# Convert to RGB quicklooks
snakemake -s Snakefile all_planet_rgb
```

Config files in `config/` define search geometry, item type, and product bundle.

### 2. Image Coregistration

Aligns images from different dates to a common reference frame using AROSICS. Results are saved as JSON with pixel shift vectors and success metrics.

```bash
python scripts/drone_coreg.py <imagedir> <outputdir> <configfile> <referenceindex>
python scripts/planet_coreg.py <imagedir> <outputdir> <configfile> <referenceindex>
```

### 3. Crown Segmentation & Classification

```bash
# Train SegFormer (WandB tracking)
python scripts/train_drone_image_segformer.py <img_dir> <mask_dir> <output_dir> [options]

# Inference
python scripts/crown_classification.py <model_path> <image_path> <crownmap_shp> <output_dir>
python scripts/apply_drone_labels.py <model_path> <crownmap_shp> <image_dir> <output_dir>
```

Batch shell scripts: `run_classification_flower.sh`, `run_classification_decid.sh`, `run_merge_flower.sh`, `run_merge_decid.sh`.

### 4. Crown Sequence Analysis

```bash
# Extract coregistered crown sequences
python scripts/coreg_crown_sequence.py \
    <drone_coreg_json> <global_coreg_json> <crownmap_shp> <planet_dir> <output_dir> <crownid>

# Time-series videos
snakemake -s Snakefile all_videos
python scripts/generate_sequence_video.py <crown_dir> <crownid> <config> <output>
```

### 5. NDVI and Phenology Analysis

```bash
python scripts/calculate_ndvi.py <input_image> <output_image>
python scripts/calc_all_ndvi.py <input_dir> <output_dir>
python scripts/crown_ndvi_scores.py <crown_dir> <crownmap_shp> <output_csv>
python scripts/ndvi_sequence_plot.py <sequence_dir> <output_plot>
python scripts/parse_phenology.py <labels_dir> <output_csv>
```

### 6. Visualization

```bash
python scripts/plot_coreg_offsets.py <coreg_json> <output_plot>
python scripts/plot_coreg_residuals.py <coreg_json> <output_plot>
python scripts/plot_coreg_success.py <coreg_json> <output_plot>
python scripts/plot_crown_stats_from_masks.py <stats_csv> <output_dir>
python scripts/crown_size_by_species.py <crownmap_shp> <output_plot>
```

## Code Organization

```
scripts/          Python analysis scripts (100+ files)
config/           YAML configs for workflows
data/             Input data (radiation, crown maps, labels)
example/          Example images
results/          Analysis outputs (generated)
drone_out/        Drone processing outputs
planet_out_*/     Planet processing outputs
```

### Key scripts by function

| Category | Scripts |
|---|---|
| Utilities | `util.py` |
| Data acquisition | `fetch_planet.py`, `select_relevant_planet_images.py` |
| Image processing | `coreg.py`, `planet_coreg.py`, `calculate_ndvi.py`, `clip_planet_image.py`, `cloud_mask_planet.py` |
| Machine learning | `train_drone_image_segformer.py`, `crown_classification.py`, `deploy_drone_image_segformer.py`, `sam2_segmentation.py` |
| Crown extraction | `crown_extractor.py`, `extract_labeled_crowns.py`, `coreg_crown_sequence.py`, `match_crowns_to_labels.py` |
| Analysis | `parse_*.py`, `*_analysis.py`, `illumination.py` |
| Visualization | `plot_*.py` |
| Video | `generate_sequence_video.py`, `crown_timelapse_mosaic.py` |

### Data flow

1. **Planet imagery** → download → clip to AOI → cloud mask → coregister → RGB conversion
2. **Drone imagery** → coregister to reference → extract crowns → train models
3. **Crown maps** (shapefiles) + **aligned images** → extract windows → classify
4. **Classifications** over time → phenology analysis → statistical summaries
5. **Crown sequences** → NDVI calculation → time series plots → videos

### Key concepts

**Coregistration:** Feature-matching alignment via AROSICS; results stored as JSON with shift vectors and success metrics.

**Crown extraction:** Crown polygon shapefiles define windows extracted from larger images (min 512×512 px, 100 px buffer).

**Classification:** Binary segmentation (flowering vs. non-flowering, deciduous vs. evergreen) using fine-tuned SegFormer on RGB or 4-band imagery.

**Cloud masking:** OmniCloudMask (`scripts/cloud_mask_planet.py`) produces 5-band uint8 GeoTIFFs (argmax class + 4 softmax probability bands) at the native 3 m grid, stored under `planet_clipped/ocm/`. Class encoding: 0=Clear, 1=Thick Cloud, 2=Thin Cloud, 3=Cloud Shadow, 255=NoData.

### Configuration files

- `config/snakemake.yml` — Snakemake workflow paths
- `config/planet_*.yml` — Planet search/download parameters
- `config/crown_video.yml` — Video generation settings
- `config/clip_*.yml` — Clipping AOI definitions

### Shell scripts

- `gen_*.sh` — Generate various outputs
- `run_classification_*.sh` — Classification pipelines
- `run_merge_*.sh` — Merge classification results
- `sam2.sh` — SAM2 segmentation workflow

## Notes

- Geospatial data in UTM projection (typically UTM 17N for BCI)
- Planet imagery: 3 m or 4.7 m resolution, 4-band (BGRN) or 8-band (SuperDove)
- Drone imagery: ~4 cm resolution
- WandB for ML experiment tracking
- Scripts use Click for CLI argument parsing
- No formal test suite; validation via visual inspection, training metrics (Dice/IoU), and comparison with litter trap data
