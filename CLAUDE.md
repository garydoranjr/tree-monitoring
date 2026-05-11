# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tree Phenology Monitoring research project from Caltech/JPL for analyzing tree flowering and deciduousness patterns using:
- High-resolution drone imagery from BCI (Barro Colorado Island) 50ha forest plot
- Planet satellite imagery (4-band and 8-band AnalyticMS)
- Machine learning models (SegFormer) for crown segmentation and classification
- Image coregistration and time-series analysis

## Environment Setup

Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate flower
```

The environment includes key dependencies: rasterio, xarray, geopandas, transformers, torch, arosics, planet SDK.

## Key Workflows

### 1. Planet Imagery Management

**Fetch Planet imagery:**
```bash
python scripts/fetch_planet.py <configfile> <year> <month> <outputdir>
```
Config files in `config/` define search parameters (geometry, item_type, product_bundle).

**Clip Planet images to study area:**
```bash
python scripts/clip_all_planet_images.py <planet_dir> <clipconfig> <outputdir>
```

**Convert to RGB for visualization:**
```bash
snakemake -s Snakefile all_planet_rgb
```

### 2. Image Coregistration

Coregistration aligns images from different dates to the same coordinate system using AROSICS.

**Drone image coregistration:**
```bash
python scripts/drone_coreg.py <imagedir> <outputdir> <configfile> <referenceindex>
```

**Planet image coregistration:**
```bash
python scripts/planet_coreg.py <imagedir> <outputdir> <configfile> <referenceindex>
```

Coregistration results are saved as JSON files containing spatial shift information for each image pair.

### 3. Crown Segmentation & Classification

**Train SegFormer model for crown detection:**
```bash
python scripts/train_drone_image_segformer.py <img_dir> <mask_dir> <output_dir> [options]
```
Uses WandB for experiment tracking.

**Run classification on images:**
```bash
python scripts/crown_classification.py <model_path> <image_path> <crownmap_shapefile> <output_dir>
```

**Apply drone labels to crowns:**
```bash
python scripts/apply_drone_labels.py <model_path> <crownmap_shapefile> <image_dir> <output_dir>
```

Batch scripts for classification workflows:
- `run_classification_flower.sh` - Run flowering classification
- `run_classification_decid.sh` - Run deciduousness classification
- `run_merge_flower.sh` - Merge flowering results
- `run_merge_decid.sh` - Merge deciduousness results

### 4. Crown Sequence Analysis

**Extract coregistered crown sequences:**
```bash
python scripts/coreg_crown_sequence.py <drone_coreg_json> <global_coreg_json> <crownmap_shp> <planet_dir> <output_dir> <crownid>
```

**Generate time-series videos:**
```bash
snakemake -s Snakefile all_videos
# Or individual:
python scripts/generate_sequence_video.py <crown_dir> <crownid> <config> <output>
```

### 5. NDVI and Phenology Analysis

**Calculate NDVI:**
```bash
python scripts/calculate_ndvi.py <input_image> <output_image>
python scripts/calc_all_ndvi.py <input_dir> <output_dir>
```

**Crown NDVI time series:**
```bash
python scripts/crown_ndvi_scores.py <crown_dir> <crownmap_shp> <output_csv>
```

**Plot NDVI sequences:**
```bash
python scripts/ndvi_sequence_plot.py <sequence_dir> <output_plot>
```

**Phenology analysis:**
```bash
python scripts/parse_phenology.py <labels_dir> <output_csv>
```

### 6. Visualization & Analysis

**Plot coregistration results:**
```bash
python scripts/plot_coreg_offsets.py <coreg_json> <output_plot>
python scripts/plot_coreg_residuals.py <coreg_json> <output_plot>
python scripts/plot_coreg_success.py <coreg_json> <output_plot>
```

**Crown analysis plots:**
```bash
python scripts/plot_crown_stats_from_masks.py <stats_csv> <output_dir>
python scripts/crown_size_by_species.py <crownmap_shp> <output_plot>
```

## Architecture & Code Organization

### Directory Structure
- `scripts/` - Python analysis scripts (100+ scripts for different tasks)
- `config/` - YAML configuration files for workflows
- `data/` - Input data (radiation, crown maps, labels)
- `example/` - Example images
- `results/` - Analysis outputs (generated)
- Various output directories: `drone_out/`, `planet_out_*/`, etc.

### Key Scripts by Function

**Core utilities:**
- `scripts/util.py` - Configuration loading and shared utilities

**Data acquisition:**
- `scripts/fetch_planet.py` - Planet API integration (async)
- `scripts/select_relevant_planet_images.py` - Filter imagery by criteria

**Image processing:**
- `scripts/coreg.py`, `scripts/planet_coreg.py` - Coregistration
- `scripts/calculate_ndvi.py` - Vegetation index calculation
- `scripts/clip_planet_image.py` - Clip to AOI

**Machine learning:**
- `scripts/train_drone_image_segformer.py` - Model training pipeline
- `scripts/crown_classification.py` - Inference on crowns
- `scripts/deploy_drone_image_segformer.py` - Model deployment
- `scripts/sam2_segmentation.py` - SAM2 segmentation

**Crown extraction:**
- `scripts/crown_extractor.py` - Extract crown chips from images
- `scripts/extract_labeled_crowns.py` - Extract crowns with labels
- `scripts/coreg_crown_sequence.py` - Build aligned time series
- `scripts/match_crowns_to_labels.py` - Match geometries to labels

**Analysis:**
- `scripts/parse_*.py` - Parse various label formats
- `scripts/*_analysis.py` - Statistical analyses
- `scripts/plot_*.py` - Visualization scripts
- `scripts/illumination.py` - Solar geometry calculations

**Video generation:**
- `scripts/generate_sequence_video.py` - Create time-lapse videos
- `scripts/crown_timelapse_mosaic.py` - Multi-crown mosaics

### Data Flow

1. **Planet imagery** → download → clip to AOI → coregister → RGB conversion
2. **Drone imagery** → coregister to reference → extract crowns → train models
3. **Crown maps** (shapefiles) + **aligned images** → extract windows → classify
4. **Classifications** over time → phenology analysis → statistical summaries
5. **Crown sequences** → NDVI calculation → time series plots → videos

### Key Concepts

**Coregistration:** Aligning images to a common reference frame using feature matching (AROSICS library). Results stored as JSON with pixel shift vectors and success metrics.

**Crown extraction:** Using crown polygon shapefiles to extract windows from larger images, typically with min 512x512px size and 100px buffer around polygons.

**Classification:** Binary segmentation (flowering vs non-flowering, deciduous vs evergreen) using fine-tuned SegFormer models on RGB or 4-band imagery.

**Phenology events:** Temporal patterns in vegetation indices or classification outputs indicating flowering, leaf-out, or senescence.

### Configuration Files

- `config/snakemake.yml` - Snakemake workflow paths
- `config/planet_*.yml` - Planet search/download parameters
- `config/crown_video.yml` - Video generation settings
- `config/clip_*.yml` - Clipping AOI definitions

### Shell Scripts

Common batch processing patterns in repository root:
- `gen_*.sh` - Generate various outputs
- `run_classification_*.sh` - Classification pipelines
- `run_merge_*.sh` - Merge classification results
- `sam2.sh` - SAM2 segmentation workflow

### Testing & Validation

No formal test suite. Validation through:
- Visual inspection of coregistration outputs
- Classification metrics (Dice, IoU) during training
- Cross-validation with litter trap data (`scripts/*trap*.py`)
- Manual labeling comparison (`scripts/compare_crown_labels.py`)

## Notes

- Large external data volumes typically mounted at `/Volumes/Earth03/flower` or `/Volumes/Luna01/flower`
- WandB integration for ML experiment tracking
- Many scripts use Click for CLI argument parsing
- Geospatial data in UTM projection (typically UTM 17N for BCI)
- Planet imagery: 3m or 4.7m resolution, 4-band or 8-band
- Drone imagery: ~4cm resolution

## Documentation invariants

### Observability pipeline docs

`docs/observability_methodology.md` and `docs/observability_methodology.mmd`
must stay in sync. Any change that adds, renames, or removes a script, data
artifact, or script argument in the observability pipeline must be reflected
in **both** files:

- the narrative + Reproduce block in the `.md`, and
- the corresponding node/edge in the `.mmd` flowchart.

When a data file's upstream producer is not documented, mark it with a `?`
prefix in the Mermaid node (e.g. `"? planet_rgb_dir"`) and a
`**Reproduce:** TODO — ...` placeholder in the doc.
