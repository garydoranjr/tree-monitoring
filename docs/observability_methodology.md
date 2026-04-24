# Canopy Observability Methodology

## Overview

This document describes the methodology used to characterize the
observability of the forest canopy from Planet satellite imagery as a
function of season, and to validate the resulting seasonal pattern against
independent ground-based measurements.

## Step 1: Scene-Level Cloud Screening with UDM2

As a preliminary analysis, the clear-sky fraction of each Planet scene was
computed from the Usable Data Mask (UDM2) products delivered with every
Planet image. The UDM2 clear band encodes the fraction of each pixel
classified as cloud-free by Planet's onboard quality pipeline. For each
scene, the average clear fraction across the 50-hectare study area was
recorded. However, this metric proved too coarse for crown-level
observability assessment: the UDM2 mask operates at the scene level and does
not capture subtler degradation such as thin haze, atmospheric distortion,
or partial cloud shadows that can render individual tree crowns
unidentifiable even when the scene is nominally classified as clear. This
motivated the development of a learned, crown-level visibility classifier.

**Scripts:**
- `scripts/cloud_coverage_analysis.py` — Computes average UDM2 clear
  fraction per scene by matching each RGB image to its corresponding UDM2
  mask file and averaging the first band (clear pixel indicator).
- `scripts/cloud_coverage_plot.py` — Visualizes clear fraction over time as
  a timeline plot showing each scene's clear fraction against the full
  2020–2024 acquisition period.
- `scripts/plot_cadence_stats.py` — Analyzes the tradeoff between
  clear-fraction threshold and observation cadence using the empirical
  survival function of UDM2 clear fractions.

## Step 2: Manual Labeling of Crown-Level Image Quality

To build training data for a crown-level visibility classifier, image
patches centered on a single reference tree crown (tag 7633) were extracted
from each Planet scene across the full time series. A human annotator
manually labeled 500 patches, classifying each as:

- **good** (n=148) — canopy clearly visible and identifiable
- **poor** (n=329) — obscured by clouds, haze, or atmospheric distortion
- **unsure** (n=23) — ambiguous quality

By labeling many images of the same crown, the annotator held spatial
context constant and focused purely on atmospheric and image quality
variation across dates.

**Label file:**
- `luis/20240723b_individual.csv` — 500 labeled patches (148 good, 329 poor, 23 unsure)

The CSV contains columns: `tag` (crown ID), `image_id` (Planet scene
identifier), and `label` (good/poor/unsure).

## Step 3: Feature Extraction with DINOv2

For each labeled image patch, a feature representation was extracted using
DINOv2 (`dinov2_vits14`), a self-supervised vision transformer pretrained on
natural images. Each Planet image crop was converted to RGB, resized to 244
pixels on the shorter side, center-cropped to 224×224 pixels, and normalized
before being passed through the model to produce a 384-dimensional embedding
vector. These embeddings capture rich semantic and textural properties of
each image that are informative of atmospheric quality beyond what simple
spectral indices can represent.

**Scripts:**
- `scripts/image_embedding.py` — Loads the DINOv2 model via PyTorch Hub,
  iterates over crown image directories, and computes embeddings for each
  image. Embeddings are saved per crown as files containing image IDs and
  their corresponding 384-dimensional vectors.

## Step 4: Visibility Classifier Training

The "unsure" labels were excluded, yielding 477 binary-labeled samples
(148 good, 329 poor) with a 2.2:1 class imbalance favoring poor images.
A Random Forest classifier with 100 estimators was trained on the
DINOv2 embeddings to predict image quality. As a baseline, NDVI — computed
from the red and near-infrared bands of the same images — was also evaluated
as a quality predictor via ROC curve analysis.

**Scripts:**
- `luis/RandomForest_train.py` — Loads DINOv2 embeddings and NDVI scores for
  crown 7633, reads manual labels, filters to binary good/poor
  classification, trains a Random Forest classifier, and generates ROC
  curves comparing the DINOv2-based classifier against the NDVI baseline.
  The trained model is serialized using the skops library.

## Step 5: Plot-Wide Visibility Assessment

The trained classifier was deployed across the entire 50-hectare plot. For
each monitored tree crown, DINOv2 embeddings were extracted from all
available Planet scenes, and the classifier was applied to produce a
visibility confidence score (the predicted probability of the "good" class).
The result is a visibility matrix of dimensions (crowns × images), where
each entry represents the model's estimated probability that a given crown
is clearly observable in a given scene.

**Scripts:**
- `scripts/visibility_assessment.py` — Loads the trained model and iterates
  over per-crown embedding files. For each crown, applies
  `predict_proba()` to obtain class-1 (good) confidence scores. Assembles
  results into a compressed NumPy archive containing the tag array, file
  array, and values matrix.

## Step 6: Observation Cadence Analysis

Using the visibility matrix, the observational cadence — the average number
of days between clear observations — was characterized as a function of time
and season. A confidence threshold of 0.5 was applied to binarize the
visibility scores. For each point in time, a rolling 60-day window centered
on that date was used to count the number of clear observations per crown,
and the cadence was computed as the window duration divided by the number of
clear observations. Percentile statistics (10th, 50th, and 90th) across all
crowns characterize the distribution of cadence at each time point.

To extract the seasonal pattern, cadence statistics were aggregated across
years 2020–2022 by day of year, pooling data from corresponding calendar
dates across multiple years to produce a single annual cycle of
observational frequency.

**Scripts:**
- `scripts/plot_assessed_cadence.py` — Computes and plots cadence over the
  full 2020–2024 time series using a 30-day half-window. Outputs a time
  series plot showing the median cadence and the 10th–90th percentile band
  across crowns.
- `scripts/plot_avg_assessed_cadence.py` — Aggregates cadence across years
  2020–2022 by day of year to produce the seasonal cycle. Outputs both a
  plot and a compressed NumPy archive of cadence percentiles for downstream
  analysis.
- `scripts/plot_planet_image_fraction_monthly.py` — Summarizes the fraction
  of images exceeding the 0.5 confidence threshold by calendar month,
  showing median and interquartile range across years.

## Step 7: Validation Against Ground-Based Solar Radiation

To independently validate that the seasonal pattern in canopy observability
reflects genuine atmospheric variation rather than an artifact of satellite
tasking or the classifier, the observation frequency was compared against
ground-based solar radiation measurements from the Lutz Tower on BCI. The
Lutz Tower pyranometers (LiCor Li200 sensors at 42m, later Kipp & Zonen
instruments at 48m) recorded incoming solar radiation at 15-minute intervals
from 2001 through March 2024.

For each day of the year, the measured solar radiation at 10am local time —
selected to approximate the Planet satellite overpass window — was averaged
across all years of record. This was compared to the theoretical clear-sky
radiation expected at BCI's coordinates (9.16°N, 79.84°W), computed using
the pysolar library's `diffuse_underclear()` function for solar geometry.
The ratio of measured to expected radiation serves as a proxy for
atmospheric transparency: values near 1.0 indicate clear skies, while lower
values indicate cloud-induced attenuation. Both the radiation fraction and
the satellite observation frequency were smoothed with a matched convolution
filter, and a linear regression was performed to quantify the correspondence
between the two independent seasonal signals.

**Scripts:**
- `scripts/illumination.py` — Loads observation count data and Lutz Tower
  radiation CSVs (east and west sensors), selects the 10am hour, computes
  daily averages by day of year, calculates expected clear-sky radiation
  using pysolar, and performs a linear regression between the smoothed
  radiation fraction and the observation rate. Produces a dual-axis plot
  showing both quantities across the annual cycle.

**Data:**
- `data/radiation/bci_lutz48m_sre_elect.csv` — East sensor average solar
  radiation at 48m (2001–2024)
- `data/radiation/bci_lutz48m_srw_elect.csv` — West sensor average solar
  radiation at 48m (2010–2024)
- `data/radiation/Methods_Electronic_Solar_Radiation_BCI.pdf` — Sensor
  methodology and calibration documentation
