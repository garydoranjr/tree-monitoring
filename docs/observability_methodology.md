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

  **Reproduce:** TODO — document canonical Planet RGB directory path.
- `scripts/cloud_coverage_plot.py` — Visualizes clear fraction over time as
  a timeline plot showing each scene's clear fraction against the full
  2020–2024 acquisition period.

  **Reproduce:** TODO — depends on `cloud_coverage.csv` path from
  `cloud_coverage_analysis.py`.
- `scripts/plot_cadence_stats.py` — Analyzes the tradeoff between
  clear-fraction threshold and observation cadence using the empirical
  survival function of UDM2 clear fractions.

  **Reproduce:** TODO — depends on `cloud_coverage.csv` path from
  `cloud_coverage_analysis.py`.

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

  **Reproduce:** TODO — document canonical crown-folder and
  embeddings-folder paths.

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

  **Reproduce:** TODO — document the trained model file and embeddings
  folder paths.

## Step 6: Observation Cadence Analysis

Using the visibility matrix, the observational cadence — the average number
of days between clear observations — was characterized as a function of time
and season. A confidence threshold of 0.5 was applied to binarize the
visibility scores. For each point in time, a rolling 10-day window centered
on that date was used to count the number of clear observations per crown,
and the cadence was computed as the window duration divided by the number of
clear observations. Percentile statistics (10th, 50th, and 90th) across all
crowns characterize the distribution of cadence at each time point.

To extract the seasonal pattern, cadence statistics were aggregated across
years 2022–2023 by day of year, pooling data from corresponding calendar
dates across multiple years to produce a single annual cycle of
observational frequency. These years were selected as they provide the most
complete temporal coverage in the Planet imagery archive (356 and 378 images
respectively, with nearly continuous year-round observations).

**Scripts:**
- `scripts/plot_assessed_cadence.py` — Computes and plots cadence over the
  full 2020–2024 time series using a 30-day half-window. Outputs a time
  series plot showing the median cadence and the 10th–90th percentile band
  across crowns.

  **Reproduce:**
  ```bash
  python scripts/plot_assessed_cadence.py \
    results/assessment.npz \
    figs/assessed_cadence.pdf
  ```
- `scripts/plot_avg_assessed_cadence.py` — Aggregates cadence across years
  2022–2023 by day of year to produce the seasonal cycle. Outputs both a
  plot and a compressed NumPy archive of cadence percentiles for downstream
  analysis.

  **Reproduce:**
  ```bash
  python scripts/plot_avg_assessed_cadence.py \
    results/assessment.npz \
    figs/avg_assessed_cadence.pdf \
    results/avg_assessed_cadence.npz
  ```
- `scripts/plot_planet_image_fraction_monthly.py` — Summarizes the fraction
  of images exceeding the 0.5 confidence threshold by calendar month,
  showing median and interquartile range across years.

  **Reproduce:** TODO — input CSV (with a `date` column) is not produced
  by any documented script.

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
across all years of record (2001–2024, encompassing 52,029 10am measurements).
Both east and west sensor records were combined and grouped by day-of-year,
producing an average seasonal pattern with approximately 142 measurements per
day-of-year. This 24-year climatological baseline was compared to the
theoretical clear-sky radiation expected at BCI's coordinates (9.16°N,
79.84°W), computed using the pysolar library's `diffuse_underclear()`
function for solar geometry. The ratio of measured to expected radiation
serves as a proxy for atmospheric transparency: values near 1.0 indicate
clear skies, while lower values indicate cloud-induced attenuation. Both the
radiation fraction and the satellite observation frequency were smoothed with
a matched convolution filter (10-day window matching the satellite analysis
window), and a linear regression was performed to quantify the correspondence
between the two independent seasonal signals.

Note that this comparison uses asymmetric time windows: the satellite
observation frequency derives from 2022–2023 imagery (2 years of high
temporal density), while the ground radiation measurements span 2001–2024
(24 years providing a long-term climatological baseline). Despite this
temporal asymmetry, the strong correlation between the two series validates
that the observed seasonal pattern in satellite observability reflects
genuine atmospheric variation consistent with BCI's multi-decadal climate.

**Scripts:**
- `scripts/illumination.py` — Loads observation count data
  (`windowed_obs_counts_05d.npz` with 10-day window) and Lutz Tower
  radiation CSVs (east and west sensors), selects the 10am hour, computes
  daily averages by day of year, calculates expected clear-sky radiation
  using pysolar, and performs a linear regression between the smoothed
  radiation fraction and the observation rate. Produces a dual-axis plot
  showing both quantities across the annual cycle.

  **Reproduce `figs/solar_radiation_comparison_v2.pdf`:**
  ```bash
  python scripts/illumination.py \
    results/windowed_obs_counts_05d.npz \
    data/radiation/bci_lutz48m_sre_elect.csv \
    data/radiation/bci_lutz48m_srw_elect.csv \
    figs/solar_radiation_comparison_v2.pdf
  ```

**Data:**
- `data/radiation/bci_lutz48m_sre_elect.csv` — East sensor average solar
  radiation at 48m (2001–2024, 781,291 records at 15-minute intervals)
- `data/radiation/bci_lutz48m_srw_elect.csv` — West sensor average solar
  radiation at 48m (2010–2024, 467,510 records at 15-minute intervals)
- `data/radiation/Methods_Electronic_Solar_Radiation_BCI.pdf` — Sensor
  methodology and calibration documentation
- `results/windowed_obs_counts_05d.npz` — Multi-year windowed observation
  counts with 10-day window (2022–2023, 365 DOYs × 4,452 crown-year
  combinations)

## Step 8: Phenological Event Observability Analysis

This analysis builds directly on the crown-level visibility assessment from
Steps 1-5. Where Step 6 characterized general observational cadence across
the annual cycle, Step 8 addresses a specific research question: given a
phenological event of known timing and duration, what is the probability of
observing it in satellite imagery? This quantifies which species and
phenological strategies are detectable via satellite monitoring.

The approach combines two independent datasets: (1) an empirical observation
model built from satellite visibility data, and (2) phenological event
characteristics extracted from BCI's 34-year litter trap record. The
empirical model interpolates observation probability as a function of event
window size and day of year, creating a 2D lookup table (364 window sizes ×
365 days of year → capture probability). The litter trap data provides
independent quantification of when flowering and fruiting events actually
occur and how long they last for each species. By querying the empirical
model with the timing and duration of real phenological events, the analysis
determines the probability of observing an event for any given crown as a
function of species.

The empirical observation model is built by processing the visibility matrix
from Step 5 with a 0.5 confidence threshold to binarize observations, then
resampling across multiple years (2022-2023) to compute observation counts
within rolling time windows for each day of the year. For each combination of
window size (1-364 days) and day of year, the model computes the fraction of
crown-year combinations with at least one clear observation, yielding a
smooth interpolation surface via SciPy's RegularGridInterpolator.

Phenological events are extracted from BCI's 200-trap litter dataset spanning
1987-2024 (`data/BCI_TRAP200_20241002_spcorrected.txt`). This dataset was
provided by project collaborators from the long-running BCI litter trap
monitoring program; citation is pending as the underlying paper has not yet
been published. Weekly collections record the presence and quantity of
flowering and fruiting material (petals, fruits, seeds) that fall from the
canopy into each trap. The data provides a time series of material counts
by species, trap, and date, with each observation representing one week's
accumulated material.

For each trap, species, and year combination, the analysis identifies whether
a reproductive event occurred and characterizes its timing and duration. The
event extraction proceeds as follows:

1. **Event presence**: Count the number of weeks with non-zero trap material
   for that species/trap/year combination. If any non-zero weeks exist, an
   event is recorded; otherwise, that species was not reproductively active
   at that trap in that year.

2. **Event duration**: Calculate as seven days times the number of non-zero
   weeks. This assumes all non-zero weeks within a single year belong to one
   continuous reproductive event, which is reasonable given that most tropical
   tree species at BCI have one reproductive pulse per year, though events
   may span multiple weeks.

3. **Event peak timing**: Apply a moving-window convolution to the weekly
   count time series using a uniform kernel with width equal to the event
   duration (number of non-zero weeks). The convolution smooths the weekly
   signal and identifies the temporal center of reproductive activity. The
   peak is defined as the date where the convolved signal reaches its
   maximum. This approach handles irregular patterns (weeks with varying
   material quantities, or occasional zero-count weeks within an event) by
   identifying the date of maximum sustained activity rather than simply the
   single week with the highest count.

The result is a table of individual phenological events, each characterized
by trap location, species identity, year, event duration (days), and event
peak timing (date). This characterizes the actual timing and duration of
reproductive events as they occur in nature, providing ground-truth
phenological patterns independent of satellite observation constraints.

A parallel analysis is performed for deciduousness (leaf-drop) events
extracted from a per-crown leaf-cover time series
(`data/df_LeafCoverTimeSeries_byTags_all_2024.csv`). This dataset was
provided by project collaborators and contains per-tag daily predictions of
percent canopy leaf cover and percent exposed branch for tagged crowns at
BCI; citation is pending as the underlying paper has not yet been published.
For each crown, deciduousness event length is defined as the number of
observations where predicted branch exposure exceeds 40%, and the event
peak is identified by a moving-window convolution over the branch-exposure
signal using a uniform kernel equal to the event length (analogous to the
trap event peak logic). The result, `results/decid_summary.csv`, is a
per-crown table of (tag, species, event length, peak date) that serves as
the deciduousness counterpart to the litter trap event table and is
consumed by the deciduousness-observability scripts described below.

For each extracted trap event, the empirical model is queried at
(event_peak_doy, event_length) to obtain the predicted capture probability
for that specific event. These probabilities are then aggregated by species,
computing the mean observation fraction across all recorded events for each
species. The result quantifies species-level observability: the probability
that a flowering or fruiting event will be detected in satellite imagery
given that species' characteristic phenological strategy (typical timing and
duration).

Note: Prior work (documented in `data/BCI_ModelSelection_Wavelets_20210106.pdf`)
fitted von Mises distributions to model seasonal phenological patterns from
litter trap data. While the von Mises approach was explored in supplementary
analyses, the final methodology uses the raw trap event data directly rather
than parametric models, as the empirical events provide more direct
characterization of actual phenological timing and duration.

**Scripts:**
- `scripts/get_annual_trap_data.py` — Preprocesses raw BCI litter trap data
  (`data/BCI_TRAP200_20241002_spcorrected.txt`), filtering to flowering or
  fruiting material codes (selected via the `-f/--fruit` flag; default is
  flower), aggregating weekly counts by species/trap/date, and structuring
  into annual time series. Outputs a compressed NumPy archive with
  per-species trap counts — `results/sp_flower_counts_annual.npz` without
  the flag, `results/sp_fruit_counts_annual.npz` with it.

  **Reproduce:**
  ```bash
  python scripts/get_annual_trap_data.py \
    data/BCI_TRAP200_20241002_spcorrected.txt \
    results/sp_flower_counts_annual.npz

  python scripts/get_annual_trap_data.py -f \
    data/BCI_TRAP200_20241002_spcorrected.txt \
    results/sp_fruit_counts_annual.npz
  ```
- `scripts/windowed_obs_counts.py` — Processes the visibility assessment
  matrix from Step 5, applies 0.5 confidence threshold to binarize
  observations, and computes rolling window observation counts for 365 days
  of year using multi-year resampling (2022-2023). For each sample date,
  counts how many satellite images fell within a specified half-window where
  each crown was visible. Outputs `windowed_obs_counts_*.npz` containing dates,
  per-crown counts, and window size (the numeric suffix on the filename tracks
  the `--halfwidth` CLI argument, so the default `halfwidth=5` produces
  `windowed_obs_counts_05d.npz` with `window_size=10.0`).

  **Reproduce:**
  ```bash
  python scripts/windowed_obs_counts.py \
    results/assessment.npz \
    results/windowed_obs_counts_05d.npz
  ```
- `scripts/fit_empirical_count_models.py` — Builds the empirical
  observability model by computing observation rates across 364 window sizes
  and 365 days of year. For each (window_size, day_of_year) cell, calculates
  the fraction of crown-year combinations with at least one observation.
  Creates 2D interpolation grid using SciPy's RegularGridInterpolator for
  smooth lookup. Outputs `empirical_model.npz` with dates, window_sizes, and
  probability matrix.

  **Reproduce:**
  ```bash
  python scripts/fit_empirical_count_models.py \
    results/assessment.npz \
    results/empirical_model.npz
  ```
- `scripts/individual_trap_analysis.py` — Extracts individual
  flowering/fruiting events from annual trap count arrays to characterize
  phenological event timing and duration. For each species/trap/year
  combination, identifies periods of non-zero trap counts. Event peak timing
  is determined by finding the date of maximum convolved counts, and event
  length is calculated as 7 days × number of consecutive non-zero weeks.
  Outputs CSV with per-event records including trap, species, year,
  event_length, and event_peak.

  **Reproduce:**
  ```bash
  python scripts/individual_trap_analysis.py \
    results/sp_flower_counts_annual.npz \
    results/sp_flower_counts_annual_stats.csv

  python scripts/individual_trap_analysis.py \
    results/sp_fruit_counts_annual.npz \
    results/sp_fruit_counts_annual_stats.csv
  ```
- `scripts/event_summary_stats.py` — Determines species-level phenological
  observability by querying the empirical model for each trap event. Loads
  the `EmpiricalCountModel` (exposing `load()` and
  `capture_prob(doy, durations)`) and event data, then for each event queries
  the model at (event_length, event_peak_doy) to get predicted capture
  probability given that event's timing and duration. Aggregates by species
  to compute mean observation fraction (the probability of observing an event
  for that species) and event count. Outputs a CSV with species-level
  observability statistics; used for the flowering, fruiting, and
  deciduousness pipelines.

  **Reproduce:**
  ```bash
  python scripts/event_summary_stats.py \
    results/empirical_model.npz \
    results/sp_flower_counts_annual_stats.csv \
    results/flower_summary_stats.csv

  python scripts/event_summary_stats.py \
    results/empirical_model.npz \
    results/sp_fruit_counts_annual_stats.csv \
    results/fruit_summary_stats.csv

  python scripts/event_summary_stats.py \
    results/empirical_model.npz \
    results/decid_summary.csv \
    results/decid_summary_stats.csv
  ```
- `scripts/plot_trap_summary.py` — Generates multi-page PDF visualization
  showing phenological event observability by species. Creates species-level
  heatmaps showing capture probability as a function of event duration and
  timing (seasonally aligned to September start), overlays actual event
  points, and produces histograms of capture probabilities. Consumes the
  per-event CSV schema (`species`, `event_length`, `event_peak`) shared by
  the flowering, fruiting, and deciduousness pipelines.

  **Reproduce:**
  ```bash
  python scripts/plot_trap_summary.py \
    results/empirical_model.npz \
    results/sp_flower_counts_annual_stats.csv \
    figs/sp_flower_counts_annual_stats.pdf

  python scripts/plot_trap_summary.py \
    results/empirical_model.npz \
    results/sp_fruit_counts_annual_stats.csv \
    figs/sp_fruit_counts_annual_stats.pdf

  python scripts/plot_trap_summary.py \
    results/empirical_model.npz \
    results/decid_summary.csv \
    figs/decid_summary.pdf
  ```
- `scripts/individual_decid_analysis.py` — Extracts individual deciduousness
  events from the per-crown leaf-cover time series
  (`data/df_LeafCoverTimeSeries_byTags_all_2024.csv`). For each tagged
  crown, counts the number of observations with predicted branch exposure
  greater than 40% to define event length, and locates the event peak date
  via uniform-kernel convolution of the branch-exposure signal (kernel
  width = event length, with wrap-around handling for seasonal continuity).
  Outputs `results/decid_summary.csv` with per-crown records (tag, species,
  event_length, event_peak) — the same schema emitted by
  `scripts/individual_trap_analysis.py`, so the deciduousness events can be
  fed directly into `event_summary_stats.py` and `plot_trap_summary.py`
  without any column-rename shim.

  **Reproduce:**
  ```bash
  python scripts/individual_decid_analysis.py \
    data/df_LeafCoverTimeSeries_byTags_all_2024.csv \
    results/decid_summary.csv
  ```
  (Downstream `event_summary_stats.py` and `plot_trap_summary.py`
  invocations for the deciduousness path are listed under those scripts
  above, alongside the flower and fruit variants.)
**Data:**
- `results/assessment.npz` — Visibility matrix from Step 5 (crowns × images ×
  confidence scores)
- `data/BCI_TRAP200_20241002_spcorrected.txt` — Raw litter trap data
  (1987-2024, weekly collections, 200 traps, flowering/fruiting material by
  species). Provided by project collaborators from the BCI litter trap
  monitoring program; citation pending as the underlying paper has not yet
  been published. Serves as the input to the fruiting/flowering event
  analysis pipeline (`scripts/get_annual_trap_data.py` →
  `scripts/individual_trap_analysis.py`).
- `results/sp_flower_counts_annual.npz` — Preprocessed annual flowering trap
  counts (species × traps × years × 52 weeks)
- `results/sp_fruit_counts_annual.npz` — Preprocessed annual fruiting trap
  counts
- `results/windowed_obs_counts_05d.npz` — Multi-year windowed observation counts
  with 10-day window (365 DOYs × crown-year combinations)
- `results/empirical_model.npz` — 2D interpolation model (364 window sizes ×
  365 DOYs → capture probability)
- `results/sp_flower_counts_annual_stats.csv` — Per-event flowering records
  (trap, species, year, event_length, event_peak)
- `results/sp_fruit_counts_annual_stats.csv` — Per-event fruiting records; same
  schema as `sp_flower_counts_annual_stats.csv`
- `data/df_LeafCoverTimeSeries_byTags_all_2024.csv` — Per-crown daily
  predicted leaf-cover and branch-exposure time series for tagged BCI
  crowns. Provided by project collaborators; citation pending as the
  underlying paper has not yet been published. Serves as the input to
  `scripts/individual_decid_analysis.py` and is the primary source of
  deciduousness event timing and duration information.
- `results/decid_summary.csv` — Per-crown deciduousness event records
  (tag, species, event_length, event_peak) extracted from the leaf-cover
  time series above.
- `results/flower_summary_stats.csv` — Species-level flowering-event
  observability statistics (species, frac_obs, n)
- `results/fruit_summary_stats.csv` — Species-level fruiting-event
  observability statistics (species, frac_obs, n)
- `results/decid_summary_stats.csv` — Species-level deciduousness-event
  observability statistics (species, frac_obs, n)
- `figs/sp_flower_counts_annual_stats.pdf` — Visualization of model vs. flowering
  events per species
- `figs/sp_fruit_counts_annual_stats.pdf` — Visualization of model vs. fruiting
  events per species
- `figs/decid_summary.pdf` — Visualization of model vs.
  deciduousness events per species
- `data/BCI_ModelSelection_Wavelets_20210106.pdf` — Von Mises model
  documentation and methodology reference
