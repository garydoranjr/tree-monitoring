# 50ha Planet training chips (drone labels → Planet scenes)

The Mask R-CNN / SegFormer training set for the BCI 50ha plot: 4-band
Planet cutouts at 0.75 m paired with a binary crown mask derived from
drone-imagery crown classifications. Built by
[`scripts/apply_drone_labels_coreg.py`](../scripts/apply_drone_labels_coreg.py),
then reduced by a human vetting pass. The AVA-plot analogue, including the
reasons that build needs a margin-then-crop workflow, is in
[ava_planet_cutouts.md](ava_planet_cutouts.md).

This pipeline is not in the `Snakefile` — it depends on drone orthomosaics
and Labelbox exports that are not reproducible from within the repo. The
command lines below are the record.

## Directory-name tokens

Sets are named `<YYYYMMDD>_<scope>_label_application_x4_coreg_4band_stretch_stats[_filt|_curated]`:

| Token | Meaning |
|---|---|
| `YYYYMMDD` | build date |
| `full` / `globus` | which label set was applied (see *Inputs*) |
| `x4` | `--resize 4`: the 3 m Planet cutout is upsampled to 0.75 m, 367 × 205 → 1468 × 820 px |
| `coreg` | per-scene AROSICS drone↔Planet shift applied to the label raster before rasterizing (`ws=(200,200)`, `max_shift=10`, matched on Red in both images) |
| `4band` | `--bands 4`: chips come from `*4band.tif` and are written as 4-band uint16 GeoTIFFs (Blue, Green, Red, NIR) |
| `stretch` | cloud-aware per-band 2/98 percentile stretch for the QA PNG — percentiles from OCM-clear pixels only, applied to all non-nodata pixels, so clouds saturate white |
| `stats` | per-scene noise/quality metrics written into `coreg_log.json` (`noise_mad_sigma`, `noise_mad_cv`, `band_corr_mean`, `band_decorrelation`) |
| `_filt` | programmatic threshold filter — [`scripts/filter_label_application.py`](../scripts/filter_label_application.py) with [`config/filter_label_application.yml`](../config/filter_label_application.yml) |
| `_curated` | human Labelbox `Quality` vetting pass — [`copy_good_planet_vetting.py`](../copy_good_planet_vetting.py) |

`stretch` and `stats` only exist on the `--bands 4` path; both
`render_rgb_from_4band()` and `compute_noise_metrics()` are on that branch
of `create_mask()`.

## Per-chip outputs

All files for a scene share the stem `<planet_scene>_4band`:

| File | Contents |
|---|---|
| `.tif` | 4-band uint16 GeoTIFF at 0.75 m, EPSG:32617 — **the training image** |
| `.mask.png` | binary uint8 (0/255) crown mask — **the label** |
| `.png` | RGB QA render with the cloud-aware stretch |
| `.drone.png` | shift-corrected drone ortho on an exact 2× subdivision of the chip grid (`--drone-scale 2`) |
| `.ocm.png` | RGBA cloud overlay from the OmniCloudMask mask |
| `coreg_log.json` | one record per (label, scene) pair, at the set root |

A scene appears only if AROSICS converged; failures are recorded in
`coreg_log.json` with `coreg_ok: false` and no chip files.
`crown_filter_by_ocm()` drops any connected crown component that touches a
non-clear OCM pixel, so partly clouded crowns are not labelled.

## Inputs

Two drone sources, disjoint in time, each with its own classification
rasters under `/Volumes/Earth03/flower/results/classifications/`. Both
classification products are 2-band float32
`(flowering_probability, deciduous_probability)` in EPSG:32617; the default
`--mode both` takes their union, `1 - Π(1 - p)`.

| | 2020–2023 | 2024–2026 (`globus`) |
|---|---|---|
| Orthos | `stri/24782016/BCI_50ha_timeseries_local_alignment/` | `stri/globus/RGB/` |
| Ortho format | 4-band uint8, ~23425 × 12697, ~1.2 GB | 4-band uint16 (R, G, B, Alpha), ~25000 × 21000, ~4 GB |
| Ortho footprint | 1056 × 572 m | 1199 × 1022 m |
| Labels | `*_local_classifications.tif` | `*_M3M_aligned_global_RGB_classifications.tif` |
| Dates | 2020-01-24 → 2023-10-24 | 2024-03-06 → 2026-01-20, 96 flights |
| Provenance | — | synced by `scripts/globus_https_sync.py`, classified by `run_crown_pipeline.sh config/pipeline_globus.sh` (see [NOTES.md](../NOTES.md)) |

`find_drone()` matches a label to its ortho by filename prefix, so
`BCI_50ha_2024_03_06_M3M_aligned_global_RGB_classifications.tif` resolves to
`BCI_50ha_2024_03_06_M3M_aligned_global_RGB.tif`.

**Label coverage differs between the two halves.** The 2020–2023 orthos
(1056 × 572 m) are *smaller* than the 1101 × 615 m chip, so those chips have
unlabelled margins that training sees as background. The globus orthos are
larger than the chip, so those chips are labelled edge to edge.

Planet side: `planet_clipped/4band/<year>/` cutouts
(`config/clip_50ha_plot.yml`) and their OmniCloudMask masks in
`planet_clipped/ocm/<year>/`. Each label date is paired with every Planet
scene within `--timewindow 2` days.

## Build

### 1. Apply labels (2020–2023, `20260608` set)

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/apply_drone_labels_coreg.py \
  /Volumes/Earth03/flower/results/classifications/*_local_classifications.tif \
  /Volumes/Earth03/flower/stri/24782016/BCI_50ha_timeseries_local_alignment \
  /Volumes/Earth03/flower/planet_clipped/4band \
  /Volumes/Earth03/flower/20260608_full_label_application_x4_coreg_4band_stretch_stats \
  -b 4 -r 4 -t 2 -d 2 \
  -k /Volumes/Earth03/flower/planet_clipped/ocm
```

323 (label, scene) pairs, 131 coregistered.

### 2. Apply labels (2024–2026 globus, `20260915` set)

Identical flags; only the label glob and the ortho directory change.

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/apply_drone_labels_coreg.py \
  /Volumes/Earth03/flower/results/classifications/*M3M_aligned_global_RGB_classifications.tif \
  /Volumes/Earth03/flower/stri/globus/RGB \
  /Volumes/Earth03/flower/planet_clipped/4band \
  /Volumes/Earth03/flower/20260915_globus_label_application_x4_coreg_4band_stretch_stats \
  -b 4 -r 4 -t 2 -d 2 \
  -k /Volumes/Earth03/flower/planet_clipped/ocm
```

96 label dates → 370 pairs (7 dates have no Planet scene within ±2 days).

`KMP_DUPLICATE_LIB_OK=TRUE` works around a torch OpenMP duplicate-runtime
abort on this machine.

**Memory.** `process_label()` holds one drone ortho for the duration of a
label date's scenes, so peak RSS is roughly the ortho size (~4.5 GB for
globus) plus the drone reprojection in `generate_drone_png()` (~2.5 GB).
On a 16 GB machine this runs but leaves little headroom; drop `-d 2` to
skip the drone PNGs if it thrashes.

### 3. Vet and assemble the curated set

Chips are rated by hand in Labelbox on a `Quality` radio
(`Poor` < `Fair` < `Good`), keyed on `data_row.external_id` = `<stem>.png`.
Exports live in `labels/` (gitignored).

```bash
DST=/Volumes/Earth03/flower/20260915_full_label_extended_x4_coreg_4band_stretch_stats_curated

# the already-vetted 2020-2023 chips, with their coreg_log.json
rsync -a /Volumes/Earth03/flower/20260706_full_label_application_x4_coreg_4band_stretch_stats_curated/ "$DST"/

# the newly vetted globus chips, merged into the log copied above
python copy_good_planet_vetting.py \
  --ndjson labels/20260915_planet_vetting_globus.ndjson \
  --src /Volumes/Earth03/flower/20260915_globus_label_application_x4_coreg_4band_stretch_stats \
  --dst "$DST"
```

`copy_good_planet_vetting.py` copies every sibling file sharing an accepted
stem and unions the two `coreg_log.json` files on `(scene, label)`, so the
merged log carries the provenance of both builds and re-runs are idempotent.
`--min-quality` defaults to `Good`; `--dry-run` lists without copying.

The earlier 50ha curated set was the same step with
`labels/20260706_planet_vetting.ndjson` (56 of 131 Good; 25 Fair, 50 Poor).

### Alternative: the programmatic filter

Where no vetting export exists, `scripts/filter_label_application.py`
applies thresholds from `config/filter_label_application.yml`
(`exact_sizes`, `min_clear_fraction: 0.10`, `min_band_corr_mean: 0.3`,
`require_coreg_ok: true`) and rewrites a filtered `coreg_log.json`:

```bash
python scripts/filter_label_application.py <src_dir> <dst_dir> config/filter_label_application.yml
```

`copy_4band_filtered.py` mirrors an RGB `_filt` selection onto the matching
4-band set.

## Consumers

Both loaders glob `*.mask.png` and pair it with the sibling `.tif`:

- `scripts/train_planet_image_segformer_4b.py` (`PlanetSegmentationDataset4B`)
- `scripts/train_planet_image_maskrcnn.py` (`PlanetMaskRCNNDataset`, plus the
  `.ocm.png` sidecar for `--ocm-masks` cloud-aware loss)

`get_split(..., 'left'|'right')` takes a 512 × 512 window abutting the
horizontal centre of the chip, which is how train and validation are kept
geographically disjoint within a scene.
