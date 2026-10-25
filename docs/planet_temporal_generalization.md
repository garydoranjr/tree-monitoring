# Temporal generalization of the Planet Mask R-CNN, and applying it between drone flights

Scripts:
- [`scripts/make_temporal_folds.py`](../scripts/make_temporal_folds.py)
- [`scripts/train_planet_image_maskrcnn.py`](../scripts/train_planet_image_maskrcnn.py), with `--folds/--fold` and `--save-epochs`
- [`scripts/eval_temporal_cv.py`](../scripts/eval_temporal_cv.py)
- [`scripts/plot_temporal_cv.py`](../scripts/plot_temporal_cv.py)
- [`scripts/plot_temporal_cv_detail.py`](../scripts/plot_temporal_cv_detail.py)
- [`scripts/coreg_planet_to_nearest_drone.py`](../scripts/coreg_planet_to_nearest_drone.py)
- [`scripts/apply_planet_maskrcnn.py`](../scripts/apply_planet_maskrcnn.py)
- [`scripts/plot_crown_species_comparison.py`](../scripts/plot_crown_species_comparison.py)
- [`scripts/make_crown_timeseries_html.py`](../scripts/make_crown_timeseries_html.py), with the page template
  [`scripts/crown_timeseries_template.html`](../scripts/crown_timeseries_template.html)
- the Slurm wrappers `scripts/slurm/planet_temporal_{cv,eval,final}.sh`,
  `scripts/slurm/planet_coreg_nearest.sh` and `scripts/slurm/planet_apply.sh`

Settings: [`config/temporal_cv_20261012.sh`](../config/temporal_cv_20261012.sh).
Folds: [`config/temporal_folds_20261012.csv`](../config/temporal_folds_20261012.csv).
Chips: `20261012_full_label_extended2_mkl_x4_coreg_4band_stretch_stats_curated`
(142 chips; see [planet_training_chips_50ha.md](planet_training_chips_50ha.md)).
A MODIS-calibrated copy of the same chips with a fixed input stretch
(`--fixed-stretch`) is set up in
[`config/temporal_cv_20261012_cal.sh`](../config/temporal_cv_20261012_cal.sh); see
[planet_radiometric_calibration.md](planet_radiometric_calibration.md#classifier-input-stretch).

This pipeline is not in the `Snakefile`. It depends on the chip set, the drone
orthos and GPU training on gattaca2, none of which can be reproduced from
within the repo. The command lines below are the record.

## Question

The model will be used for **temporal interpolation over the same plot**:
1. It is trained on Planet chips within ±2 days of a drone flight, labelled
   from that flight's crown classifications.
2. It is then applied to the Planet scenes in between, which are more than a
   few days from any flight but no more than about a month from a training
   chip.

How well does it do on such a scene?

Every earlier result used the east/west split (`get_split`, 'left' and
'right'). That split trains on the left 512 × 512 window of every chip and tests
on the right window. **Every date is in both train and test**, so the split
measures transfer across ~400 m of forest on dates the model has seen. It says
nothing about new dates. The opposite holds for deployment: the ground is the
same, and the date is new.

## Procedure in brief

1. **Temporal CV.** Hold out contiguous blocks of dates, train on the rest, and
   score the held-out chips. Record each held-out chip's distance in days to
   the nearest training chip, and break the error down by that distance.
2. **Pick the training length.** Choose the epoch with the best pooled
   held-out score.
3. **Train the final model** on all 142 chips for that many epochs.
4. **Apply it.** Coregister each Planet scene away from the drone dates to the
   nearest drone ortho, then predict per scene and per crown, flagging scenes
   more than 30 days from any training chip.

## Fold design

`make_temporal_folds.py` builds the folds in four steps:

1. **Read the dates.** Each chip's Planet date comes from its stem, and its
   drone flight date(s) from the `coreg_ok` records of `coreg_log.json`.
2. **Group by flight.** Chips that share a flight share a label, so they are
   grouped (union-find) and never split across folds.
3. **Cut blocks.** The groups are cut, in time order, into contiguous **blocks
   spanning at most 14 days** of Planet dates (`--block-days`).
4. **Assign blocks to folds.** The blocks are dealt, in time order, to
   **6 folds** (`--n-folds`). Each block goes to the fold with the fewest chips
   so far, excluding the fold of the previous block. So:
   - folds are balanced;
   - neighbouring blocks are never held out together;
   - every held-out gap has training data on both sides unless the record
     itself has a gap there.

For each chip, `days_to_nearest_train` is the distance in days from its Planet
date to the nearest Planet date of a chip in another fold. That is its distance
to training data when its fold is held out.

**Why 14-day rather than 30-day blocks.** Counting every Planet scene from
January 2021 to January 2026 (before cloud filtering), the distance to the
nearest training chip is distributed as follows:

| Days to nearest training chip | 0–3 | 4–7 | 8–14 | 15–30 | >30 |
|---|---|---|---|---|---|
| Planet scenes | 545 | 242 | 327 | 401 | 202 |
| Held-out chips, 14-day blocks | 7 | 27 | 51 | 35 | 22 |
| Held-out chips, 30-day blocks | 0 | 3 | 25 | 52 | 62 |

With 30-day blocks the CV almost never sees a chip 4–7 days from training data,
yet that is a common case in deployment. With 14-day blocks, every bin from 4
to 30 days has a usable sample.

**The resulting folds** (instances are the left- plus right-window crowns at
`--min-instance-size 64`):

| Fold | Blocks | Chips | Instances | Median days to train | Max | Chips > 30 d |
|---|---|---|---|---|---|---|
| 0 | 6 | 24 | 2,229 | 10 | 55 | 1 |
| 1 | 5 | 25 | 1,791 | 30 | 58 | 11 |
| 2 | 6 | 23 | 1,885 | 12 | 75 | 2 |
| 3 | 9 | 24 | 1,404 | 11 | 75 | 5 |
| 4 | 10 | 24 | 1,461 | 12 | 92 | 2 |
| 5 | 7 | 22 | 1,337 | 13 | 31 | 1 |

This gives 43 blocks, 142 chips and 62 flights in total. Every flight is in
exactly one fold, the longest block spans 13 days, and no two adjacent blocks
share a fold. Fold 1 holds most of the sparse 2021 flights, so it is the fold
furthest from its training data.

**What a fold trains and tests on.** In fold mode both the left and right
512 × 512 windows of every chip are samples, for training and testing alike.
Whole chips are not used, for two reasons:
- torchvision's detection transform rescales inputs to `min_size=800`, so a
  1468 × 820 chip would be seen at a different scale from the 512 windows;
- the 2020–23 phantom chips have unlabelled margins.

A held-out chip shows the same trees as the training chips on other dates,
which is the deployment situation.

## Procedure

### 1. Make the folds (Mac, done)

```bash
F=/Volumes/Earth03/flower
python scripts/make_temporal_folds.py \
  $F/20261012_full_label_extended2_mkl_x4_coreg_4band_stretch_stats_curated \
  config/temporal_folds_20261012.csv --block-days 14 --n-folds 6
```

The CSV is committed (force-added past the `*.csv` ignore rule), so the
cluster trains on exactly these folds. It has these columns:
- `chip`, `planet_date`, `drone_dates`
- `block`, `block_start`, `block_end`
- `fold`, `days_to_nearest_train`, `n_instances`

### 2. Copy the inputs to gattaca2 (Mac, done)

Steps 3–7 all run on gattaca2 and read from `/scratch/tree-monitoring`. The
drone orthos (`stri/`) and the crown map are already there. Copy the chip set
and the clipped Planet scenes with their OmniCloudMask masks:

```bash
F=/Volumes/Earth03/flower
rsync -a $F/20261012_full_label_extended2_mkl_x4_coreg_4band_stretch_stats_curated \
  gattaca2:/scratch/tree-monitoring/
rsync -a $F/planet_clipped/4band $F/planet_clipped/ocm \
  gattaca2:/scratch/tree-monitoring/planet_clipped/   # 1.2 GB + 0.26 GB
```

The jobs stage the chip set onto `/local/gdoran_flower` themselves
(`scripts/slurm/stage_chips.sh`) and verify it by checksum.

### 3. Temporal CV training (gattaca2)

On gattaca2, from the repo root at the commit that adds this doc:

```bash
mkdir -p logs
source config/temporal_cv_20261012.sh
CV=$(sbatch --parsable scripts/slurm/planet_temporal_cv.sh)
sbatch --dependency=afterok:$CV scripts/slurm/planet_temporal_eval.sh
```

Each array task trains one fold:
- settings `--ocm-masks --min-instance-size 64 --lr 1e-4`, flat LR, batch 4;
- 40 epochs, with checkpoints at epochs 1, 2, 3, 5, 8, 12, 17, 25 and 40.

A fold trains on about 118 chips × 2 windows. On an A100 with chips staged to
`/local`, a fold took 30–35 min (about 50 s/epoch). With the 2-GPU-per-user
limit (see [gattaca2_cluster.md](gattaca2_cluster.md)) the array runs one or
two folds at a time, so the six folds take about 2–3 h.

40 epochs is enough: every earlier run peaked between epoch 4 and 12 and then
overfit, and each epoch here has twice as many samples. If the pooled curve
is still rising at 40, extend `TEMPORAL_EPOCH_GRID` and rerun.

The eval job writes `$TEMPORAL_RUNS/temporal_cv.json` and the per-window
`temporal_cv_windows.csv` (every epoch). It prints three tables:
- the pooled metrics per epoch;
- the selected epoch, picked by pooled `test_map/map_50`;
- the breakdowns by distance bin and by fold.

**Pooling.** Metrics are pooled over all folds: one IoU, mAP and TP/FP/FN
accumulator across every held-out window. They are not averages of per-fold
scores, because a fold holds only 22–25 chips.

### 4. Plot (gattaca2)

```bash
source config/temporal_cv_20261012.sh
python scripts/plot_temporal_cv.py \
  $TEMPORAL_RUNS/temporal_cv.json $TEMPORAL_RUNS/temporal_cv.png
```

`plot_temporal_cv_detail.py` draws four more figures that show IoU, precision
and recall next to mAP@50. It writes `temporal_cv_metrics`, `temporal_cv_bins`,
`temporal_cv_chips` and `temporal_cv_season` (PNG and PDF):

```bash
python scripts/plot_temporal_cv_detail.py $TEMPORAL_RUNS/temporal_cv.json \
  $TEMPORAL_RUNS/temporal_cv_windows.csv $TEMPORAL_RUNS/temporal_cv
```

The per-chip values sum the left and right windows of each chip. The script
checks that the per-chip TP/FP/FN add up to the pooled counts. The figures
were made on the Mac from the Earth03 copy (`/Volumes/Earth03/flower/202610_temporal/`).

### 5. Train the final model (gattaca2)

```bash
source config/temporal_cv_20261012.sh
export TEMPORAL_FINAL_EPOCHS=$(python -c "import json; print(json.load(open('$TEMPORAL_RUNS/temporal_cv.json'))['best_epoch'])")
sbatch scripts/slurm/planet_temporal_final.sh
```

`--fold all` trains on all 142 chips, using both windows, with no test set.
The model is `$TEMPORAL_RUNS/final_e<E>/last_model.pth`. Its `params` record
the training chips, and the apply step uses them to compute each scene's
distance to training data.

### 6. Coregister the scenes away from drone dates (gattaca2)

This step needs only the CPU, and it can run while the CV trains:

```bash
source config/temporal_cv_20261012.sh
sbatch scripts/slurm/planet_coreg_nearest.sh
```

The job runs `coreg_planet_to_nearest_drone.py` on
`$APPLY_PLANETDIR` and `$APPLY_OCMDIR`, with one `--drone-dir` per entry of
`$TEMPORAL_DRONE_DIRS` (`stri/globus/RGB`, then
`stri/C3KW2X/BCI_50ha_timeseries_local_alignment`, then
`stri/24782016/BCI_50ha_timeseries_local_alignment`), and passes
`--exclude-within-days 2 --min-clear-fraction 0.1 --workers 8`. It writes
`$APPLY_COREG_JSON`.

**Scene selection.** `--dry-run` reports the selection without
coregistering. It found 798 scenes in 104 ortho groups. Of the 2,146 clipped
scenes, it skipped:
- 766 within 2 days of a flight;
- 580 below 10 % clear;
- 2 without an OCM mask.

| Year | Scenes | Median days to a flight | Max |
|---|---|---|---|
| 2020 | 160 | 12 | 42 |
| 2021 | 167 | 9 | 17 |
| 2022 | 182 | 9 | 17 |
| 2023 | 88 | 3 | 14 |
| 2024 | 57 | 3 | 4 |
| 2025 | 29 | 3 | 7 |
| 2026 | 115 | 84 | 203 |

The distance here is to any drone flight, including the 2018–2020 phantom
flights that have no training chips. The 2020 scenes and most of the 2026
scenes are far from any training chip (the chips span 2021-01 to 2026-01), and
the apply step flags them `in_envelope = false`.

**Coregistration.** Each scene is matched to the ortho nearest in time with
`compute_coreg_shift`, using the training build's AROSICS settings: Red band,
`ws=(200, 200)`, `max_shift=10`. Each ortho group runs in a fresh process, and
a mavic ortho takes about 1 min to load. The JSON is rewritten after each
group, so an interrupted run resumes where it stopped.

In a test on 2025-02-12 to 02-28, 1 of 2 scenes coregistered (shift 7.3 m E,
1.4 m N). The full run took 72 min with 8 workers, and 546 of the 798 scenes
(68 %) coregistered, more than the training build's 40 %. A scene that
fails is still predicted, but its crown rows use unshifted polygons and are
flagged `coreg_ok = false`.

### 7. Apply (gattaca2)

```bash
source config/temporal_cv_20261012.sh
export TEMPORAL_FINAL_EPOCHS=<E>
sbatch scripts/slurm/planet_apply.sh
# when the array has finished:
python scripts/apply_planet_maskrcnn.py concat $APPLY_OUT
```

**Chip and grid.** Each scene is reprojected (bilinear) onto the grid of a
training chip (`--grid-like`). For the normal 367 × 205 px cutouts this
reproduces the training chips byte for byte. It also brings the few scenes
delivered on a 353 × 191 px grid onto the common 1468 × 820 grid.

**Tiled prediction.**
- The model runs on 512 × 512 tiles at stride 384.
- A detection is kept only from the tile whose core (overlaps split at their
  midpoint) holds its box centre, and dropped if that centre is cloudy, as in
  `evaluate()`.
- Masks are binarized at 0.5.
- Every detection the model returns (score ≥ 0.05) is kept in the rasters.
  `--det-thresh` (default 0.5) applies only to the crown statistics and the QA
  outlines.

Then copy the results back to Earth03:

```bash
rsync -a gattaca2:/scratch/tree-monitoring/202610_temporal_apply/ \
  /Volumes/Earth03/flower/202610_temporal_apply/
```

## Outputs

Under `$APPLY_OUT` (`202610_temporal_apply/`):

| File | Contents |
|---|---|
| `coreg_nearest_drone.json` | one record per selected scene. Fields: `scene`, `planet_rel` (path under `planet_clipped/4band`), `date`, `clear_fraction`, `drone_date`, `days_to_drone`, `drone_file`, `coreg_ok`, `x_shift_m`, `y_shift_m`. The shift is the AROSICS (drone → Planet) translation, as in `coreg_log.json` |
| `pred/<scene>.pred.tif` | float32, 3 bands on the raw Planet chip grid (EPSG:32617, 0.75 m). Band 1 is the instance id, where 1 is the highest-scoring instance, the best-scoring instance wins on overlaps, and 0 is none. Band 2 is the highest instance score covering the pixel. Band 3 is clear (OCM clear and not nodata) |
| `pred/<scene>.pred_coreg.tif` | the same bands, with the transform translated by −shift into drone / crown-map geometry (coregistered scenes only) |
| `qa/<scene>.png` | RGB render (cloud-aware 2/98 % stretch) with outlines of instances scoring ≥ `--det-thresh` in red, and cloud and nodata tinted blue |
| `crowns/<scene>.csv`, joined into `crown_timeseries.csv` | one row per crown in `BCI_50ha_2022_09_29_crownmap_improved.shp` (2,280) per scene; columns below |
| `scenes/<scene>.json`, joined into `scene_summary.csv` | per scene: the shared fields plus `drone_date`, `clear_fraction`, `valid_fraction`, `n_detections`, `n_detections_above_thresh`, `positive_px`, `det_thresh`, `model`, `model_epoch` |

Columns of `crown_timeseries.csv`:
- **shared with `scene_summary.csv`:** `scene`, `date`, `days_to_drone`,
  `days_to_nearest_train`, `in_envelope` (true when the scene is 30 days or
  less from a training chip), `coreg_ok`, `x_shift_m`, `y_shift_m`
- **crown identity:** `tag`, `mnemonic`
- **statistics:** `crown_px`, `crown_clear_frac`, `pos_frac`, `max_score`,
  `n_instances`

The crown statistics are computed on the crown polygons translated by +shift
onto the Planet grid:
- `crown_clear_frac` is the clear fraction of the crown's pixels.
- `pos_frac` is the fraction of the crown's clear pixels covered by an
  instance scoring at least `--det-thresh`.
- `max_score` is the highest instance score on a clear crown pixel.
- `n_instances` counts the distinct instances above the threshold that touch
  the crown.

Where polygons overlap, a pixel belongs to the smaller crown.

## Results

Temporal CV run on gattaca2 on 2026-10-17 and 18 (`$TEMPORAL_RUNS/temporal_cv.json`;
figure `temporal_cv.png`). The pooled held-out `map_50` peaks at
**epoch 12 (E\* = 12)**. It then falls slightly (0.336–0.340 from epoch 17 to
40), while IoU and precision keep rising and recall falls. So the later epochs
trade detections for cleaner masks rather than generalizing better.

| Epoch | 1 | 2 | 3 | 5 | 8 | **12** | 17 | 25 | 40 |
|---|---|---|---|---|---|---|---|---|---|
| Pooled `map_50` | 0.169 | 0.247 | 0.269 | 0.303 | 0.335 | **0.354** | 0.337 | 0.340 | 0.336 |
| Pooled `test_iou` | 0.199 | 0.209 | 0.203 | 0.227 | 0.280 | **0.352** | 0.381 | 0.407 | 0.422 |

| | Pooled `map_50` | `test_iou` | Precision | Recall |
|---|---|---|---|---|
| Temporal CV, epoch 12 | 0.354 | 0.352 | 0.242 | 0.477 |

| Days to nearest training chip | Chips | `map_50` | `test_iou` | Precision | Recall |
|---|---|---|---|---|---|
| 0–3 | 7 | 0.512 | 0.416 | 0.305 | 0.600 |
| 4–7 | 27 | 0.435 | 0.420 | 0.298 | 0.549 |
| 8–14 | 51 | 0.372 | 0.385 | 0.278 | 0.496 |
| 15–30 | 35 | 0.258 | 0.272 | 0.185 | 0.386 |
| >30 | 22 | 0.203 | 0.217 | 0.124 | 0.347 |

Precision and recall are the event metrics (`test_event/*`).

**Skill falls steadily with distance from training data.** `map_50` drops
from about 0.5 within 3 days of a training chip, to 0.37 at 1–2 weeks, to 0.26
at 2–4 weeks. Within the 30-day envelope that the apply step flags, the
weakest interpolated scenes therefore score about half as well as scenes near
a flight.

**Skill is seasonal.** By calendar month at epoch 12 (`temporal_cv_season.png`;
all years together, counts summed over chips), the dry-season chips
(December to April, 110 chips) score far above the wet-season ones (May to
November, 32 chips):

| Season | Days to nearest training chip | Chips | `test_iou` | Precision | Recall |
|---|---|---|---|---|---|
| Dry (Dec–Apr) | ≤ 14 | 85 | 0.399 | 0.287 | 0.522 |
| Dry (Dec–Apr) | 15–30 | 19 | 0.325 | 0.247 | 0.422 |
| Dry (Dec–Apr) | > 30 | 6 | 0.314 | 0.216 | 0.459 |
| Wet (May–Nov) | 15–30 | 16 | 0.187 | 0.117 | 0.323 |
| Wet (May–Nov) | > 30 | 16 | 0.124 | 0.051 | 0.193 |

August to October is the weakest stretch (IoU 0.11–0.13, precision
0.04–0.06). No wet-season chip is within 14 days of training data: their
median distance is 30.5 days, against 12 for the dry season. Distance alone
does not explain the gap, though. At the same 15–30 days, wet-season IoU is
0.19 against 0.33 in the dry season. Scenes are cloudier then, there are fewer
flights and the canopy is in full leaf, so the wet-season interpolation is the
least reliable part of the apply output.

By fold at epoch 12, `map_50` is 0.365–0.376 for folds 0 and 2–5 and
0.292 for fold 1. Fold 1 holds most of the sparse 2021 flights and is the
furthest from its training data (median 30 days).

**Apply run** (2026-10-18, model `final_e12`, 4 shards of about 8 min each).
All 798 selected scenes were predicted, giving 1,819,440 crown rows
(798 × 2,280). Of these scenes, 546 have a coregistered raster as well as the
raw one. 478 scenes are `in_envelope` (30 days or less from a training chip),
almost all of them from 2021 to 2025. Of the 160 scenes from 2020 only 3 are
in the envelope, and of the 115 from 2026 only 14.

| Year | Scenes | In envelope | `coreg_ok` | Median days to train | Median detections ≥ 0.5 |
|---|---|---|---|---|---|
| 2020 | 160 | 3 | 107 | 250 | 22.5 |
| 2021 | 167 | 159 | 124 | 14 | 28 |
| 2022 | 182 | 133 | 122 | 15.5 | 19 |
| 2023 | 88 | 85 | 66 | 9 | 18.5 |
| 2024 | 57 | 55 | 38 | 4 | 32 |
| 2025 | 29 | 29 | 19 | 4 | 38 |
| 2026 | 115 | 14 | 70 | 98 | 27 |

**MODIS-calibrated chips** (2026-10-24, `config/temporal_cv_20261012_cal.sh`,
`~/Documents/tree-flowering/202610_temporal_cal/`). The same folds, chips,
labels and training settings, but with the chips as MODIS-calibrated surface
reflectance and `--fixed-stretch config/planet_fixed_stretch.yml` (see
[planet_radiometric_calibration.md](planet_radiometric_calibration.md#classifier-input-stretch)).
Folds took 34–41 min each and the pooled evaluation 24 min. Pooled `map_50`
again peaks at **epoch 12**, so both runs are compared there
(`compare_temporal_cv.py`, figure `temporal_cv_toa_vs_cal.png`):

| Run | Pooled `map_50` | 0–3 d | 4–7 d | 8–14 d | 15–30 d | >30 d |
|---|---|---|---|---|---|---|
| TOA, epoch 12 | 0.354 | 0.512 | 0.435 | 0.372 | 0.258 | 0.203 |
| Calibrated, epoch 12 | 0.346 | 0.516 | 0.399 | 0.377 | 0.239 | 0.200 |

| Run | Pooled `test_iou` | 0–3 d | 4–7 d | 8–14 d | 15–30 d | >30 d |
|---|---|---|---|---|---|---|
| TOA, epoch 12 | 0.352 | 0.416 | 0.420 | 0.385 | 0.272 | 0.217 |
| Calibrated, epoch 12 | 0.361 | 0.410 | 0.419 | 0.392 | 0.292 | 0.226 |

**Calibration does not improve temporal transfer.** Pooled `map_50` falls by
0.008 and IoU rises by 0.009, both within noise. The far bins, where a
different atmosphere should matter most, do not improve either: at 15–30 days
`map_50` falls by 0.019 while IoU rises by 0.020, and beyond 30 days both are
flat. Fold 1 (the sparse 2021 flights) drops from 0.292 to 0.268 in `map_50`;
the other folds move by −0.04 to +0.02. The calibrated run is ahead early in
training (`map_50` 0.308 against 0.269 at epoch 3) and its precision is higher
at every epoch but epoch 5 (0.263 against 0.242 at epoch 12), with recall lower from
epoch 8 on. The wet season stays the weak spot (IoU 0.10–0.13 from August to
October). The spatial benchmark, by contrast, gains about 0.02 in both
metrics ([planet_training_chips_50ha.md](planet_training_chips_50ha.md#modis-calibrated-chips-full_mkl_cal)).

**Calibrated apply run** (2026-10-24, model `final_e12` in
`202610_temporal_cal/`, trained in 12 min, 4 shards of about 8 min each,
output `/scratch/tree-monitoring/202610_temporal_apply_cal/`). It reads
`planet_clipped_calfill/4band` and reuses the TOA coregistration
(`202610_temporal_apply/coreg_nearest_drone.json`):

```bash
source config/temporal_cv_20261012_cal.sh
export TEMPORAL_FINAL_EPOCHS=12
FINAL=$(sbatch --parsable -A tree-monitoring scripts/slurm/planet_temporal_final.sh)
sbatch -A tree-monitoring --dependency=afterok:$FINAL scripts/slurm/planet_apply.sh
python scripts/apply_planet_maskrcnn.py concat $APPLY_OUT
```

792 of the 798 scenes were predicted, giving 1,805,760 crown rows
(792 × 2,280). The 6 skipped scenes are Dove Classic `1105` scenes from 2020
to 2022 with no SR product, so they are absent from `planet_clipped_calfill`.
Of the 792, 475 are in the envelope and 543 have a coregistered raster. The
calibrated model returns more confident detections than the TOA one in
2021–2023 and 2026, and slightly fewer in 2024–2025:

| Year | Scenes | Median detections ≥ 0.5, TOA | Calibrated |
|---|---|---|---|
| 2020 | 157 | 22.5 | 23 |
| 2021 | 166 | 28 | 30.5 |
| 2022 | 180 | 19 | 29 |
| 2023 | 88 | 18.5 | 38.5 |
| 2024 | 57 | 32 | 30 |
| 2025 | 29 | 38 | 30 |
| 2026 | 115 | 27 | 32 |

(The TOA medians are over all of that year's scenes, including the 6 the
calibrated run skips.) The temporal CV found no skill difference between the
two models. Against the drone labels the calibrated output is slightly worse,
entirely on scenes with a borrowed offset
([below](#calibrated-apply-run-against-the-drone-labels)), and it also lacks
the 6 Dove Classic scenes. So the TOA apply output stays the reference.

## Crown-level comparison with the drone classifications

`plot_crown_species_comparison.py` compares the apply run's per-crown calls
with the drone crown classifications, by species. It runs on the Mac, from the
Earth03 copy, in about 5 s:

```bash
python scripts/plot_crown_species_comparison.py \
  /Volumes/Earth03/flower/202610_temporal_apply/crown_species
```

It writes `crown_species_{timeseries,season,agreement}.{png,pdf}` and
`crown_species_agreement.csv`.

**The drone label is the training-mask rule applied per crown.** The rule is
`apply_drone_labels_coreg.py --mode both`: a crown is an *event* (flowering or
deciduous) when `1 − (1 − p_flowering)(1 − p_deciduous) > 0.5`. Here it is
applied to the per-crown mean probabilities in `results/crown_classifications.nc4`
(phantom, 2018 to 2024-03-18) and `crown_classifications_mavic.nc4` (mavic,
from 2024-03-19). The classification rasters are almost constant inside a
crown: the within-crown standard deviation of `p_flowering` is about 0.01 on a
2022-03-30 sample. So the crown mean gives the same answer as the per-pixel
rule.

**The Planet call.** A Planet crown is an event when at least 25 % of its
clear pixels lie under a detection scoring 0.5 or more (`pos_frac ≥ 0.25`,
`--pos-frac`). On both sides, only these crowns are used:
- crowns of at least 64 Planet pixels, the training `--min-instance-size`
  (2,000 of the 2,227 crown tags);
- Planet crowns at least 50 % clear.

The choice of rule was checked on the matched set below:

| Planet rule | Precision | Recall | Share of crowns called |
|---|---|---|---|
| `pos_frac ≥ 0.10` | 0.47 | 0.41 | 4.2 % |
| **`pos_frac ≥ 0.25`** | **0.56** | **0.39** | **3.3 %** |
| `pos_frac ≥ 0.50` | 0.68 | 0.29 | 2.0 % |
| `max_score ≥ 0.50` | 0.35 | 0.42 | 5.8 % |

The drone label calls 4.8 % of the same crown-scenes events. `max_score`
ranks crowns slightly better (AUROC 0.75, against 0.70 for `pos_frac`), but
at any useful cutoff it calls far more false events.

**Species.** The twelve species with the most drone event crown-dates over
the apply period are shown.

**Agreement.** Each Planet scene is matched to the drone flight it was
coregistered to (`drone_date`). The match uses only scenes that are
coregistered, in the envelope, and 3 to 7 days from that flight (the apply
step excludes scenes within 2 days). That gives 304,890 crown-scenes in 203
scenes, matched to 74 flights.

| Species | Drone events | Precision | Recall | Drone share | Planet share |
|---|---|---|---|---|---|
| *Cavanillesia platanifolia* | 1,204 | 0.90 | 0.61 | 0.50 | 0.34 |
| *Tachigali panamensis* | 361 | 0.82 | 0.62 | 0.08 | 0.06 |
| *Dipteryx oleifera* | 989 | 0.75 | 0.70 | 0.23 | 0.22 |
| *Hura crepitans* | 1,268 | 0.74 | 0.60 | 0.14 | 0.11 |
| *Platypodium elegans* | 586 | 0.73 | 0.50 | 0.32 | 0.22 |
| *Handroanthus guayacan* | 373 | 0.72 | 0.46 | 0.16 | 0.11 |
| *Ceiba pentandra* | 572 | 0.72 | 0.59 | 0.15 | 0.12 |
| *Tabebuia rosea* | 423 | 0.70 | 0.41 | 0.16 | 0.10 |
| *Cordia alliodora* | 326 | 0.64 | 0.27 | 0.19 | 0.08 |
| *Jacaranda copaia* | 1,363 | 0.54 | 0.17 | 0.08 | 0.03 |
| *Zanthoxylum ekmanii* | 636 | 0.51 | 0.08 | 0.07 | 0.01 |
| *Quararibea stenophylla* | 798 | 0.30 | 0.18 | 0.03 | 0.02 |
| other species | 5,590 | 0.39 | 0.32 | 0.025 | 0.021 |
| all species | 14,489 | 0.56 | 0.39 | 0.048 | 0.033 |

**Interactive per-crown viewer.** `make_crown_timeseries_html.py` writes one
self-contained page (about 21 MB, with plotly.js inlined so it opens offline):

```bash
python scripts/make_crown_timeseries_html.py \
  /Volumes/Earth03/flower/202610_temporal_apply/crown_timeseries_viewer.html
```

- **The map** is the 2022-09-29 crown map, and a crown is selected by
  clicking it. It colours crowns by one of:
  - the selected species;
  - the drone event probability at a flight;
  - the Planet `max_score` or `pos_frac` in a scene.

  The ◀ ▶ buttons (or the arrow keys) step through flights or scenes.
- **The crown panel** plots, for the selected crown:
  - the drone `p_flowering` and `p_deciduous`, and the event probability
    1 − (1 − p_f)(1 − p_d), at every flight from 2018 to 2026;
  - the Planet `max_score` at every scene, with `pos_frac` available from
    the legend.

  Grey points are outside the 30-day envelope, and open circles are scenes
  that were not coregistered. Hovering over a point gives its clear fraction
  and its distances to the nearest drone flight and training chip. Clicking a
  point moves the map to that date.
- **The species panel** repeats the species shares of
  `crown_species_timeseries.png` for the selected crown's species.
- **Filtering.** A minimum Planet clear fraction (default 0.5) filters both
  panels and the map. The page includes only crown-scenes at least 10 % clear
  (`--store-min-clear`).
- **The URL** keeps the selected tag (`#<tag>`), so a crown can be linked
  directly.

**What the comparison shows.**
- **Large deciduous and mass-flowering canopy species track the drone
  closely.** *Dipteryx*, *Hura*, *Ceiba* and *Cavanillesia* reach precision
  0.7–0.9 and recall 0.6–0.7, and their Planet shares follow the drone's
  dry-season peaks year by year.
- **Planet under-calls everywhere** (3.3 % against 4.8 %), most of all for
  *Zanthoxylum*, *Jacaranda*, *Cordia* and *Quararibea*. Their events are
  likely smaller in area or lower in contrast at 3 m.
- ***Zanthoxylum's* September–October drone peak is missed entirely.** This
  matches the wet-season weakness in the temporal CV.
- Before 2021 and after 2026-01, almost every Planet scene is outside the
  envelope (grey in the time series). There the Planet shares still rise
  with the drone in season, but they are not validated.

### Calibrated apply run against the drone labels

The same comparison was run on the calibrated apply output (2026-10-25).
Its viewer is in the same directory:

```bash
python scripts/plot_crown_species_comparison.py \
  /Volumes/Earth03/flower/202610_temporal_apply_cal/crown_species \
  --apply-dir /Volumes/Earth03/flower/202610_temporal_apply_cal
python scripts/make_crown_timeseries_html.py \
  /Volumes/Earth03/flower/202610_temporal_apply_cal/crown_timeseries_viewer.html \
  --apply-dir /Volumes/Earth03/flower/202610_temporal_apply_cal
python scripts/compare_crown_agreement.py \
  /Volumes/Earth03/flower/202610_temporal_apply_cal/crown_agreement_toa_vs_cal
```

`compare_crown_agreement.py` scores both runs on the same crown-scenes. A
crown-scene enters only if it passes the filters in both runs, which drops the
6 Dove Classic scenes. A few tags are shared by several polygons, so crowns
are paired polygon by polygon. Scenes 3 to 7 days from their coregistered
flight form two strata, in the 30-day envelope and outside it. The script
writes `crown_agreement_toa_vs_cal.{csv,png,pdf}` and `_rules.csv`, and
prints the breakdowns below. The bootstrap intervals resample scenes
(2,000 draws).

| Stratum | Scenes | Drone events | Precision, TOA → cal | Recall, TOA → cal | 95 % interval, Δ precision | Δ recall |
|---|---|---|---|---|---|---|
| In envelope | 202 | 14,478 | 0.561 → 0.542 | 0.392 → 0.368 | −0.033 to −0.005 | −0.046 to 0.000 |
| — own MODIS fit | 148 | 12,199 | 0.562 → 0.552 | 0.409 → 0.398 | −0.024 to +0.002 | −0.031 to +0.012 |
| — borrowed offset | 54 | 2,279 | 0.553 → 0.459 | 0.299 → 0.209 | −0.159 to −0.030 | −0.160 to −0.003 |
| Outside envelope | 48 | 2,228 | 0.489 → 0.436 | 0.349 → 0.342 | −0.096 to −0.014 | −0.042 to +0.028 |
| — own MODIS fit | 33 | 1,842 | 0.502 → 0.466 | 0.403 → 0.385 | −0.081 to +0.007 | −0.054 to +0.019 |
| — borrowed offset | 15 | 386 | 0.319 → 0.237 | 0.093 → 0.140 | −0.209 to +0.020 | −0.040 to +0.126 |

The in-envelope TOA row is the [table above](#crown-level-comparison-with-the-drone-classifications)
restricted to the shared scenes. The 18 flights outside the envelope are
mostly in 2020 (26 scenes) and 2022 (16).

- **On scenes with their own MODIS fit, the two models agree with the drone
  equally well.** The differences are about 0.01 and within noise.
- **The loss comes from scenes that borrow an offset.** Precision drops by
  about 0.09 and recall by 0.09 in the envelope. A borrowed offset removes
  the sensor and seasonal bias but not that day's atmosphere
  ([planet_radiometric_calibration.md](planet_radiometric_calibration.md),
  caveat 11). The fixed stretch passes that residual on to the model, whereas
  the TOA per-image stretch absorbs it.
- **Calibration does not help outside the envelope either.** This was the
  case the temporal CV could not test. Precision falls (0.489 → 0.436) and
  recall is flat, and the 2020 scenes account for most of the drop
  (precision 0.488 → 0.415).
- **The two models disagree on few calls.** Their calls differ on 3.0 % of
  in-envelope crown-scenes (Cohen's κ 0.53). Planet calls 3.3 % and 3.2 % of
  crown-scenes events, against 4.8 % for the drone. By species, the largest
  in-envelope drops are *Zanthoxylum* precision (0.51 → 0.43) and
  *Cavanillesia* and *Tabebuia* recall (0.61 → 0.55, 0.41 → 0.35). *Ceiba*
  recall rises (0.59 → 0.65).
- **The Planet rule still holds.** For the calibrated run, `pos_frac ≥ 0.25`
  gives precision 0.54 and recall 0.37, and `max_score ≥ 0.5` gives
  0.33 / 0.40. AUROC is 0.69 for `pos_frac` and 0.76 for `max_score`, as for
  TOA (0.70, 0.75).

So the TOA apply output stays the reference. Calibration could only help if
offsets were fitted for more scenes, or if borrowed-offset scenes fell back
to a per-image stretch.

## Why calibration did not help

Candidate explanations for the null temporal-CV result, tested cheapest first:

| | Explanation | Test |
|---|---|---|
| H1 | The training augmentation already covers the radiometric variation. Every run applies `ColorJitter(brightness=0.2, contrast=0.2)` to the RGB channels, hard-coded on. There are no flips or rotations, copy-paste is off, and the input is RGB only. | Jitter ablation |
| H2 | Too few scenes are calibrated directly (39 % of date-satellite groups) | CV split by offset source |
| H3 | The architecture is the limit | DINOv3 satellite backbone |
| H4 | The model uses within-chip contrast, which a per-scene linear correction leaves unchanged | Test-time offset sensitivity |
| H5 | Calibration removes little of the input variation. What remains is seasonal canopy change, PS2 colour casts, local haze, and sun and view geometry. | Per-chip skill drivers |
| H6 | Skill comes partly from the persistence of the same trees' state on nearby dates | Persistence baseline from the drone labels |
| H7 | Label noise caps both runs equally | Per-chip skill by label source and grade |
| H8 | Training run-to-run variance is unmeasured; each arm was run once with no seed | Chip bootstrap; repeat run |
| H9 | The calibrated chips (from the SR product) are shifted or blurred relative to the TOA chips the labels were coregistered to | Chip-pair shift and sharpness |
| H10 | ImageNet normalization and hyperparameters were set on TOA inputs | Conditional retrain |

### CV split by offset source, with a chip bootstrap (H2, H8)

`compare_temporal_cv.py --windows ... --chip-meta` pools the per-window counts
of each run (`temporal_cv_windows.csv`) at epoch 12 over strata of chips. It
also gives a paired chip-bootstrap 95 % interval (2,000 resamples) of
calibrated minus TOA. The interval covers chip sampling only, not training
variance. mAP cannot be pooled from window counts, so the strata report IoU and
crown precision and recall only. The pooled and per-bin values reproduce
`temporal_cv.json` exactly.

```bash
python scripts/compare_temporal_cv.py \
  --run toa=/Volumes/Earth03/flower/202610_temporal/temporal_cv.json \
  --run cal=/Volumes/Earth03/flower/202610_temporal_cal/temporal_cv.json \
  --windows toa=/Volumes/Earth03/flower/202610_temporal/temporal_cv_windows.csv \
  --windows cal=/Volumes/Earth03/flower/202610_temporal_cal/temporal_cv_windows.csv \
  --chip-meta /Volumes/Earth03/flower/20261012_full_label_extended2_mkl_x4_coreg_4band_cal_curated/chips.csv
```

| Chips | n | IoU, TOA → cal | Δ IoU 95 % | Precision, TOA → cal | Δ precision 95 % | Recall, TOA → cal | Δ recall 95 % |
|---|---|---|---|---|---|---|---|
| All | 142 | 0.352 → 0.361 | +0.002 to +0.018 | 0.242 → 0.263 | +0.010 to +0.031 | 0.477 → 0.462 | −0.031 to −0.000 |
| Own MODIS fit | 130 | 0.356 → 0.368 | +0.004 to +0.020 | 0.245 → 0.269 | +0.015 to +0.035 | 0.484 → 0.471 | −0.028 to +0.004 |
| Borrowed, same instrument | 10 | 0.266 → 0.253 | −0.033 to +0.010 | 0.189 → 0.157 | −0.065 to +0.008 | 0.370 → 0.302 | −0.109 to −0.032 |
| Borrowed, same satellite | 2 | 0.324 → 0.276 | −0.056 to −0.014 | 0.241 → 0.159 | −0.119 to −0.021 | 0.369 → 0.282 | −0.232 to +0.085 |
| 0–3 d | 7 | 0.416 → 0.410 | −0.029 to +0.016 | 0.305 → 0.320 | −0.018 to +0.042 | 0.600 → 0.614 | −0.009 to +0.035 |
| 4–7 d | 27 | 0.419 → 0.419 | −0.015 to +0.011 | 0.298 → 0.318 | −0.002 to +0.038 | 0.549 → 0.516 | −0.052 to −0.014 |
| 8–14 d | 51 | 0.384 → 0.391 | −0.005 to +0.022 | 0.278 → 0.289 | −0.005 to +0.027 | 0.496 → 0.494 | −0.030 to +0.031 |
| 15–30 d | 35 | 0.271 → 0.292 | +0.004 to +0.039 | 0.185 → 0.205 | −0.005 to +0.049 | 0.386 → 0.356 | −0.061 to +0.003 |
| >30 d | 22 | 0.217 → 0.226 | −0.004 to +0.020 | 0.124 → 0.147 | +0.010 to +0.037 | 0.347 → 0.327 | −0.034 to −0.008 |

- **On chips with their own fit, calibration gives a small gain that chip
  sampling resolves.** IoU rises by 0.012 and precision by 0.024, and recall
  falls by 0.013 (not resolved). It is the same precision-for-recall trade as in
  the pooled numbers.
- **The 12 borrowed-offset chips go the other way.** Recall falls by 0.07–0.09,
  as on the borrowed scenes in the drone comparison. They are 8 % of the chips,
  and they pull the pooled gain down to +0.009 IoU.
- So coverage (H2) does matter, but full coverage would add only about 0.003 to
  the pooled IoU gain. The own-fit gain is small either way, and whether it
  exceeds training noise (H8) needs a repeat run.

### Persistence baseline (H6)

`persistence_baseline.py` uses no imagery. It predicts one drone flight's crown
events from another flight's labels, with the labelling rule and the 64 px crown
minimum of `plot_crown_species_comparison.py` (2,000 crowns, 200 flights). It
runs in about a minute on the Mac:

```bash
python scripts/persistence_baseline.py --folds config/temporal_folds_20261012.csv \
  --cv-json /Volumes/Earth03/flower/202610_temporal/temporal_cv.json
```

The CV analogue predicts each chip's flight from the flight of the
nearest-dated chip in the other folds, which is the training data that fold's
model saw. "Area Jaccard" weights crowns by their area (crown pixels that are
events on both dates over those that are events on either), as an
approximation of the CV's binary IoU.

| Days to nearest training chip | Chips | Persistence precision | Persistence recall | Persistence area Jaccard | CV IoU, TOA e12 |
|---|---|---|---|---|---|
| 0–3 | 7 | 0.804 | 0.810 | 0.740 | 0.416 |
| 4–7 | 27 | 0.791 | 0.785 | 0.686 | 0.419 |
| 8–14 | 51 | 0.667 | 0.699 | 0.597 | 0.384 |
| 15–30 | 35 | 0.479 | 0.487 | 0.376 | 0.271 |
| >30 | 22 | 0.274 | 0.232 | 0.161 | 0.217 |
| Pooled | 142 | 0.644 | 0.650 | 0.539 | 0.352 |

Over all flight pairs up to 60 days apart, persistence falls in the same way:
area Jaccard is 0.65 at 4–7 d, 0.54 at 8–14 d, 0.39 at 15–30 d and 0.25 at
31–60 d.

- **A crown's state on the nearest training date predicts its state on the
  held-out date better than the model does,** out to 30 days. The comparison
  flatters persistence. It is scored on drone crowns with exact geometry, while
  the CV is scored on 3 m Planet pixels with coregistration error. So
  persistence is an upper bound on what remembering the same crowns could give,
  not a like-for-like competitor.
- **The CV's decline with distance has the same shape as persistence.** A model
  that reads events from the image alone would have no reason to lose skill
  with distance from the training dates, apart from seasonal shift. So part of
  the temporal-CV skill is probably the model recognizing the same crowns and
  their recent state (H6). Beyond 30 days, the model beats persistence (0.217
  vs 0.161), so it does read some events from the imagery.
- **The spatial benchmark points the same way.** On new ground at seen dates,
  the TOA model peaks at IoU 0.293. On seen ground at new dates, the temporal
  CV pooled IoU is 0.352 (0.42 within a week). The test chips and label sources
  differ, so this is indicative only.
- Skill that comes from memory of the same crowns does not depend on input
  radiometry, which helps explain why calibration changes the temporal CV so
  little. The temporal CV therefore overstates skill on dates far from any
  flight. A crown-level CV score (model calls per crown, as in the drone
  comparison) would allow a like-for-like comparison with persistence.

### What drives per-chip skill (H5, H7)

`temporal_cv_chip_drivers.py` joins the per-chip IoU of both runs at epoch 12 to
properties of each chip and scene. It reports Spearman correlations, raw and
partial on days to the nearest training chip, and IoU pooled by group. The
persistence column comes from `persistence_baseline.py --out-prefix P`:

```bash
python scripts/persistence_baseline.py --folds config/temporal_folds_20261012.csv --out-prefix P
python scripts/temporal_cv_chip_drivers.py \
  --windows toa=/Volumes/Earth03/flower/202610_temporal/temporal_cv_windows.csv \
  --windows cal=/Volumes/Earth03/flower/202610_temporal_cal/temporal_cv_windows.csv \
  --persistence P_cv.csv
```

With 142 chips, |ρ| above about 0.17 is significant at 5 %.

| Property | ρ with TOA IoU | partial | ρ with cal − TOA | partial |
|---|---|---|---|---|
| Days to nearest training chip | −0.66 | | +0.12 | |
| Persistence area Jaccard | +0.64 | +0.12 | −0.09 | +0.03 |
| Labelled crowns per chip | +0.62 | +0.48 | −0.02 | +0.04 |
| Crown contrast-to-noise, TOA red input | +0.52 | +0.46 | −0.10 | −0.07 |
| Day of year | −0.42 | −0.18 | +0.14 | +0.09 |
| Clear fraction | +0.28 | +0.25 | −0.01 | +0.01 |
| Sun elevation | −0.25 | −0.14 | +0.16 | +0.14 |
| TOA red input median (haze) | −0.13 | −0.06 | +0.08 | +0.06 |
| Scene calibration offset, mean abs RGB | −0.05 | −0.03 | +0.16 | +0.15 |
| Saturated share, fixed stretch | +0.07 | −0.01 | +0.05 | +0.06 |
| View angle | −0.09 | −0.13 | −0.08 | −0.08 |
| Coregistration shift | −0.07 | +0.05 | +0.05 | +0.04 |

| Group | Chips | IoU TOA | IoU cal |
|---|---|---|---|
| Dry season (Jan–Apr) | 101 | 0.385 | 0.391 |
| Wet season (May–Dec) | 41 | 0.202 | 0.211 |
| PSB.SD | 133 | 0.357 | 0.368 |
| PS2 | 7 | 0.214 | 0.216 |
| phantom labels | 72 | 0.372 | 0.377 |
| mavic labels | 70 | 0.325 | 0.339 |

- **The scene's radiometric state does not predict TOA skill.** Haze (TOA input
  level), the size of the calibration correction, and saturation all have
  |ρ| ≤ 0.13 with TOA IoU. If radiometric differences between scenes were
  limiting the TOA model, its worst chips would be the hazy or strongly
  corrected ones. They are not, so there was little for calibration to fix.
- **Skill follows the crowns' contrast in the image and the label content.**
  Crown contrast-to-noise (ρ +0.46 after removing distance) and the number of
  labelled crowns (+0.48) are the strongest predictors after distance and
  persistence. Contrast-to-noise is unchanged by a per-scene linear correction
  (H4). Chips with few events score low IoU partly because IoU is unstable with
  small denominators.
- **The calibration gain is only weakly patterned.** It is a little larger
  where the correction was larger (ρ +0.16) and where the sun is higher
  (+0.16), both at the edge of significance. It is about the same in both
  seasons (+0.006 dry, +0.009 wet), so calibration does not fix the
  wet-season weakness.

### Calibrated vs TOA chip geometry (H9)

The calibrated chips are SR-product clips resampled onto the TOA chip grids,
while the labels were coregistered on TOA. `compare_chip_sets.py` checks each
pair for a shift, extra blur and per-band agreement over valid, OCM-clear pixels
(`results/chip_sets_toa_vs_cal.csv`, about 5 min on the Mac):

```bash
python scripts/compare_chip_sets.py \
  /Volumes/Earth03/flower/20261012_full_label_extended2_mkl_x4_coreg_4band_stretch_stats_curated \
  /Volumes/Earth03/flower/20261012_full_label_extended2_mkl_x4_coreg_4band_cal_curated \
  results/chip_sets_toa_vs_cal.csv
```

The shift is an integer phase-correlation start refined by masked
Lucas–Kanade. Plain phase correlation underestimated a known shift: on the
`--self-test 1.6` check (1.2 m per axis), it returned 1.05 m. The cause is that
nodata and cloud pixels sit at the same place in both copies. The refined
estimate recovers 1.200 m on every chip.

**There is no geometric confound.** The median red-band shift is 0.005 m
(90th percentile 0.25 m, maximum 0.51 m, below one 0.75 m chip pixel and a
sixth of a Planet pixel). The mean shift is 0.01 m in each axis. The sharpness
ratio (Tenengrad over variance, calibrated over TOA) is 0.99–1.03 (10th–90th
percentile; full range 0.95–1.11). Per-band correlation is at least 0.976
(blue) and 0.989 (green, red, NIR) on every chip. The calibrated chips differ
from the TOA chips only radiometrically.

### Test-time offset sensitivity (H4, H1)

`radiometric_sensitivity.py` re-scores each fold's epoch-12 checkpoint on its
held-out chips with an offset added to the raw chip before the model's own
stretch. The offset is k times the across-scene SD of the fitted
Planet–MODIS offsets (blue / green / red 0.0158 / 0.0149 / 0.0158 in
reflectance), with the same sign in every band, as haze would add. This is the
size of the scene-to-scene error that calibration removes. The TOA chips store
radiance counts, so each band's offset is converted per chip with the slope of
the TOA band against the calibrated copy over clear pixels.

The run took about 2 h on the Mac CPU. MPS hit GPU command-buffer errors and a
float64 failure, so use `--device cpu` there; CUDA is much faster.

```bash
python scripts/radiometric_sensitivity.py \
  --chip-dir /Volumes/Earth03/flower/20261012_full_label_extended2_mkl_x4_coreg_4band_stretch_stats_curated \
  --cal-chip-dir /Volumes/Earth03/flower/20261012_full_label_extended2_mkl_x4_coreg_4band_cal_curated \
  --folds config/temporal_folds_20261012.csv \
  --run-template '/Volumes/Earth03/flower/202610_temporal/fold{fold}' \
  --units toa --device cpu --k=-1,0,1 --out-csv sensitivity_toa.csv
```

At k = 0 it reproduces `temporal_cv.json` (pooled IoU 0.3518, map_50 0.3544 vs
0.3541).

| TOA model, offset | Pooled `map_50` | IoU | Precision | Recall |
|---|---|---|---|---|
| −1 SD | 0.3515 | 0.3512 | 0.252 | 0.474 |
| none | 0.3544 | 0.3518 | 0.242 | 0.478 |
| +1 SD | 0.3491 | 0.3447 | 0.233 | 0.475 |

**The TOA model hardly notices errors as large as those calibration removes.**
Offsets of ±1 SD cost at most 0.005 in pooled `map_50` and 0.007 in pooled
IoU. Within distance bins the largest IoU change is 0.011 (0–3 d, +1 SD). The per-image stretch and
the training jitter already make the model close to invariant to scene-level
offsets. So calibration could recover at most about this much, and the observed
gain on own-fit chips (+0.012 IoU) is of that order. This confirms H4, and H1
in its weak form. Whether jitter specifically is what buys the invariance is
the question for the ablation below. The calibrated folds' checkpoints are only
on gattaca2. Run the same command there with
`--chip-dir <cal chips> --run-template '<202610_temporal_cal>/fold{fold}' --units reflectance`
to see how much the fixed stretch passes offsets on, which is the
borrowed-offset loss.

### Augmentation ablation and repeat run (H1, H8; pending on gattaca2)

`train_planet_image_maskrcnn.py --no-color-jitter` turns off the training
ColorJitter, and `--seed N` seeds Python, NumPy and torch. Both are stored in
`params`. Their defaults (jitter on, no seed) reproduce every earlier run.
There are three CV-only configs, which unset the apply settings so they cannot
overwrite the parent runs' output:

| Config | Run | Tests |
|---|---|---|
| `config/temporal_cv_20261012_nojit.sh` | TOA, no jitter | H1: if TOA drops clearly more than calibrated without jitter, jitter was standing in for calibration |
| `config/temporal_cv_20261012_cal_nojit.sh` | Calibrated, no jitter | H1 |
| `config/temporal_cv_20261012_seed1.sh` | TOA, `--seed 1` | H8: the run-to-run spread the other differences are judged against |

```bash
for C in nojit cal_nojit seed1; do
  source config/temporal_cv_20261012_$C.sh
  CV=$(sbatch --parsable -A tree-monitoring scripts/slurm/planet_temporal_cv.sh)
  sbatch -A tree-monitoring --dependency=afterok:$CV scripts/slurm/planet_temporal_eval.sh
done
```

Each run takes about 2–3 h within the 2-GPU limit, so the three take about a
day if run one after another. Compare them with `compare_temporal_cv.py`
(`--run` and `--windows` for each).

## Caveats

- **The epoch choice is slightly optimistic.** E* is chosen on the same pooled
  held-out score that is reported, so that score is mildly optimistic. With
  only nine grid epochs and a broad optimum the bias should be small, but the
  per-bin numbers carry it too.
- **The >30-day bin is outside the envelope.** The 22 chips more than 30 days
  from training data are mostly the sparse 2021 flights. Report them, but do
  not use them to judge interpolation.
- **Held-out chips see the same trees.** This is intended: it measures change
  over time, not transfer across space. The east/west benchmark still answers
  the spatial question.
- **The label sources are mixed.** Phantom chips (2021–24) and mavic chips
  (2024–26) carry different label characteristics: mavic masks halved after
  the colour transfer, and median crown fraction is 2.65 % for mavic against
  4.03 % for phantom. A fold dominated by one source is scored against that
  source's labels.
- **Tiling changes the model's input context.** The CV scores fixed 512 × 512
  windows, but deployment predicts on overlapping tiles. The tiling code is
  exact: a single aligned tile reproduces the window prediction, IoU 1.000.
  The model itself, however, is shift-sensitive. With the 20260824 e012
  checkpoint:
  - moving the window by 64 px gives a union-mask IoU of 0.61–0.73 with the
    unshifted prediction;
  - tiled and window predictions agree at IoU 0.44–0.65, with similar total
    positive area.
  
  Individual crown calls in a scene are therefore noisier than a smooth time
  series would suggest. Aggregating over dates per crown, or test-time shift
  averaging, would reduce this.
- **Coregistration fails on cloudy scenes, and is cross-date.** About 60 % of
  scenes failed AROSICS in the training builds. Failed scenes keep raw Planet
  geometry, which has a median offset of 5.3 m and a 90th percentile of 17 m
  over the training chips. Their crown rows can be filtered on `coreg_ok`.
  Matching to an ortho days to weeks away assumes the canopy texture is
  stable over that gap.
- **Event metrics on MPS are not reproducible.** Two runs of
  `compare_maskrcnn_runs.py` on the same checkpoint and chips (Apple MPS)
  gave event TP 168 and 189, with identical IoU and mAP. Use the CUDA numbers
  from gattaca2 for reporting.

## Figure captions

`temporal_cv.png`:

> Held-out performance of the Planet Mask R-CNN under blocked temporal
> cross-validation of the 142 chips of the 20261012 curated set, 2021-01 to
> 2026-01. The set is split into 6 folds of contiguous blocks of at most 14
> days of Planet acquisition dates, and each block is held out with training
> data on both sides. Left: mask mAP at IoU 0.5 against training epoch (log
> scale) for the windows of all six held-out folds pooled together (black,
> two 512 × 512 windows per chip, crowns of at least 64 px, cloud-centred
> predictions removed). Each fold scored on its own is also shown (grey), and
> the dashed line marks the epoch selected for the full-record model. Right:
> the same pooled metric at the selected epoch, split by the number of days
> between a held-out chip's Planet acquisition and the nearest training chip's
> acquisition, with the number of held-out chips in each bin printed above
> its bar.

`temporal_cv_metrics.png`:

> Held-out mask and crown metrics of the Planet Mask R-CNN against training
> epoch (log scale) under the blocked temporal cross-validation of
> `temporal_cv.png` (142 chips of the 20261012 curated set, 2021-01 to
> 2026-01, 6 folds of contiguous blocks of at most 14 days). The four panels
> show mask mAP at IoU 0.5 (top left), binary IoU of the union of predicted
> and labelled crown masks over clear pixels (top right), crown precision
> (bottom left) and crown recall (bottom right). For the last two, a predicted
> crown counts as a true positive when it matches a labelled crown at mask
> IoU 0.5 or more, after greedy matching. Each panel shows the metric for the
> windows of all six held-out folds pooled together (black, two 512 × 512
> windows per chip, crowns of at least 64 px, cloud-centred predictions
> removed), with each fold scored on its own in grey. The dashed line marks
> the epoch selected by pooled mAP@50 (epoch 12).

`temporal_cv_bins.png`:

> Held-out performance of the Planet Mask R-CNN at the selected epoch 12 by
> the number of days between a held-out chip's Planet acquisition and the
> nearest training chip's acquisition, under the blocked temporal
> cross-validation of the 142 chips of the 20261012 curated set (2021-01 to
> 2026-01). The bars in each bin show, from dark to light: mask mAP at IoU
> 0.5, binary IoU of the predicted and labelled crown masks over clear
> pixels, crown precision and crown recall (a match is a mask IoU of at least
> 0.5). All are pooled over every held-out window in the bin. The number of
> held-out chips in each bin is given under its label. The hatched bin
> (more than 30 days) lies outside the 30-day envelope used when the model is
> applied.

`temporal_cv_chips.png`:

> Per-chip held-out binary IoU (top row, predicted and labelled crown masks
> over clear pixels), crown precision (middle) and crown recall (bottom; a
> match is a mask IoU of at least 0.5) of the Planet Mask R-CNN at the
> selected epoch 12, under the blocked temporal cross-validation of the 142
> chips of the 20261012 curated set (2021-01 to 2026-01). Each point is one
> chip, with its left and right 512 × 512 windows summed, coloured and shaped
> by the fold in which it was held out. Left: against the number of days to
> the nearest training chip's Planet acquisition. The black segments are the
> values pooled over each distance bin (0–3, 4–7, 8–14, 15–30 and more than
> 30 days), and the dashed line is the 30-day envelope. Right: against the
> chip's Planet acquisition date. The dotted line is the value pooled over all
> held-out chips.

`temporal_cv_season.png`:

> Seasonal variation in the per-chip held-out binary IoU (top, predicted and
> labelled crown masks over clear pixels), crown precision (middle) and crown
> recall (bottom; a match is a mask IoU of at least 0.5) of the Planet Mask
> R-CNN at the selected epoch 12, under the blocked temporal cross-validation
> of the 142 chips of the 20261012 curated set. The x-axis is the chip's
> Planet acquisition day of year, with chips from 2021-01 to 2026-01 shown
> together. Each blue point is one chip, with its left and right 512 × 512
> windows summed. The black segments are the values pooled over each
> calendar month, computed from the TP/FP/FN counts and mask pixels summed
> over that month's chips, and the number of chips in each month is printed
> above the top panel.

`temporal_cv_toa_vs_cal.png`:

> Held-out performance of the Planet Mask R-CNN under the blocked temporal
> cross-validation of `temporal_cv.png` (142 chips of the 20261012 curated
> set, 2021-01 to 2026-01, 6 folds of contiguous blocks of at most 14 days)
> trained on TOA chips with a per-image 0th–99.9th percentile stretch per band
> (blue) and on the same chips as MODIS-calibrated surface reflectance with
> one fixed per-band stretch (green). The first four panels show, against
> training epoch (log scale), mask mAP at IoU 0.5, binary IoU of the union of
> predicted and labelled crown masks over clear pixels, crown precision and
> crown recall (a match is a mask IoU of at least 0.5), each pooled over the
> windows of all six held-out folds (two 512 × 512 windows per chip, crowns of
> at least 64 px, cloud-centred predictions removed); circles mark each run's
> epoch selected by pooled mAP@50 (epoch 12 for both). The right panel gives
> mAP@50 at that epoch by the number of days between a held-out chip's Planet
> acquisition and the nearest training chip's acquisition.

`crown_species_timeseries.png`:

> Share of the crowns of each of the twelve species with the most drone
> event crown-dates that are flowering or deciduous, from 2020 to 2026. Each
> panel is one species, with the number of crowns of at least 64 Planet
> pixels (0.75 m) in the plot. The drone series (black, one point per drone
> flight) counts a crown as an event when its mean classification
> probabilities give 1 − (1 − p_flowering)(1 − p_deciduous) > 0.5. That is
> the rule used to make the training masks, applied to the phantom record to
> 2024-03-18 and the mavic record after it. The Planet series has one point
> per Planet scene between flights. It counts a crown as an event when at
> least 25 % of its clear pixels lie under a detection of the epoch-12
> temporal Mask R-CNN scoring 0.5 or more. It uses only crowns at least 50 %
> clear, and scenes with at least 5 such crowns of the species. Scenes 30 days
> or less from a training chip are blue, and scenes further away grey.

`crown_species_season.png`:

> Seasonal variation in the share of the crowns of each of the twelve species
> with the most drone event crown-dates that are flowering or deciduous, by
> calendar month. The drone flights (black) and the Planet scenes 30 days or
> less from a training chip (blue) both cover 2020-12 to 2026-02. Each month's
> value pools the crown-dates of that month across all years. The drone
> series counts a crown as an event when its mean classification
> probabilities give 1 − (1 − p_flowering)(1 − p_deciduous) > 0.5, the
> training-mask rule. The Planet series counts a crown as an event when at
> least 25 % of its clear pixels lie under a detection of the epoch-12
> temporal Mask R-CNN scoring 0.5 or more, for crowns at least 50 % clear.
> Only crowns of at least 64 Planet pixels (0.75 m) are used, and the number
> of such crowns of each species is in its title.

`crown_species_agreement.png`:

> Crown-level agreement between the Planet Mask R-CNN's event calls and the
> drone classifications, by species. The Planet scenes are 203 coregistered
> scenes 3 to 7 days from the drone flight they were coregistered to and 30
> days or less from a training chip (304,890 crown-scenes, 74 flights,
> 2021-2026). Left: precision (orange, share of Planet event calls that the
> drone also calls an event) and recall (green, share of drone events that
> Planet calls). Right: the share of crown-scenes called an event by the
> drone label (black) and by Planet (blue). A drone event is
> 1 − (1 − p_flowering)(1 − p_deciduous) > 0.5 on the crown's mean
> classification probabilities. A Planet event is at least 25 % of the
> crown's clear pixels under a detection scoring 0.5 or more. Only crowns of
> at least 64 Planet pixels (0.75 m) and at least 50 % clear are used. The
> twelve species with the most drone event crown-dates are shown above the
> rule, with the remaining species and all species pooled below it. The
> number of drone events for each row is given next to its name.

`crown_agreement_toa_vs_cal.png`:

> Crown-level agreement with the drone classifications of the epoch-12
> temporal Planet Mask R-CNN trained and applied on TOA scenes with a
> per-image 0th–99.9th percentile stretch per band (blue), and on the same
> scenes as MODIS-calibrated surface reflectance with one fixed per-band
> stretch (green), by species. Both runs are scored on the same crown-scenes:
> coregistered Planet scenes 3 to 7 days from the drone flight they were
> coregistered to, matched to that flight, 2020-2026. Top row: scenes 30 days
> or less from a training chip (202 scenes, 74 flights). Bottom row: scenes
> more than 30 days from one (48 scenes, 18 flights, mostly 2020 and 2022).
> Left: precision (share of Planet event calls that the drone also calls an
> event). Middle: recall (share of drone events that Planet calls). Right:
> the share of crown-scenes called an event by the drone label (black) and by
> each model. A drone event is 1 − (1 − p_flowering)(1 − p_deciduous) > 0.5
> on the crown's mean classification probabilities. A Planet event is at
> least 25 % of the crown's clear pixels under a detection scoring 0.5 or
> more. Only crowns of at least 64 Planet pixels (0.75 m) and at least 50 %
> clear in both runs are used. The twelve species with the most drone event
> crown-dates are shown above the rule, with the remaining species and all
> species pooled below it. The number of drone events in each row is given
> next to its name.
