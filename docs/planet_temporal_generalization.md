# Temporal generalization of the Planet Mask R-CNN, and applying it between drone flights

Scripts:
- [`scripts/make_temporal_folds.py`](../scripts/make_temporal_folds.py)
- [`scripts/train_planet_image_maskrcnn.py`](../scripts/train_planet_image_maskrcnn.py), with `--folds/--fold` and `--save-epochs`
- [`scripts/eval_temporal_cv.py`](../scripts/eval_temporal_cv.py)
- [`scripts/plot_temporal_cv.py`](../scripts/plot_temporal_cv.py)
- [`scripts/coreg_planet_to_nearest_drone.py`](../scripts/coreg_planet_to_nearest_drone.py)
- [`scripts/apply_planet_maskrcnn.py`](../scripts/apply_planet_maskrcnn.py)
- the Slurm wrappers `scripts/slurm/planet_temporal_{cv,eval,final}.sh`,
  `scripts/slurm/planet_coreg_nearest.sh` and `scripts/slurm/planet_apply.sh`

Settings: [`config/temporal_cv_20261012.sh`](../config/temporal_cv_20261012.sh).
Folds: [`config/temporal_folds_20261012.csv`](../config/temporal_folds_20261012.csv).
Chips: `20261012_full_label_extended2_mkl_x4_coreg_4band_stretch_stats_curated`
(142 chips; see [planet_training_chips_50ha.md](planet_training_chips_50ha.md)).

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

A fold trains on about 118 chips × 2 windows. The October runs took 70 s/epoch
for 142 single-window samples, so expect about 2 min/epoch and about 1.5 h per
fold on an A100.

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
1.4 m N). Expect a success rate near the training build's 40 %. A scene that
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

*To be filled in after the gattaca2 runs.*

| | Pooled `map_50` | `test_iou` | Precision | Recall |
|---|---|---|---|---|
| Temporal CV, epoch E* | | | | |

| Days to nearest training chip | Chips | `map_50` | `test_iou` | Precision | Recall |
|---|---|---|---|---|---|
| 0–3 | 7 | | | | |
| 4–7 | 27 | | | | |
| 8–14 | 51 | | | | |
| 15–30 | 35 | | | | |
| >30 | 22 | | | | |

The per-fold spread is printed by `eval_temporal_cv.py`.

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
