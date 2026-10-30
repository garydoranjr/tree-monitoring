# Crown temporal model: PSE-LTAE on cloud-free Planet crown series

This is a crown-level classifier that predicts the flowering-or-deciduous event
for every mapped crown on every Planet date. It reads the crown's whole
cloud-free Planet series around that date. The model choice is argued in
[temporal_crown_models.md](temporal_crown_models.md). It is evaluated against
the persistence baseline of
[planet_temporal_generalization.md](planet_temporal_generalization.md#persistence-baseline-h6),
on the same temporal folds.

Scripts:
- [`scripts/build_crown_pixel_series.py`](../scripts/build_crown_pixel_series.py)
- [`scripts/train_crown_pse_ltae.py`](../scripts/train_crown_pse_ltae.py)
- [`scripts/eval_crown_temporal_cv.py`](../scripts/eval_crown_temporal_cv.py)
- [`scripts/coreg_planet_to_nearest_drone.py`](../scripts/coreg_planet_to_nearest_drone.py), reused

Folds: [`config/temporal_folds_20261012.csv`](../config/temporal_folds_20261012.csv).

This pipeline is not in the `Snakefile`. It needs the clipped Planet scenes,
their OmniCloudMask rasters and the drone orthos on `/Volumes/Earth03`. The
command lines below are the record. Everything ran on the Mac (Apple GPU via
MPS), so there are no Slurm wrappers.

## How cloud, shadow and nodata are kept out

The exclusion happens at two levels.

1. **Pixels** (`build_crown_pixel_series.py`). A crown pixel is stored only if
   it has data and its OmniCloudMask class is clear. It must also lie more than
   `--dilate` pixels (default 1, i.e. 3 m) from any thick-cloud, thin-cloud or
   cloud-shadow pixel. OCM runs at 10 m, so its edges are coarse. Pixels that
   fail are never written, so nothing downstream can train or evaluate on them.
2. **Crown-dates** (`train_crown_pse_ltae.py`). A crown observation is valid
   only when two conditions hold:
   - at least `--min-pixels` (2) of its pixels are clear;
   - at least `--min-clear-frac` (0.5) of its valid pixels are clear, the same
     50 % rule the apply step uses.

   Invalid observations are not in the sequence at all, so they are neither
   inputs, nor attention keys, nor targets. Within a batch, padding is masked
   out of both the pixel pooling and the attention. A unit check confirmed
   this: setting padded dates and padded pixels to 1e3 leaves every unpadded
   output unchanged.

Haze that OCM calls clear is not caught. The crown-visibility random forest of
[observability_methodology.md](observability_methodology.md) is not wired in.
Terrain and illumination shade is not masked either.

## Procedure

### 1. Coregister every scene, including those near flights

Labelled observations are the scenes within 2 days of a flight. The apply
step's `coreg_nearest_drone.json` excludes exactly those scenes
(`--exclude-within-days 2`). It was therefore copied, and the missing scenes
were added to the copy with the same AROSICS settings, each against the ortho
nearest in time:

```bash
D=/Volumes/Earth03/flower
mkdir -p $D/crown_series_20261029
cp $D/202610_temporal_apply/coreg_nearest_drone.json \
   $D/crown_series_20261029/coreg_all_scenes.json
python scripts/coreg_planet_to_nearest_drone.py \
    $D/planet_clipped/4band $D/planet_clipped/ocm \
    $D/crown_series_20261029/coreg_all_scenes.json \
    --drone-dir $D/stri/globus/RGB \
    --drone-dir $D/stri/C3KW2X/BCI_50ha_timeseries_local_alignment \
    --drone-dir $D/stri/24782016/BCI_50ha_timeseries_local_alignment \
    --exclude-within-days -1 --min-clear-fraction 0.1 --workers 2
```

`--exclude-within-days -1` excludes nothing. The script skips scenes already
in the JSON, so only the 445 near-flight scenes that are at least 10 % clear
were coregistered (139 ortho groups, 2 h 7 min with 2 workers). AROSICS
succeeded on 294 of the 445, by year: 2020 8/13, 2021 28/40, 2022 15/25,
2023 68/104, 2024 89/140, 2025 79/115, 2026 7/8. The JSON then holds 1,243
scenes, 840 of them coregistered. The other 1,301 of the 2,141 scenes use
unshifted polygons and carry `coreg_ok = false`.

### 2. Extract the crown pixel series

```bash
python scripts/build_crown_pixel_series.py run \
    $D/planet_clipped/4band $D/planet_clipped/ocm $D/crown_series_20261029 \
    --crowns $D/stri/24784053/BCI_50ha_2022_09_29_crownmap_improved/BCI_50ha_2022_09_29_crownmap_improved.shp \
    --coreg-json $D/crown_series_20261029/coreg_all_scenes.json --workers 6
python scripts/build_crown_pixel_series.py concat $D/crown_series_20261029
```

**Inputs.**
- **Scenes:** the native 3 m 4-band TOA scenes, not the ×4 training chips.
- **Crown map:** the one used by the apply step, 2,280 crowns.
- **Polygons:** shrunk by `--buffer 1` m, then shifted onto each scene by its
  coreg shift. A scene that did not coregister uses unshifted polygons and is
  flagged `coreg_ok = false`, as in the apply step.

**Check against the apply step.** With `--buffer 0 --dilate 0`, the builder
was run on the 13 coregistered January 2023 scenes and compared with
`crown_rows` in the apply CSV on 28,685 crown-scenes:
- per-crown clear fraction: correlation 0.9997, mean absolute difference
  0.001;
- pixel count × 16 against the 0.75 m count: median ratio 1.00.

The whole build takes about 1 min with 6 workers.

### 3. Temporal CV

```bash
for k in 0 1 2 3 4 5; do
  python scripts/train_crown_pse_ltae.py \
      $D/crown_series_20261029/crown_pixel_series.npz \
      $D/crown_series_20261029/cv/fold$k \
      --folds config/temporal_folds_20261012.csv --fold $k --epochs 15
done
python scripts/eval_crown_temporal_cv.py $D/crown_series_20261029/cv \
    --folds config/temporal_folds_20261012.csv \
    --out-prefix $D/crown_series_20261029/cv/eval
```

**Model** (after Garnot et al. 2020).
- **Pixel-set encoder.** Each valid observation contributes up to 16 clear
  pixels: all of them if the crown has 16 or fewer, otherwise a random draw.
  Each pixel carries B, G, R, N, NDVI and the green-red index, standardised.
  A shared MLP (6 → 32 → 64) is followed by masked mean and standard-deviation
  pooling. The pooled vector is joined with log crown area, clear fraction,
  log clear-pixel count, `coreg_ok` and a SuperDove flag, then passed through
  a second MLP to d = 128.
- **Temporal transformer.** Each observation's position is the sinusoidal
  encoding of its day offset from the window centre (τ = 1000, as in the
  paper) plus a projection of day-of-year sin and cos. Then come 2
  pre-norm transformer layers (4 heads) with the padding mask, and one event
  logit per observation. The original L-TAE pools the whole series into one
  output instead.

The window is relative, not absolute, so the model cannot tie events to
calendar dates beyond the season.

**Training.**
- **Samples.** One sample per labelled (crown, training flight): a 120-day
  window centred on the flight, with ±30 days of jitter, holding at most 96
  observations.
- **Augmentation.** 20 % of the window's observations are dropped at random,
  as extra simulated cloud.
- **Loss and optimiser.** BCE on the labelled observations only. AdamW with
  a one-cycle learning rate peaking at 1e-3, batch 64.

**Labels.** Each crown's drone event
`1-(1-p_flower)(1-p_decid) > 0.5` (`load_drone`) is put on every valid
observation within 2 days of the flight. Crowns enter the labels only if they
pass two tests:
- area ≥ 36 m², the 64 px at 0.75 m of `persistence_baseline.py`;
- a tag that is unique in the crown map.

**Folds.** Fold K withholds the labels of its chips' flights. The imagery of
every date stays in the input, as it would when interpolating between flights.
By default only the 62 flights of the fold CSV give labels, as in the Mask
R-CNN CV. Two options extend this:
- `--all-flights` adds every other drone flight more than 7 days from a
  withheld flight;
- `--crown-folds N --crown-fold J` also withholds the crowns in one of N
  east-west strips.

**Prediction.** Windows are centred every 30 days. Each observation takes its
score from the window whose centre is nearest, so every prediction sees at
least 45 days of context on each side, except at the ends of the record.

**Evaluation.** Each withheld flight is scored on the crowns that have a label
there and on the nearest training flight. A crown also needs at least one valid
observation within 2 days, and its score is the mean `p_event` over those
observations. Persistence (the label on the nearest training flight) is scored
on exactly the same crown-flights.

## Results

Six-fold temporal CV, 15 epochs per fold. Training took about 90 s per epoch
on MPS, about 25 min per fold with prediction. Each fold's withheld flights
are scored on the crowns that are labelled there and on the nearest training
flight, and that have a clear view within 2 days. That gives 119,446
crown-flights on 62 flights, 5.0 % of them events. Rows are binned by the
flight's distance to the nearest training flight of the fold CSV. "Model at k"
calls the top-scoring crowns in the row, as many as persistence calls.

| Days to nearest training flight | Flights | Crown-flights | Events | Persistence P / R / F1 | Persistence area J | Model (0.5) P / R / F1 | Model (0.5) area J | Model at k P | Model at k area J | Model AP |
|---|---|---|---|---|---|---|---|---|---|---|
| 4–7 | 14 | 27,573 | 1,985 | 0.799 / 0.798 / 0.799 | **0.705** | 0.801 / 0.579 / 0.672 | 0.611 | 0.695 | 0.630 | 0.753 |
| 8–14 | 16 | 31,514 | 1,936 | 0.681 / 0.682 / 0.682 | **0.607** | 0.736 / 0.547 / 0.628 | 0.606 | 0.621 | 0.593 | 0.670 |
| 15–30 | 21 | 40,456 | 1,556 | 0.510 / 0.490 / 0.500 | 0.382 | 0.681 / 0.343 / 0.456 | 0.430 | 0.492 | **0.431** | 0.495 |
| >30 | 11 | 19,903 | 465 | 0.215 / 0.256 / 0.234 | 0.150 | 0.750 / 0.155 / 0.257 | 0.268 | 0.307 | **0.302** | 0.286 |
| Dry season (Jan–Apr) | 38 | 74,602 | 4,980 | 0.695 / 0.676 / 0.685 | 0.583 | 0.753 / 0.535 / 0.626 | 0.592 | 0.643 | **0.594** | 0.682 |
| Wet season | 24 | 44,844 | 962 | 0.374 / 0.438 / 0.403 | **0.297** | 0.703 / 0.155 / 0.254 | 0.258 | 0.310 | 0.290 | 0.300 |
| Pooled | 62 | 119,446 | 5,942 | 0.634 / 0.637 / 0.636 | 0.524 | 0.750 / 0.474 / 0.581 | 0.540 | 0.590 | **0.541** | 0.624 |

**The plan's success test is not met.** That test was to beat persistence in
the 8–30-day bins and in the wet season.
- **Within 14 days, persistence wins.** At matched calls it is 0.10 more
  precise at 4–7 days and 0.06 at 8–14 days.
- **From 15 days on, the model wins on area.**
  - At 15–30 days: area Jaccard 0.43 against 0.38. Precision at matched
    calls is 0.49 against 0.51, so the model is right more often on the large
    crowns.
  - Beyond 30 days: area Jaccard 0.30 against 0.15, and precision 0.31
    against 0.22. At that range the event is read from the imagery, not
    remembered.
- **In the wet season the two are level.** Area Jaccard is 0.29 for the model
  against 0.30, with wet-season events at 2.1 %.

**Threshold.** At 0.5 the model under-calls, with precision 0.75 and recall
0.47 pooled. Its mean score matches the event rate (0.048 against 0.050), but
its best pooled F1 comes at a threshold of 0.35 (0.60 against 0.58), and in the
wet season at 0.15 (0.35 against 0.25). A threshold tuned per season on the
training flights would recover part of the wet-season gap. It still would not
reach persistence's 0.40.

**Against Mask R-CNN.** The crown-level area Jaccard is well above the Mask
R-CNN temporal CV's pixel IoU in every bin of
[the persistence section](planet_temporal_generalization.md#persistence-baseline-h6):

| | 4–7 d | 8–14 d | 15–30 d | >30 d |
|---|---|---|---|---|
| Model area J | 0.61 | 0.61 | 0.43 | 0.27 |
| Mask R-CNN IoU, TOA e12 | 0.42 | 0.38 | 0.27 | 0.22 |

The comparison is not like for like:
- the units differ (crown area against pixels);
- this model is scored only on crowns with a clear view within 2 days.

**Model plus persistence.** The two are complementary. Averaging the model
score with the persistence label (0.5 each) ranks crowns better than either
alone, judged against the model's AP:
- pooled AP 0.70 against 0.62;
- 4–7 days: 0.85 against 0.75;
- 15–30 days: 0.56 against 0.50;
- wet season: 0.41 against 0.30.

Beyond 30 days the blend is worse (0.24 against 0.29), because there the
persistence label is mostly wrong.

So the natural next model takes the nearest-flight label as an input
alongside the imagery: a per-crown "last known state" token, with its age in
days. Interpolating between flights is how the model would be used in any
case.

**Caveats.**
- Each fold was run once with seed 0, so run-to-run spread is unmeasured.
- No crown hold-out (`--crown-folds`) run yet. The model sees every crown's
  labels on the training flights, so part of its skill near the flights may
  be memory of those crowns, as with Mask R-CNN (H6).
- 15 epochs was not tuned. The training loss was still falling slowly (0.08
  to 0.09 at epoch 15 in every fold).

Outputs are in `$D/crown_series_20261029/cv/`:
- `fold{0..5}/` for each fold: `model.pt`, `predictions.csv`, `params.json`
  and the log;
- `eval_pairs.csv` and `eval_summary.csv`.
