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
were coregistered (139 ortho groups).

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

Pending.
