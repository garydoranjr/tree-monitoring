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

Labeled observations are the scenes within 2 days of a flight. The apply
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
  Each pixel carries B, G, R, N, NDVI and the green-red index, standardized.
  A shared MLP (6 → 32 → 64) is followed by masked mean and standard-deviation
  pooling. The pooled vector is joined with log crown area, clear fraction,
  log clear-pixel count, `coreg_ok` and a SuperDove flag, then passed through
  a second MLP to d = 128.
- **Temporal transformer.** Each observation's position is the sinusoidal
  encoding of its day offset from the window center (τ = 1000, as in the
  paper) plus a projection of day-of-year sin and cos. Then come 2
  pre-norm transformer layers (4 heads) with the padding mask, and one event
  logit per observation. The original L-TAE pools the whole series into one
  output instead.

The window is relative, not absolute, so the model cannot tie events to
calendar dates beyond the season.

**Training.**
- **Samples.** One sample per labeled (crown, training flight): a 120-day
  window centered on the flight, with ±30 days of jitter, holding at most 96
  observations.
- **Augmentation.** 20 % of the window's observations are dropped at random,
  as extra simulated cloud.
- **Loss and optimizer.** BCE on the labeled observations only. AdamW with
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

**Prediction.** Windows are centered every 30 days. Each observation takes its
score from the window whose center is nearest, so every prediction sees at
least 45 days of context on each side, except at the ends of the record.

**Evaluation.** Each withheld flight is scored on the crowns that have a label
there and on the nearest training flight. A crown also needs at least one valid
observation within 2 days, and its score is the mean `p_event` over those
observations. Persistence (the label on the nearest training flight) is scored
on exactly the same crown-flights.

## Results

Six-fold temporal CV, 15 epochs per fold. Training took about 90 s per epoch
on MPS, about 25 min per fold with prediction. Each fold's withheld flights
are scored on the crowns that are labeled there and on the nearest training
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

### By event type

The event merges two processes, flowering and leaf loss. Each true event is
typed from the drone probabilities on the withheld flight:
- **Flowering:** `p_flower` > 0.5 and at least `p_decid`.
- **Leaf loss:** `p_decid` > 0.5 and greater than `p_flower`.
- **Mixed:** neither component is above 0.5, but the combined event is.

Each type is scored against its own events plus every non-event.
Precision therefore counts every false call and is not comparable across
types. The table gives recall and area Jaccard, with the model at
persistence's number of calls.

| Type | Events | Days / season | Persistence recall | Model recall at k | Persistence area J | Model area J at k | Model AP |
|---|---|---|---|---|---|---|---|
| Leaf loss | 3,969 | 4–7 | **0.876** | 0.769 | **0.728** | 0.658 | 0.791 |
| | | 8–14 | **0.785** | 0.756 | 0.634 | **0.644** | 0.756 |
| | | 15–30 | 0.625 | 0.628 | 0.409 | **0.474** | 0.594 |
| | | >30 | 0.371 | **0.609** | 0.144 | **0.288** | 0.435 |
| | | dry | 0.774 | 0.743 | 0.608 | **0.638** | 0.750 |
| | | wet | **0.635** | 0.567 | **0.320** | 0.299 | 0.428 |
| | | pooled | 0.757 | 0.730 | 0.554 | **0.584** | 0.713 |
| Flowering | 1,076 | 4–7 | **0.639** | 0.449 | **0.195** | 0.120 | 0.306 |
| | | 8–14 | **0.469** | 0.235 | **0.117** | 0.056 | 0.126 |
| | | 15–30 | **0.259** | 0.148 | **0.059** | 0.039 | 0.035 |
| | | >30 | 0.125 | **0.175** | 0.031 | **0.098** | 0.031 |
| | | dry | **0.472** | 0.304 | **0.119** | 0.077 | 0.160 |
| | | wet | **0.208** | 0.108 | 0.046 | **0.058** | 0.018 |
| | | pooled | **0.406** | 0.266 | **0.095** | 0.071 | 0.125 |
| Mixed | 897 | pooled | **0.386** | 0.319 | 0.103 | 0.100 | 0.096 |

- **Leaf loss is two-thirds of the events, and there the model is level with
  or ahead of persistence from 8 days on.**
  - Area Jaccard at matched calls: 0.64 against 0.63 at 8–14 days, 0.47
    against 0.41 at 15–30 days, and 0.29 against 0.14 beyond 30 days.
  - Persistence keeps only the first week, and the wet season (0.32 against
    0.30, on 490 events).
  - Restricted to leaf loss, the plan's 8–30-day test is met. The wet-season
    test is not.
- **The model hardly sees flowering.**
  - Flowering AP is 0.13 pooled and 0.02 in the wet season.
  - At threshold 0.5 its flowering recall is 0.18 overall and 0.004 in the
    wet season.
  - Persistence finds 64 % of flowering events at 4–7 days and 47 % at 8–14
    days. The model beats it only beyond 30 days, on 160 events.
  - The combined-event gap within 14 days is therefore mostly flowering.
- **Mixed events are weak for both.** These are 15 % of events, crowns that
  pass 0.5 only in combination, and they sit near the label threshold by
  construction.

**Do persistence's hits come from crowns the drone classifier always calls
events?** This was checked once, from `eval_pairs.csv` and the drone tables.
For each crown, the check took the share of all drone flights on which it is
an event.
- **Flowering: no.** Persistence's flowering hits and misses are on crowns
  with similar event rates (median 0.09 against 0.08). Its flowering lead
  therefore looks like real persistence of flowering beyond a week, not a
  repeated per-crown classifier error.
- **Leaf loss: unclear.** Its hits are on crowns that are events far more
  often (median 0.26 against 0.14). 35 % of leaf-loss events are on crowns
  that are events on at least 30 % of flights. Persistence recalls 0.90 of
  those, against 0.68 of the rest. The model also scores those crowns high
  (mean p 0.77 against 0.52). So these may be genuinely long or recurring
  leaf loss, or a classifier bias that both methods pick up. Only
  independent labels can separate the two.

**Caveats.**
- Each fold was run once with seed 0, so run-to-run spread is unmeasured.
- In the temporal CV the model sees every crown's labels on the training
  flights. Part of its skill near the flights is memory of those crowns, as
  with Mask R-CNN (H6). See
  [Spatial generalization](#spatial-generalization-west-to-east).
- 15 epochs was not tuned. The training loss was still falling slowly (0.08
  to 0.09 at epoch 15 in every fold).

Outputs are in `$D/crown_series_20261029/cv/`:
- `fold{0..5}/` for each fold: `model.pt`, `predictions.csv`, `params.json`
  and the log;
- `eval_pairs.csv`, `eval_summary.csv` and `eval_by_type.csv`.

## Spatial generalization (west to east)

The question is how the model does on crowns it never saw labels for.

### The split

The split copies the Mask R-CNN spatial experiments
([planet_training_chips_50ha.md](planet_training_chips_50ha.md)).
- **The line.** All 142 chips share one footprint. `_split_window` cuts
  every one of them at x = 626281.5 m (UTM 17N): the model trains on the left
  window and is tested on the right one.
- **Crowns.** Each crown goes to the side of its centroid. The crown map has
  1,170 crowns to the west and 1,110 to the east; 25 straddle the line.
- **Event rates are similar on the two sides:** 4.8 % of the scored west
  crown-flights and 5.1 % of the east.

### Three runs on the same crowns

All three runs are scored on the same 60,089 eastern crown-flights over the
62 flights. The truth, persistence and day bins are identical across them,
because spatial runs are scored fold by fold like the temporal CV.

| Run | Eastern crowns' labels | Test flight's labels |
|---|---|---|
| Temporal CV (the run above, rescored on the east) | seen, on other flights | withheld |
| **Spatial** (`--fold all --split-x 626281.5`) | never seen | seen, for western crowns |
| **Spatio-temporal** (`--fold K --split-x 626281.5`) | never seen | withheld |

Persistence reads the eastern crowns' own labels. It is a reference here,
not a competitor: no such labels exist off the drone footprint.

```bash
D=/Volumes/Earth03/flower/crown_series_20261029
S=$D/crown_pixel_series.npz; F=config/temporal_folds_20261012.csv; X=626281.5
python scripts/train_crown_pse_ltae.py $S $D/cv_spatial/west_east --folds $F \
    --fold all --split-x $X --test-side east --epochs 15
for k in 0 1 2 3 4 5; do
  python scripts/train_crown_pse_ltae.py $S $D/cv_spatiotemporal/fold$k \
      --folds $F --fold $k --split-x $X --test-side east --epochs 15
done
python scripts/eval_crown_temporal_cv.py $D/cv --folds $F --split-x $X \
    --side east --benchmark-chips config/spatial_test_chips_56.txt \
    --out-prefix $D/cv/eval_east
for r in cv_spatial cv_spatiotemporal; do
  python scripts/eval_crown_temporal_cv.py $D/$r --folds $F \
      --benchmark-chips config/spatial_test_chips_56.txt --out-prefix $D/$r/eval
done
```

**Training data.** The spatial run had 58,873 crown-flight samples, against
about 100,000 per temporal fold. The spatio-temporal folds had about 50,000.
All runs used 15 epochs, so the spatial runs took fewer optimizer steps.
Each spatial run took about 15 min on MPS.

### Results

"Benchmark flights" are the 25 flights of the 56 phantom 2020–23 chips of
the Mask R-CNN spatial benchmark. Area J and recall are at persistence's
number of calls.

| Days to nearest training flight | Temporal AP | Spatial AP | Spatio-temporal AP | Temporal area J | Spatial area J | Spatio-temporal area J | Persistence area J (reference) |
|---|---|---|---|---|---|---|---|
| 4–7 | **0.761** | 0.607 | 0.579 | **0.649** | 0.581 | 0.546 | 0.745 |
| 8–14 | **0.702** | 0.613 | 0.578 | **0.636** | 0.595 | 0.551 | 0.634 |
| 15–30 | **0.573** | 0.562 | 0.514 | 0.493 | 0.488 | **0.495** | 0.434 |
| >30 | 0.342 | **0.380** | 0.292 | 0.375 | **0.397** | 0.295 | 0.156 |
| Dry season | **0.713** | 0.613 | 0.578 | **0.630** | 0.588 | 0.553 | 0.618 |
| Wet season | **0.376** | **0.376** | 0.337 | 0.332 | **0.351** | 0.321 | 0.330 |
| Benchmark flights | **0.720** | 0.611 | 0.571 | **0.600** | 0.565 | 0.531 | 0.545 |
| Pooled | **0.667** | 0.576 | 0.540 | **0.580** | 0.548 | 0.516 | 0.562 |

- **Not seeing a crown's labels costs about 0.09 in pooled AP** (0.667 to
  0.576). The whole cost is near the flights:
  - 0.15 at 4–7 days and 0.09 at 8–14 days;
  - nothing from 15 days on, where the spatial run is level (15–30) or ahead
    (beyond 30 days).
  - So the temporal CV's edge within two weeks is memory of the crowns' state
    on nearby flights (H6), and the imagery-only skill is what the spatial
    run shows.
- **Not seeing the date's labels costs a further 0.04** (0.576 to 0.540).
  The spatial run learns, from the western crowns, what that flight's events
  look like in the imagery.
- **On entirely new crowns and dates, area Jaccard is 0.52 pooled.** That is
  0.55 within two weeks of a training flight, and 0.50 at 15–30 days.
  - The spatio-temporal model is above persistence beyond 15 days, even
    though persistence reads the crowns' own labels.
  - The spatial run is above persistence on the benchmark flights (0.565
    against 0.545).
- **Against Mask R-CNN:** on its spatial benchmark, Mask R-CNN peaked at
  pixel IoUs of 0.27–0.32 across training sets: 0.293 for `full_mkl`, the
  142-chip TOA run, and 0.318 calibrated. The crown-level spatial area Jaccard
  on the same 25 flights is 0.57. This is context, not a like-for-like comparison:
  - pixels against crowns;
  - the 384 m right window against all 1,110 eastern crowns;
  - every labeled crown pixel against crowns with a clear view within 2 days.

**By event type**, pooled over the 62 flights:

| Type | Temporal AP | Spatial AP | Spatio-temporal AP | Temporal area J | Spatial area J | Spatio-temporal area J | Persistence area J |
|---|---|---|---|---|---|---|---|
| Leaf loss (2,175 events) | 0.746 | 0.646 | 0.618 | 0.621 | 0.586 | 0.565 | 0.588 |
| Flowering (431 events) | 0.083 | 0.060 | 0.027 | 0.058 | 0.064 | 0.034 | 0.081 |

- **Leaf loss transfers to new crowns.**
  - On new crowns at new dates: AP 0.62, area Jaccard 0.57.
  - Beyond 15 days the spatio-temporal run beats persistence by area Jaccard
    (0.56 against 0.46 at 15–30 days).
- **Flowering does not transfer.**
  - Already weak in the temporal CV (AP 0.08).
  - On new crowns at new dates its AP is 0.027. The base rate in that subset
    is 0.0075, so the model ranks flowering crowns only a little better than
    chance.
  - What flowering skill there is depends on having seen the crown.

**Caveats.**
- One seed per run.
- The spatial runs had fewer training samples and steps than the temporal
  CV at the same 15 epochs.
- Not run:
  - east to west, which would check whether the direction matters;
  - a crown-level Mask R-CNN score on the same eastern crown-flights, which
    would need its spatial checkpoints applied on gattaca2.

Outputs:
- `$D/crown_series_20261029/cv_spatial/west_east/`;
- `$D/crown_series_20261029/cv_spatiotemporal/fold{0..5}/`;
- `eval_*` in each of the two;
- `cv/eval_east_*`: the temporal CV rescored on the eastern crowns.
