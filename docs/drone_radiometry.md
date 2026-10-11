# Phantom vs mavic colour calibration and the flowering classifier

Script: [`scripts/compare_drone_radiometry.py`](../scripts/compare_drone_radiometry.py).
Outputs (October 2026 run): `/Volumes/Earth03/flower/figs/202610_updates/radiometry/`
(`radiometry_*.csv`, `radiometry_paired_fits.json`, `transforms.json`,
`mavic_linear_<date>.yml`, `renorm_cdf_distances.csv`, `reclassify_*.csv`,
`*.png/.pdf`).

## Question

On the two days both drones flew (2024-03-06 and 2024-03-18), the drone
SegFormer flowering model calls about 5× more crowns flowering on mavic than on
phantom. Taking mavic P > 0.5 and phantom (local) P < 0.2, 42 crown-dates
flower on mavic only, against 1 on phantom only and 16 on both
(`plot_update_figures.py same-date`).

The model was trained on uint8 phantom chips. The uint16 mavic mosaics reach it
through the fixed per-band gain and offset in
[`config/crown_classification_mavic.yml`](../config/crown_classification_mavic.yml).
Those coefficients were fitted by matching p2 and p98 pooled over *different*
dates of each camera.

The hypothesis tested here is that the excess comes from colour calibration:
the two cameras lack colour constancy, and the fixed mapping does not correct
for it. Because both drones imaged the same canopy on the same days, the two
dates make a controlled test.

## Result

Colour calibration accounts for most of the excess mavic flowering calls.

A transform that matches the mavic colour distribution to phantom's removes
**33 of the 42** mavic-only positives (`mkl`, fitted on the other date). It
keeps **13 of the 16** crowns flowering on both cameras, and it creates no
positives among 60 control crowns. A per-band histogram match does about the
same: it removes 36 of 42 and keeps 11 of 16.

A transform that matches only each band's mean and standard deviation does
little: it removes 14 of 42. The difference lies in the tails of each band and
in the colour (cross-band) structure, not in overall brightness.

### 1. Same-day RGB distributions (`distributions`)

The comparison uses the joint 50 ha footprint in the uint8 values the model
receives.

**Overall brightness differs by about as much as one camera varies from day to
day.**

| Comparison | Mean \|CDF difference\| |
|---|---|
| phantom vs mavic, same day | 0.052 (03-06), 0.062 (03-18) |
| phantom vs itself, adjacent flights | 0.030, 0.042, 0.084 |
| mavic vs itself (03-06 vs 03-18) | 0.069 |

**The hue shift is consistent on both dates.**

| | phantom 03-06 → 03-18 | mavic − phantom, 03-06 | mavic − phantom, 03-18 |
|---|---|---|---|
| g = G/(R+G+B) | −0.002 | +0.020 | +0.022 |
| b = B/(R+G+B) | −0.003 | −0.018 | −0.011 |
| blue median, phantom → mavic | | 94 → 72 | 110 → 93 |

Mavic is greener and has a darker blue band. It also has more contrast: a
wider p2–p98 range in every band, and a broader chromaticity distribution, i.e.
more saturated. Both cameras shift the same way on both days. The shift is 3–4×
larger than phantom's own hue change between the two dates.

**The scenes also differ in illumination.** Matching 1 m blocks give phantom ~
mavic R² of only 0.28–0.53. A full 3×3 colour matrix adds less than 0.01 R²
over per-band fits. In the example chips the phantom flights look visibly
hazier and flatter than mavic, most of all on 2024-03-18.

**The disagreeing crowns are not shifted more than the rest.** The 40
mavic-only crowns in the per-crown table have the same colour shift as the
crowns where neither camera says flowering. Their green shift is, if anything,
smaller (Mann–Whitney p = 0.002). The colour cast is global, and the
classifier's response to it is what concentrates on particular crowns.

### 2. Renormalization (`fit-renorm`)

Three mavic-uint16 → uint8 maps were fitted to the same-day phantom
distribution over the joint footprint:

| Name | Method |
|---|---|
| `linear` | per-band gain and offset matching each band's mean and standard deviation |
| `histmatch` | per-band quantile mapping |
| `mkl` | linear Monge–Kantorovich colour transfer (Pitié & Kokaram 2007), matching the mean and the full 3×3 covariance |

Each was also fitted in reverse, mapping phantom onto mavic-as-the-model-sees-it.

Each map was applied in two ways:
- **in-sample:** fitted on the date being corrected, as an upper bound;
- **cross-date:** fitted on the other date, as the honest test.

In-sample, every map brings the CDF distance below 0.012. The maps fitted on the
two dates differ, however: the red gain is 0.00386 on 03-06 and 0.00273 on
03-18. Cross-date, the distance falls only from 0.062 to 0.033–0.037 on 03-18
and from 0.052 to 0.032–0.048 on 03-06. So per-flight exposure and illumination
vary on top of any fixed camera difference.

### 3. Re-classification (`reclassify`)

The flowering and deciduous models were re-run locally on CPU, following the
production path in [`scripts/crown_classification.py`](../scripts/crown_classification.py):
- the same `centered_window` crop;
- `SegformerImageProcessor` at 512;
- sigmoid of the class-1 logit;
- mean over the crown polygon.

The crowns were 121 crown-dates: every disagreement case, plus 60 controls
where both cameras give P(flowering) < 0.1.

**Reproduction check.** The local baseline matches the archived HPC crown means
from `same_date_crowns.csv`:

| | Spearman ρ | Median \|Δ\| |
|---|---|---|
| phantom flower | 0.95 | 0.002 |
| mavic flower | 0.97 | 0.009 |
| phantom decid | 0.95 | 0.001 |
| mavic decid | 0.90 | 0.001 |

**Crowns with P(flowering) > 0.5 on mavic, by transform:**

| mavic transform | mavic only (of 42) | both (of 16) | controls (of 60) | median P, mavic-only |
|---|---|---|---|---|
| baseline (current config) | 42 | 16 | 0 | 0.66 |
| linear, cross-date | 28 | 16 | 0 | 0.55 |
| linear, in-sample | 32 | 16 | 0 | 0.56 |
| histmatch, cross-date | 6 | 11 | 0 | 0.10 |
| histmatch, in-sample | 5 | 14 | 0 | 0.19 |
| mkl, cross-date | 9 | 13 | 0 | 0.15 |
| mkl, in-sample | 6 | 13 | 0 | 0.15 |

The full table, including the deciduous cases, is in
`reclassify_summary.csv`. The deciduous outputs barely change under any
transform.

**The suppression is selective.** Under `mkl` cross-date, 21% of the mavic-only
positives survive, against 81% of the crowns flowering on both cameras. The
survivors look plausibly real:
- 6 of the 9 are *Jacaranda copaia* (crowns 748, 1386, 1461, 1766, 2269), a
  dry-season flowerer that was already suspected of real early flowering.
- The others are *Trichilia tuberculata* 1150, *Quararibea asterolepis* 606
  and *Miconia argentea* 873. Crown 1150 shows orange flower colour in the mavic
  RGB.

Among the 16 crowns flowering on both cameras, the two lost are *Cordia
alliodora* 747 and *Dipteryx oleifera* 1941.

**The reverse test is only partial.** Making phantom look like mavic raises
phantom's positives among the 42 mavic-only crowns from 0 to 5–9 (`mkl` and
`histmatch`). Colour explains why mavic over-fires, but it does not fully
explain why phantom stays quiet. The hazier, lower-contrast phantom texture
likely also suppresses some flowering signal.

## Production config

[`config/crown_classification_mavic_mkl.yml`](../config/crown_classification_mavic_mkl.yml)
holds a single fixed `mkl` colour transfer for every mavic date. It is fitted
on the two shared dates pooled, and written by `fit-renorm`.
[`scripts/crown_classification.py`](../scripts/crown_classification.py) reads
it through the existing `--scaling-config` option: a `color_transfer` key
selects the 3×3 transform, and the per-band `uint16_to_uint8` form still works.

Over both dates, the pooled transform brings the CDF distance to phantom from
0.052 to 0.018 and from 0.062 to 0.025. Run through the unchanged production
code path (`extract_centered_window` → `preprocess` → `apply_model`) on the same
121 crown-dates, it calls flowering on:
- 6 of 42 mavic-only crowns;
- 14 of 16 crowns flowering on both cameras;
- 0 of 60 controls.

With the old config, the same path reproduces the `reclassify` baseline to
within 3e-8.

To re-run the whole mavic series on the HPC:
```bash
./run_crown_pipeline.sh config/pipeline_mavic_mkl.sh
```
This writes to `/scratch/tree-monitoring/results/globus_mkl`, so the per-band
results in `results/globus` stay in place. The file names match the old run's
(`<stem>_classifications.tif`), so keep the two runs in separate directories
when copying them back to Earth03.

## Caveats and next steps

- **The target is phantom C3KW2X on two dates, not the training domain.** The
  model was trained on 24782016 phantom chips, and phantom on 2024-03-18 is hazy.
  For production, refit `mkl` or `histmatch` against a pooled 24782016 sample;
  this extends the current p2/p98 match in the config to the full distribution
  and covariance. Then re-run the full mavic time series and check whether the
  flowering fraction falls to phantom levels.
- **Only disagreement and control crowns were re-run.** A full-plot run (2,280
  crowns × 2 dates) would give the corrected crown fractions and Spearman ρ for
  the whole plot.
- **Environment:** the checkpoints are pickled whole models and need
  `transformers<5`, which is now pinned in `environment.yml`. They are stored
  locally under `/Volumes/Earth03/flower/results/models/drone_{flower,decid}_geo_models/`.

## Figures

**`rgb_cdfs.png`:** Per-band distribution of drone pixel values as received by
the flowering classifier, for phantom (green, C3KW2X global-alignment uint8
mosaic) and mavic (red, M3M uint16 mosaic mapped to uint8 by the fixed gains
and offsets in `crown_classification_mavic.yml`). Each panel plots the
cumulative fraction of pixels against the uint8 value of one band (R, G or B);
dotted lines restrict the sample to pixels inside the 2,280 crowns of the
2022-09-29 crown map. Pixels are every 4th native pixel (about 4.5–4.7 cm) in
each direction within the joint valid footprint of both mosaics inside the
crown-map hull, about 15–16 M pixels per mosaic, on the two shared flight dates
2024-03-06 (top) and 2024-03-18 (bottom).

**`rgb_paired_hexbin.png`:** Pixel-level agreement between phantom and mavic on
the same day, as 1 m block means of the uint8 classifier input. Each panel shows
the log-scaled density of the 532,547 joint 1 m blocks inside the crown-map hull
for one band (R, G or B), with mavic on the x-axis and phantom on the y-axis.
The grey line is 1:1 and the coloured line is the ordinary-least-squares fit of
phantom on mavic. Block means are area averages of the native-resolution
mosaics resampled to a common 1 m grid. Data are the shared flight dates
2024-03-06 (top) and 2024-03-18 (bottom).

**`chromaticity.png`:** Colour balance of the two drone cameras. The left and
centre panels show density contours (enclosing the densest 50%, 90% and 99% of
the pixel distribution) of chromaticity, r = R/(R+G+B) on the x-axis against
g = G/(R+G+B) on the y-axis, computed from the uint8 classifier input for
phantom (green) and mavic (red). Pixels are the same native-resolution samples
as `rgb_cdfs.png`, for 2024-03-06 (left) and 2024-03-18 (centre). The
right panel shows each crown's mavic-minus-phantom change in mean r (x-axis)
and mean g (y-axis). Crown means are taken over the 1 m blocks of each crown in
the 2022-09-29 crown map, pooled over both dates. Points are coloured by the
archived classifier outcome: grey, neither camera P(flowering) > 0.5; black,
both; red, mavic > 0.5 and phantom < 0.2; green, phantom > 0.5 and mavic < 0.3.

**`reclassify_examples.png`:** Crowns flowering on mavic but not phantom,
re-classified after colour transfer. Each row is one crown-date, labelled with
its date, crown index in the 2022-09-29 crown map and species; the crowns are
the 8 with the highest archived mavic P(flowering) among the 42 mavic-only
crown-dates. From left to right, the six panels are:
1. phantom RGB, the C3KW2X local-alignment uint8 mosaic;
2. phantom P(flowering);
3. mavic RGB under the current fixed uint16 → uint8 scaling;
4. mavic P(flowering) under that scaling;
5. mavic RGB after linear Monge–Kantorovich colour transfer to the same-day
   phantom distribution, fitted on the *other* shared date;
6. mavic P(flowering) after that transfer.

Each RGB panel is the classifier's own input window: at least 512 × 512 pixels
centred on the crown, with the crown outline in cyan. Each probability panel
shows the flowering SegFormer output (sigmoid of the class-1 logit; black = 0,
yellow-white = 1) over that window. The value in its title is the mean over the
crown polygon, from a local CPU re-run of the production checkpoint.

## Reproduce

```bash
python scripts/compare_drone_radiometry.py distributions   # ~17 min from Earth03
python scripts/compare_drone_radiometry.py fit-renorm      # seconds, uses cached samples
python scripts/compare_drone_radiometry.py reclassify      # ~25 min on CPU
```

`distributions` and `reclassify` read `same_date_crowns.csv`, which is written
by `python scripts/plot_update_figures.py same-date`.
