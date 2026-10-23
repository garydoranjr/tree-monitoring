# Planet radiometric calibration against MODIS

PlanetScope imagery of the 50ha plot comes from 180+ Dove sensors of three
generations (Dove Classic `PS2`, Dove-R `PS2.SD`, SuperDove `PSB.SD`), each with
its own radiometric calibration, viewed under sun/view geometries that change
through the year. This page describes a replication of the cross-calibration of
PlanetScope surface reflectance (SR) to BRDF-adjusted MODIS developed by Jin
Wu's group at HKU:

- Wang, J., Yang, D., Detto, M., Nelson, B.W., Chen, M., Guan, K., Wu, S., Yan,
  Z., Wu, J. (2020). Multi-scale integration of satellite remote sensing
  improves characterization of dry-season green-up in an Amazon tropical
  evergreen forest. *Remote Sensing of Environment* 246, 111865.
  doi:10.1016/j.rse.2020.111865 (method paper; Tapajós k67).
- Wu, S., Wang, J., et al. (2021). Monitoring tree-crown scale autumn leaf
  phenology in a temperate forest with an integration of PlanetScope and drone
  remote sensing observations. *ISPRS J. Photogramm. Remote Sens.* 171, 36–48.
  doi:10.1016/j.isprsjprs.2020.10.017 (Sect. 3.3 gives the step-by-step method
  used here).
- Reused in Wang et al. 2023 RSE (doi:10.1016/j.rse.2022.113429), Zhao et al.
  2022 RSE (doi:10.1016/j.rse.2022.113310) and Song et al. 2024 RSE
  (doi:10.1016/j.rse.2024.114027).

## Method

1. **Planet SR over a large box.** The scenes behind the 50ha clips are
   re-ordered as `analytic_sr_udm2`, clipped to the same BCNM box (~12 × 14 km,
   `config/planet_search_sr.yml`) as the TOA orders. 2,138 of the 2,146 scenes
   behind `planet_clipped/` have 4-band SR; the 8 without are Dove Classic
   scenes from satellite `1105`.
2. **Cloud and water masks.** OmniCloudMask runs on each SR box clip
   (`cloud_mask_planet.py`, unchanged). A static water mask
   (`build_bcnm_water_mask.py`) flags pixels whose clear NIR SR is below 0.08
   in more than half of their clear observations; 42% of the box is water.
3. **MODIS reference** (`build_modis_reference.py`). MCD43A1 v061 kernel
   weights for MODIS B3/B4/B1/B2 (blue/green/red/NIR) are evaluated with the
   RossThick-LiSparse-Reciprocal model at nadir view and 45° solar zenith
   (Kvol = −0.0459, Kgeo = −1.1068). Quality control keeps mandatory quality 0
   or 1 and requires QA30 (more than 30% of an 11 × 11 pixel, ~5 km window
   passes). Each pixel, band and year is gap-filled daily by the number N of
   good days: N ≥ 250, shape-preserving (PCHIP) cubic through the pixel;
   50 ≤ N < 250, the 5 km neighbourhood seasonality (that year, Gaussian-kernel
   smoothed with σ = 8 days, if it has ≥ 250 good days, else the 2005–2026
   day-of-year mean) shifted to the pixel's good days; N < 50, the
   neighbourhood day-of-year mean. The neighbourhood series is the pixel's
   long-term median plus the mean anomaly of good neighbours of the same
   cover class (land where the long-term median NIR is ≥ 0.15).
4. **Histogram matching** (`calibrate_planet_modis.py`). Scenes are grouped by
   date and satellite and mosaicked on a 3 m grid over the box; clear land
   reflectance is aggregated onto the MODIS 500 m sinusoidal grid. MODIS pixels
   that are ≥ 99% land, whose 3 × 3 neighbourhood is ≥ 90% land, and that are
   ≥ 95% clear in Planet are used. Per band, offset = μ_MODIS − gain · μ_Planet
   with gain fixed at 1 (default, `--gain-mode fixed`); the HKU gain
   σ_MODIS / σ_Planet is available as `--gain-mode sd` and is always reported.
   Groups with fewer than 30 usable pixels stay uncalibrated. The correction
   is applied to the 3 m SR and each scene is clipped to the 50ha plot.

### Deviations from HKU

- **Box:** the existing BCNM order box (~12 × 14 km, mostly lake) instead of a
  10 × 10 km box centred on the site.
- **Grouping:** one fit per date *and satellite* instead of per date, so two
  sensors are never forced to share a gain.
- **Cloud mask:** OmniCloudMask instead of the group's STI-ACSS.
- **Climatology:** MODIS 2005–2026 (HKU: the 20 years before their study year).
- **Water:** masked automatically from NIR SR rather than by hand in ENVI, with
  an explicit one-MODIS-pixel land buffer.
- **Gain:** fixed at 1 (mean matching only) instead of σ_MODIS / σ_Planet; see
  *Sources of uncertainty*.
- **Gap-fill neighbourhood and smoothers.** HKU's neighbourhood is the plain
  mean of good pixels in the 5 km window, smoothed or interpolated with cubic
  splines. Around BCI ~40% of each window is lake, and the share of water
  among the good pixels changes with the season (59% in July vs ~40% in other
  months for the 50ha pixel), so the plain mean followed the land/water mix,
  not the forest. Wet-season fills were also wild: Jun–Sep 2022 at the 50ha
  pixel ranged 0.08–1.19 NIR (observed 0.29–0.33), and over all land pixels
  the 1st/99th percentiles of flag-2 fills were −1.69/1.88. Two causes: the
  splines overshot across long gaps, and the neighbourhood mean divided by a
  good-pixel fraction that `uniform_filter` leaves at ~1e-17 instead of 0 when
  no pixel in the window is good, giving values up to 0.86 NIR on days with
  no good neighbour (a 0.44 NIR hump over the plot in July 2021). The
  neighbourhood is now a same-class anomaly mean that needs at least half a
  good pixel, and the splines are replaced by PCHIP and a Gaussian-kernel
  mean. In the rebuilt 2005–2026 reference, land NIR 0.1st/50th/99.9th
  percentiles are 0.084/0.293/0.411 observed, 0.115/0.303/0.395 for flag 2
  and 0.130/0.304/0.390 for flag 3, and the plot-mean NIR stays within
  0.307–0.334 in July 2021 (earlier versions kept as
  `reference_nadir_sza45_v1_spline.nc` and `_v2_roundoff.nc`).

## Sources of uncertainty

Tracked per date-satellite group and band in `results/planet_modis_calibration.csv`
where they can be quantified; the rest are known but not yet quantified.

1. **Gain (σ ratio) — not applied.** The box's usable land is nearly uniform
   forest: in the Feb-2024 pilot the spatial SD across matched 500 m pixels was
   0.004 (Planet) / 0.008 (MODIS) in blue and ~0.015 in NIR, so σ_MODIS /
   σ_Planet mostly measures noise and Planet–MODIS misregistration, not a
   radiometric scale. Pilot gains: blue 1.5–2.6, green 1.2–2.2, red 1.0–1.6,
   NIR 0.7–1.0, with bootstrap 95% interval widths 0.3–0.6 — comparable to
   the day-to-day scatter. Applying such gains would stretch 3 m crown
   contrast by up to 2.5×. Reported as `gain_sd_ratio` with
   `gain_sd_ratio_ci_lo/hi`. With gain fixed at 1, any true gain error g − 1
   leaves an error (g − 1)(ρ − μ_Planet) at reflectance ρ: zero at the box
   land mean, growing for crowns far from it. Dropping the gain costs almost
   nothing in fit: on the 11 calibrated pilot groups the median out-of-sample
   (checkerboard) RMSE against MODIS is 0.0060 / 0.0055 / 0.0050 / 0.0152
   (blue / green / red / NIR) with gain 1 vs 0.0062 / 0.0058 / 0.0054 / 0.0145
   with the σ ratio, down from 0.0125 / 0.0129 / 0.0102 / 0.0527 uncalibrated.
2. **Offset sampling error.** Pixel-bootstrap 95% interval of the applied
   offset (`offset_ci_lo/hi`): ±~0.001 in the visible and ±~0.003 in NIR
   (~1% of forest NIR) for ~100 pixels in the pilot; in the full run the
   median half-widths are 0.0015 (visible) and 0.0047 (NIR) with a median of
   63 pixels per fit. The bootstrap treats
   pixels as independent, so with spatial autocorrelation these are lower
   bounds.
3. **Pixel count and representativeness.** At most 165 MODIS pixels pass the
   land criteria (of 708 fully inside the box; 42% of the box is water);
   per-date counts depend on clouds (full run: 30–165 on calibrated groups,
   median 63). The
   50ha plot touches 8 MODIS pixels and is assumed to share the box-wide
   correction; a larger box would add pixels but little reflectance range
   (more forest) unless it took in other land covers.
4. **Coverage / selection.** Groups below 30 usable pixels stay uncalibrated
   (pilot: 24 of 35 in February, ~16 from cloud over the land, ~7 from strip
   footprints). Calibrated dates are biased toward clear days: 61% of the
   1,651 groups stay uncalibrated, and the rate is 0.44–0.57 in
   December–April but 0.18–0.28 in May–November, so the calibrated series is
   thinnest when wet-season phenology happens.
5. **MODIS reference.** Errors of the MCD43A1 BRDF inversion (only QA 0/1 and
   QA30 days kept: 45–48% of pixel-days, 2005–2026), and of the gap fill on
   the other days (`fill_flag` 1–4 in the reference). Fewer than 0.5% of
   filled days come from the pixel's own series (N ≥ 250 good days in a year
   is rare in Panama); nearly all are the neighbourhood seasonality shifted
   to the pixel, 22–30% from that year and 69–78% from the multi-year mean
   (by band), so wet-season references are largely modelled. HKU's
   scenarios i + ii (N ≥ 50) cover 99.8–99.9% of filled days, as in their
   >99%. Each fit records the share of
   its matched MODIS pixels that were observed that day
   (`modis_observed_frac`): 0.78–0.81 on average over calibrated fits, and
   below 0.5 for 18–20% of them. Observed days are not error-free either: the
   plot-mean reference NIR drops to 0.23–0.26 (median 0.306) on 19 days since
   2020 (early November 2022 and 2024, August and late September 2026) when
   50–100% of the plot's pixels pass QC, probably residual cloud or haze in
   the BRDF inversion window; four calibrated groups fall on such days
   (2024-11-05, 2026-08-08). The kernel evaluation itself is exact: MCD43A1
   kernels at the local-noon SZA reproduce MCD43A4 NBAR to within its 0.001
   quantization (median |Δ| 0.0003, ~3,000 pixels on each of 2021-12-05 and
   2024-02-15). The reference is at nadir and SZA 45°, while Planet is at the
   actual sun/view geometry; the offset absorbs the box-mean BRDF difference,
   assumed uniform across land covers. This difference is seasonal: the PSB.SD
   NIR offset runs from −0.04 in December–January to −0.08 to −0.09 in
   April–June and correlates with sun elevation (r = −0.57). So part of the
   offset is geometry, not sensor calibration, and the calibrated series
   carries the reference's fixed SZA 45° view rather than the forest's
   appearance at the actual sun angle.
6. **Geolocation and point-spread function.** MODIS geolocation error
   (~50 m) and a point-spread function wider than the nominal 463 m pixel blur
   the reference and contaminate it with lake reflectance near shores; the
   ≥ 99% land and 3 × 3 ≥ 90% land rules limit, but do not remove, this.
7. **Cloud mask.** Thin cloud, haze or shadow missed by OmniCloudMask biases
   the Planet box mean; mitigated by the 2-pixel dilation and the ≥ 95% clear
   requirement per MODIS pixel.
8. **Spectral bandpasses.** PS2, PS2.SD and PSB.SD bands differ from each
   other and from MODIS B3/B4/B1/B2; the offset removes the mean difference
   over the box, not the part that depends on spectral shape (e.g. forest vs
   bare soil, or changing leaf colour).
9. **Missing scenes.** 8 Dove Classic (`1105`) scenes behind `planet_clipped/`
   have no SR product and are absent from the calibrated set. One box scene
   (an 18 × 546 px sliver) has no OCM mask because OmniCloudMask failed on it,
   and is skipped.
10. **Clip grids.** About 3% of the `planet_clipped/ocm` masks are 191 × 353
   instead of 205 × 367 (7 px inside the usual clip). The evaluation reads
   every version on the mask's own bounds, so plot means for those clips
   cover a slightly smaller area.

## Data locations

| What | Where |
|---|---|
| SR orders (BCNM box) | `/Volumes/Earth03/flower/planet/planet_bci_orders_sr/` (scene list `scene_ids.csv`, `fetch.log`) |
| Staged SR scenes, index, OCM, water mask | `/Volumes/Earth03/flower/planet_bcnm_sr/{4band,ocm,aux}/`, `index.csv` |
| MCD43A1 subsets (OPeNDAP, tile h10v08 rows 177–230, cols 251–311) | `/Volumes/Earth03/flower/modis/mcd43a1_bcnm/opendap/{daily,yearly}/` |
| MODIS reference | `/Volumes/Earth03/flower/modis/mcd43a1_bcnm/reference_nadir_sza45.nc` |
| 50ha clips, raw SR | `/Volumes/Earth03/flower/planet_clipped_sr/4band/<YYYY>/` |
| 50ha clips, calibrated | `/Volumes/Earth03/flower/planet_clipped_cal/4band/<YYYY>/` (uint16 reflectance × 10⁴, nodata 0; same grid as `planet_clipped/ocm`) |
| Fit table, matched pixels | `results/planet_modis_calibration.csv`, `results/planet_modis_calibration_pixels/` |
| Evaluation | `results/planet_calibration_eval.csv`, `figs/planet_calibration/` |

## Commands

```bash
# 1. SR orders (scene IDs: planet_clipped inventory rows with status ok/skipped_existing
#    whose PSScene item has ortho_analytic_4b_sr)
python scripts/fetch_planet.py config/planet_search_sr.yml \
    /Volumes/Earth03/flower/planet/planet_bci_orders_sr \
    --missing-csv /Volumes/Earth03/flower/planet/planet_bci_orders_sr/scene_ids.csv \
    --order-name bcnm_sr

# 2. stage, cloud-mask, water mask
python scripts/stage_planet_sr.py /Volumes/Earth03/flower/planet/planet_bci_orders_sr
python scripts/cloud_mask_planet.py --input-dir /Volumes/Earth03/flower/planet_bcnm_sr/4band \
    --output-dir /Volumes/Earth03/flower/planet_bcnm_sr/ocm
python scripts/build_bcnm_water_mask.py

# 3. MODIS reference (Earthdata login from ~/.netrc). fetch_modis_opendap.py reads the
#    AOI window of each daily granule from NASA's cloud OPeNDAP (~160 KB, a few s/day);
#    fetch_modis_appeears.py gets the same layers via AppEEARS but its queue can stall.
python scripts/fetch_modis_opendap.py --start 2005-01-01 --end 2026-10-15 --workers 6 --combine
python scripts/build_modis_reference.py /Volumes/Earth03/flower/modis/mcd43a1_bcnm/opendap/yearly/*.nc

# 4. calibrate and cut 50ha clips; 5. evaluate
python scripts/calibrate_planet_modis.py
python scripts/evaluate_planet_calibration.py
python scripts/plot_planet_calibration_examples.py                  # Fig. 7
python scripts/plot_planet_calibration_examples.py --stretch fixed  # Fig. 8
```

## Results

Full run, 2020-01 to 2026-08 (`results/planet_modis_calibration.csv`, reference
rebuilt after the `uniform_filter` fix; the first run on the earlier reference
calibrated the same groups and its offsets differ by at most 0.017 NIR, median 0).

**Coverage.** 638 of 1,651 date-satellite groups (39%) are calibrated, covering
875 of 2,176 box scenes, with 30–165 usable MODIS pixels per fit (median 63).
The rate follows the dry season: 0.44–0.57 in December–April and 0.18–0.28 in
May–November (Fig. 4). By instrument: PS2 42% of 193 groups, PS2.SD 34% of 112,
PSB.SD 39% of 1,346.

**Offsets.** Median applied offsets are −0.017 / −0.015 / −0.013 / −0.062
(blue / green / red / NIR) for PSB.SD, −0.028 / −0.030 / −0.031 / −0.065 for
PS2.SD and −0.026 / −0.023 / −0.037 / −0.002 for PS2: raw Planet SR is brighter
than the MODIS reference in every band except PS2 NIR. Median bootstrap 95%
half-widths are 0.0015 in the visible and 0.0047 in NIR. The PSB.SD NIR offset
has an annual cycle, from −0.04 in December–January to −0.08 to −0.09 in
April–June, and correlates with sun elevation (r = −0.57; median 48° in
December, 65–70° in April–July): Planet at higher sun (SZA 20–30°) looks
brighter than the reference fixed at SZA 45°, as expected from forest BRDF.
The σ-ratio gain, not applied, is 1.5–2.5 in the visible and scatters around
1 in NIR, rising to a median of 1.46 for PSB.SD in 2026 (Fig. 3).

**Agreement with MODIS at 500 m.** Median out-of-sample RMSE (fit on one half
of a checkerboard of MODIS pixels, scored on the other) drops from
0.0192 / 0.0174 / 0.0158 / 0.0615 to 0.0065 / 0.0066 / 0.0061 / 0.0194
(blue / green / red / NIR); in-sample values are within 0.0004 of these, so
the offset is not overfit. Pooled over all fits, Planet still spans a
narrower range than MODIS in blue and green (Fig. 5), the scale difference
the fixed gain leaves in place.

**50ha plot time series.** 431 calibrated clips (350 date-satellite passes)
are ≥ 80% OCM-clear and exist in all three versions. Plot-mean NIR p10/50/p90
is 0.278/0.347/0.397 (TOA), 0.305/0.354/0.404 (SR) and 0.291/0.303/0.317
(CAL); NDVI 0.574/0.746/0.786, 0.714/0.828/0.881 and 0.846/0.874/0.898.

| version | NIR jitter | NIR residual SD | NDVI jitter | NDVI residual SD | EVI residual SD |
|---|---|---|---|---|---|
| TOA | 0.0274 | 0.0337 | 0.0251 | 0.0691 | 0.0943 |
| SR | 0.0270 | 0.0285 | 0.0273 | 0.0533 | 0.0550 |
| CAL | 0.0053 | 0.0083 | 0.0101 | 0.0141 | 0.0127 |

Jitter is the median |Δ| between passes ≤ 3 days apart; residual SD is around
a centred 31-day rolling median. In 2020–2022, the median NIR residual per
instrument is −0.025 / +0.032 / +0.010 (PS2 / PS2.SD / PSB.SD) in TOA,
−0.019 / +0.028 / +0.009 in SR and within ±0.0005 in CAL; for NDVI,
−0.070 / 0.000 / +0.020 (TOA), −0.035 / −0.025 / +0.019 (SR) and
≤ 0.001 (CAL). The CAL residual SD is highest in May–July for NIR (0.015–0.020,
vs 0.004–0.008 in the dry season) and in July, August and October for NDVI
(0.019–0.025) (Fig. 2).

**Caveat: the jitter reduction is partly by construction.** Each calibrated
scene's box-land mean is set to that day's MODIS reference, which is smooth
in time, and the plot shares the box-wide offset, so the plot-mean series
inherits the reference's smoothness. The test that does not depend on this
is the spatial checkerboard RMSE. Day-to-day changes in the plot that differ
from the box-land mean (e.g. leaf flush in the plot relative to the
surrounding forest) are kept; changes common to the whole box are replaced by
MODIS. Uncalibrated scenes (61% of groups, mostly wet season) have no CAL
version.

### Figures (`figs/planet_calibration/`)

1. `timeseries_plot_mean.png` — Plot-mean NIR reflectance (top row) and NDVI
   (bottom row) of the 50ha plot for top-of-atmosphere reflectance from the
   existing clips and metadata coefficients (TOA, left), raw Planet surface
   reflectance (SR, centre) and SR cross-calibrated to MODIS (CAL, right),
   with points coloured by instrument (blue, Dove Classic PS2; orange, Dove-R
   PS2.SD; green, SuperDove PSB.SD) and the MODIS MCD43A1 reference at nadir
   view and 45° solar zenith averaged over the 8 MODIS pixels touching the plot
   (black line, daily, gap-filled). Each point is the mean over OCM-clear
   pixels of one date-satellite pass, averaged over its scenes; only the 431
   calibrated clips that are ≥ 80% clear (350 passes, 2020-01 to 2026-08) are
   shown, and the MODIS line covers the same period.
2. `residual_sd_by_month.png` — Seasonal pattern of short-term noise in the
   plot-mean series: standard deviation, by calendar month, of residuals of
   plot-mean NIR reflectance (left) and NDVI (right) from a centred 31-day
   rolling median, for TOA (blue), raw SR (orange) and calibrated SR (green).
   Residuals are pooled over the 350 date-satellite passes of Fig. 1,
   2020-01 to 2026-08, and over years for each month.
3. `gains_offsets.png` — Per-fit radiometric corrections over time: the
   HKU σ-ratio gain (left column, σ_MODIS / σ_Planet over matched 500 m
   pixels; reported but not applied) and the applied offset (right column,
   μ_MODIS − μ_Planet in reflectance units) for blue, green, red and NIR
   (rows), one point per calibrated date-satellite group, coloured by
   instrument as in Fig. 1, with the no-correction values (gain 1, offset 0)
   as grey lines. Covers the 638 calibrated groups, 2020-01 to 2026-08.
4. `calibration_success.png` — Seasonality of calibration coverage: share of
   date-satellite groups by calendar month with at least 30 usable MODIS
   pixels (≥ 99% land, 3 × 3 neighbourhood ≥ 90% land, ≥ 95% clear in
   Planet), with the number of groups in each month above the bar; all
   1,651 groups, 2020-01 to 2026-08, pooled over years.
5. `matched_pixels.png` — Agreement of Planet with the MODIS reference at
   500 m before (top row, raw SR) and after (bottom row, calibrated)
   calibration: log-scaled hexbin density of Planet reflectance averaged
   from 3 m clear land pixels onto each MODIS sinusoidal pixel against the
   MODIS reference for the same pixel and day, per band (columns), with the
   1:1 line (black) and the pooled RMSE in each title. Pools all matched
   pixels of the 638 calibrated groups, 2020-01 to 2026-08; axes span the
   0.5–99.5th percentiles.
6. `example_histograms.png` — Histogram matching for the calibrated group
   with the most usable pixels (2021-03-05, SuperDove 227a, 165 MODIS
   pixels): distributions over matched 500 m pixels of the MODIS reference
   (black), raw Planet SR (orange) and calibrated Planet SR (green), per band
   (panels), in 30 equal reflectance bins spanning all three.
7. `example_clips_model.png` (`plot_planet_calibration_examples.py`) — The
   50ha plot as the Mask R-CNN classifier sees it, for two pairs of clear
   clips (≥ 97% OCM-clear) from different instruments 2 days apart in January
   and February 2021 (columns; date, instrument and satellite in each title),
   chosen as the cross-instrument pairs ≤ 3 days apart with the largest
   raw-SR plot-mean NIR difference. Rows are the existing TOA clip the current
   models use (top), raw Planet SR (middle) and SR cross-calibrated to MODIS
   (bottom), each rendered as R, G, B after `build_input_channels`, the
   training and apply-time per-image, per-band stretch of the 0th–99.9th
   percentile to [0, 1]; the label in each panel is the plot-mean NIR
   reflectance of that version.
8. `example_clips_fixed.png` — The same four clips as Fig. 7 as raw SR and
   calibrated SR, in true colour (R, G, B) and false colour (NIR, R, G)
   (rows), with one linear stretch per band shared by all panels (1st–99th
   percentile of all pixels shown, both versions pooled: blue 0.004–0.057,
   green 0.016–0.078, red 0.007–0.065, NIR 0.149–0.448 reflectance), so
   brightness differences between panels are real reflectance differences.

**What the classifier sees.** The model stretch starts at each band's
minimum over the whole clip, nodata included. All 142 chips of the current
training set and 2,103 of the 2,146 TOA 50ha clips contain some nodata (0)
pixels, so in practice each band is divided by its own 99.9th percentile.
That keeps part of the calibration offset: in Fig. 7 the calibrated inputs
are more alike across instruments than the raw-SR or TOA ones, and the largest
per-pixel change between raw-SR and calibrated inputs is 0.57 of the [0, 1]
range. The 43 TOA clips without nodata (2020: 22, 2021: 18, 2022: 3) are
instead stretched from their true minimum, so they get a different, higher
contrast at apply time than any training chip (Fig. 7, columns 1 and 4, top
row).
