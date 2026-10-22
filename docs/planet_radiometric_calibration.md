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
  not the forest. The splines also overshot across long wet-season gaps:
  Jun–Sep 2022 fills at the 50ha pixel ranged 0.08–1.19 NIR (observed
  0.29–0.33), and over all land pixels the 1st/99th percentiles of flag-2
  fills were −1.69/1.88. The neighbourhood is now a same-class anomaly mean,
  and the splines are replaced by PCHIP and a Gaussian-kernel mean. In the
  rebuilt 2005–2026 reference, land NIR 1st/50th/99th percentiles are
  0.125/0.293/0.375 observed, 0.150/0.302/0.378 for flag 2 and
  0.155/0.304/0.368 for flag 3, and the monthly medians of every fill type at
  the 50ha pixel stay within 0.27–0.32 (the spline version is kept as
  `reference_nadir_sza45_v1_spline.nc`).

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
   (~1% of forest NIR) for ~100 pixels in the pilot. The bootstrap treats
   pixels as independent, so with spatial autocorrelation these are lower
   bounds.
3. **Pixel count and representativeness.** At most 161 MODIS pixels pass the
   land criteria (of 708 fully inside the box; 42% of the box is water);
   per-date counts depend on clouds (pilot: 39–148 on calibrated dates). The
   50ha plot touches 8 MODIS pixels and is assumed to share the box-wide
   correction; a larger box would add pixels but little reflectance range
   (more forest) unless it took in other land covers.
4. **Coverage / selection.** Groups below 30 usable pixels stay uncalibrated
   (pilot: 24 of 35 in February, ~16 from cloud over the land, ~7 from strip
   footprints). Calibrated dates are biased toward clear days, and the
   wet-season rate will be lower.
5. **MODIS reference.** Errors of the MCD43A1 BRDF inversion (only QA 0/1 and
   QA30 days kept: 45–48% of pixel-days, 2005–2026), and of the gap fill on
   the other days (`fill_flag` 1–4 in the reference). Fewer than 0.5% of
   filled days come from the pixel's own series (N ≥ 250 good days in a year
   is rare in Panama); nearly all are the neighbourhood seasonality shifted
   to the pixel, 31–42% from that year and 58–68% from the multi-year mean
   (by band), so wet-season references are largely modelled. HKU's
   scenarios i + ii (N ≥ 50) cover 99.8–99.9% of filled days, as in their
   >99%. Each fit records the share of
   its matched MODIS pixels that were observed that day
   (`modis_observed_frac`). The kernel evaluation itself is exact: MCD43A1
   kernels at the local-noon SZA reproduce MCD43A4 NBAR to within its 0.001
   quantization (median |Δ| 0.0003, ~3,000 pixels on each of 2021-12-05 and
   2024-02-15). The reference is at nadir and SZA 45°, while Planet is at the
   actual sun/view geometry; the offset absorbs the box-mean BRDF difference,
   assumed uniform across land covers.
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
   have no SR product and are absent from the calibrated set.

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
```

## Results

(pending)
