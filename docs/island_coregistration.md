# Whole-island Planet coregistration and the case for local warping

Script: [`scripts/island_coreg.py`](../scripts/island_coreg.py).
Outputs (November 2026 run): tables in `/Volumes/Earth03/flower/results/island_coreg/`,
aligned rasters in `/Volumes/Earth03/flower/planet_bci_aligned/L/`, figures and videos in
`/Volumes/Earth03/flower/figs/20261110_updates/` (`island_coreg_*`).

## Question

Every Planet registration so far is one rigid shift per scene over the 50ha plot
([handoff_rerun_coregistration.md](handoff_rerun_coregistration.md)), and
`measure_chip_local_offsets.py` found that at the plot's 1 km scale the per-frame
georeferencing error is essentially rigid. Over the whole 5.9 × 5.4 km island,
orthorectification with Planet's elevation model, the view angle and the strip
geometry could leave residuals that one shift cannot remove. This run aligns every
clear scene over the island to one master Planet scene and asks: after the best
translation, how much coherent misalignment is left, does an affine or a local
(non-rigid) warp remove it, and is what it removes real (geometric) or matching
noise?

## Result

Run of 2026-11-07: 760 scenes measured, 718 pass the global match gate (the 41
rejected are half-clear scenes, median 58 % island-clear, with an NCC peak below
0.2: haze or thin cloud over the land; 3 of them hit the search edge), 1.31 million
tile vectors, a median of 1,718 tiles per scene.

- **Whole-scene offsets are large and vary from scene to scene.** Relative to the
  master, the median offset is 7.1 m, the quartiles 5.0–11.4 m, the 90th
  percentile 17 m and the 99th 44 m; 91 % of scenes are off by more than one
  pixel, 31 % by more than 10 m and 3 % by more than 30 m (the old AROSICS cap).
  The archive's median offset from the master is (5.0, −1.8) m east, north
  (`island_coreg_global_shifts`).
- **After the best translation, 2 m of residual remains, and most of it is
  real.** The median tile residual after a per-scene translation is 1.96 m over
  all tiles (per-scene median of medians 1.99 m); 85 % of scenes have a 90th
  percentile residual above one pixel. The red–NIR noise floor is 1.10 m, and the
  red and NIR affine-residual fields of a scene correlate at 0.46 (median), so about
  half of the residual variance is geometric rather than matching noise
  (`island_coreg_residuals`).
- **An affine takes it to 1.64 m; a local warp to 1.17 m (hold-out), at the noise
  floor.** The in-sample local residual is 0.98 m, the honest hold-out one 1.17 m,
  and it barely depends on the smoothing length (1.23–1.28 m per-scene median for
  σ of 1–3 tiles). The per-scene gain of the local warp over the translation has a
  median of 0.67 m and a 90th percentile of 1.98 m; it exceeds 1 m in 31 % of
  scenes and 0.5 m in 64 %. With the local warp, 44 % of scenes still have a 90th
  percentile residual above one pixel, against 85 % with a translation and 74 %
  with an affine.
- **The coherent residual is not orthorectification parallax.** Averaged over
  scenes, the translation residual forms a 2–3 m band along the island's central
  ridge that points the same way whether the satellite was east or west of the
  island (340 vs 378 scenes; `island_coreg_residual_map`). Parallax from an
  elevation error reverses with the viewing side, so this field is common to all
  scenes, which with a single-scene reference means it is the master's own
  distortion relative to the archive. Nothing depends on the view angle (0–5.2°
  in the fitted set, median 3.3°): Spearman ρ with view angle is 0.03 for the
  affine amplitude, 0.04 for the local-warp gain and 0.03 for the coherence
  (`island_coreg_view_geometry`). Per scene, the along-view residual component has
  no consistent dependence on elevation (slope median 0.22 m per 100 m of
  elevation, 10th–90th percentiles −3.2 to +3.7 m, centered on zero and not
  growing with tan(view angle); `island_coreg_terrain`), and the U-shaped
  dependence of the mean residual on elevation (1.9 m at 80 m a.s.l., 3.6–3.8 m at
  the lowest and highest tiles) is the same common field seen through the terrain.
  With view angles this small, a 1 m displacement needs about 17 m of DEM error,
  and the orthorectification DEM is evidently better than that.
- **Subtracting that common field is worth as much as an affine, not as much as
  the local warp.** Translation plus one fixed archive-mean field gives 1.66 m,
  the same as the affine (1.64 m); the per-scene local warp gives 1.17 m. So the
  value of the local warp is mostly per-scene distortion, not a fixed correction
  that re-referencing could absorb.
- **The worst scenes are near-nadir SuperDove scenes with 20–36 m of affine
  distortion** (`island_coreg_worst_cases`): 20201010_155324_99_2412 (view 0.1°,
  translation residual 6.7 m, affine 3.7 m, local 1.8 m, coherence 0.87),
  20230328_144849_31_241a (view 0.6°, 5.5 / 3.0 / 0.9 m, coherence 0.95: the scene
  behind the guayacan missed-event example in
  [handoff_rerun_coregistration.md](handoff_rerun_coregistration.md), whose field
  is a rotation about a point in the island's west) and 20240117_155649_55_24d1 (view
  3.3°, 5.6 / 4.4 / 1.9 m). Their fields are smooth, strongly coherent between
  bands, and unrelated to terrain or view angle: scene-level geometric
  (attitude or model) errors in Planet's orthorectification.
- **By generation,** the per-scene median residual after translation / affine /
  hold-out local warp is 1.4 / 1.3 / 1.1 m for Dove Classic (68 scenes), 2.4 / 1.7 /
  1.2 m for Dove-R (49) and 2.1 / 1.8 / 1.3 m for SuperDove (601). The NIR band sits
  about 0.5–0.7 px (median by generation; up to 1.4 px on some Dove Classic
  scenes) from the red band on the same scene, so one geometric model per scene
  leaves a band-to-band misregistration of that size whichever model is used.
- **Independent check at the 50ha plot.** For the 596 scenes that also have a
  Planet-to-drone shift in `crown_series_20261101/coreg_all_scenes.json`, the
  island models evaluated at the plot center disagree with the drone shift (after
  removing the constant master-to-drone offset) by a median of 3.4 m for the
  translation, 3.3 m for the affine and 2.5 m for the local warp, with 64 % of
  scenes within 3 m under the local warp against 41 % under the translation
  (`results/island_coreg/plot_crosscheck.csv`). The local field moves toward the
  drone's answer; the remaining disagreement includes the drone shift's own error
  (one AROSICS window per scene, drone mosaics of different dates).
- **Re-measured after warping** (the 288 video scenes, each warped with each model
  and matched again against the master; `remeasure.csv`): the per-scene median
  residual is 1.92 m after the translation, 1.68 m after the affine and 1.29 m
  after the local warp, against 1.92 / 1.63 / 1.18 m predicted by the fit (hold-out
  for the local warp; its in-sample 0.99 m is, as expected, optimistic). The local
  warp beats the translation on 84 % of scenes and the affine on 74 %; the share
  of scenes whose 90th-percentile residual stays above one pixel is 87 % / 77 % /
  45 %. The re-measured residual is what a viewer of the aligned sequence sees,
  and it is also what the frame labels of `island_coreg_sequence.mp4` show.

**Answer.** Over the whole island a translation is not enough: it leaves 2 m
(0.65 px) of median residual with 85 % of scenes above one pixel at the 90th
percentile, and about half of that residual is coherent geometry. The coherent
part is not terrain parallax (view angles are too small and nothing depends on
them); it is scene-level distortion of Planet's orthorectification (the affine part alone
exceeds 10 m across the island on 32 % of scenes, 20 m on 8 % and 30 m on 3 %),
plus a fixed component from the master. A local warp fitted per scene removes it down to the matching noise
floor (1.2 m, 0.4 px), whereas an affine gets only halfway (1.6 m). For
crown-scale work (a 10 m crown is 3 px) the local warp is the right default for
whole-island products; for the 50ha plot alone, where the earlier finding of a
rigid error still holds at 1 km scale, a translation remains adequate on most
scenes, but the scenes with tens of meters of distortion (a 20 m stretch across
5.9 km is 3.4 m across the 1 km plot) and the band-to-band offset apply there too.

## Method

**Scenes.** `select` joins the island clear fraction of every scene
(`whole_island/bci_clear_fraction.csv`, from `bci_clear_fraction.py`) with the SR
scene index (`planet_bcnm_sr/index.csv`: instrument, satellite, view angle,
satellite azimuth, sun elevation) and the per-scene MODIS offsets
(`results/planet_calibration/planet_scene_offsets.csv`, fit or borrowed). Scenes at
least 50 % island-clear with an SR product are measured (760); those at least 80 %
clear with offsets make the video (293). Matching and rendering both use the SR
product, whose geometry agrees with the TOA clips to 0.005 m
([planet_radiometric_calibration.md](planet_radiometric_calibration.md), item 13).

**Grid and master.** Everything is read onto the island grid of
`planet_bci_clipped` (`config/clip_bci_margin.yml`: EPSG:32617, origin
(623979, 1015020), 3 m, 1958 × 1802 px). The SR scenes, their OmniCloudMask
masks and the static water mask all sit on one 3 m lattice (checked on every file
used) with varying extents, so the island is a boundless window read with no
resampling. The master is `20230227_145542_89_24be`: SuperDove, 100 % island-clear,
view angle 1.8°, February 2023 (the middle of the 2020–2026 archive, dry season),
chosen by hand. Alignment is therefore relative; the master's own georeferencing
error is inherited by the whole sequence, and the archive's median offset from it
is recorded with the global-shift figure. Matching uses land pixels only: the
water mask eroded by 2 px (the shoreline moves with the lake level), and only
pixels that are OCM-clear and have data in both images.

**Global shift.** The red band of the scene, z-scored over its valid pixels and
zero elsewhere, is matched to the master by normalized cross-correlation
(`skimage.feature.match_template`) over ±30 px (90 m), the same range as the
Planet-to-drone NCC check in `coreg_ncc.py`. The peak must be at least 0.2 and
not on the search edge. The peak-to-second-peak ratio used in `coreg_ncc.py` is
recorded but not used as a gate: at 3 m against another Planet scene the NCC
surface is smooth and the ratio is about 1.3 everywhere, confident or not.

**Tile field.** After removing the integer global shift, 64 px (192 m) tiles on a
32 px stride are matched in red and in NIR separately: a tile is kept if at least
90 % of its pixels are valid in both images, its ±6 px NCC search has a peak of at
least 0.35 that is not on the edge, and the sub-pixel refinement succeeds. The
refinement is masked Lucas–Kanade (least squares on the image gradients, valid
pixels only, cubic resampling, started from the NCC peak, as in
`compare_chip_sets.shift_px`), with the parabolic interpolation of the NCC peak as
a fallback. In a self-test on a known shift, Lucas–Kanade is exact, the parabolic
fit has a 0.02 px bias and `phase_cross_correlation(upsample_factor=20)` a
0.08 px bias; 128 px tiles do not raise the NCC peaks, and ~1,500 tiles per band
per scene take about 20 s. Tiles whose vector differs from the median of their
3 × 3 lattice neighbors by more than 2 px are dropped as outliers before fitting.

**Displacement convention.** A tile vector d = (dy, dx) in pixels means the content
at master pixel p appears at p + d in the scene. The correction samples the scene
at p + d; a pure translation moves the scene's transform by −d. This is the
convention of `apply_drone_labels_coreg` (a reference point p appears at
p + shift in the Planet frame). `selftest` applies a known affine (0.1 % scale,
0.5 px shift) plus a 2 px Gaussian bump to the master and recovers the tile field
to a median 0.004 px, the affine terms within 6 %, and a re-measured residual of
0.017 px after the local warp.

**Models.** Per scene and band (and pooled, with the NIR tiles referred to the red
band's translation, because Dove Classic NIR sits more than a pixel from red on
some scenes):

- **T, translation:** the median tile vector.
- **A, affine:** least squares of each component on (1, row, col) with two passes
  of 3·MAD clipping. Its non-translational amplitude is the largest difference in
  displacement between two points of the island that the scale, shear and rotation
  terms imply.
- **L, local warp:** the affine plus a normalized Gaussian smoothing (σ = 1.5
  tiles = 144 m) of the affine residual over the tile lattice, evaluated on pixels
  by bilinear interpolation; where no tile is within reach the field falls back to
  the affine. A thin-plate spline was considered and not used: the tiles already
  lie on a lattice, cloud gaps are handled by the weights, and the one
  regularization length is in meters.

**Honest evaluation.** T and A are 2- and 6-parameter fits to ~1,500 tiles, so their
in-sample residuals are already honest. L is scored by hold-out: every 4th tile in
each lattice direction (16 phases) is predicted from a field fitted without it and
without its 3 × 3 ring, so training tiles share no pixels with the test tile. The
hold-out residual barely depends on σ between 1 and 3 tiles. Two further checks:
the **noise floor**, the median over tiles of the red-versus-NIR disagreement of the
translation residuals divided by √2 (the two bands see the same geometry with
independent texture and noise), and the **coherence**, the Pearson correlation of
the red and NIR affine-residual fields of a scene (near 1 the residual is
geometric; near 0 it is matching noise). Finally `remeasure` warps the video scenes
with each model and runs the tile matcher again against the master: the residual a
viewer of the sequence sees, independent of the fit.

**Terrain.** `dem` puts the Copernicus GLO-30 tile `N09_W080` (public COG) on the
island grid with slope and aspect. Island land spans 25–190 m above sea level. A
DEM error of h meters in Planet's orthorectification displaces a pixel by
h·tan(view angle) toward or away from the satellite; with view angles of 0–7.4°
(median 3.2°) over BCI, a 1 m displacement needs about 18 m of DEM error at the
median angle. The terrain figure bins the residual by tile elevation and slope and
fits, per scene, the along-view residual component against elevation.

**Aligned rasters and video.** `warp` resamples the four SR bands bilinearly at
p + d (OCM by nearest neighbor; pixels whose source footprint is not fully valid
set to nodata) and writes uint16 reflectance × 10⁴ on the island grid, uncalibrated,
to `planet_bci_aligned/L/<YYYY>/`; the MODIS offsets are in its `index.csv` and are
applied at render time (they are negative, so adding them before storing in uint16
would clip dark pixels to nodata). `video` renders true color (R, G, B) of the
calibrated reflectance on the fixed stretch of `config/planet_fixed_stretch.yml`
for every frame, 6 frames per second, through the same ffmpeg raw-video pipe as
`make_registration_figures.py`. The comparison video shows a 900 × 900 m window
(where the local and translation models disagree most on average) as delivered,
after translation and after the local warp.

## Caveats

- Alignment is relative to one Planet scene. The whole sequence inherits its
  georeferencing error, and its own orthorectification distortion is the zero of
  every residual map: a pattern common to all scenes could be the master's.
- The noise floor from red–NIR disagreement is a lower bound where both bands share
  the same false matches (cloud-shadow edges, the shoreline); the hold-out and the
  re-measured residuals are the primary numbers.
- Wet-season and Dove Classic scenes differ most from the February SuperDove master
  in phenology and sensor, which thins their tiles rather than biasing them; tile
  counts per instrument are in `scene_models.csv`.
- Thin peninsulas and partial strips have few 90 %-valid tiles, so the local field
  falls back to the affine there.

## Figures and videos (`figs/20261110_updates/`)

Each has a `<name>_caption.md` with the full caption and the numbers behind it.

1. `island_coreg_global_shifts` — whole-scene offset of every scene relative to the
   master: vector scatter by Dove generation, magnitude over time, histogram.
2. `island_coreg_residuals` — the headline: tile-residual CDFs after each model
   with the noise floor, per-scene medians by generation, and the per-scene
   local-warp gain.
3. `island_coreg_residual_map` — mean residual vector field over the island after
   translation and after affine, split by viewing side, over the DEM hillshade.
4. `island_coreg_view_geometry` — affine amplitude, local-warp gain and coherence
   against view angle.
5. `island_coreg_terrain` — residual against elevation and slope; per-scene
   along-view residual–elevation slope against tan(view angle).
6. `island_coreg_worst_cases` — the three video scenes with the largest local-warp
   gain: tile field and 300 m zooms (master / translation / local warp) with the
   master's edges overlaid; blink GIFs in `gifs/`.
7. `island_coreg_master` — the master scene with the land mask and tile grid.
8. `island_coreg_sequence.mp4` — all 288 video scenes aligned with the local warp,
   MODIS-calibrated true color on one fixed stretch, 6 fps (`_frame0.png`,
   `_frame144.png` are stills).
9. `island_coreg_comparison.mp4` — a 900 m window as delivered / translation /
   local warp, side by side, 3 fps. The window is chosen as the block where the
   local and translation models disagree most on average (1.04 px); it lands on
   UTM 625737–626637 E, 1011690–1012590 N, which is the 50ha plot, because the plot
   sits on the central-ridge band where the coherent residual is largest.

## Data locations

| What | Where |
|---|---|
| Scene list, master, matching parameters | `results/island_coreg/scenes.csv`, `master.json` |
| Global shifts, tiles (per scene cache and concatenated) | `results/island_coreg/global_shifts.csv`, `tiles/<scene>.csv`, `tiles.csv.gz` |
| Models and residuals | `results/island_coreg/scene_models.csv` (one row per scene × band: red, nir, both), `tile_residuals.csv.gz`, `models/<scene>.npz` (T, A coefficients, L field) |
| Re-measured residuals after warping | `results/island_coreg/remeasure.csv` |
| Island models at the 50ha plot center vs the Planet-to-drone shifts | `results/island_coreg/plot_crosscheck.csv` |
| DEM on the island grid (elevation, slope, aspect) | `dem/bci_glo30_3m.tif` (source `dem/Copernicus_DSM_COG_10_N09_00_W080_00_DEM.tif`) |
| Aligned SR scenes and OCM, index with offsets | `planet_bci_aligned/L/<YYYY>/<scene>_{4band,ocm}.tif`, `planet_bci_aligned/L/index.csv` |
| Figures, captions, videos | `figs/20261110_updates/island_coreg_*` |

## Commands

```bash
python scripts/island_coreg.py selftest
python scripts/island_coreg.py select
python scripts/island_coreg.py measure --jobs 5          # ~45 min for 760 scenes
python scripts/island_coreg.py fit
python scripts/island_coreg.py dem
python scripts/island_coreg.py remeasure --jobs 5        # video scenes, T/A/L
python scripts/island_coreg.py warp --model L --jobs 5   # ~8 GB
python scripts/island_coreg.py video
python scripts/island_coreg.py figures
```
