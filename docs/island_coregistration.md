# Whole-island Planet coregistration: translation, affine, local warp and the elevation term

Scripts: [`scripts/island_coreg.py`](../scripts/island_coreg.py) (alignment) and
[`scripts/island_coreg_diagnose.py`](../scripts/island_coreg_diagnose.py) (diagnosis of the
central band, drone anchoring).
Outputs: tables in `/Volumes/Earth03/flower/results/island_coreg/` (diagnostics in
`diagnose/`), aligned rasters in `/Volumes/Earth03/flower/planet_bci_aligned/Ha/` (recommended;
the earlier local-warp set is in `L/`), figures and videos in
`/Volumes/Earth03/flower/figs/20261110_updates/` (`island_coreg_*`) and
`/Volumes/Earth03/flower/figs/20261116_updates/` (`island_diag_*`, re-rendered videos).

**Summary of the 2026-11-16 revision.** The band of residual across the center of the
island is not match error and not a product of Planet's orthorectification resampling. It
is the island's central plateau, and what it shows is that every scene carries a
displacement proportional to terrain elevation (median 2.8 m per 100 m of elevation, 90th
percentile 9.9 m). An affine plus that one elevation term (model **H**, 8 parameters per
scene) aligns better than the local warp (re-measured median residual 1.04 m against
1.29 m, better on 95 % of scenes), and the drone orthomosaic of 2024-10-14 anchors the
result to the ground (model **Ha**). Details in
[The central band](#the-central-band-match-error-master-or-orthorectification).

**Addition of 2026-11-17.** The elevation term is the tilt of each frame's RPC. The line
of sight of the RPC at the frame center, minus the view direction in Planet's metadata,
equals the term measured from the imagery (slope 0.975, 97.5 % of the variance, 544
scenes; for the master +2.13 m east, +2.36 m north per 100 m against +2.00 and +2.43
measured with the drone). The term can be read from metadata without image matching.
Planet's documented process, the literature and the checks are in
[planet_geolocation_literature.md](planet_geolocation_literature.md).

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
- **The coherent residual is not parallax along the nominal view direction, but it
  is proportional to elevation** (revised 2026-11-16; see
  [The central band](#the-central-band-match-error-master-or-orthorectification)).
  Averaged over
  scenes, the translation residual forms a 2–3 m band along the island's central
  ridge that points the same way whether the satellite was east or west of the
  island (340 vs 378 scenes; `island_coreg_residual_map`). Parallax from an
  elevation error in the DEM reverses with the viewing side, so the mean field is
  common to all scenes: it is the master's own elevation term (+2.4 m north,
  +2.0 m east per 100 m of elevation). Each scene has its own such term, in a
  direction unrelated to its reported view vector, which is why
  nothing depends on the view angle (0–5.2°
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

**Where the distortion comes from** (2026-11-14, from the affine terms in
`scene_models.csv`; the affine part of each scene's field decomposed into
cross-track scale `jac_xx_m`, along-track scale `jac_yy_m`, rotation
(`jac_xy − jac_yx`)/2 and shear (`jac_xy + jac_yx`)/2, all in meters across the
island):

- It is **per frame, not per satellite.** The archive spreads over ~180 Doves, so
  only 3 satellites have 8 or more passes; within them 68–91 % of the variance
  of each term is pass-to-pass, not a fixed camera signature. Stronger: in the
  108 cases where the same satellite imaged the island twice on the same day
  (consecutive frames seconds apart), the range of each term across those frames
  (1.8 / 5.3 / 2.3 / 2.2 m for cross-track scale / along-track scale / rotation /
  shear) is as large as the terms themselves (2.0 / 4.1 / 1.6 / 1.8 m). Consecutive
  frames from one camera carry different distortions, which rules out lens or
  focal-plane calibration and anything that varies slowly, and leaves the
  geometric solution of each individual frame: the satellite's attitude and
  position estimate for that frame and its refinement against ground control.
- The largest term is the **along-track scale** (median |3.2| m, mean −1.7 m,
  against 1.9 m cross-track, 1.3 m rotation, 1.4 m shear). Doves fly near-polar
  orbits, so along-track is roughly north–south; a scale error along the
  direction of flight is what an error in the velocity or pitch-rate part of the
  frame's attitude model produces.
- **Cloudier scenes are more distorted** (Spearman ρ = −0.27 between affine
  amplitude and island-clear fraction; no dependence on date, 0.02, or sun
  elevation, 0.14). Planet refines each frame's geometry by matching tie points
  to its reference basemap; clouds remove tie points, and around BCI most of the
  footprint is lake anyway, so the refinement over the island rests on few,
  distant points and the attitude-model residual shows through.
- **More than half of what the local warp removes is beyond affine** (median
  gain over translation 0.22 m from the affine part, 0.38 m from the local part).
  The 2026-11-16 analysis shows what that part is: not a bend of the frame but a
  displacement proportional to terrain elevation, which an affine in map
  coordinates cannot represent and a smooth local field only approximates.

This is the island-scale face of what the 50ha work already saw: adjacent frames
of one strip registered to the drone with shifts more than 10 m apart
([planet_training_chips_50ha.md](planet_training_chips_50ha.md)). The error is
assigned frame by frame, and within a frame it is a smooth field of a few meters
to a few tens of meters across 6 km.

## The central band: match error, master or orthorectification?

Added 2026-11-16 (`island_coreg_diagnose.py`). The mean residual map has a 2–3 m band
across the center of the island, over the 50ha plot, and by eye the alignment there
looked wrong. Three explanations were tested.

**1. False matches on ambiguous canopy texture: no.**

- The band is 184 of 1,520 tiles. Its across-scene spread is higher than elsewhere
  (2.33 m against 1.73 m, robust), but the red-versus-NIR disagreement of the same
  tile and scene is the same (1.40 m against 1.39 m) and so is the NCC peak (0.62
  against 0.61). False matches would make the two bands disagree
  (`island_diag_band`, panel c).
- The ±6 px NCC surfaces of 60 scenes against 4 masters (612,000 tiles) are
  single-peaked: 31 % have any secondary local maximum at all, and where one exists
  its value is a median 3 % of the main peak (1 % in the band).
- Triangle closure. With the master M0 and three alternate masters Mk (a wet-season
  SuperDove, a Dove Classic and another dry-season SuperDove, all fully clear),
  d(S→M0) − d(Mk→M0) − d(S→Mk) was formed per tile for 60 scenes (417,000
  triangles). Real displacements close to zero whatever they are; false matches do
  not. The median closure error is 0.64 m in red and 0.39 m in NIR, the same inside
  the band (0.61 m) as outside (0.64 m), and exceeds 3 m on 1.8 % of red tiles
  (0.9 % in the band). Swapping scene and reference between two masters reverses the
  band-mean field to within 0.03 m (`island_diag_band`, panel d).

**2. The master's own distortion: partly.** Under the original master the band-mean
residual is 2.9 m; under the three alternates it is 1.1–1.7 m, in different
directions, and the masters differ from one another by 1.5–3.9 m in the band against
0.3–0.5 m outside it. So the band is not one scene's defect: it is a place where every
scene is displaced differently, and the across-scene spread there is high under every
master (2.2–2.6 m against 1.4–1.6 m).

**3. Planet's orthorectification resampling: no.** The basic (level 1B,
non-orthorectified) scenes with their RPCs are in
`csdap/planet/PSScene-<id>/assets/basic_analytic_4b/`; 542 of the 718 fitted scenes
have one. `ortho1b` projects them with `gdalwarp -rpc` (cubic, `RPC_DEM` = the
Copernicus GLO-30 tile) onto the island lattice. For 64 scenes (the 60-scene sample
and the 4 masters) our RPC ortho and Planet's ortho of the same image agree to a
median tile residual of 0.042 m after a constant offset of 0.9 m (worst scene
0.064 m; affine amplitude 0.23 m across the island; NCC peak 0.99), against 1.7 m of
scene-to-master residual for the same scenes. Planet's ortho product is the RPC model
applied with a DEM equivalent to GLO-30, and nothing else: no tie-point mesh, no
resampling artifact. Re-measuring scene against scene in our own RPC orthos would
therefore reproduce the Planet-ortho results to 0.04 m and was not run. The error is
upstream of the orthorectification, in the geometric model (the RPC) of each frame.

**What it is: a displacement proportional to elevation.** The band is the island's
central plateau (band tiles average 155 m above sea level, the rest 80 m;
`island_diag_band`, panel b). Adding a term k·(h − 100 m) to the per-scene affine,
with h the GLO-30 elevation and k a 2-vector per scene, removes it:

| Model (per scene) | Parameters | Fit residual, 718 scenes (m) | Re-measured after warping, 288 scenes (m) | Same, plateau tiles ≥ 120 m (m) | 90th percentile above 1 px (share of scenes, re-measured) |
|---|---|---|---|---|---|
| Translation (T) | 2 | 1.99 | 1.92 | 2.95 | 87 % |
| Affine (A) | 6 | 1.68 | 1.68 | 2.71 | 77 % |
| Local warp (L) | field | 1.23 (hold-out) | 1.29 | 1.25 | 45 % |
| Affine + elevation (H) | 8 | 1.11 | 1.04 | 1.02 | 28 % |
| H + local (HL) | field | 1.15 (hold-out) | 1.26 | 1.19 | 42 % |

(Medians over scenes of the per-scene median tile residual; noise floor 1.10 m.)
H reaches the noise floor with 8 parameters, beats the local warp on 95 % of the
re-measured scenes (85 % on the plateau tiles) and the affine on 96 %. Adding a local
field on top of H makes it worse: what is left after H is matching noise, and a
smooth field fitted to noise adds error when applied.

- **Size.** Referred to the ground (see the anchoring below), the term has a median
  magnitude of 2.8 m per 100 m of elevation and a 90th percentile of 9.9 m; it
  exceeds 5 m per 100 m on 28 % of scenes and 10 m on 10 %. Between the lake shore
  (27 m) and the top of the plateau (165 m) that is 3.9 m at the median and 13.8 m at
  the 90th percentile. It is larger along track (standard deviation 5.7 m per 100 m
  north–south against 3.0 m east–west) and similar across generations (median 1.8 m
  for Dove Classic, 2.9 m for Dove-R and SuperDove).
- **It is per frame.** In the 108 same-satellite same-day groups the term differs
  between consecutive frames by a median of 2.1 m (north) and 1.5 m (east) per 100 m.
  It is uncorrelated with the reported view angle (ρ = −0.03), the clear fraction
  (−0.06) and the sun elevation (0.06), and it does not follow the reported view
  vector: regressing each tile's residual on tan(view angle) along the satellite
  azimuth, or on the sun's shadow vector, explains 3 % of its variance.
- **Likely mechanism.** A displacement of height × angle is what an error in a
  frame's pointing leaves after orthorectification: 2.8 m per 100 m is 1.6° and
  9.9 m is 5.6°. Planet's ground-control adjustment can absorb a pointing error as a
  shift at the height of its tie points, which around BCI are mostly near lake
  level, and the error then reappears in proportion to height above them. This fits
  the data but the adjustment itself was not examined.
- **Mechanism confirmed from the RPCs (2026-11-17).** `island_coreg_diagnose.py
  metadata` computes, for the 544 fitted scenes with a basic RPC, how far the ground
  position of the frame-center pixel moves per 100 m of assumed height (the RPC's line
  of sight) and subtracts the line of sight given by the metadata incidence angle and
  satellite azimuth. That tilt has a median of 3.03 m per 100 m (1.73°) and a 90th
  percentile of 10.65 m (6.08°), and it equals the elevation term: term = 0.975 × tilt,
  97.5 % of the variance of the east and north components (98.8 % for SuperDove,
  92.7 % for Dove-R, 57.8 % for Dove Classic), median difference 0.51 m per 100 m. For
  the master the tilt is +2.13 m east and +2.36 m north, against the +2.00 and +2.43
  measured with the drone. The term is unrelated to the difference between a scene's
  reported line of sight and the master's (0.0 % of variance) and to where the island
  falls in the frame (0.1 %). All 718 ortho XMLs record the same elevation model
  (`FineDEM`) and processor version (4.1.4). The six `ground_control` false scenes are
  13–82 m from the master. See
  [planet_geolocation_literature.md](planet_geolocation_literature.md).
- **Independent confirmation against the drone.** The 2024-10-14 whole-island drone
  orthomosaic (20 cm, averaged to the 3 m grid; red and green bands) was matched to
  the 120 Planet scenes nearest in date that are at least 90 % clear; 54 pass the global gate (tile NCC peak
  0.42). The elevation term of a scene measured against the drone and the one
  measured against the Planet master agree at r = 0.98 (north) and 0.97 (east), with
  a constant difference that is the master's own term: +2.43 m north (MAD 0.26) and
  +2.00 m east (MAD 0.29) per 100 m from 53 scenes, +2.60 and +2.17 m measured
  directly on the master (`island_diag_elevation`, panel b). The archive consensus
  (minus the median of all scenes' terms relative to the master) gives +3.4 and
  +1.4 m.
- **At the 50ha plot.** Against the independent Planet-to-drone shifts of 596
  scenes, the island models evaluated at the plot center disagree by a median of
  3.40 m (translation), 3.31 m (affine), 2.52 m (local warp), 2.30 m (H) and 2.50 m
  (HL); within 3 m on 41 / 44 / 61 / 68 / 63 % of scenes
  (`diagnose/plot_crosscheck.csv`).

**Anchoring to the ground (model Ha).** Alignment to a master inherits the master's
errors: an offset of 6.5 m from the drone mosaic, affine terms of up to 3.8 m across the
island and the elevation term above. `island_coreg_diagnose.py drone` estimates the
master's own affine + elevation model as the median over 53 scenes of (scene against
drone) − (scene against master) and writes it to `results/island_coreg/master_abs.json`;
model `Ha` adds it to each scene's H coefficients (the two fields add to first order).
Checked on 6 scenes re-measured against the drone after the `Ha` warp: offset 0.1–0.6 m
(6–13 m as delivered), elevation term within ±0.8 m per 100 m of zero (one scene
−1.5 m east), median tile residual 1.1–1.3 m, which is the cross-sensor matching
noise. `planet_bci_aligned/Ha/` holds the 288 video scenes warped this way.

**Not done from the original plan.** The known-shift test on real scene pairs was
dropped because the triangle closure answers the same question without a synthetic
shift. The scene-to-scene measurement in RPC-ortho geometry was dropped for the reason
given under 3. Matcher changes (coarse-to-fine, joint red + NIR, ambiguity weighting)
were not needed: the matches were not the problem.

**Answer (revised 2026-11-16).** Use affine + elevation (H; Ha where absolute position
matters) for whole-island products. It is the physically motivated model, has 8
parameters, reaches the matching noise floor everywhere including the plateau, and
agrees best with the drone. The local warp of the first run was approximating the
elevation term with a smooth field and should not be used on top of H. The paragraph
below is the first run's answer, kept for the record.

**Answer (first run).** Over the whole island a translation is not enough: it leaves 2 m
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

- **H, affine + elevation** (2026-11-16): least squares of each component on
  (1, row, col, h − 100 m), h the mean GLO-30 elevation of the tile
  (`dem/bci_glo30_3m.tif`), with the same clipping. Applied per pixel with the 3 m
  DEM. `k_north_m100`, `k_east_m100` in `scene_models.csv` are the elevation term in
  meters per 100 m of elevation, relative to the master.
- **HL:** H plus the smoothed H residual, scored by the same hold-out as L.
- **Ha:** H plus the master's own affine + elevation model relative to the drone
  orthomosaic (`master_abs.json`), so the output is in the drone's frame.

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

- The elevation in H is the Copernicus GLO-30 surface model resampled from 30 m; it
  does not resolve individual crowns or gaps, and the term is fitted to 192 m tiles.
- The anchor rests on one drone mosaic (2024-10-14) and on 53 scenes; its affine
  terms are known to about 0.5–1.1 m across the island (MAD over scenes) and its
  elevation term to about 0.3 m per 100 m. Against the drone a common local field
  remains (hold-out residual 0.65 m with it, 0.98 m without), which may be the
  mosaic's own local distortion; it is not applied.
- The pointing-error mechanism is confirmed by the RPCs (the tilt of each frame's RPC
  against the reported view direction equals the term), but why Planet's refinement
  leaves that tilt is not known; the reading in
  [planet_geolocation_literature.md](planet_geolocation_literature.md) (tilt and shift
  are nearly interchangeable for a narrow field of view over flat tie points) is ours.

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

Added 2026-11-16 (`figs/20261116_updates/`):

10. `island_diag_band` — tile maps of the mean residual, elevation, red–NIR
    disagreement and triangle closure error, with the band outlined.
11. `island_diag_elevation` — the elevation term of every scene, its agreement
    between the Planet master and the drone orthomosaic, and the re-measured residual
    by model on plateau and lower tiles.
12. `island_coreg_sequence.mp4` — the 288 video scenes aligned with `Ha`.
13. `island_coreg_comparison.mp4` — the 900 m window over the 50ha plot under
    translation / local warp / affine + elevation, side by side.

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
| Figures, captions, videos | `figs/20261110_updates/island_coreg_*`, `figs/20261116_updates/` |
| Master's model relative to the drone mosaic | `results/island_coreg/master_abs.json` |
| Aligned scenes, affine + elevation, drone-anchored | `planet_bci_aligned/Ha/<YYYY>/`, `planet_bci_aligned/Ha/index.csv` |
| Diagnostics: per-tile spread and band tiles | `results/island_coreg/diagnose/spread_tiles.csv` |
| Diagnostics: sample, alternate masters | `diagnose/sample.csv`, `diagnose/masters.json` |
| Diagnostics: tile vectors and NCC-surface statistics for every scene–master pair | `diagnose/multi_planet.csv.gz` (per pair in `diagnose/pairs_planet/`) |
| Diagnostics: triangle closure, mean field per master | `diagnose/closure_planet.csv.gz`, `closure_tiles_planet.csv`, `master_fields_planet.csv`, `master_pairs_planet.csv` |
| RPC orthos of basic scenes (red, NIR, udm2 clear; island grid + 40 px margin) | `planet_bci_1b_ortho/<YYYY>/<scene>_1b_ortho.tif` |
| RPC ortho against Planet ortho | `diagnose/orthocmp_scenes.csv`, `orthocmp_tiles.csv.gz` |
| Drone mosaic at 3 m, scene-to-drone tiles and fits | `diagnose/drone_20241014_3m.tif`, `drone_tiles.csv.gz`, `drone_fits.csv` |
| Island models at the plot center against Planet-to-drone shifts, all models | `diagnose/plot_crosscheck.csv` |
| Tables of the first run, before the H models were added | `diagnose/scene_models_before_H.csv`, `remeasure_before_H.csv` |
| Planet quality flags, ortho XML fields, reported and RPC line of sight per fitted scene | `diagnose/metadata_checks.csv` |

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

# 2026-11-16: diagnosis, elevation models, drone anchoring
python scripts/island_coreg_diagnose.py spread
python scripts/island_coreg_diagnose.py sample
python scripts/island_coreg_diagnose.py multi --jobs 4      # 252 pairs, ~25 min
python scripts/island_coreg_diagnose.py closure
python scripts/island_coreg_diagnose.py ortho1b --jobs 3    # needs gdalwarp on PATH
python scripts/island_coreg_diagnose.py orthocmp --jobs 3
python scripts/island_coreg_diagnose.py drone --jobs 5      # writes master_abs.json
python scripts/island_coreg_diagnose.py crosscheck
python scripts/island_coreg.py remeasure --jobs 5           # T, A, L, H, HL
python scripts/island_coreg.py warp --model Ha --jobs 3
python scripts/island_coreg.py video --model Ha --which sequence --outdir <figs>/20261116_updates
python scripts/island_coreg.py video --model H --which comparison --panels T,L,H --outdir <figs>/20261116_updates
python scripts/island_coreg_diagnose.py figures

# 2026-11-17: Planet metadata and RPC line of sight against the fitted terms
python scripts/island_coreg_diagnose.py metadata            # writes diagnose/metadata_checks.csv
```

`island_coreg_diagnose.py` also has `island1b` and `multi --source rpc` (the standard
measurement in RPC-ortho geometry); they were not run, see the central-band section.
