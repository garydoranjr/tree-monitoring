# PlanetScope geolocation: Planet's process, published limitations, and what our scenes show

Background and literature for [island_coregistration.md](island_coregistration.md). That
document measured a per-frame displacement proportional to terrain elevation in every
PlanetScope scene over Barro Colorado Island and inferred, without evidence from Planet's
side, that it came from each frame's pointing. This document covers three things: what
Planet documents about how scenes are geolocated and what accuracy it promises; what
independent assessments and other users have found and how they correct it; and four
checks of those explanations against Planet's own metadata and RPCs for our scenes
(`scripts/island_coreg_diagnose.py metadata`, 2026-11-17).

**Summary.**

- Planet promises less than 10 m RMSE at the 90th percentile, against its own reference
  imagery. It makes no promise about scene-to-scene alignment, and none that the error is a
  rigid shift. Over BCI the delivered scenes are at about twice that figure (20 m at the
  90th percentile against the drone mosaic, 17 m against one master scene).
- Planet's orthorectification is two steps: a per-scene geometric model (RPC) refined
  against reference imagery, then projection of that model onto a 30–90 m elevation model.
  For our scenes the second step is not where the error is: every scene records the same
  elevation model and processor version, and our own RPC + GLO-30 orthorectification
  reproduces Planet's to 0.04 m.
- **The elevation term is the tilt of each frame's RPC.** The line of sight of the RPC at
  the frame center differs from the view direction the satellite reported by a median of
  1.7° (90th percentile 6.1°), and that difference, as meters of ground displacement per
  100 m of height, equals the elevation term we measured from the imagery: slope 0.975,
  97.5 % of the variance, 544 scenes. For the master scene the RPC predicts +2.13 m east,
  +2.36 m north per 100 m, and the drone orthomosaic measured +2.00 and +2.43.
- The term can therefore be predicted from two small metadata files, to a median of 0.5 m
  per 100 m of height on the 544 scenes where image matching could check it. The shift and
  affine parts of the error still need image matching.
- Published work describes the same error class (RPCs "not accurate enough" for subpixel
  registration; "strong relief-related artifacts" in ortho scenes; residual "stereoscopic
  effects" removed with a polynomial that includes elevation) but none we found measures
  the RPC tilt against the reported view direction.

## What Planet documents

Source: *PlanetScope Product Specifications* (Planet, June 2026) unless noted.

| Step | What the specification says |
|---|---|
| Sensor model | Basic (level 1B) scenes: "Spacecraft-related effects are corrected using attitude telemetry and best available ephemeris data, and refined using GCPs." They ship with rational polynomial coefficients (RPCs) so that users can orthorectify. |
| Tie-down | "The rectification tiedown process wherein tie points are identified across the source images and a collection of reference images (ALOS, NAIP, OSM, Landsat) and RPCs are generated." Feature points come from "OpenCV's STAR keypoint detector and FREAK keypoint extractor"; matching uses "a combination of RANSAC, phase correlation and mutual information." |
| Orthorectification | "The actual orthorectification of the scenes using the RPCs, to remove terrain distortions." The terrain model "is derived from multiple sources (SRTM, Intermap, and other local elevation datasets) which are periodically updated", with a post spacing of 30 to 90 m, and snapshots are archived so the model used for a scene can be identified. |
| Accuracy | Positional accuracy "Less than 10 m RMSE at 90th percentile", for basic and ortho scenes alike. "Ground Control Points (GCPs) are used in the creation of every image and the accuracy of the product will vary from region to region based on available GCPs." |
| Quality flag | `ground_control`: "If the image meets the positional accuracy specifications this value will be true. If the image has uncertain positional accuracy, this value will be false." A Planet product manager stated on the community forum (2025-04-29) that true requires at least 200 ground control points, that the 10 m figure is for cloud-free images, and that cloudier images can be worse. |
| Composite frames | Dove-R (PS2.SD) and SuperDove (PSB.SD) scenes "are produced by an image registration process that involves compositing multiple frames ahead and behind an 'anchor frame'. Band to band alignment is dependent on accurate ground-lock for the anchor frame and can vary according to scene content", with lower yield over "open water, mountainous terrain, or cloudy areas." |
| Metadata | `view_angle` is the "Spacecraft across-track off-nadir viewing angle"; no along-track (pitch) angle is reported. The ortho XML field `elevationCorrectionApplied` names the elevation model used. |

Two further details come from papers rather than the specification. Ghuffar (2018) reports
that "A bundle adjustment type RPC refinement is already performed by Planet [8] for each
individual strip (consecutive images acquired from a single satellite) using reference
imagery." Kääb et al. (2017, with a Planet co-author) give the size of the problem the
tie-down has to solve: "Current pointing error for the satellites is of the order of 5 km
prior to georectification."

**Planet's remedies.** The Orders API Coregister tool aligns items to one anchor item
chosen by the user. Its documentation describes it as "designed to support coregistration
of small areas of interest - contained within a single scene"; it reports one `pixel_shift`
per item (method `imreg_dft`), skips items whose matching score is already 0.8 or higher,
and delivers items it cannot improve unchanged. A single shift per scene is what our
translation model (T) does, and it leaves a median 1.96 m of residual over the island. A
community request (2024-05-02, open) asks for an anchor other than a single scene. Planet
staff also point to a Sentinel Hub post on co-registering temporal stacks (Peressutti 2023),
which we could not retrieve. Basemap mosaics are the other suggested route to consistent
geometry; they are not daily. Mueting and Bookhagen (2024) advise against correlating
the custom multi-scene composites that Planet Explorer can build, and recommend aligning
scenes separately. We have not evaluated basemaps or composites.

## Independent assessments

| Source | Data | Finding |
|---|---|---|
| Semple, Tan and Lin (2023), NASA CSDA | 175 SuperDove scenes at 25 sites against WorldView panchromatic | CE90 13.76 m globally (7.21 m North America, 7.26 m South America, 25.98 m Europe); 3.80 m after removing each site's mean offset. The site means are attributed to "offsets in their internal Ground Control Points (GCPs)/reference imagery". NIR against red: mean 1.13 m, CE90 2.36 m. Sites were cities and airports with "minimal tree cover, low buildings, and no clouds": "a best-case scenario", and teams at "remote locations that are harder to match ... should expect worse". |
| Kääb et al. (2017) | Dove Classic, New Zealand | Specification 10 m RMSE; 6.5 m "according to validation measurements by Planet". Scene-to-scene matching noise 0.2–0.3 px after a polynomial co-registration. |
| Mueting and Bookhagen (2024) | SuperDove, two Andean landslides | Uncorrected ortho pairs show "higher and more variable disparities within the bounds of the < 10 m relative geolocation accuracy"; after correction, zero median and about 0.5 px interquartile range over stable terrain. |
| This project | 718 scenes of all three generations over BCI against one master | Median offset 7.1 m, 90th percentile 17 m; 31 % of scenes beyond 10 m. Against the 2024-10-14 drone mosaic the delivered scenes are 6–13 m off. NIR against red 0.5–0.7 px. |

As delivered, our scenes do not meet the specification over BCI. In its own form
(per-scene RMSE, then the 90th percentile over scenes) the 718 scenes have 9.5 m at the
median and 20.4 m at the 90th percentile against the ground, 7.8 and 17.3 m against the
master scene, and 7.2 and 14.3 m for the 53 scenes at least 90 % clear that were matched
directly to the drone mosaic; 46 % of scenes are above 10 m. The comparison is not exact:
Planet's figure is against its own reference imagery and for cloud-free scenes, ours is
against the drone mosaic (so it includes any offset between the two references) and
includes partly cloudy scenes. NASA's site means of up to tens of meters show the same
kind of reference offset. Beyond the size of the error, the specification does not address
what matters for crown-scale time series: the error differs from frame to frame and is
not a shift. Details in [island_coregistration.md](island_coregistration.md#model-r-the-elevation-term-from-the-rpc).

## Published error sources and how they map onto ours

| Error source | Published evidence | In our scenes |
|---|---|---|
| Reference imagery (absolute offset) | Site-dependent mean offsets of up to tens of meters (Semple et al. 2023) | A constant offset common to all scenes would be invisible in scene-to-master work; the master is 6.5 m from the drone mosaic. |
| Elevation model error × view angle | "the size of which is proportional to the DEM error and the off-nadir viewing angle" (Kääb et al. 2017); the difference in view direction between two scenes "is a robust predictor" of the bias, so pairs are restricted to 0.6° (Mueting and Bookhagen 2024) | Not the cause. One elevation model and processor for all 718 scenes; the term is unrelated to the reported view direction (check 3); at 3.6° median incidence, 1 m of displacement needs 16 m of elevation error. |
| Relief-related artifacts in ortho scenes | "topography-related artifacts should be negligible given the long focal length and quasi-nadir acquisition. However, we observe strong relief-related artifacts. The observed pattern suggests a mixture of artifacts due to the DEM inaccuracies used during the orthorectification process and the change in attitude between the frames when generating the composite scenes" (Aati et al. 2022a) | The same observation. Here it is one tilt per scene, present in Dove Classic (single frames) as well as in the composite generations, and it is in the RPC. |
| Sensor orientation after refinement (RPC error) | Georeferencing errors "from imperfect satellite positions and attitude angles, and their refinement from registering the images to reference images" expected to exceed the elevation-model effect (Kääb et al. 2017); "the RFM models delivered with commercial data are commonly not accurate enough for 3-D extraction, subpixel registration or ground deformation measurements" (Aati and Avouac 2020); a topographic pattern in ortho disparity maps "pointing to inaccuracies in the RPCs" (Mueting and Bookhagen 2024) | **The cause of the elevation term** (check 4): the RPC line of sight is tilted from the reported one, by 1.7° at the median. |
| Unrecorded pitch | An outlying acquisition attributed to "a slightly different satellite pitch angle, which has not been recorded in the metadata" (Mueting and Bookhagen 2024) | The tilt is larger along track (SD 6.3 m per 100 m north against 3.2 m east). But a real pitch would be modeled correctly by the RPC and leave no displacement; ours appears in the imagery, so it is a tilt of the model, not of the satellite. |
| Ramps (shift, rotation, scale between scenes) | "ramp errors are the dominating error source in the offset field", corrected per footprint with polynomials (Feng et al. 2019); first- and second-order polynomials (Kääb et al. 2017, 2019) | The affine part of model H: 7.4 m median amplitude across the island. |
| Sub-frame misalignment (stripes) | About 1 px in SuperDove, mostly north–south, not correctable without level 1A data (Mueting and Bookhagen 2024) | Not isolated. It would be part of the 1.0–1.1 m left after H, which is at the matching noise floor. |
| Band-to-band offset | NIR CE90 2.36 m (Semple et al. 2023) | 0.5–0.7 px NIR against red, up to 1.4 px on some Dove Classic scenes; not corrected by any of our models. |
| Cloud and scarce tie points | Accuracy "will vary from region to region based on available GCPs"; cloudier scenes worse (Planet forum) | Offset correlates with cloud cover (ρ = +0.25); the RPC tilt only weakly (ρ = +0.14). |

## How others correct it

| Approach | Used by | Relation to our models |
|---|---|---|
| One shift per scene (AROSICS, Planet Coregister) | Fitts et al. (2025) for tree crowns of a few pixels in the Sahel, registering to a Maxar mosaic; Dille et al. (2021) as cited by Mueting and Bookhagen | Model T. Fitts et al. note that "minor geolocation shifts (around half a pixel) impact smaller crowns more than larger ones". |
| First- or second-order polynomial per pair or per footprint | Kääb et al. (2017, 2019); Feng et al. (2019) | Model A (first order). |
| Polynomial in position **and elevation** per pair | Mueting and Bookhagen (2024): ramps and stereoscopic effects "can be efficiently modeled through a polynomial fit that takes into account X and Y position and local elevations" | Model H, arrived at independently. They apply it to displacement maps; we apply it to the imagery. |
| Pair only scenes with the same view direction | Mueting and Bookhagen (2024) | Does not apply: our term does not follow the reported view direction. |
| Refine the RPC (bias-compensate or re-fit the rational function model from tie points) and orthorectify the basic scenes | Aati and Avouac (2020): reprojection error 0.38 px for PlanetScope without ground control. Aati et al. (2022a): an affine "pointing error" correction per image from tie points projected at a range of heights. Ghuffar (2018): an affine bias per image; he found the delivered RPC bias "typically less than 0.5 pixels", but that is a residual at tie points in image space and does not test the model's dependence on height. Aati et al. (2022b) develop the approach for other sensors. | Not tried. Check 4 shows which correction is needed (a tilt) and gives its value per scene. |
| No correction | Frazier and Hemingway (2021), reviewing PlanetScope applications: "the majority of studies we reviewed did not report any additional geometric correction, raising questions about the spatial integrity and uncertainty of data used in those applications" | — |
| Orthorectify basic scenes on a newer or self-made elevation model | Mueting and Bookhagen (2024), for terrain that changed after the SRTM mission | No effect here: RPC + GLO-30 reproduces Planet's ortho to 0.042 m. |

## Checks against our scenes

`python scripts/island_coreg_diagnose.py metadata` joins the fitted terms of the 718 scenes
(`results/island_coreg/scene_models.csv`) with each scene's Planet item properties
(`<scene>_metadata.json`), ortho XML and, for the 544 with a basic asset on disk, RPC, and
writes `results/island_coreg/diagnose/metadata_checks.csv`.

**1. Planet's quality flags.** Six scenes have `ground_control` false. All six are
`quality_category` "test", their offsets from the master are 13–82 m (median 27 m against
7.0 m for the other 712), and their elevation terms are about twice as large. Filtering on
the flag is worthwhile but removes under 1 % of scenes and none of the ordinary error: 30 %
of `ground_control` true scenes are more than 10 m from the master. "Test" scenes (167, all
but 4 SuperDove) are slightly worse than "standard" ones (median translation residual 2.56
against 1.90 m). Offset, affine amplitude and translation residual rise with Planet's
`cloud_cover` (ρ = +0.25, +0.23, +0.22); the elevation term does not (+0.07).

**2. Elevation model.** All 718 ortho XMLs record `elevationCorrectionApplied` =
`FineDEM`, `geoCorrectionLevel` = `Precision Geocorrection` and processor version 4.1.4,
for acquisitions from 2020 to 2026. With the 0.042 m agreement between our RPC + GLO-30
orthos and Planet's, this rules out differences in the elevation model between dates as
the source of scene-to-scene misalignment here. The field names a product, not a dated
snapshot, so a silent update of the model is excluded by the reproduction, not by the XML.

**3. Reported view direction.** Each scene's line of sight, as ground displacement per
100 m of height, is 100·tan(incidence) along the satellite azimuth (median 6.3 m per
100 m; incidence from the XML, median 3.6°). If the term came from elevation error times
view direction, it would be proportional to the difference between a scene's line of sight
and the master's. It is not: slope −0.02, 0.0 % of variance explained (718 scenes).

**4. RPC line of sight.** For each RPC we computed how far the ground position of the
frame-center pixel moves when the assumed height rises by 100 m (from 27 m, lake level).
This is the model's line of sight in the same units as the elevation term. Its difference
from the reported line of sight of check 3 is the tilt of the model.

| Quantity (m per 100 m of height) | Value |
|---|---|
| Tilt, median / 90th percentile (544 scenes) | 3.03 / 10.65 (1.73° / 6.08°) |
| Tilt by generation, median: Dove Classic / Dove-R / SuperDove | 1.88 / 2.75 / 3.45 |
| Share of scenes with tilt above 5 / above 10 | 30 % / 11 % |
| Elevation term measured from imagery, median / 90th percentile | 3.05 / 10.62 |
| Elevation term = c × tilt: c, variance explained | 0.975, 97.5 % |
| Same, Dove Classic (68) / Dove-R (49) / SuperDove (427) | 0.76, 57.8 % / 0.92, 92.7 % / 0.99, 98.8 % |
| Elevation term minus tilt, median / 90th percentile | 0.51 / 1.42 |
| Mean of that difference, east / north | −0.00 / −0.22 |
| Master scene: tilt, east / north | +2.13 / +2.36 |
| Master scene: term measured against the drone, east / north | +2.00 / +2.43 |
| `ground_control` false (6) / true (538), median tilt | 9.29 / 3.02 |

The elevation term here is the absolute one (each scene's term relative to the master plus
the master's term against the drone mosaic), and "variance explained" is for the east and
north components together after removing their means. Three points follow.

- The image-derived term and the metadata-derived tilt are independent measurements. Their
  agreement means the reported view direction is close to the truth and the RPC is not: the
  orthorectification is correct at the height where the tie points constrained it and
  wrong by height × tilt elsewhere. The mechanism proposed in
  [island_coregistration.md](island_coregistration.md) is confirmed, and the master's
  elevation term, until now resting on one drone mosaic, is confirmed to 0.13 m per 100 m.
- The look direction legitimately varies across a frame (median 2.0 m per 100 m between the
  frame center and the island center). That variation explains 0.1 % of the term, so the
  term is the same wherever the island falls in the frame, as a tilt of the whole model
  would be.
- The fit is weaker for Dove Classic, whose tilts are smaller and whose terms are noisier
  (median difference 0.88 m per 100 m, against 0.47 for SuperDove).

Why the refinement leaves a tilt is not something these data show. With a field of view of
a few degrees and tie points on nearly flat ground (around BCI, the lake shore), a tilt of
the camera and a shift of its position have almost the same effect on the tie points, so a
solution can trade one for the other at no cost to its own residuals. That is our reading,
consistent with the term being unrelated to cloud cover; Planet's adjustment was not
examined. The translation and affine parts of the error (7 m and 7 m median) are not
explained by the tilt (ρ = 0.19 and 0.28) and remain attributed to the reference imagery
and tie-point distribution.

## What this changes

1. **The elevation term can be read from metadata.** It is the RPC line of sight at the
   frame center minus the reported one, good to a median 0.5 m per 100 m of height (0.7 m
   over the island's 138 m of relief). That figure is the disagreement with image matching
   on the 544 scenes that have both, so it bounds the error of the reported view direction
   and the matching noise together; using the tilt on a scene that could not be matched
   assumes the same holds there. The shift and the affine part still have to be measured
   from imagery. Model H currently needs enough clear
   tiles to fit eight parameters; with the term fixed from metadata, cloudy scenes need only
   a shift or an affine. It needs the basic asset's RPC file (on disk for 544 of the 718
   fitted scenes; the files are a few kB each).
2. **It reaches the 50ha plot.** The plot is on the plateau, roughly 95–135 m above the
   lake, so the term appears there as a shift of about 1.15 times its value: 3 m at the
   median and 11 m at the 90th percentile, different for each frame. Part of what the
   Planet-to-drone registration at the plot measures as a rigid shift is this, and for
   scenes that fail to register it could be predicted.
3. **Re-orthorectifying with a different elevation model will not help**, and neither will
   pairing scenes by view angle. Correcting the RPC would: the published route is RPC
   refinement on the basic scenes (Aati and Avouac 2020), for which check 4 supplies the
   dominant correction directly.
4. **Filter `ground_control` false scenes.** It costs six scenes and removes the worst
   offsets.

Done since (2026-11-17): the metadata-derived term is model R in `island_coreg.py`. It
matches the fitted model H (re-measured 1.04 against 1.03 m) with two fewer fitted
parameters per scene and is the recommended correction; see
[island_coregistration.md](island_coregistration.md#model-r-the-elevation-term-from-the-rpc).
Not done: testing it at the plot against the Planet-to-drone registration beyond the
cross-check there; RPC refinement; Planet's Coregister tool and basemaps on our scenes;
getting the RPC files of the 174 scenes that lack one.

## Figures (`figs/20261117_updates/`)

From `python scripts/make_island_alignment_figures.py geometry | tilt | rpcmap | model3d`;
each static figure has a `<name>_caption.md`.

1. `island_rpc_geometry` — one scene (2021-03-05, tilt 7.0°): the reported and the RPC line
   of sight from orbit (camera positions 61 km apart at 496 km), at the 50ha plot (a crown
   on the plateau placed 17 m away) and in map view with the measured elevation term.
2. `island_rpc_tilt` — all 544 scenes: reported and RPC lines of sight in map view, the
   elevation term against the tilt, and the distribution of the tilt in degrees.
3. `island_rpc_map` — the same scene over the island: elevation-related displacement
   measured by image matching, predicted from RPC and metadata, and their difference
   (median 2.4, 2.4 and 0.7 m).
4. `island_rpc_model.html` — a self-contained interactive 3-D page of the geometry at
   three scales (orbit, island, one tree crown) for the master scene, the example, a
   median scene of each generation and a 90th-percentile scene (template
   `scripts/island_rpc_model.html`).

## Reading status

Read in the relevant sections, with quotations checked against the library text: Mueting
and Bookhagen (2024; the library copy is the discussion version), Kääb et al. (2017, 2019),
Feng et al. (2019), Ghuffar (2018), Aati and Avouac (2020), Aati et al. (2022a), Frazier
and Hemingway (2021), Fitts et al. (2025), and the Planet specification and NASA
assessment from their PDFs. Aati et al. (2022b) and Huang et al. (2022) were only searched
for their treatment of PlanetScope RPCs: the first is not specific to PlanetScope and the
second does not discuss the RPCs beyond using them. The Planet forum posts and the
Coregister documentation were read through a text summary of the pages. Not retrieved:
Planet's quarterly L1 data quality reports and the Sentinel Hub post.

## References

Library ids are Scholium ids (tag `planet-geolocation`).

- Aati and Avouac 2020, *Optimization of Optical Image Geometric Modeling, Application to
  Topography Extraction and Topographic Change Measurements Using PlanetScope and SkySat
  Imagery*, Remote Sensing 12, 3418, doi:10.3390/rs12203418 — library id `b8507bacd2da1bf7`
- Aati et al. 2022a, *Potential and Limitation of PlanetScope Images for 2-D and 3-D Earth
  Surface Monitoring With Example of Applications to Glaciers and Earthquakes*, IEEE TGRS
  60, doi:10.1109/TGRS.2022.3215821 — `dac863ee158e9835`
- Aati et al. 2022b, *A new approach for 2-D and 3-D precise measurements of ground
  deformation from optimized registration and correlation of optical images and ICA-based
  filtering of image geometry artifacts*, Remote Sensing of Environment 277, 113038,
  doi:10.1016/j.rse.2022.113038 — `5f3b405778fb5a87`
- Feng et al. 2019, *A Block Ramp Errors Correction Method of Planet Subpixel Offset:
  Application to the 2018 Mw 7.5 Palu Earthquake, Indonesia*, IEEE Access 7,
  doi:10.1109/ACCESS.2019.2956198 — `459fdc3620d702c7`
- Fitts et al. 2025, *Using PlanetScope NDVI time series to detect the phenology of
  individual trees in the Sahel*, Remote Sensing of Environment,
  doi:10.1016/j.rse.2025.114650 — `d98cdadd7f56c243`
- Frazier and Hemingway 2021, *A Technical Review of Planet Smallsat Data: Practical
  Considerations for Processing and Using PlanetScope Imagery*, Remote Sensing 13, 3930,
  doi:10.3390/rs13193930 — `0f9e161642c24bc2`
- Ghuffar 2018, *DEM Generation from Multi Satellite PlanetScope Imagery*, Remote Sensing
  10, 1462, doi:10.3390/rs10091462 — `3fd70d5970ac0b01`
- Huang et al. 2022, *An evaluation of PlanetScope images for 3D reconstruction and change
  detection – experimental validations with case studies*, GIScience & Remote Sensing 59,
  doi:10.1080/15481603.2022.2060595 — `fbd1f2555c15ce2c`
- Kääb et al. 2017, *Coseismic displacements of the 14 November 2016 Mw 7.8 Kaikoura, New
  Zealand, earthquake using the Planet optical cubesat constellation*, NHESS 17, 627–639,
  doi:10.5194/nhess-17-627-2017 — `ce4ba4496ad7e07c`
- Kääb et al. 2019, *River-ice and water velocities using the Planet optical cubesat
  constellation*, HESS 23, 4233–4247, doi:10.5194/hess-23-4233-2019 — `3e1b2c4d370257d2`
- Mueting and Bookhagen 2024, *Tracking slow-moving landslides with PlanetScope data: new
  perspectives on the satellite's perspective*, Earth Surface Dynamics 12, 1121–1143,
  doi:10.5194/esurf-12-1121-2024 — `121e9860ad8b16f3`
- Scheffler et al. 2017, *AROSICS: An Automated and Robust Open-Source Image
  Co-Registration Software for Multi-Sensor Satellite Data*, Remote Sensing 9, 676,
  doi:10.3390/rs9070676 — `232aca1025303b16`
- Semple, Tan and Lin 2023, *SuperDove Geometric Quality Assessment Summary*, NASA CSDA
  Program, issue 1.0, 2023-11-13,
  <https://www.earthdata.nasa.gov/s3fs-public/2024-05/NASA-PlanetSuperDoveGeometricAssessment-04092024%20Report.pdf>
- Planet 2026, *PlanetScope Product Specifications*, June 2026,
  <https://assets.planet.com/docs/Planet_PSScene_Imagery_Product_Spec_letter_screen.pdf>
- Planet, *Orders API tools: Coregister*,
  <https://docs.planet.com/develop/apis/orders/tools/#coregister>
- Planet Community 2025, *Slight spatial misalignment in PlanetScope/RapidEye imagery*,
  <https://community.planet.com/planetscope-76/slight-spatial-misalignment-in-planetscope-rapideye-imagery-6172>
- Planet Community 2024, *Improve co-registration tool in APIs for higher average
  positional accuracy*,
  <https://community.planet.com/ideas/improve-co-registration-tool-in-apis-for-higher-average-positional-accuracy-5860>
- Peressutti 2023, *How to Co-Register Temporal Stacks of Satellite Images*, Sentinel Hub
  blog, <https://medium.com/sentinel-hub/how-to-co-register-temporal-stacks-of-satellite-images-5167713b3e0b>
