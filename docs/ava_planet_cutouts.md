# AVA-plot Planet cutouts

Clipped Planet time-series cutouts around the **AVA plot** on BCI, the
analogue of the existing 50ha `planet_clipped` product. Produced 2026-07-26.

## Plot extent

The AVA plot is defined by the drone orthomosaics in
`/Volumes/Earth03/flower/ava/global/` (`BCI_ava_YYYY_MM_DD_orthomosaic.tif`)
and the crown map `/Volumes/Earth03/flower/ava/BCI_ava_crownmap_timeseries.gpkg`.
Both are in **EPSG:32617** (UTM 17N). The footprint is consistent across
all dates (2018–2026) and matches the crown-map extent:

- x: **626391.186 – 626594.030**, y: **1012307.552 – 1012617.796**
- ~203 m × 310 m, located just north of the 50ha plot.

## Clip configs

Mirror the 50ha configs (`config/clip_50ha_plot.yml` /
`config/clip_50ha_plot_margin.yml`), same GeoJSON ring ordering:

- `config/clip_ava_plot.yml` — AVA ortho footprint **+ 20 m** all sides
  (drives the cutouts).
- `config/clip_ava_plot_margin.yml` — base **+ 300 m** all sides (created
  for parity; not used for the current build).

## Build

Generated with the unchanged `scripts/build_planet_clipped_50ha.py`,
reusing the same Planet sources as the 50ha build
(`--csdap-root /Volumes/Earth03/flower/csdap`,
`--planet-root /Volumes/Earth03/flower/planet`):

```bash
python scripts/build_planet_clipped_50ha.py \
  --clip-config config/clip_ava_plot.yml \
  --output-root /Volumes/Earth03/flower/ava/planet \
  --no-api-check
```

Default date range 2020-01-01 → today. `--no-api-check` skips the Planet
API missing-files report; drop it if API credentials are configured.

## Output

Written to `/Volumes/Earth03/flower/ava/planet/{4band,udm2,rgb}/<year>/`
plus `inventory.csv`. Each 4-band cutout is ~82 × 117 px at 3 m,
EPSG:32617, extent ≈ 626370–626616 × 1012287–1012638.

From the 2026-07-26 run (4982 candidate scenes):

| status | count |
|---|---|
| `ok` | 1925 |
| `no_intersection` | 2214 |
| `nodata` | 840 |
| other | 3 |

Yielding 1927 `4band` / 1927 `rgb` / 2768 `udm2` cutouts. The high
`no_intersection` count is expected — many whole-island source scenes
do not cover this small plot; slightly fewer `ok` scenes than the 50ha
build (2051 RGB) because AVA sits a bit north.

## OCM cloud masks

OmniCloudMask (OCM) cloud/shadow masks live at
`/Volumes/Earth03/flower/ava/planet/ocm/<year>/<prefix>_ocm.tif`,
one per base `4band` cutout, on the identical 82 × 117 px grid.
5-band uint8 (band 1 = argmax class `0=Clear,1=Thick Cloud,2=Thin
Cloud,3=Cloud Shadow,255=NoData`; bands 2–5 = per-class softmax prob,
scale 1/255), matching the 50ha `planet_clipped/ocm` product.

### Why a margin-then-crop workflow

`scripts/cloud_mask_planet.py` runs OCM inference at 10 m and requires
≥ 32 × 32 px there. The base AVA cutout is only ~24 × 35 px at 10 m
(width < 32), so masking the base cutouts directly fails on every
scene. Instead we mask a **larger extent** and crop back:

1. Build margin-extent 4-band cutouts with `config/clip_ava_plot_margin.yml`
   (~843 × 950 m → 282 × 317 px ≈ 85 × 95 px at 10 m):
   ```bash
   python scripts/build_planet_clipped_50ha.py \
     --clip-config config/clip_ava_plot_margin.yml \
     --output-root /Volumes/Earth03/flower/ava/planet_margin --no-api-check
   ```
   (2047 cutouts.)
2. Run OCM on the margin cutouts:
   ```bash
   KMP_DUPLICATE_LIB_OK=TRUE python scripts/cloud_mask_planet.py \
     --input-dir /Volumes/Earth03/flower/ava/planet_margin/4band \
     --output-dir /Volumes/Earth03/flower/ava/planet_margin/ocm
   ```
   (1990 masks; 57 errored — thin partial-swath scenes still < 32 px.)
   `KMP_DUPLICATE_LIB_OK=TRUE` works around an OpenMP duplicate-runtime
   abort seen with torch on this machine.
3. Crop the margin masks to the base grid with
   `scripts/crop_ocm_to_base.py`. For each scene the base and margin
   cutouts are `rasterio.mask` crops of the same source scene, so they
   share one pixel grid and the crop is a pure integer-offset window
   read (no resampling), preserving band descriptions/scales/tags:
   ```bash
   python scripts/crop_ocm_to_base.py
   ```

### Coverage

**1877 of 1927** base scenes have masks (2026-07-26 run). The 50
uncovered scenes are thin edge slivers whose Planet swath barely clips
the plot (≤ 2.5 % plot coverage; base cutout as small as 82 × 3 px) —
their margin cutout is still < 32 px tall at 10 m, and they carry no
usable plot imagery, so they are skipped. Full coverage from 2023 on;
misses are concentrated in 2020–2021.

The `/Volumes/Earth03/flower/ava/planet_margin/` tree is an
intermediate needed only to regenerate masks and can be deleted once
the cropped `ocm/` product is accepted.

## Training set (drone labels → Planet chips)

AVA analogue of the 50ha drone-label→Planet training chips, built to
augment the existing 50ha set. Final chips live in
`/Volumes/Earth03/flower/ava/train/` as 4-band uint16 GeoTIFFs
(`<scene>_4band.tif`) with a paired binary crown mask
(`<scene>_4band.mask.png`) plus RGB/OCM/drone QA PNGs, matching the
50ha training-dir layout so the two sets concatenate directly.

### Why margin-then-crop-then-pad

Two size constraints drive a three-stage build:

- **COREG needs ≥ 200 px.** `scripts/apply_drone_labels_coreg.py` runs
  AROSICS with `ws=(200,200)`; the base AVA cutout (82 × 117 px) is too
  small, so alignment is done on the **margin** cutout (282 × 317 px,
  `config/clip_ava_plot_margin.yml`).
- **Labels are only valid inside the plot.** The AVA drone ortho
  (`ava/global/*.tif`) covers only the plot, so on the margin grid the
  surrounding 300 m of real canopy would be reprojected to *no label*
  and poison training as background. Chips are therefore **cropped back
  to the base plot extent** after label application.
- **Scale parity with 50ha.** Both products are 3 m/px, but the AVA base
  footprint (82 × 117) is far smaller than a 50ha cutout (367 × 205), so
  the SegFormer dataloader's resize-to-512 would upsample AVA crowns
  ~4–6× more. Cropped chips are **center-padded to 367 × 205** (image =
  nodata, mask = 0) so a crown covers a comparable post-resize pixel
  count in both datasets.

### Build

1. **Apply labels on the margin cutouts** (unchanged
   `scripts/apply_drone_labels_coreg.py`), reusing the margin OCM masks
   for crown filtering:
   ```bash
   KMP_DUPLICATE_LIB_OK=TRUE python scripts/apply_drone_labels_coreg.py \
     /Volumes/Earth03/flower/ava/results/*_classifications.tif \
     /Volumes/Earth03/flower/ava/global \
     /Volumes/Earth03/flower/ava/planet_margin/4band \
     /Volumes/Earth03/flower/ava/train_margin \
     --bands 4 --timewindow 2 \
     --maskdir /Volumes/Earth03/flower/ava/planet_margin/ocm
   ```
   `KMP_DUPLICATE_LIB_OK=TRUE` works around the torch OpenMP
   duplicate-runtime abort on this machine. Writes margin-extent chips +
   masks + QA PNGs and a `coreg_log.json`.
2. **Crop to base and pad to the 50ha footprint** with the new
   `scripts/crop_train_to_base.py` (patterned on `crop_ocm_to_base.py`:
   integer-offset window read against the paired base 4-band cutout, no
   resampling; then a constant center-pad):
   ```bash
   python scripts/crop_train_to_base.py
   ```
   Defaults: margin `.../ava/train_margin` → base grids
   `.../ava/planet/4band` → output `.../ava/train`, `--pad-to 367x205`.
   Use `--no-pad` for base-extent-only chips.

### Yield (2026-07-26 run)

150 AVA drone label dates (2018-11-26 → 2026-01-20) × Planet scenes
within ±2 days gave **616 candidate pairs**, of which **198 coregistered**
(32 %; the rest fail AROSICS, largely cloudy/low-texture scenes). All 198
became 367 × 205 chips. Usable counts after cloud filtering (clear =
OCM class-0 fraction over the **base plot**):

| base-plot clear ≥ | chips |
|---|---|
| 0.25 | 159 |
| 0.50 | 147 |
| **0.80** | **135** |
| 0.90 | 130 |
| 0.95 | 126 |

So ~**135 usable chips at the reliable ≥ 0.80 clear threshold**,
meaningfully augmenting the 50ha set. `coreg_log.json` in
`train_margin/` records per-scene `coreg_ok`, shift, and (margin-extent)
`clear_fraction` for provenance.

The `/Volumes/Earth03/flower/ava/train_margin/` tree is an intermediate
(margin-extent chips) needed only to rebuild `train/` and can be deleted
once the cropped product is accepted.
