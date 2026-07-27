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
