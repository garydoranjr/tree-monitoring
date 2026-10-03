# Data Source Information

The following publication includes the 2025 crown maps for the BCI 50-ha plot,
as well as the 10-ha plot directly north of it, the AVA plot, the 25-ha plot,
and multiple other large plots within the BCNM. The 2024-2025 drone data
includes the 10-ha plot, the AVA plot, and about half the 25-ha plot in
addition to the 50-ha plot. 
 
> Arauz, F., M. Hernandez, P. Villarreal, P. Ramos, A. Agrazal, M. Perez, M.
> Demarsan, and H. C.  Muller-Landau. 2026. AVUELO: Tropical Forest Crown Maps
> with Tree Species Names, Panama (Version 1). ORNL Distributed Active Archive
> Center. https://doi.org/10.3334/ORNLDAAC/2517 

This is a single, one-time crown map (no per-date geometry): one polygon per
tagged tree, delineated from one orthomosaic per site (for BCI, a Trinity Pro
flight on 2024-07-16) and field-verified 2025-2026. Phenology attributes are a
single observation at each crown's field-visit date (`Census_date_2025`).
 
 
And here are the drone data publications:

> Vásquez, V., M. García, M. Hernández, and H. Muller-Landau. 2024. Barro
Colorado Island AVA Plot Aerial Photogrammetry (2018-2024): Orthomosaics,
Digital Surface Models, Point Clouds, and Raw Images. Smithsonian Research Data
Repository. https://doi.org/10.60635/C3G59D

> Vásquez, V., M. García, M. Hernández, and H. Muller-Landau. 2024. Barro
> Colorado Island 50-ha Plot Aerial Photogrammetry (2018-2024): Orthomosaics,
> Digital Surface Models, Point Clouds, Raw Images, and Globally/Locally Aligned
> Timeseries. Smithsonian Research Data Repository.
> https://doi.org/10.60635/C3KW2X .  (This is an updated version of the 2023
> aligned-timeseries publication available at
> https://doi.org/10.25573/data.24782016)

**Caveat for C3KW2X:** on the download server
(https://smithsonian.dataone.org/datasets/BCI_50ha_drone_products/) the
directory names are swapped relative to the file names:
`orthomosaic_aligned_global/` holds `*_local.tif` and
`orthomosaic_aligned_local/` holds `*_global.tif`. The file suffixes are
correct (verified 2026-10-02: `*_global.tif` is a constant rigid offset from
the unaligned orthomosaic across the plot, while `*_local.tif` varies by up to
~2 m). The 2024 release also reprocessed every date, so its files are not
pixel-identical to the 2023 (24782016) versions of the same dates.

Earlier (2023) figshare releases that are also used locally:

> Barro Colorado Island 50-ha plot aerial photogrammetry orthomosaics and
> digital surface models for 2018-2023: Globally and locally aligned time
> series. Smithsonian Tropical Research Institute, figshare.
> https://doi.org/10.25573/data.24782016

> Barro Colorado Island 50-ha plot crown maps: manually segmented and instance
> segmented. Smithsonian Tropical Research Institute, figshare.
> https://doi.org/10.25573/data.24784053

> Barro Colorado whole-island aerial photogrammetry products for 2018-2023.
> Smithsonian Tropical Research Institute, figshare.
> https://doi.org/10.25573/data.24757284

## Dataset names

For internal record keeping the 50-ha drone mosaics are grouped by the drone
that flew them:

| Name | Drone / camera | Sources | Dates |
|---|---|---|---|
| **phantom** | DJI Phantom 4 Pro (`FC6310`) | 24782016 (90 dates) + C3KW2X (16 dates), global and local alignments | 2018-04-04 – 2024-03-18 |
| **mavic** | DJI Mavic 3M (M3M) | the STRI Globus share in `stri/globus/RGB/`, global alignment only | 2024-03-06 – 2026-01-20 |

The two overlap on 2024-03-06 and 2024-03-18. These names are used in
code identifiers, configs (`config/pipeline_mavic.sh`,
`config/crown_classification_mavic.yml`, `config/pipeline_phantom_ext_*.sh`),
figure labels and new output directories. Existing on-disk names are **not**
renamed, since coreg logs, checkpoints and docs record them: `stri/globus/`,
`*_M3M_aligned_global_RGB*`, `/scratch/tree-monitoring/results/globus`, and
the `20260915_globus_*` chip sets all refer to the mavic data. "Globus" on its
own still means the transfer service (`scripts/globus_https_sync.py`,
`NOTES.md`). Older figures and the September 2026 deck call phantom "STRI"
and mavic "globus".

## Local copies

Drone data lives under `/Volumes/Earth03/flower/stri/`. Gattaca2 has copies
of some of these directories under `/scratch/tree-monitoring/stri/` (e.g.
`globus/RGB`, `24784053`, `C3KW2X`). `avuelo/` was fetched 2026-10-02 to
Earth03 only. Together, the three 50-ha aligned-timeseries sources cover
2018-04 through 2026-01 (near-monthly flights through early 2023, near-weekly
after; the longest gap in the C3KW2X stretch is 2023-12-12 to 2024-01-03).

| Local directory | Source | Contents |
|---|---|---|
| `stri/24782016/` | doi:10.25573/data.24782016 (2023) | 50-ha global and local aligned 4-band (RGB+DSM) timeseries, 90 dates 2018-04-04 – 2023-10-24; LiDAR and plot outline in `aux_files/` |
| `stri/C3KW2X/` | doi:10.60635/C3KW2X (2024) | Only the 16 aligned dates **not** in 24782016: 2023-10-31 – 2024-03-18, same layout and grid as `24782016/`. Files are filed by suffix, not by the swapped server directory names; see `README_local.md` and `manifest.tsv` there |
| `stri/globus/RGB/` | STRI Globus share `SI_STRI_ForestLandscapes_Shares:/UAVSHARE/paula_mavic_products/Aligned/Product_global/RGB/` (not a formal publication), fetched with `scripts/globus_https_sync.py` | 96 globally aligned 50-ha M3M mosaics, 2024-03-06 – 2026-01-20 (uint16). Overlaps C3KW2X on 2024-03-06 and 2024-03-18, but from a **different camera** (Mavic 3M, against the Phantom 4 Pro `FC6310` behind 24782016 and C3KW2X) as well as different processing — so the shared dates are not a like-for-like comparison |
| `stri/24784053/` | doi:10.25573/data.24784053 (2023) | 50-ha crown maps for 2020-08-01 and 2022-09-29 (raw and improved) |
| `stri/24757284/` | doi:10.25573/data.24757284 (2023) | Whole-island eBee orthomosaics, DSMs, point clouds, 8 dates 2018-2023 |
| `stri/avuelo/` | doi:10.3334/ORNLDAAC/2517 (2026) | Complete dataset: `AVUELO_combined_crownmaps_2025.gpkg` (7,688 crowns, 9 plots; 2,429 in `p50`), the three source orthomosaics, and the user guide. The BCI orthomosaic is stored as uint16 with values 0-255 (nodata 65535) |

### Crown classifications of the C3KW2X dates

All 32 C3KW2X mosaics (16 dates x both alignments) were classified on
2026-10-02 into `/scratch/tree-monitoring/results/phantom_ext/`, with
`run_crown_pipeline.sh` and `config/pipeline_phantom_ext_{global,local}.sh`.
Both alignments share that one output directory because each per-date
subdirectory is named from the image stem, which already carries the
`_global`/`_local` suffix. Products per date follow the usual layout:
`<stem>_flower.tif`, `<stem>_decid.tif`, the 2-band
`<stem>_classifications.tif` (band 1 flowering probability, band 2
deciduous), and the `flower/`+`decid/` per-crown tile directories.

Unlike the mavic (globus) mosaics, these need **no `--scaling-config`**: they are
4-band uint8 (23425x12697, EPSG:32617), the same dtype as the 24782016
imagery the SegFormer models were trained on, so they are passed through
untouched. `config/crown_classification_mavic.yml` applies only to the
uint16 mavic rasters.

**Do not read a phenology step at the 2023-10-24 / 2023-10-31 boundary as
biology.** That boundary is where the series crosses from the 2023 release
(24782016) to the 2024 one (C3KW2X), and the 2024 release reprocessed every
date — RGB correlation on the dates present in both is only ~0.27. The
footprints agree to within 9 mm, so the crown map projects correctly onto
both; it is the pixel content, not the georeferencing, that changes.

AVA plot imagery used by the pipeline is in `/Volumes/Earth03/flower/ava/`
(aligned `global/` and `local/` orthomosaics, 2018-11-26 – 2026-01-20, plus
`BCI_ava_crownmap_timeseries.gpkg`). It already covers every C3G59D date, so
C3G59D was not downloaded separately. The local copy extends past C3G59D's
2024-03-18 end date and includes aligned products C3G59D does not publish, so
its exact provenance is not recorded here.
