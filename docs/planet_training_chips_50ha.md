# 50ha Planet training chips (drone labels → Planet scenes)

The Mask R-CNN / SegFormer training set for the BCI 50ha plot: 4-band
Planet cutouts at 0.75 m paired with a binary crown mask derived from
drone-imagery crown classifications. Built by
[`scripts/apply_drone_labels_coreg.py`](../scripts/apply_drone_labels_coreg.py),
then reduced by a human vetting pass. The AVA-plot analogue, including the
reasons that build needs a margin-then-crop workflow, is in
[ava_planet_cutouts.md](ava_planet_cutouts.md).

This pipeline is not in the `Snakefile` — it depends on drone orthomosaics
and Labelbox exports that are not reproducible from within the repo. The
command lines below are the record.

## Directory-name tokens

Sets are named `<YYYYMMDD>_<scope>_label_application_x4_coreg_4band_stretch_stats[_filt|_curated]`:

| Token | Meaning |
|---|---|
| `YYYYMMDD` | build date |
| `full` / `globus` / `phantomext` | which label set was applied (see *Inputs*): `full` = phantom 24782016 (2020–2023), `globus` = mavic (2024–2026), `phantomext` = phantom C3KW2X (2023–2024); `full` in a `_curated` name means a merged set. Dataset names are explained in [data_sources.md](data_sources.md#dataset-names) |
| `x4` | `--resize 4`: the 3 m Planet cutout is upsampled to 0.75 m, 367 × 205 → 1468 × 820 px |
| `coreg` | per-scene AROSICS drone↔Planet shift applied to the label raster before rasterizing (`ws=(200,200)`, `max_shift=10`, matched on Red in both images) |
| `4band` | `--bands 4`: chips come from `*4band.tif` and are written as 4-band uint16 GeoTIFFs (Blue, Green, Red, NIR) |
| `stretch` | cloud-aware per-band 2/98 percentile stretch for the QA PNG — percentiles from OCM-clear pixels only, applied to all non-nodata pixels, so clouds saturate white |
| `stats` | per-scene noise/quality metrics written into `coreg_log.json` (`noise_mad_sigma`, `noise_mad_cv`, `band_corr_mean`, `band_decorrelation`) |
| `_filt` | programmatic threshold filter — [`scripts/filter_label_application.py`](../scripts/filter_label_application.py) with [`config/filter_label_application.yml`](../config/filter_label_application.yml) |
| `_curated` | human Labelbox `Quality` vetting pass — [`copy_good_planet_vetting.py`](../copy_good_planet_vetting.py) |

`stretch` and `stats` only exist on the `--bands 4` path; both
`render_rgb_from_4band()` and `compute_noise_metrics()` are on that branch
of `create_mask()`.

## Per-chip outputs

All files for a scene share the stem `<planet_scene>_4band`:

| File | Contents |
|---|---|
| `.tif` | 4-band uint16 GeoTIFF at 0.75 m, EPSG:32617 — **the training image** |
| `.mask.png` | binary uint8 (0/255) crown mask — **the label** |
| `.png` | RGB QA render with the cloud-aware stretch |
| `.drone.png` | shift-corrected drone ortho on an exact 2× subdivision of the chip grid (`--drone-scale 2`) |
| `.ocm.png` | RGBA cloud overlay from the OmniCloudMask mask |
| `coreg_log.json` | one record per (label, scene) pair, at the set root |

A scene appears only if AROSICS converged; failures are recorded in
`coreg_log.json` with `coreg_ok: false` and no chip files.
`crown_filter_by_ocm()` drops any connected crown component that touches a
non-clear OCM pixel, so partly clouded crowns are not labelled.

## Inputs

Three drone label sources, each with its own classification rasters under
`/Volumes/Earth03/flower/results/classifications/`: the phantom (Phantom 4
Pro) record in two releases, 24782016 and C3KW2X, and the mavic (Mavic 3M)
record. All classification products are 2-band float32
`(flowering_probability, deciduous_probability)` in EPSG:32617; the default
`--mode both` takes their union, `1 - Π(1 - p)`.

| | phantom 2020–2023 (`full`) | phantom 2023–2024 (`phantomext`) | mavic 2024–2026 (`globus`) |
|---|---|---|---|
| Orthos | `stri/24782016/BCI_50ha_timeseries_local_alignment/` | `stri/C3KW2X/BCI_50ha_timeseries_local_alignment/` | `stri/globus/RGB/` |
| Ortho format | 4-band uint8, ~23425 × 12697, ~1.2 GB | same as 24782016 | 4-band uint16 (R, G, B, Alpha), ~25000 × 21000, ~4 GB |
| Ortho footprint | 1056 × 572 m | same as 24782016 | 1199 × 1022 m |
| Labels | the 24782016 dates of `*_local_classifications.tif` | the 16 C3KW2X dates of `*_local_classifications.tif` | `*_M3M_aligned_global_RGB_classifications.tif` |
| Dates | 2020-01-24 → 2023-10-24 | 2023-10-31 → 2024-03-18, 16 flights | 2024-03-06 → 2026-01-20, 96 flights |
| Provenance | doi:10.25573/data.24782016 | doi:10.60635/C3KW2X, classified by `run_crown_pipeline.sh config/pipeline_phantom_ext_local.sh` | synced by `scripts/globus_https_sync.py`, classified by `run_crown_pipeline.sh config/pipeline_mavic.sh` (see [NOTES.md](../NOTES.md)) |

**`*_local_classifications.tif` now matches both phantom releases** (106
dates). The two releases keep their orthos in different directories, and
`find_drone()` looks only in the one `DRONEDIR` it is given, so a build must
pass the labels of one release at a time. The commands below build the
label list from the release's ortho directory.

`find_drone()` matches a label to its ortho by filename prefix, so
`BCI_50ha_2024_03_06_M3M_aligned_global_RGB_classifications.tif` resolves to
`BCI_50ha_2024_03_06_M3M_aligned_global_RGB.tif`.

**Label coverage differs between the two halves.** The 2020–2023 orthos
(1056 × 572 m) are *smaller* than the 1101 × 615 m chip, so those chips have
unlabelled margins that training sees as background. The globus orthos are
larger than the chip, so those chips are labelled edge to edge.

Planet side: `planet_clipped/4band/<year>/` cutouts
(`config/clip_50ha_plot.yml`) and their OmniCloudMask masks in
`planet_clipped/ocm/<year>/`. Each label date is paired with every Planet
scene within `--timewindow 2` days.

## Build

### 1. Apply labels (phantom 2020–2023, `20260608` set)

When this set was built the glob `*_local_classifications.tif` matched only
the 90 24782016 dates; with the C3KW2X rasters alongside it, list the
24782016 labels explicitly:

```bash
ORTHO=/Volumes/Earth03/flower/stri/24782016/BCI_50ha_timeseries_local_alignment
LABELS=()
for f in $ORTHO/BCI_50ha_*_local.tif; do
  LABELS+=(/Volumes/Earth03/flower/results/classifications/$(basename "$f" .tif)_classifications.tif)
done
KMP_DUPLICATE_LIB_OK=TRUE python scripts/apply_drone_labels_coreg.py \
  "${LABELS[@]}" \
  $ORTHO \
  /Volumes/Earth03/flower/planet_clipped/4band \
  /Volumes/Earth03/flower/20260608_full_label_application_x4_coreg_4band_stretch_stats \
  -b 4 -r 4 -t 2 -d 2 \
  -k /Volumes/Earth03/flower/planet_clipped/ocm
```

323 (label, scene) pairs, 131 coregistered.

### 2. Apply labels (mavic 2024–2026, `20260915` set)

Identical flags; only the label glob and the ortho directory change.

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/apply_drone_labels_coreg.py \
  /Volumes/Earth03/flower/results/classifications/*M3M_aligned_global_RGB_classifications.tif \
  /Volumes/Earth03/flower/stri/globus/RGB \
  /Volumes/Earth03/flower/planet_clipped/4band \
  /Volumes/Earth03/flower/20260915_globus_label_application_x4_coreg_4band_stretch_stats \
  -b 4 -r 4 -t 2 -d 2 \
  -k /Volumes/Earth03/flower/planet_clipped/ocm
```

96 label dates → 370 pairs (7 dates have no Planet scene within ±2 days).

`KMP_DUPLICATE_LIB_OK=TRUE` works around a torch OpenMP duplicate-runtime
abort on this machine.

**Yield.** 370 pairs, **154 coregistered (41.6 %)**, against 131/323
(40.6 %) for the 2020–2023 build — the two eras agree closely, so the
failure rate is weather and not a footprint or band-order problem. Applied
shifts have median magnitude 4.30 m and max 27.83 m against the 30 m
(`max_shift=10` px) bound, with mean (x, y) = (+2.25, −1.35) m: a
sub-pixel systematic drone↔Planet georeferencing offset at the 3 m Planet
scale, which is what the coregistration step exists to remove. Chip crown
fraction has median 4.07 % (max 8.55 %), against 4.15 % for the 56 curated
2020–2023 chips.

Chip geometry is 820 × 1468 px for 152 of the 154; two partial-swath
scenes come out 728 × 1468 and 820 × 1204. `get_split()` needs
`height >= size` and `width >= 2 * size` — 512 and **1024** at the default
size, since the left and right windows must not overlap — and all three
shapes clear it, the 820 × 1204 chip with the least room (+180 px of
width). A narrower partial swath would raise `ValueError` rather than
silently return an overlapping crop, so it fails loudly.

Both loaders were smoke-tested over all 154 chips: every `*.mask.png`
resolves to its `.tif`, `left` and `right` both crop to 512 × 512, and
`PlanetMaskRCNNDataset` yields 14 794 crown instances on the left split
(11 chips give zero instances — the fully clouded ones, which the vetting
pass drops).

**Runtime and memory.** 2 h 49 m for the 370 pairs on a 64 GB / 16-core
machine (~23 s per pair). `process_label()` holds one drone ortho for the
duration of a label date's scenes, and **measured peak RSS is ~20 GB** —
well above the ~7 GB that the ortho (~4.5 GB) plus the
`generate_drone_png()` reprojection (~2.5 GB) would suggest. A 16 GB
machine drives this deep into swap and runs roughly 4× slower per pair
(~95 s); drop `-d 2` to skip the drone PNGs if that is the only option.

**One Planet scene can pair with two label dates.** Chip stems are keyed
on the Planet scene alone, so when two flights fall within `--timewindow`
of one scene the second date's chip overwrites the first and only one of
the two pairs keeps its output. In the globus set this affects 6 scenes /
12 pairs (370 pairs → 364 unique scenes). `coreg_log.json` still records
both pairs; only the files on disk collide.

**If the final `json.dump` fails,** `scripts/reconstruct_coreg_log.py`
rebuilds the log from the chips without re-running the build. Everything
except the AROSICS shift is recoverable, and rebuilt records are flagged
`reconstructed: true` with a null shift. This was needed once, for the
float32 crash fixed in `compute_coreg_shift`; the recovered log was later
confirmed to agree with a full re-run on all 370 `coreg_ok` values.

### 2b. Apply labels (phantom 2023–2024, C3KW2X, `20261002` set)

The 16 C3KW2X dates fill what used to be the gap between the two records.
Same flags; local alignment, like the 2020–2023 phantom build:

```bash
ORTHO=/Volumes/Earth03/flower/stri/C3KW2X/BCI_50ha_timeseries_local_alignment
LABELS=()
for f in $ORTHO/BCI_50ha_*_local.tif; do
  LABELS+=(/Volumes/Earth03/flower/results/classifications/$(basename "$f" .tif)_classifications.tif)
done
KMP_DUPLICATE_LIB_OK=TRUE python scripts/apply_drone_labels_coreg.py \
  "${LABELS[@]}" \
  $ORTHO \
  /Volumes/Earth03/flower/planet_clipped/4band \
  /Volumes/Earth03/flower/20261002_phantomext_label_application_x4_coreg_4band_stretch_stats \
  -b 4 -r 4 -t 2 -d 2 \
  -k /Volumes/Earth03/flower/planet_clipped/ocm
```

No new cloud masks were needed. All 175 50ha Planet scenes acquired
2023-10-29 to 2024-03-20 already had readable OCM masks, which were checked
before the build.

**Yield.** 16 label dates → 91 pairs, all on distinct Planet scenes
(2023-12-05 has no scene within ±2 days), and **46 coregistered (50.5 %)**.
Median applied shift is 7.37 m (max 29.2 m), close to the 8.45 m of the
2020–2023 phantom build and above the mavic 4.30 m. The global-alignment
builds (step 2c) show that this is a phantom-vs-Planet offset shared by
both alignments, not something the local warp adds. Several adjacent Planet frames from the same strip got
shifts that differ by more than 10 m (e.g. `20240209_155640_21_24ad`
19.5 m vs `20240209_155642_52_24ad` 7.9 m). `measure_chip_local_offsets.py`
nevertheless finds both well aligned after their shift (median window
residual 0.2 m and 0.8 m), so the per-scene Planet georeferencing really
does differ by that much. Over all 46 chips the median window residual is
0.59 m. Only two chips are misregistered (12.5 m and 19.6 m residual),
both mostly clouded.

Runtime was 62 min on the 16 GB laptop, about 41 s per pair. The orthos are
the 1.2 GB uint8 ones, not the 4 GB mavic ones, but the machine still ran
deep into swap while other jobs were reading the volume.

**Chip names collide with the mavic set.** The 2024-03-06 and 2024-03-18
phantom flights pair with five Planet scenes that the mavic build also
used. Those chips have identical stems, all five are Good in both sets,
and they are in the 126-chip extended set:
`20240304_155459_24_24f6`, `20240306_150044_82_24ba`,
`20240306_150046_96_24ba`, `20240307_150321_14_24a8` and
`20240317_150402_66_2455`. They are handled at assembly (step 5).

### 2c. Apply labels (phantom, global alignment, `20261003` sets)

Phantom training chips use the local alignment. To compare alignments,
both phantom releases were also built from their globally aligned mosaics
and global classifications, with the same flags. One build per release,
because `find_drone()` looks in a single `DRONEDIR`:

```bash
F=/Volumes/Earth03/flower
for spec in "24782016 fullglobal" "C3KW2X phantomextglobal"; do
  set -- $spec; REL=$1; TOK=$2
  ORTHO=$F/stri/$REL/BCI_50ha_timeseries_global_alignment
  LABELS=()
  for f in $ORTHO/BCI_50ha_*_global.tif; do
    LABELS+=($F/results/classifications/$(basename "$f" .tif)_classifications.tif)
  done
  KMP_DUPLICATE_LIB_OK=TRUE python scripts/apply_drone_labels_coreg.py "${LABELS[@]}" \
    $ORTHO $F/planet_clipped/4band \
    $F/20261003_${TOK}_label_application_x4_coreg_4band_stretch_stats \
    -b 4 -r 4 -t 2 -d 2 -k $F/planet_clipped/ocm
done
```

| Build | Pairs | Coregistered | Median shift | Local-alignment counterpart |
|---|---|---|---|---|
| `20261003_fullglobal_...` (24782016) | 323 | 128 (39.6 %) | 8.41 m | 131 (40.6 %), 8.45 m |
| `20261003_phantomextglobal_...` (C3KW2X) | 91 | 43 (47.3 %) | 7.40 m | 46 (50.5 %), 7.37 m |

Runtime was 3 h 03 m and 40 min on the 16 GB laptop. These sets are
analysis-only and were not vetted. They are compared with the local chips
on the same Planet scenes, using the local ratings, because ratings judge
Planet image quality and not the drone alignment. 151 scenes coregistered
under both alignments, and AROSICS gave each almost the same shift either
way (median difference in magnitude +0.01 m). See `plot_update_figures.py
coreg-stats-alignment`.

### 3. Review the coregistration

Before rating anything, check that the drone ortho, the Planet chip and the
crown mask applied from the drone actually line up. `--model` is optional;
without it the viewer drops the prediction layer and is just a layer
comparator:

```bash
python scripts/deploy_planet_image_maskrcnn_interactive.py \
  /Volumes/Earth03/flower/20260915_globus_label_application_x4_coreg_4band_stretch_stats \
  --split whole
```

`--split whole` shows the entire chip rather than the 512 px training window.
Tick *Show drone overlay*, then either **Blend** it against the chip on the
opacity slider or **Swipe** it in from the left behind a hard edge; `b`
blinks the layer on and off, which is the fastest way to see a shift. Arrow
keys step through the set. The Δx/Δy readout comes from `coreg_log.json` and
is display-only — a log rebuilt by `reconstruct_coreg_log.py` reads `n/a
(reconstructed)` rather than a fabricated zero.

A chip with no `.mask.png` (any scene whose coregistration failed) simply
loses the ground-truth layer.

### 4. Vet the chips

Chips are rated `Poor` < `Fair` < `Good` on image quality — cloud, haze,
partial scene coverage, nodata. This used to be a Labelbox `Quality` radio
keyed on `data_row.external_id` = `<stem>.png`, with exports in `labels/`
(gitignored); that access is gone, so rate them locally instead:

```bash
python scripts/vet_planet_chips.py \
  /Volumes/Earth03/flower/20260915_globus_label_application_x4_coreg_4band_stretch_stats
```

A contact sheet of every chip, `1`/`2`/`3` or the P/F/G buttons to rate,
click a thumbnail for the full-resolution view with an optional cloud-mask
overlay and a note field. Ratings land in `<imagedir>/vetting.json` (or
`-o`) after every click, so the pass is resumable; *Show: unrated* narrows
the sheet to what is left.

**Phantom 2023–2024 (`20261002_phantomext_...`).** Claude pre-rated the
46 coregistered chips on 2026-10-02, writing them into that set's
`vetting.json` with notes prefixed `[Claude pre-rating]`. The pre-ratings
were calibrated by viewing a sample of the mavic ratings. They judge image
quality the same way, plus a residual-offset check on chips with unusually
large AROSICS shifts. The ratings were then reviewed in
`vet_planet_chips.py`, which moved the two partly shadowed
`20240220_1459*_24ca` chips from Good to Fair. **Final: 21 Good, 11 Fair,
14 Poor.**

### 5. Assemble the curated set

```bash
DST=/Volumes/Earth03/flower/20260915_full_label_extended_x4_coreg_4band_stretch_stats_curated

# the already-vetted 2020-2023 chips, with their coreg_log.json
rsync -a /Volumes/Earth03/flower/20260706_full_label_application_x4_coreg_4band_stretch_stats_curated/ "$DST"/

# the newly vetted globus chips, merged into the log copied above
python copy_good_planet_vetting.py \
  --vetting /Volumes/Earth03/flower/20260915_globus_label_application_x4_coreg_4band_stretch_stats/vetting.json \
  --src /Volumes/Earth03/flower/20260915_globus_label_application_x4_coreg_4band_stretch_stats \
  --dst "$DST"
```

`--vetting` takes either a `vetting.json` from `vet_planet_chips.py` or a
`*.ndjson` Labelbox export, chosen by extension, so the historical exports
still work (`--ndjson` remains as an alias).

`copy_good_planet_vetting.py` copies every sibling file sharing an accepted
stem and unions the two `coreg_log.json` files on `(scene, label)`, so the
merged log carries the provenance of both builds and re-runs are idempotent.
`--min-quality` defaults to `Good`; `--dry-run` lists without copying.

The earlier 50ha curated set was the same step with
`labels/20260706_planet_vetting.ndjson` (56 of 131 Good; 25 Fair, 50 Poor).

The mavic ratings actually live in
`20260915_globus_label_application_x4_coreg_4band_stretch_stats_rerun/vetting.json`
(70 Good, 35 Fair, 49 Poor). The `_rerun` directory has the same 370-pair /
154-chip log as the first build.

**Phantom 2023–2024 sets (`20261002`).** Two sets add the vetted C3KW2X
chips:

- a phantom-only extension of the 56 base chips, which tests same-sensor
  extra data against the mavic extension;
- an all-sources set extending the 126.

`--no-overwrite` keeps the mavic copy of the five colliding scenes in the
all-sources set, so that set's mavic half stays identical to the 126-chip
set:

```bash
F=/Volumes/Earth03/flower
PE=$F/20261002_phantomext_label_application_x4_coreg_4band_stretch_stats

DST=$F/20261002_phantom_label_extended_x4_coreg_4band_stretch_stats_curated
rsync -a $F/20260706_full_label_application_x4_coreg_4band_stretch_stats_curated/ "$DST"/
python copy_good_planet_vetting.py --vetting $PE/vetting.json --src $PE --dst "$DST" --no-overwrite

DST=$F/20261002_full_label_extended2_x4_coreg_4band_stretch_stats_curated
rsync -a $F/20260915_full_label_extended_x4_coreg_4band_stretch_stats_curated/ "$DST"/
python copy_good_planet_vetting.py --vetting $PE/vetting.json --src $PE --dst "$DST" --no-overwrite
```

Run each copy with `--dry-run` first. The second one lists the five
colliding stems as kept.

Result (2026-10-02):

| Set | Chips | `coreg_log.json` records | Left-split crown instances (`--min-instance-size 64`) |
|---|---|---|---|
| `20261002_phantom_label_extended_..._curated` | 56 + 21 = **77** | 323 + 91 = 414 | 3,258 |
| `20261002_full_label_extended2_..._curated` | 126 + 16 = **142** | 693 + 91 = 784 | 7,908 |

In the all-sources set the five colliding chips are byte-identical to
their mavic copies. Every chip in both sets crops to 512 × 512 on both
halves, and none has zero instances.

### Alternative: the programmatic filter

Where no vetting export exists, `scripts/filter_label_application.py`
applies thresholds from `config/filter_label_application.yml`
(`exact_sizes`, `min_clear_fraction: 0.10`, `min_band_corr_mean: 0.3`,
`require_coreg_ok: true`) and rewrites a filtered `coreg_log.json`:

```bash
python scripts/filter_label_application.py <src_dir> <dst_dir> config/filter_label_application.yml
```

`copy_4band_filtered.py` mirrors an RGB `_filt` selection onto the matching
4-band set.

## Training head-to-head on Gattaca2 (October 2026)

The September comparison (`figs/202609_updates/headtohead_sweep.json`)
scored two runs on the right (test) halves of the 56 base chips:

- `base`, trained on the 56 base chips: `/home/gdoran/Documents/tree-flowering/20260824_maskrcnn_out_min064`;
- `ext`, trained on the 126 chips: `.../202609_full_curated_maskrcnn_out_min064`.

October adds two runs on the `20261002` sets above, and keeps the existing
`base` and `ext` checkpoints:

- `phantom`, trained on the 56 base chips plus the C3KW2X Good chips;
- `full`, trained on the 126 chips plus those Good chips.

The two training commands were not recorded, so the settings were read
back from `params` in `epoch_001.pth` of each base checkpoint. Both
`base` and `ext` agree on all sixteen recorded settings: RGB
(`channel_kinds: [red, green, blue]`, `n_channels: 3`,
`fourth_band: none`), `use_ocm_masks: true`, `min_instance_size: 64`,
lr 1e-4 flat, batch 4, 200 epochs, no copy-paste — and
**`extra_train_dirs: []`**, so neither passed `--extra-train-dir`. The
AVA directory belongs to the separate local run
`full_curated_maskrcnn_out_min064`, not to `ext`, and no AVA chip set
exists under `/scratch/ecopro-hpc/flower`. The October runs therefore
drop that flag; their `epoch_001.pth` params match the two base runs
exactly.

Both `20261002` sets already live on `/scratch/ecopro-hpc/flower`. Stage
them, and the 56-chip test set, onto node-local disk before training:
reading a chip's 512 x 512 window off GPFS costs **540-675 ms**, against
**14-17 ms** from `/local`, a ~40x gap that leaves the GPU at 0% and the
process in `cxiWaitEventWait`. The loader touches every chip twice per
epoch (train split, then eval split), so GPFS alone accounted for ~85 s
of each 140 s epoch; from `/local` epochs took 36 s.

```bash
mkdir -p /local/gdoran_flower
for d in 20261002_phantom_label_extended_x4_coreg_4band_stretch_stats_curated \
         20261002_full_label_extended2_x4_coreg_4band_stretch_stats_curated \
         20260706_full_label_application_x4_coreg_4band_stretch_stats_curated; do
  cp -r /scratch/ecopro-hpc/flower/$d /local/gdoran_flower/
done
```

`/local` is node-local, so this ties the run to one node and the copies
should be checksummed against `/scratch` before use.

Then train. Per-epoch checkpoints stay on, because the sweep scores
twelve of them, and `--no-wandb` is needed because wandb is on by
default. Run on a GPU node inside the `flower` env; the two runs go
back to back, since a one-core allocation serialises the in-process
data loading and gains nothing from overlapping them:

```bash
L=/local/gdoran_flower
O=/home/gdoran/Documents/tree-flowering
python scripts/train_planet_image_maskrcnn.py \
  $L/20261002_phantom_label_extended_x4_coreg_4band_stretch_stats_curated \
  $O/202610_phantom_curated_maskrcnn_out_min064 \
  --ocm-masks --min-instance-size 64 --lr 1e-4 --num-epochs 200 --no-wandb
python scripts/train_planet_image_maskrcnn.py \
  $L/20261002_full_label_extended2_x4_coreg_4band_stretch_stats_curated \
  $O/202610_full2_curated_maskrcnn_out_min064 \
  --ocm-masks --min-instance-size 64 --lr 1e-4 --num-epochs 200 --no-wandb
```

On one A100 with one CPU core: 1 h 30 m for `phantom` (36 s/epoch) and
3 h 04 m for `full` (70 s/epoch), 35 G of checkpoints each.

Score all four runs on the same 56 base test halves, at the September
epochs:

```bash
CK=()
for e in 001 002 003 005 008 012 017 025 040 060 100 200; do
  CK+=(--checkpoint base_e$e=$O/20260824_maskrcnn_out_min064/epoch_$e.pth
       --checkpoint ext_e$e=$O/202609_full_curated_maskrcnn_out_min064/epoch_$e.pth
       --checkpoint phantom_e$e=$O/202610_phantom_curated_maskrcnn_out_min064/epoch_$e.pth
       --checkpoint full_e$e=$O/202610_full2_curated_maskrcnn_out_min064/epoch_$e.pth)
done
python scripts/compare_maskrcnn_runs.py \
  --chip-dir $L/20260706_full_label_application_x4_coreg_4band_stretch_stats_curated \
  "${CK[@]}" --ocm-masks --min-instance-size 64 --json-out headtohead_sweep_202610.json
```

Scoring the 48 checkpoints took 16 min. `base` and `ext` are re-scored
rather than carried over, which makes the JSON self-consistent and
checks the harness: their 144 metric values reproduce the September
sweep to a maximum absolute difference of 0, i.e. bit-identical.

Result (2026-10-03), peak over the twelve scored epochs:

| Run | Chips | Peak `map_50` | at | Peak `test_iou` | at | P/R at `map_50` peak |
|---|---|---|---|---|---|---|
| `base` | 56 | **0.1681** | e012 | **0.3076** | e017 | 0.129 / 0.320 |
| `ext` | 126 | 0.1513 | e005 | 0.2686 | e025 | 0.070 / 0.342 |
| `phantom` | 77 | 0.1562 | e005 | 0.2988 | e017 | 0.111 / 0.293 |
| `full` | 142 | 0.1637 | e008 | 0.2730 | e025 | 0.095 / 0.329 |

The C3KW2X Good chips help the extended set — `ext` (126) to `full`
(142) gains 0.0124 `map_50` and 0.0044 `test_iou` — but hurt the base
set, `base` (56) to `phantom` (77) losing 0.0119 and 0.0088. The same
chips cannot both help and hurt, so on 56 test chips a ~0.01 `map_50`
difference is within noise, and the honest reading is that none of the
three larger sets beats the 56-chip baseline on this benchmark.

Two caveats before concluding that more chips do not pay. First, the
test halves are the right halves of the 56 base chips, whose left halves
are in *every* run's training set; `base` draws its entire training
distribution from those scenes while `ext`/`full` dilute it with mavic
2024-26 scenes, so the benchmark structurally favours `base` and says
nothing about generalisation to new dates. Second, all four runs peak
between e005 and e012 on the scored grid and then collapse (`phantom`
falls from 0.1562 at e005 to 0.0475 at e200), so these are
early-stopping scores on a heavily
overfitting fit, and the 200-epoch budget is far past useful. (On the
full 200-epoch grid each run's own validation puts the best even
earlier: epoch 6 for `phantom`, epoch 4 for `full`.)

The JSON lands at `logs/headtohead_sweep_202610.json` on the cluster;
the plotters hardcode `FLOWER = Path('/Volumes/Earth03/flower')`, so
copy it to `figs/202610_updates/headtohead_sweep.json` and plot on the
Mac:

```bash
python scripts/plot_update_figures.py maskrcnn-summary \
  --label base="phantom 2020-23 only (56)" --label ext="+ mavic 2024-26 (126)" \
  --label phantom="+ phantom 2023-24 (77)" --label full="all three (142)"
python scripts/plot_headtohead_epochs.py \
  /Volumes/Earth03/flower/figs/202610_updates/headtohead_sweep.json \
  /Volumes/Earth03/flower/figs/202610_updates/maskrcnn_headtohead.png --label ...
```

## Consumers

Both loaders glob `*.mask.png` and pair it with the sibling `.tif`:

- `scripts/train_planet_image_segformer_4b.py` (`PlanetSegmentationDataset4B`)
- `scripts/train_planet_image_maskrcnn.py` (`PlanetMaskRCNNDataset`, plus the
  `.ocm.png` sidecar for `--ocm-masks` cloud-aware loss)

`get_split(..., 'left'|'right')` takes a 512 × 512 window abutting the
horizontal centre of the chip, which is how train and validation are kept
geographically disjoint within a scene.
