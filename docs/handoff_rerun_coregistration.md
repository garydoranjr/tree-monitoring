# Handoff: re-run all Planet → drone image registrations

Written 2026-10-30, updated 2026-11-01.

| step | status |
|---|---|
| 1. Classify the failures | **done** 2026-11-01: [results](#step-1-results-2026-11-01) |
| 2. Choose new settings and change the code | **done** 2026-11-01: [code changes](#step-2-code-changes-2026-11-01); [decision: use the NCC fallback](#decision-use-the-ncc-fallback-2026-11-01) |
| 3. Re-run every registration | **not started**: [procedure](#step-3-re-run-the-registrations) |
| 4. Rebuild the downstream products and experiments | **not started**: [procedure](#step-4-rebuild-downstream-products) |

## Why

AROSICS is capped at `max_shift=10` in `compute_coreg_shift` (`scripts/apply_drone_labels_coreg.py`).
That is 10 Planet pixels, or 30 m along each axis (successful shifts reach 36 m in magnitude), and it is also used by `coreg_planet_to_nearest_drone.py`.
Some Planet scenes are mis-georeferenced by more than 30 m. For those scenes AROSICS cannot find
the true match, reports failure (`ignore_errors=True`), and the pipeline falls back to:
- **no shift**, as in `build_crown_pixel_series.py`, which uses the unshifted polygons for a
  failed scene; or
- **dropping the scene**, as in the chip builds.

Either way, the scene is silently misaligned or lost.

## Evidence

The evidence comes from the guayacan example in `scripts/plot_phenology_cadence_figures.py`
(`missed-event`). It concerns scene `20230328_144849_31_241a_4band`, against the same-day phantom
local mosaic `BCI_50ha_2023_03_28_local.tif`:

| method | shift (x, y) m | result |
|---|---|---|
| `compute_coreg_shift` settings (`max_shift=10`) | – | `success: False`; stored as `coreg_ok: false` in `crown_series_20261029/coreg_all_scenes.json` |
| same settings, `max_shift=15` | (−10.4, −31.8) | `success: True` |
| brute-force NCC of Planet RGB brightness against the drone mosaic at 1 m, whole 50 ha | (−12, −33), r = 0.54 | clear peak |

The NCC method was checked on scenes where AROSICS succeeded and agrees within about 2 m:
- 20230404_154320_37_2413: NCC (+5, −6), AROSICS (+7.0, −7.3).
- 20230405_152749_56_2438: NCC (+5, −9), AROSICS (+4.4, −7.3).

## Scale

`coreg_ok` / success counts in the existing logs:

| file | ok / records |
|---|---|
| `crown_series_20261029/coreg_all_scenes.json` | 840 / 1243 |
| `202610_temporal_apply/coreg_nearest_drone.json` | 546 / 798 |
| `20260706_full_label_application_x4_coreg_4band_stretch_stats_curated/coreg_log.json` | 131 / 323 |
| `20261002_full_label_extended2_x4_coreg_4band_stretch_stats_curated/coreg_log.json` | 331 / 784 |
| `20260915_globus_label_application_x4_coreg_4band_stretch_stats_rerun/coreg_log.json` | 154 / 370 |

Other chip builds show similar rates. It is unknown how many failures are caused by the
`max_shift` cap and how many by clouds, haze or low contrast; finding that out is the first job.

## Original plan (2026-10-30)

Steps 1 and 2 are done. Steps 3 and 4 are replaced by the detailed
procedures further down.

1. **Classify the failures.** For every failed scene in `coreg_all_scenes.json`, rerun AROSICS with
   `max_shift` 15 and 20, and also run the NCC check. Count how many succeed, and look at the
   distribution of shift magnitudes. Is the tail real geolocation error, or false matches?
2. **Choose new settings.** Raise `max_shift` in `compute_coreg_shift`, perhaps with a reliability
   or NCC sanity check so a larger search window does not admit false matches. Record the reason
   in the function.
3. **Re-run every registration** with the new settings:
   - all chip builds' `coreg_log.json`, through the chip-build scripts in
     `docs/planet_training_chips_50ha.md`;
   - `coreg_nearest_drone.json` (`coreg_planet_to_nearest_drone.py`,
     `docs/planet_temporal_generalization.md`);
   - `crown_series_20261029/coreg_all_scenes.json`.
4. **Rebuild downstream products** that consume the shifts:
   - the training chip sets and their vetting/curation;
   - `crown_series_*/crown_pixel_series.npz` and the PSE-LTAE temporal CV results
     (`docs/crown_temporal_model.md`);
   - `apply_planet_maskrcnn.py` outputs;
   - any figure that reads the shift JSONs, including `plot_study_area_figure.py` and
     `plot_phenology_cadence_figures.py`. The latter has a hand-entered override for the scene
     above that can be removed once the JSON is fixed.

## Step 1 results (2026-11-01)

`scripts/diagnose_coreg_max_shift.py` reran AROSICS on every failed scene in
`crown_series_20261029/coreg_all_scenes.json`:
- 403 failed scenes, against the same ortho, with only `max_shift` changed: 20 and 30 Planet px
  (60 m and 90 m);
- the NCC check, searching ±90 m at 1 m, run on each scene;
- 149 successful scenes as controls: one per ortho group with a failure, plus every success
  above 20 m on either axis.

```bash
cd scripts; D=/Volumes/Earth03/flower
python -W ignore diagnose_coreg_max_shift.py $D/crown_series_20261029/coreg_all_scenes.json \
    $D/planet_clipped/4band $D/planet_clipped/ocm \
    $D/crown_series_20261029/coreg_max_shift_diagnosis.csv --workers 3
```

Per-scene output is in
`crown_series_20261029/coreg_max_shift_diagnosis.csv`. The cause assigned to each failure is in
`coreg_max_shift_failure_causes.csv` (same directory).

### Controls
- **Wider search.** It changes nothing for scenes that already succeed: all 149 succeed at 20 and
  30 px with the same shift, within 1e-5 m.
- **NCC agreement.** NCC agrees with AROSICS within 3 m on 134 of the 147 controls it could score.
  - The 13 disagreements are mostly cloudy (median clear fraction 0.25) and have weak NCC peaks
    (median 0.14).
  - NCC is *confident* when its peak is ≥ 0.2 and at least 2.5× the best value more than 15 m
    away. On confident controls it agrees on 93 of 95.

### Failures by cause
"Usable" is the share of the ortho that the scene covers and calls clear. It is the scene's
coverage of the ortho × `clear_fraction`.

| cause | scenes | rule |
|---|---|---|
| cap, recovered | **35** | succeeds at 30 px, and NCC agrees within 3 m |
| cap, unverified | 9 | succeeds at 30 px, NCC disagrees |
| cap, not recovered | 2 | still fails at 30 px; confident NCC beyond 30 m |
| AROSICS does not converge | 37 | still fails; confident NCC within 30 m |
| cloud or coverage | 214 | still fails; usable < 0.25 |
| no confident match | 106 | still fails; NCC not confident |
| **total** | **403** | |

- **Cap, recovered.**
  - All 35 have a shift beyond 30 m on at least one axis: 31–82 m, median 38 m. Most of the
    shift is north–south.
  - 30 of them succeed at 20 px; the other 5 need 30 px (shifts of 64–82 m).
  - They span every year from 2020 to 2026, and 15 have usable ≥ 0.75. The 20230328 scene from
    the evidence table is one of them.
  - `build_crown_pixel_series.py` currently uses these scenes **unshifted**. That puts their
    polygons 31–82 m from the crowns, more than a crown width.
- **Cap, unverified.** These are false matches from the wider window:
  - NCC disagrees, its peak is weak (≤ 0.28, ratio ≤ 3.0), and 6 of the 9 have usable < 0.25.
  - AROSICS reports shifts such as (63, 38) m or (−59, −34) m.
  - A wider `max_shift` without a check would admit them.
- **Cap, not recovered.** 20211122_145420_50 has a confident NCC shift of (51, 60) m.
  20230928_150012_89 has its NCC peak at the edge of the search (−4, −90) m, which is not
  trustworthy.
- **AROSICS does not converge.** These scenes are not limited by the cap.
  - Many are fully clear and cover the whole ortho. 20240109_145901_63 is one: clear, same-day,
    NCC peak 0.55 at (3, −11) m.
  - With `q=False`, AROSICS alternates between two candidate shifts, (−1, 0) and (0, −4) px,
    until it gives up with "No match found in the given window."
  - A wider `max_shift` does not help; an NCC fallback would.
- **Cloud or coverage.** 64 of these 214 scenes cover less than a quarter of the ortho.
  - This includes slivers whose `clear_fraction` is 1.0 because it is computed over their few
    valid pixels.
  - The rest are cloudy.

### Recommendation for step 2
- **Search range.** Raise `max_shift` to 30 px (90 m). 20 px would miss 5 of the 35.
- **Gate.** Accept a shift beyond the old 30 m limit only when NCC agrees with it within 3 m.
  - This rejects all 9 unverified matches.
  - It leaves every shift at or under 30 m, and so every current success, unchanged.
  - Gain: 35 more coregistered scenes in the crown series (840 → 875 of 1,243).
- **Separate change (optional).** Use the NCC shift as a fallback when AROSICS fails and NCC is
  confident. Replaying the rule on the diagnosis CSV, it adds 39 scenes: the 37 non-converging
  scenes, 20211122_145420_50, and one more.
- **Not checked.** The other coreg JSONs were not diagnosed. They draw on the same Planet
  archive, so similar proportions are expected.

## Step 2: code changes (2026-11-01)

**`scripts/apply_drone_labels_coreg.py`.**
- **`arosics_shift`.** This is the old `compute_coreg_shift` body. It runs AROSICS alone and
  returns `(x, y, success)`. Only `max_shift` changed: `COREG_MAX_SHIFT = 30` px (90 m), up from
  10.
- **`compute_coreg_shift`.** It now returns a dict of log fields, not a tuple:
  - Up to `NCC_CHECK_ABOVE_M` = 30 m on both axes, the AROSICS shift is used as before
    (`coreg_method: 'arosics'`).
  - Beyond that, the shift is kept only if the NCC shift agrees within `NCC_AGREE_M` = 3 m on
    both axes (`'arosics+ncc'`). Otherwise the scene fails.
  - With `ncc_fallback=True`, a scene that fails takes the NCC shift when NCC is confident
    (`'ncc'`).
- **New log fields.** Every record now also carries `coreg_method`, `arosics_x_m` and
  `arosics_y_m` (AROSICS's own shift), and `ncc_x_m`, `ncc_y_m`, `ncc_peak`, `ncc_second` and
  `ncc_clear`. The NCC fields are null where NCC was not run, which is every scene that AROSICS
  registers within 30 m.
- **`create_mask`.** It passes the scene's OCM mask to the check. So for a shift beyond 30 m,
  success can now depend on `-k`, against the assumption in
  `smoke_test_apply_drone_labels_coreg.py`. All training builds pass `-k`.
- **New CLI options.**
  - `--ncc-fallback`.
  - `--redo-failed LOG`, which builds only the `(scene, label)` pairs that LOG records as
    failed. Successful pairs need no rerun, since the wider search leaves their shift unchanged
    (all 149 controls).

**`scripts/coreg_ncc.py` (new).** This is the NCC check, moved out of the diagnosis script:
- `ncc_shift`: the shift search.
- `ncc_confident`: peak ≥ 0.2, at least 2.5× the best value more than 15 m from the peak, and
  not on the edge of the ±90 m search.
- `drone_red_1m`: the ortho's red band at 1 m, cached per ortho.

Reading the ortho takes about 45 s (phantom) or 2.5 min (mavic), once per ortho and only when a
check is needed. The match itself takes about 0.4 s. On an ortho no wider than 180 m, such as
the plot-only AVA ortho, there is no template. The check then cannot run, so shifts beyond 30 m
are rejected there.

**`scripts/coreg_planet_to_nearest_drone.py`.**
- It passes each scene's OCM mask to the check, and its records gain the new fields.
- New flags: `--ncc-fallback`, and `--redo-failed`, which reruns the scenes OUT_JSON records as
  failed and keeps the rest.
- Keep the scene-selection flags of the original run. A failed scene that is no longer selected
  loses its record.

**`copy_good_planet_vetting.py`.** In `merge_coreg_log`, the destination record still wins,
except that a failed destination record is replaced by a successful source record for the same
`(scene, label)`. Without this, a curated set's log would keep `coreg_ok: false` for a chip
added from a redo build. The rule was unit-tested, including idempotence on a second merge.

**`scripts/diagnose_coreg_max_shift.py`.** It now uses `arosics_shift` and `coreg_ncc`. Its
output is unchanged.

**Checks.**
- **Replay.** The gate and fallback were replayed on `coreg_max_shift_diagnosis.csv`:

  | | controls (149) | failures (403) |
  |---|---|---|
  | gate only | 149 `arosics`, unchanged | 35 `arosics+ncc`, 368 fail |
  | gate + `--ncc-fallback` | 149 `arosics`, unchanged | 35 `arosics+ncc`, 39 `ncc`, 329 fail |

  Fallback shifts are 2–60 m on the larger axis, median 8 m. On the 95 controls where NCC is
  confident, it agrees with AROSICS on 93. The other two miss the 3 m tolerance by at most
  0.3 m on one axis.
- **Selection.** `coreg_planet_to_nearest_drone.py --redo-failed --dry-run` on a copy of
  `coreg_all_scenes.json` selects exactly the 403 failed scenes, in 123 ortho groups.
- **Live run.** The new `compute_coreg_shift` was run on three failures and two controls, with
  and without the fallback:

  <a id="live-check-of-compute_coreg_shift"></a>

  | scene | diagnosis cause | gate only | with `--ncc-fallback` |
  |---|---|---|---|
  | 20230328_144849_31_241a | cap, recovered | `arosics+ncc` (−10.38, −31.75) | same |
  | 20241208_160233_72_2514 (mavic) | cap, unverified | fails: AROSICS (−59.4, −33.9) rejected, NCC peak 0.05 | fails |
  | 20240109_145901_63_24c4 | no convergence | fails | `ncc` (3, −11) |
  | 20230326_145322_83_24bc | control | `arosics` (−4.75, −25.20), as before | same |
  | 20240107_154142_74_2473 | control | `arosics` (7.13, −1.33), as before | same |

  On mavic orthos GDAL prints "Value 0 in the source dataset has been changed to 1" during the
  run. This is harmless.
- **End to end.** `apply_drone_labels_coreg.py` was run with
  `--redo-failed .../20260608_full_..._stretch_stats/coreg_log.json --ncc-fallback` on the
  2023-03-28 label alone, into a scratch directory.
  - It built only that date's 2 failed pairs of 6.
  - `20230328_144849_31_241a` came out `arosics+ncc` at (−10.38, −31.75).
  - `20230326_152812_41_249e` came out `ncc` at (4, 25).
  - Both have the full set of chip files.
  - The run printed empty "Error in sys.excepthook" lines at exit, with exit code 0. The
    pre-change script prints them too (5 on the same date), so they are not from this change.

## Decision: use the NCC fallback (2026-11-01)

**Every registration in step 3 runs with `--ncc-fallback`.** The user chose this on 2026-11-01.
- **Gate alone.** It recovers the 35 scenes that hit the search limit. Every shift beyond 30 m
  it accepts was found by AROSICS and confirmed by NCC.
- **What the fallback adds.** About 39 more scenes on which AROSICS fails to converge. Their
  shift is the 1 m NCC estimate, which no second method confirms.
  - On the 95 confident controls, NCC agrees with AROSICS within about 3 m.
  - In the crown series a failed scene is used **unshifted**. These scenes' shifts have a median
    of 8 m (up to 60 m), so leaving them unshifted would put their crown outlines off by several
    3 m pixels.
- **Tracing them.** `coreg_method: 'ncc'` marks every fallback scene, so they can be dropped or
  compared later. The flag is not the default in the code, so it must be passed explicitly.
  Use it in every step, so (a), (b) and (c) stay consistent.

Fix a date tag for the new outputs, e.g. `export RUN=20261103` (the day the re-run starts). The
Python snippets below read it from the environment.

## Step 3: re-run the registrations

There are three kinds of coreg output:
- **(a)** the chip builds' `coreg_log.json`, from `apply_drone_labels_coreg.py`;
- **(b)** the apply step's `202610_temporal_apply/coreg_nearest_drone.json`;
- **(c)** the crown series' `crown_series_20261029/coreg_all_scenes.json`.

(b) and (c) are both written by `coreg_planet_to_nearest_drone.py`.

Every scene AROSICS already registered keeps its shift. The 149 controls show this, and the gate
changes nothing at or under 30 m. Each step therefore redoes only the failed scenes
(`--redo-failed`), and keeps the old records next to the new ones.

### 3a. Crown series JSON, (c): Mac, about 3–4 h

```bash
D=/Volumes/Earth03/flower; OLD=$D/crown_series_20261029; NEW=$D/crown_series_$RUN
mkdir -p $NEW && cp $OLD/coreg_all_scenes.json $NEW/coreg_all_scenes.json
python -W ignore scripts/coreg_planet_to_nearest_drone.py \
    $D/planet_clipped/4band $D/planet_clipped/ocm $NEW/coreg_all_scenes.json \
    --drone-dir $D/stri/globus/RGB \
    --drone-dir $D/stri/C3KW2X/BCI_50ha_timeseries_local_alignment \
    --drone-dir $D/stri/24782016/BCI_50ha_timeseries_local_alignment \
    --exclude-within-days -1 --min-clear-fraction 0.1 --redo-failed --ncc-fallback --workers 2
```

- **Selection.** The selection flags are the original ones (`docs/crown_temporal_model.md`,
  step 1). The dry run selected 403 scenes in 123 ortho groups.
- **Run time.** The original 445 scenes took 2 h 07 min with 2 workers. Add about 45 s
  (phantom) or 2.5 min (mavic) per ortho group in which NCC runs. Each worker holds one ortho of
  up to 4 GB, so stay at 2 workers on the 16 GB Mac.
- **Resuming.** The JSON is rewritten after every group. Until the run finishes, it lacks the
  failed records of the groups not yet redone, so do not use it before then. To resume after an
  interruption, rerun the same command *without* `--redo-failed`, which picks up exactly the
  missing scenes. With the flag, it would also redo every scene that failed again in the groups
  already finished.

**Check** against the diagnosis:

```bash
python - <<'EOF'
import json, os, pandas as pd
D = '/Volumes/Earth03/flower'; RUN = os.environ['RUN']
old = pd.DataFrame(json.load(open(f'{D}/crown_series_20261029/coreg_all_scenes.json'))).set_index('scene')
new = pd.DataFrame(json.load(open(f'{D}/crown_series_{RUN}/coreg_all_scenes.json'))).set_index('scene')
cause = pd.read_csv(f'{D}/crown_series_20261029/coreg_max_shift_failure_causes.csv').set_index('scene')
assert set(old.index) == set(new.index)
ok = old.index[old.coreg_ok]
assert (new.loc[ok, ['x_shift_m', 'y_shift_m']] == old.loc[ok, ['x_shift_m', 'y_shift_m']]).all().all()
redo = new.loc[old.index[~old.coreg_ok]].join(cause['cause'])
print(pd.crosstab(redo['cause'], redo['coreg_method'].fillna('fail')))
EOF
```

Expected:
- `cap_recovered`: all 35 are `arosics+ncc`, with the shifts of `x_30`/`y_30` in the causes CSV.
- `cap_unverified`: all 9 fail.
- 39 more are `ncc`: all 37 `arosics_no_converge`, 20211122_145420_50 (`cap_not_recovered`) and
  one `cloud_or_coverage` scene.

AROSICS is deterministic, so a cause that differs from the table needs a look before going on.

### 3b. Apply JSON, (b): derived from (c), no rerun

(c) began as a copy of (b) with 445 near-flight scenes added. So the new (b) is the new (c)
restricted to (b)'s 798 scenes; rerunning them on the cluster would repeat 3a. On today's
files this restriction reproduces (b) exactly: all 798 records are equal. The consumers of
(b) use `planet_rel`, the date fields, `coreg_ok` and the shifts. Redone records carry Mac paths
in `drone_file` and `planet_file`, which nothing reads.

```bash
A=$D/202610_temporal_apply
cp $A/coreg_nearest_drone.json $A/coreg_nearest_drone_maxshift10.json
python - <<'EOF'
import json, os, sys
sys.path.insert(0, 'scripts')
from apply_drone_labels_coreg import write_coreg_log
D = '/Volumes/Earth03/flower'; RUN = os.environ['RUN']
b = json.load(open(f'{D}/202610_temporal_apply/coreg_nearest_drone_maxshift10.json'))
c = {r['scene']: r for r in json.load(open(f'{D}/crown_series_{RUN}/coreg_all_scenes.json'))}
write_coreg_log([c[r['scene']] for r in b], f'{D}/202610_temporal_apply/coreg_nearest_drone.json')
print(sum(not r['coreg_ok'] and c[r['scene']]['coreg_ok'] for r in b), 'newly coregistered')
EOF
```

Then copy both files to `/scratch/tree-monitoring/202610_temporal_apply/` on gattaca2. The cluster
copy is the one `config/temporal_cv_20261012*.sh` read (`APPLY_COREG_JSON`).

### 3c. Chip builds, (a): Mac, several hours each

One redo build per training source, each into a new directory. Use the flags of
`docs/planet_training_chips_50ha.md` (steps 1, 2 and 2b) and add `--redo-failed`:

```bash
F=/Volumes/Earth03/flower; C=$F/results/classifications
for spec in "24782016 full 20260608_full_label_application_x4_coreg_4band_stretch_stats" \
            "C3KW2X phantomext 20261002_phantomext_label_application_x4_coreg_4band_stretch_stats"; do
  set -- $spec; REL=$1; TOK=$2; OLDSET=$3
  ORTHO=$F/stri/$REL/BCI_50ha_timeseries_local_alignment
  LABELS=(); for f in $ORTHO/BCI_50ha_*_local.tif; do LABELS+=($C/$(basename "$f" .tif)_classifications.tif); done
  KMP_DUPLICATE_LIB_OK=TRUE python scripts/apply_drone_labels_coreg.py "${LABELS[@]}" \
    $ORTHO $F/planet_clipped/4band $F/${RUN}_${TOK}_label_application_x4_coreg_4band_stretch_stats_redo \
    -b 4 -r 4 -t 2 -d 2 -k $F/planet_clipped/ocm --redo-failed $F/$OLDSET/coreg_log.json --ncc-fallback
done
KMP_DUPLICATE_LIB_OK=TRUE python scripts/apply_drone_labels_coreg.py \
  $C/*M3M_aligned_global_RGB_classifications.tif $F/stri/globus/RGB $F/planet_clipped/4band \
  $F/${RUN}_globus_mkl_label_application_x4_coreg_4band_stretch_stats_redo \
  -b 4 -r 4 -t 2 -d 2 -k $F/planet_clipped/ocm \
  --redo-failed $F/20260915_globus_label_application_x4_coreg_4band_stretch_stats_rerun/coreg_log.json --ncc-fallback
```

**Failed pairs to redo:** 192 (phantom 2020–23), 45 (C3KW2X) and 216 (mavic).

**Mavic masks.**
- `results/classifications` now holds the colour-transfer mavic classifications: section 2d of
  the chips doc reapplied them from there.
- So the mavic redo build writes the `mkl` masks directly, and no `reapply_drone_labels.py` step
  is needed.
- Check one mask against a chip of
  `20261012_globus_mkl_label_application_x4_coreg_4band_stretch_stats` before vetting.

**Expected yield.** Only part of this can be estimated from the diagnosis, which covers the
failed pairs whose scenes are in the crown series:

| build | failed pairs | in the diagnosis | cap recovered | no convergence (fallback) |
|---|---|---|---|---|
| phantom 2020–23 | 192 | 61 | 9 | 7 |
| C3KW2X | 45 | 16 | 2 | 4 |
| mavic | 216 | 77 | 5 | 2 |

- 295 of the other 299 failed pairs are scenes below 10 % clear, which will stay failed.
- So expect about 29 new chips: 16 from the search limit, 13 from the fallback.
- About 40–50 % of chips have been rated Good in past vetting, so perhaps 8–15 new Good chips.

**Run time.** About 23 s per pair on 64 GB. On the 16 GB Mac the mavic build swaps, at about
95 s per pair and roughly 6 h. Run them one at a time.

## Step 4: rebuild downstream products

The steps are in priority order. 4a matters most: today the crown series uses the 74
recovered scenes *unshifted*. Their crown outlines are 31–82 m off on the 35 that hit the search
limit, and up to 60 m (median 8 m) off on the 39 fallback scenes.

### 4a. Crown series and the PSE-LTAE experiments (Mac)

Build into the **new** directory. `build_crown_pixel_series.py run` skips scenes whose
`scenes/<scene>.npz` exists (it has no `--overwrite` by default). The npz stores no shift, so a
rerun into the old directory would label the recovered scenes `coreg_ok = true` while keeping
their unshifted pixels.

```bash
python scripts/build_crown_pixel_series.py run \
    $D/planet_clipped/4band $D/planet_clipped/ocm $NEW \
    --crowns $D/stri/24784053/BCI_50ha_2022_09_29_crownmap_improved/BCI_50ha_2022_09_29_crownmap_improved.shp \
    --coreg-json $NEW/coreg_all_scenes.json --workers 6
python scripts/build_crown_pixel_series.py concat $NEW
```

Then rerun every experiment in `docs/crown_temporal_model.md` with `D=$NEW`, using the same
commands, folds (`config/temporal_folds_20261012.csv`) and seeds:
- **Temporal CV** (step 3 of that doc): 6 folds at about 23 min each on MPS, then
  `eval_crown_temporal_cv.py`.
- **By event type** and the **Mask R-CNN comparison and blend**. These come from the eval
  outputs.
- **Spatial generalization** ("Three runs on the same crowns"): `cv_spatial/west_east` and
  `cv_spatiotemporal/fold0..5` at about 15 min each, then the three evals, including
  `cv/eval_east`.

**Checks.**
- `scenes` in the new npz has about 840 + 74 = 914 `coreg_ok` scenes.
- For every scene whose coreg record is unchanged, the new `scenes/<scene>.npz` arrays equal
  the old ones (`np.array_equal` on every key). Only the recovered scenes may differ.

**Docs.** Update the procedure counts in `docs/crown_temporal_model.md` (step 1: 840 → new) and
every results table: bins and season, by event type, the Mask R-CNN comparison, and the spatial
tables. Keep the old numbers in a short "before re-registration" note, so the effect of the
fix is visible.

**Figure constants.** These scripts read the crown series by a hard-coded path. Point them at
`crown_series_$RUN` after the check:
- `plot_phenology_cadence_figures.py:52`;
- `plot_ocm_observability.py:48`;
- `plot_study_area_figure.py:53`.

`plot_ocm_observability.py` is part of the observability pipeline: keep
`docs/observability_methodology.md` (lines 30–38) and the `Snakefile` in sync, per CLAUDE.md.

### 4b. Temporal apply step (gattaca2)

`apply_planet_maskrcnn.py run` skips a scene whose `crowns/<scene>.csv` exists, and
`scripts/slurm/planet_apply.sh` passes no `--overwrite`. For the scenes that are newly
coregistered in the new (b):
1. Move their `crowns/<scene>.csv` to `<APPLY_OUT>/pre_recoreg/`.
2. Resubmit the apply job. It predicts only those scenes and writes their `pred_coreg.tif` and
   shifted crown rows.
3. Rerun `concat`.

Do this for each of the three apply outputs. All three read the same (b):

| config | `APPLY_OUT` |
|---|---|
| `config/temporal_cv_20261012.sh` | `202610_temporal_apply` (TOA) |
| `config/temporal_cv_20261012_cal.sh` | `202610_temporal_apply_cal` |
| `config/temporal_cv_20261012_cal_offjit1.sh` | `202610_temporal_apply_cal_offjit1` |

```bash
source config/temporal_cv_20261012.sh   # then _cal.sh, then _cal_offjit1.sh
python - <<'EOF'
import json, os, shutil
out = os.environ['APPLY_OUT']; a = '/scratch/tree-monitoring/202610_temporal_apply'
old = {r['scene']: r['coreg_ok'] for r in json.load(open(f'{a}/coreg_nearest_drone_maxshift10.json'))}
new = json.load(open(f'{a}/coreg_nearest_drone.json'))
os.makedirs(f'{out}/pre_recoreg', exist_ok=True)
for r in new:
    p = f"{out}/crowns/{r['scene']}.csv"
    if r['coreg_ok'] and not old[r['scene']] and os.path.exists(p):
        shutil.move(p, f'{out}/pre_recoreg/')
EOF
sbatch scripts/slurm/planet_apply.sh
python scripts/apply_planet_maskrcnn.py concat $APPLY_OUT
```

The model checkpoints do not change. The temporal CV trains on the chip list in
`config/temporal_folds_20261012.csv`, not on the chip directory. Then rsync the outputs back
to Earth03 (`docs/planet_temporal_generalization.md`, "rsync -a gattaca2:...").

**Regenerate** the products of `crown_timeseries.csv` and `scene_summary.csv` with the commands
in `docs/planet_temporal_generalization.md`:
- `plot_crown_species_comparison.py`;
- `make_crown_timeseries_html.py`;
- `compare_crown_agreement.py`;
- the per-crown viewer, `serve_crown_timeseries.py`. It reads
  `<apply_dir>/coreg_nearest_drone.json`; `202610_temporal_apply_cal/` on Earth03 has no copy of
  that file, which looks like an existing bug in the documented `--apply-dir ..._cal` command.

Update the counts in `docs/planet_temporal_generalization.md` (546 / 798 coregistered) and the
settings it quotes (`ws=(200, 200)`, `max_shift=10`, about line 267).

### 4c. Chip sets and Mask R-CNN (optional: small gain)

1. **Review the coregistration.** Open each redo set in the viewer
   (`deploy_planet_image_maskrcnn_interactive.py <set> --split whole`, chips doc step 3). Look
   hardest at the `arosics+ncc` and `ncc` chips, whose shifts are large or new.
2. **Vet.** Run `vet_planet_chips.py <set>` on each redo set (chips doc step 4). Only the new
   chips need rating.
3. **Assemble** a new curated set from the current 142-chip TOA set:

   ```bash
   DST=$F/${RUN}_full_label_extended3_mkl_x4_coreg_4band_stretch_stats_curated
   rsync -a $F/20261012_full_label_extended2_mkl_x4_coreg_4band_stretch_stats_curated/ "$DST"/
   for S in ${RUN}_full ${RUN}_globus_mkl ${RUN}_phantomext; do
     SRC=$F/${S}_label_application_x4_coreg_4band_stretch_stats_redo
     python copy_good_planet_vetting.py --vetting $SRC/vetting.json --src $SRC --dst "$DST" --no-overwrite
   done
   ```

   Run each with `--dry-run` first. The merge now replaces a failed record with its redone
   success and prints how many it replaced. The 142 existing chips stay byte-identical.
4. **Calibrated copy.** Run `build_calibrated_chips.py "$DST" ${DST/stretch_stats_curated/cal_curated}`
   (`docs/planet_radiometric_calibration.md`).
5. **Mask R-CNN spatial run.** Copy `config/spatial_20261012_cal.sh` to a new config with
   `SPATIAL_CHIPS` at the new calibrated set and new `SPATIAL_RUN`, `SPATIAL_LABEL` and
   `SPATIAL_SWEEP_JSON`. Then submit `planet_spatial_train.sh` and `planet_spatial_compare.sh`
   as in the chips doc.
   - The 56-chip benchmark is fixed by `config/spatial_test_chips_56.txt`, so the scores
     compare directly with `full_mkl_cal`.
   - Only worth running if vetting adds a meaningful number of Good chips. The ~0.01 noise on
     this benchmark is larger than a handful of chips is likely to move it.
6. **Mask R-CNN temporal CV.** Leave it as it is. The folds CSV fixes its chips, so new chips
   enter only if `make_temporal_folds.py` is rerun. That reshuffles the greedy block assignment
   and breaks comparison with every earlier temporal-CV result.

Sync any new set to `/scratch/tree-monitoring` before cluster runs. `stage_chips.sh` stages with
`rsync -a` and no `--delete`.

### 4d. Figures and remaining docs

- **`plot_phenology_cadence_figures.py`.** Remove the hand-entered override for
  `20230328_144849_31_241a_4band` (`shift_overrides`, lines 222–227 and 247–248). The new JSON
  carries the same shift. Rerun `missed-event` and `durations`.
- **`plot_study_area_figure.py`.** Rerun `candidates --select` first, since `figure` reads the
  shifts cached in `materials/selected_crown.json`. Then rerun `figure`.
- **`make_registration_figures.py shifts`.** The axis limit (`lim = 35`) and the dashed ±30 m
  "search limit" box describe the old setting.
  - Widen the axes to the new shifts, up to 82 m.
  - Draw the ±90 m search, and the ±30 m line beyond which NCC must agree.
  - Color or mark the points by `coreg_method`.
  - Rerun `blink` only if scene `20240209_155640_21_24ad` changes (it should not).
- **`plot_ocm_observability.py`.** Rerun it after the path change in 4a.
- **`plot_update_figures.py coreg-stats` / `coreg-stats-alignment`.** These read the raw builds'
  logs. Regenerate them only if the yield figures are reused, pointing them at the merged
  old + redo logs.
- **`docs/planet_training_chips_50ha.md`.**
  - The `coreg` token row (line 24) and the yield text quote `max_shift=10`. Add the new
    setting and the redo builds.
  - The `20261003` global-alignment sets and the AVA set (`docs/ava_planet_cutouts.md`) are
    analysis-only. Leave them on the old setting, and say so.
  - The AVA ortho is too small for the NCC check, so a redo there could only reproduce the old
    result.

## Scripts that call AROSICS or use these shifts

- `apply_drone_labels_coreg.py`: `compute_coreg_shift` (gated, see step 2), `arosics_shift`,
  `apply_shift_to_label`.
- `coreg_ncc.py`: the NCC check.
- `coreg_planet_to_nearest_drone.py`
- `build_crown_pixel_series.py` (`load_coreg`)
- `apply_planet_maskrcnn.py` (`crown_rows`)
- `drone_coreg.py`, `drone_coreg_align.py`, `planet_coreg.py`, `measure_chip_local_offsets.py`,
  `reconstruct_coreg_log.py`, `plot_coreg_*.py`
