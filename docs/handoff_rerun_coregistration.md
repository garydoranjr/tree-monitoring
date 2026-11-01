# Handoff: re-run all Planet → drone image registrations

Written 2026-10-30. **Step 1 (classify the failures) done 2026-11-01; see
[Step 1 results](#step-1-results-2026-11-01).** Steps 2–4 are not started.

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

## Scale (not yet broken down by failure cause)

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

## Suggested plan

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
  confident. This would add the 37 non-converging scenes and 20211122_145420_50.
- **Not checked.** The other coreg JSONs were not diagnosed. They draw on the same Planet
  archive, so similar proportions are expected.

## Scripts that call AROSICS or use these shifts

- `apply_drone_labels_coreg.py`: `compute_coreg_shift` (`max_shift=10`), `apply_shift_to_label`.
- `coreg_planet_to_nearest_drone.py`
- `build_crown_pixel_series.py` (`load_coreg`)
- `apply_planet_maskrcnn.py` (`crown_rows`)
- `drone_coreg.py`, `drone_coreg_align.py`, `planet_coreg.py`, `measure_chip_local_offsets.py`,
  `reconstruct_coreg_log.py`, `plot_coreg_*.py`
