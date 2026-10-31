# Handoff: re-run all Planet → drone image registrations

Written 2026-10-30. **Not started.** This is deferred to a dedicated session.

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

## Scripts that call AROSICS or use these shifts

- `apply_drone_labels_coreg.py`: `compute_coreg_shift` (`max_shift=10`), `apply_shift_to_label`.
- `coreg_planet_to_nearest_drone.py`
- `build_crown_pixel_series.py` (`load_coreg`)
- `apply_planet_maskrcnn.py` (`crown_rows`)
- `drone_coreg.py`, `drone_coreg_align.py`, `planet_coreg.py`, `measure_chip_local_offsets.py`,
  `reconstruct_coreg_log.py`, `plot_coreg_*.py`
