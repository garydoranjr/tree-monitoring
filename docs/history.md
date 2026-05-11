# Historical / Superseded Analyses

This document preserves context on exploratory analyses whose scripts and
figures have since been removed from the active codebase. The work is
retained here for provenance — the findings motivated subsequent design
choices in the active observability pipeline
(`docs/observability_methodology.md`).

## UDM2 Scene-Level Cloud Screening

Before the crown-level visibility classifier was developed (see Steps 2–5 of
`docs/observability_methodology.md`), an initial line of work characterized
observability at the **scene** level using the Usable Data Mask (UDM2)
products delivered with every Planet image. The UDM2 clear band encodes the
fraction of each pixel classified as cloud-free by Planet's onboard quality
pipeline.

The core artifact was `results/cloud_coverage.csv`, a two-column table of
`(File, PercentClear)` with one row per Planet RGB scene, where
`PercentClear` is the average of the UDM2 clear band over the 50 ha study
area.

### Why this approach was superseded

The scene-level UDM2 mask proved too coarse for crown-level observability
assessment: it does not capture subtler degradation such as thin haze,
atmospheric distortion, or partial cloud shadows that can render individual
tree crowns unidentifiable even when the scene is nominally classified as
clear. This limitation directly motivated the development of the learned,
crown-level visibility classifier documented in Steps 2–5 of the
observability methodology, whose output (`results/assessment.npz`) replaces
`cloud_coverage.csv` as the primary observability signal for all downstream
analyses (Steps 6–8).

### Removed scripts

- **`scripts/cloud_coverage_analysis.py`** — Generator of
  `results/cloud_coverage.csv`. Took a Planet RGB directory, matched each
  `*_rgb.tif` to its `*_udm2*.tif` sibling via the helper
  `find_matching_mask()`, averaged the first (clear) band of each mask, and
  wrote `File,PercentClear` rows to CSV.

- **`scripts/cloud_coverage_plot.py`** — Consumer of
  `results/cloud_coverage.csv`. Produced a timeline plot with one vertical
  bar per scene, shading the green portion up to that scene's clear fraction
  over a red background (the unusable fraction). Also exported the helper
  `load_data(inputfile)`, which parsed scene dates out of filenames; this
  helper was reused by the two cadence scripts below.

- **`scripts/plot_cadence_stats.py`** — Consumer of
  `results/cloud_coverage.csv` (via `load_data`). Analyzed the tradeoff
  between a clear-fraction threshold and the resulting observation cadence
  using the empirical survival function of UDM2 clear fractions. Produced a
  two-line plot of "Average Days between Observations" vs. "Clear Fraction
  Threshold" with separate curves for "Individual Crown" (threshold-weighted
  by cumulative average clear fraction) and "Full Observation" (survival
  function only). A `get_stats()` function was stubbed in but never
  implemented — the script was functionally identical to
  `plot_average_cadence.py` below. Output: `figs/quant_cadence.pdf`.

- **`scripts/plot_average_cadence.py`** — Functionally identical duplicate
  of `plot_cadence_stats.py` (same labels, same plotting logic; differed
  only in the absent `get_stats` stub). Output: `figs/avg_cadence.pdf`.

### Related removed script

- **`scripts/masked_coreg.py`** — Prototype coregistration script that
  reused `find_matching_mask()` from `cloud_coverage_analysis.py` to locate
  UDM2 masks for the reference and target Planet scenes, applied the mask
  to exclude cloudy pixels from the AROSICS `COREG` feature-matching
  window, and plotted the shifted result. This was an experiment in using
  UDM2 information to improve coregistration robustness; the production
  coregistration pipeline (`scripts/planet_coreg.py`,
  `scripts/drone_coreg.py`) does not rely on UDM2 masks. Removed together
  with the UDM2 scripts because it depended on `find_matching_mask`.

### Preserved artifacts

The following outputs remain in the repository even though their producing
scripts have been removed:

- `results/cloud_coverage.csv` — UDM2 scene-level clear fractions, kept as
  a historical data record.
- `figs/avg_cadence.pdf` — Cadence vs. clear-fraction tradeoff
  (`plot_average_cadence.py` output, May 2024).
- `figs/quant_cadence.pdf` — Cadence vs. clear-fraction tradeoff
  (`plot_cadence_stats.py` output, June 2024; visually identical to
  `avg_cadence.pdf` due to the two scripts being duplicates).
- `figs/cloud_coverage_plot.pdf` — Timeline of per-scene UDM2 clear
  fraction (`cloud_coverage_plot.py` output).
