# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tree phenology monitoring research (Caltech/JPL) using drone and Planet satellite imagery over the BCI 50ha forest plot. See [README.md](README.md) for full project description, environment setup, workflow commands, and code organization.

## Git workflow

After implementing any significant change, automatically create a git
commit for it. A "significant change" is one that meaningfully alters
behavior, adds or removes functionality, fixes a bug, or modifies more
than a trivial amount of code or documentation. Trivial edits (fixing
a typo, tweaking a comment, adjusting formatting) do not need their
own commit.

If it is unclear whether a change is significant enough to warrant a
commit, ask the user before committing. When committing, follow the
existing commit-message style in `git log` and stage only the files
relevant to the change (do not use `git add -A`).

Commits from this repo feed a weekly-report generator that summarizes
recent progress for the section's Confluence page. Write commit
subjects that would read usefully on a weekly report: start with a
verb, name the concrete thing that changed (script, model, dataset,
figure, config), and — when the outcome matters — state the result
or motivation in one clause. Prefer "Train SegFormer on 4-band
imagery; F1 0.82 on held-out crowns" over "update training script".
The body is included too — use it for detail, caveats, numbers,
and context that won't fit in a scannable subject line.

## Figure captions

Captions for figures in papers and methodology docs should be
self-contained so a reader can interpret the figure without the
surrounding text. Use a single compact prose block (no bullets,
headings, or sub-labels) and follow these conventions:

1. **Open with what the figure shows** — a one-clause description of
   the relationship or quantity being illustrated (e.g. "Seasonal
   variation in observability and measured solar radiation").
2. **Identify each plotted series inline** by pairing its visual
   encoding with a precise definition, in parentheses immediately
   after the series name. Include units or the exact quantity being
   plotted, not a vague label. Example: "observability (red, number
   of clear Planet images per day per year)".
3. **State any processing applied to the plotted values** — smoothing
   windows, running means, normalizations, aggregations across years,
   etc. Be specific about the window size and what is being averaged
   over (e.g. "10-day running means over years for each day of year").
4. **State the temporal or spatial coverage for each series**, and
   split them out if they differ (e.g. "observability data extending
   over 2022-2023, and the radiation for 2001-2024"). Do not leave
   the reader to guess the date range.
5. **Prefer concrete quantities over generic labels** — "proportion of
   the radiation expected on a clear day" beats "normalized
   radiation".

Reference example (for `solar_radiation_comparison.pdf`):

> Seasonal variation in observability (red, number of clear Planet
> images per day per year) and measured solar radiation as a
> proportion of the radiation expected on a clear day (black). The
> plotted data are the smoothed 10-day running means over years for
> each day of year, with the observability data extending over
> 2022-2023, and the radiation for 2001-2024.

## Documentation invariants

### Observability pipeline docs

`docs/observability_methodology.md` and the top-level `Snakefile` must
stay in sync. Any change that adds, renames, or removes a script, data
artifact, or script argument in the observability pipeline must be
reflected in **both** files: the narrative in the `.md` and the
corresponding rule (inputs, outputs, shell command) in the `Snakefile`.

When a script is not yet wired into the `Snakefile` (e.g. because its
upstream inputs are undocumented), list it in the top-level
"Reproducibility" section of `docs/observability_methodology.md` so
readers know which parts of the pipeline are not yet automated.
