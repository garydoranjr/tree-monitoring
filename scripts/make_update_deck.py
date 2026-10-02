#!/usr/bin/env python
"""Build a monthly TreeFlowering update deck from a rendered figure set.

The deck is assembled by cloning an existing deck in the series (for its
theme, slide master, and layouts), stripping its slides, and rebuilding from
the ``SLIDES`` spec below. Terse bullets go on the slides; the long-form
figure captions and speaker notes go in the notes pane.

Requires ``python-pptx``, which lives in the *base* conda env rather than
``flower``::

    /opt/miniconda3/bin/python scripts/make_update_deck.py

With no arguments this reproduces the 2026-09 deck at
``~/Documents/powerpoint/flower/20260929-TreeFlowering.pptx`` from the figures
in ``/Volumes/Earth03/flower/figs/202609_updates``. For a later month, point
``--figdir``/``--out``/``--date`` at the new material and edit ``SLIDES``.
"""

import argparse
import os
import re
import sys

from pptx import Presentation
from pptx.enum.text import MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Inches, Pt

# Layout indices in the series' slide master (stock Office set).
LAYOUT_TITLE = 0
LAYOUT_CONTENT = 1
LAYOUT_SECTION = 2
LAYOUT_TWO_CONTENT = 3

DEFAULT_TEMPLATE = "~/Documents/powerpoint/flower/20260818-TreeFlowering.pptx"
DEFAULT_FIGDIR = "/Volumes/Earth03/flower/figs/202609_updates"
DEFAULT_OUT = "~/Documents/powerpoint/flower/20260929-TreeFlowering.pptx"
DEFAULT_DATE = "September 29, 2026"

DECK_TITLE = ("Monitoring the phenology, distribution, and mortality of "
              "selected tropical trees from space")

# Geometry, in inches. Two figure arrangements cover almost every slide:
# near-square figures sit to the right of the bullets, wide figures sit below.
TITLE_BOX = (0.92, 0.30, 11.50, 0.75)      # one-line title
TITLE_PT = 28                              # master default (~40pt) wraps and
                                           # collides with the body below
SIDE_TEXT = (0.92, 1.30, 4.90, 5.40)       # bullets, figure-right slides
SIDE_FIG = (6.05, 1.25, 6.75, 5.50)        # figure, figure-right slides
BELOW_TEXT = (0.92, 1.15, 11.50, 2.15)     # bullets, figure-below slides
BELOW_FIG = (0.70, 3.40, 11.90, 3.50)      # figure, figure-below slides
FULL_TEXT = (0.92, 1.30, 11.50, 5.40)      # bullets only
TWO_TEXT = (0.92, 1.30, 5.67, 5.40)        # left column, two-content slides
TWO_TEXT2 = (6.75, 1.30, 5.67, 5.40)       # right column, two-content slides


# --------------------------------------------------------------------------
# Slide spec
# --------------------------------------------------------------------------
# kind:     content | title | section | two_content | figrow | figstack
# bullets:  list of (level, text); "**bold**" spans become bold runs
# figures:  paths relative to --figdir
# captions: one short label per figure (figrow/figstack only)
# notes:    speaker-notes text (figure captions from slides.md go here)

SLIDES = [
    dict(
        kind="title",
        title=DECK_TITLE,
        subtitle=None,  # filled from --date
    ),
    dict(
        kind="content",
        title="Updates",
        bullets=[
            (0, "New drone data: 96 globus 50ha mosaics, 13 whole-island orthomosaics"),
            (0, "Planet imagery and cloud masks brought up to date"),
            (0, "Drone crown classification on the HPC"),
            (0, "Drone labels transferred to Planet: 70 new training chips"),
            (0, "Result: the extended training set does not help Mask R-CNN"),
            (0, "Diagnosing why: registration vs. label semantics"),
        ],
        notes="Covers the work since the new datasets arrived: the whole-island "
              "orthomosaics and the extended (globus) 50ha record.",
    ),
    dict(
        kind="content",
        title="Summary",
        bullets=[
            (0, "Ingested two new drone datasets"),
            (1, "96 globus 50ha mosaics, 2024-03 \u2192 2026-01 (355 GiB)"),
            (1, "13 full-resolution whole-island orthomosaics, 2024-06 \u2192 2025-09 (856 GB)"),
            (0, "Planet imagery current through 2026-08-11; new scenes cloud-masked with OmniCloudMask"),
            (0, "Classified all 96 globus mosaics for flowering / deciduous crowns on the HPC, "
                "using a new unified Slurm pipeline"),
            (0, "Transferred the drone labels onto Planet: 70 new \u201cGood\u201d training chips (2024\u201326). "
                "The 50ha training set grows from 56 to 126 chips"),
            (0, "**Result: the extended training set does not improve Mask R-CNN.** Registration in the "
                "new chips is not worse than in the old ones. The better lead is the labels themselves: "
                "on globus mosaics the flowering classifier fires about 3.5\u00d7 as often as on the STRI record"),
        ],
        notes="The data engineering is done and the new record is usable end to end. "
              "The disappointing part is the model result, and the rest of the talk "
              "works through why.",
    ),
    dict(
        kind="content",
        title="New Data and How It Fits Together",
        layout="below",
        bullets=[
            (0, "**Globus 50ha record:** 96 DJI M3M flights, 2024-03-06 \u2192 2026-01-20. "
                "4-band uint16, globally aligned. 355 GiB over HTTPS in 5.6 h, byte-exact"),
            (0, "**Whole-island orthomosaics:** 13 dates, 2024-06-11 \u2192 2025-09-15, "
                "each about 150k \u00d7 139k px. 856 GB from Drive in 37 min"),
            (0, "**Gap:** no 50ha mosaic between 2023-10-24 (last STRI) and 2024-03-06 (first globus)"),
            (0, "**Crown outlines:** dated outlines stop at 2024-03-18; the 94 later globus dates use "
                "the static 2022-09-29 crown map (2,280 crowns)"),
        ],
        figures=["coverage.png"],
        notes="Temporal coverage of the data feeding the 50ha training set. The top "
              "panel shows the number of Planet scenes intersecting the 50ha plot per "
              "month. Blue bars are scenes available before this month (2,051); red "
              "bars are the 95 scenes added in the 2026-09-11 ingest. The black line "
              "is the number of whole-island Planet scenes per month that are at least "
              "50% clear according to OmniCloudMask. Planet coverage runs from 2020-01 "
              "to 2026-08-11. The bottom panel shows the acquisition date of each drone "
              "mosaic as a tick: the 90-date STRI 50ha locally aligned series "
              "(2018-04-04 to 2023-10-24); the 96-date globus M3M globally aligned "
              "series (2024-03-06 to 2026-01-20); the 13 whole-island orthomosaics "
              "(2024-06-11 to 2025-09-15); and the Planet acquisition dates of the 56 "
              "existing and 70 new training chips. The bottom bar marks the period "
              "covered by dated crown outlines (2018-04-04 to 2024-03-18). The hatched "
              "bar marks the globus period, where the static 2022-09-29 crown map was "
              "used instead. Grey shading marks the 2023-10-24 to 2024-03-06 gap with "
              "no 50ha mosaic.\n\n"
              "The two drone records are disjoint: no globus date appears in the STRI "
              "set. Only two globus dates, 2024-03-06 and 2024-03-18, have dated crown "
              "outlines.",
    ),
    dict(
        kind="content",
        title="Planet Imagery Through the Present",
        bullets=[
            (0, "Ordered and ingested 145 new PlanetScope scenes over BCI (orders to 2026-08-11); "
                "95 of them intersect the 50ha plot"),
            (0, "Holdings are now 3,184 island scenes and 2,146 50ha scenes "
                "(2020-01-02 \u2192 2026-08-11). Nothing is missing at Planet"),
            (0, "Fixed a silent failure: three truncated `_4band.tif` clips had passed a "
                "\u201cfile exists and is non-empty\u201d check for two months"),
            (1, "Outputs are now validated by opening the raster before a file is skipped"),
        ],
        notes="The red bars on the coverage figure (previous slide) are these 95 new "
              "50ha scenes.",
    ),
    dict(
        kind="content",
        title="OmniCloudMask Cloud Masks for the New Scenes",
        bullets=[
            (0, "Ran OmniCloudMask on the new scenes"),
            (1, "95 new 50ha masks, in 1 h 45 min"),
            (1, "147 new whole-island masks, in 7 h 37 min"),
            (1, "23 failures, all tiles smaller than 32 px, which OCM cannot process"),
            (0, "Recomputed island-wide clear fraction for 3,166 scenes "
                "(feeds observability and flight-coverage analyses)"),
            (0, "Masks drive chip selection, and a crown touching a non-clear pixel is "
                "dropped from the labels"),
        ],
        figures=["ocm_example.png"],
        notes="OmniCloudMask classification of two new 2026 PlanetScope scenes over the "
              "50ha plot. The left column shows the scene RGB (2\u201398% stretch). The "
              "right column shows the same scene with thick cloud (white), thin cloud "
              "(orange) and cloud shadow (purple) overlaid, and the fraction of valid "
              "pixels classified clear in the title. OmniCloudMask runs at 10 m and is "
              "upsampled nearest-neighbour to the 3 m Planet grid, so mask edges are "
              "blocky at about 10 m. The scenes are 20260517_163455_99_2518 "
              "(2026-05-17, 61% clear) and 20260618_155047_98_2555 (2026-06-18, 50% "
              "clear).",
    ),
    dict(
        kind="content",
        title="Drone Crown Classification on the HPC",
        bullets=[
            (0, "One config-driven Slurm pipeline, `run_crown_pipeline.sh`, replaces six "
                "near-duplicate array scripts: classify \u2192 merge \u2192 concat as dependent "
                "arrays, restartable"),
            (0, "Ran it on all 96 globus mosaics"),
            (1, "about 17 min per mosaic (about 2.2 crowns/s)"),
            (1, "merge step 7.7\u00d7 faster after a fix, numerically identical output"),
            (0, "Globus mosaics are uint16; the SegFormer models were trained on 8-bit STRI "
                "mosaics. A fixed per-band gain/offset fitted on p2/p98 over about 63 M pixels "
                "cuts histogram distance from 0.146 to 0.033"),
            (0, "The 94 post-2024-03-18 dates use the static 2022-09-29 crown map to place "
                "the classification windows"),
        ],
        figures=["classification_example.png"],
        notes="HPC SegFormer crown classification of one globus 50ha mosaic from "
              "2025-06-24, the globus date with the most crowns classified as flowering. "
              "The panels are the RGB mosaic (uint16 converted to uint8 with the fixed "
              "per-band gain/offset used for classification), the per-pixel flowering "
              "probability, and the per-pixel deciduous probability. Crown outlines are "
              "the static 2022-09-29 crown map (2,280 crowns). The top row shows the "
              "whole plot read at 1/16 resolution (about 0.75 m pixels). The bottom row "
              "is a 120 m zoom, at 1/2 resolution (about 9 cm), centred on the crown "
              "with the highest mean flowering probability. The green box marks the zoom "
              "on the top row.\n\n"
              "The classifier runs per pixel inside a window around each crown polygon, "
              "and the polygon is not used to clip the output. Stale 2022 polygons "
              "therefore change *where* the model looks, and can miss new or "
              "much-changed crowns. They do not shift the labels off the crowns that "
              "are classified.",
    ),
    dict(
        kind="content",
        title="Flag: Flowering Rate Runs 3.5\u00d7 Higher on Globus",
        layout="below",
        bullets=[
            (0, "The deciduous seasonal cycle continues across the gap"),
            (0, "The globus **flowering** rate sits well above the STRI record all year: "
                "a median of 1.8% of crowns vs 0.5%, never below 0.6%"),
            (0, "That points to a sensor or radiometry shift in the flowering classifier, "
                "not a real change in phenology"),
        ],
        figures=["classification_timeseries.png"],
        notes="Share of the 50ha plot's crowns classified as flowering (top, pink) or "
              "deciduous (bottom, brown) at each drone mosaic date. A crown counts when "
              "the mean classifier probability over its 2022-09-29 crown-map polygon "
              "exceeds 0.5; 2,279 of the 2,280 crowns are large enough to score. "
              "Probabilities were averaged from the classification rasters read at 1/16 "
              "resolution (about 0.75 m). Circles are the 90 STRI locally aligned "
              "mosaics (2018-04-04 to 2023-10-24), classified with dated crown "
              "outlines. Squares are the 96 globus M3M mosaics (2024-03-06 to "
              "2026-01-20), classified on the HPC with the static 2022-09-29 crown map "
              "and the fixed uint16 \u2192 uint8 radiometric mapping. Dashed lines are the "
              "median for each series. Grey shading marks the 2023-10-24 to 2024-03-06 "
              "gap with no 50ha mosaic.",
    ),
    dict(
        kind="content",
        title="Spot-Check: What Does the Classifier Call \u201cFlowering\u201d?",
        bullets=[
            (0, "Sampled 32 globus crown-dates with mean P(flowering) > 0.5 over 16 dates, "
                "plus 16 STRI crown-dates as a reference, and checked them against the drone RGB"),
            (0, "**True positives exist and look right:** lilac, cream/tan, white and pink "
                "flowering crowns (e.g. 2025-06-17 crown 854, 2024-06-04 crown 1523, "
                "2026-01-20 crown 1881)"),
            (0, "**Many are clear false positives**, of two kinds"),
            (1, "bright yellow-green crowns, probably new-leaf flush, sometimes with bare branches"),
            (1, "plain green or shaded crowns with no visible flowers"),
            (0, "Roughly half the globus positives show no flowers. STRI positives are more "
                "often visibly flowering, though several are ambiguous at that resolution"),
            (0, "Fits a radiometric domain shift: under the fixed uint16 \u2192 uint8 mapping the "
                "M3M greens come out paler and yellower than the STRI training imagery"),
            (0, "The 2022 crown polygons also visibly misfit some globus crowns (e.g. 2025-01-28 "
                "crown 164, 2025-06-17 crown 605)"),
        ],
        notes="This is a visual spot-check, not a blind labelling; the proportions are "
              "rough. A proper estimate needs a random sample rated blind at full "
              "resolution, and the chip vetting tool could be adapted for that.\n\n"
              "The labels are per-pixel, so polygon misfit affects which crowns get "
              "scored, not where the label pixels sit.",
    ),
    dict(
        kind="figstack",
        title="Globus Crowns Labelled Flowering, With and Without Flowers",
        figures=[
            "spotcheck_flowering/zoom_globus_likely_tp.png",
            "spotcheck_flowering/zoom_globus_suspect_fp.png",
        ],
        captions=[
            "Visible flowers (lilac, cream, pink, white) \u2014 likely true positives",
            "No visible flowers \u2014 suspected false positives",
        ],
        notes="Globus drone crowns labelled flowering by the HPC SegFormer classifier "
              "(mean P(flowering) over the crown polygon > 0.5, shown in each title "
              "with the flight date and crown ID), displayed at native mosaic "
              "resolution (about 4.7 cm) after the fixed per-band uint16 \u2192 uint8 "
              "mapping used for classification. Cyan outlines are the static 2022-09-29 "
              "crown map. The top figure shows crowns with visible flowers (lilac, "
              "cream, pink, white). The bottom figure shows crowns with no visible "
              "flowers that were nonetheless labelled flowering.",
    ),
    dict(
        kind="content",
        title="Drone Labels \u2192 Planet Training Chips (2024\u201326)",
        layout="below",
        bullets=[
            (0, "Paired each globus flight with Planet scenes within \u00b12 days: 370 pairs over "
                "96 flight dates (7 dates had no scene)"),
            (0, "One AROSICS shift per scene: 154 coregistered (41.6%, against 40.6% for the "
                "2020\u201323 build). Median applied shift 4.3 m (about 1.4 Planet pixels), "
                "against 8.4 m for 2020\u201323"),
            (0, "Rated the chips locally with a new vetting tool that replaces Labelbox: "
                "70 Good, 35 Fair, 49 Poor. The Good chips join the 56 existing ones \u2192 126 chips"),
            (0, "Labels look statistically like the old ones: median crown fraction 4.07% vs "
                "4.15%, and 14,794 crown instances in the new chips"),
        ],
        figures=["coreg_stats.png"],
        notes="Yield and shift magnitude of the drone-label \u2192 Planet-chip transfer for "
              "the 2020\u201323 build (blue, STRI locally aligned mosaics) and the 2024\u201326 "
              "build (red, globus globally aligned mosaics). Left: number of "
              "drone\u2013Planet pairs within \u00b12 days, the number successfully coregistered "
              "by AROSICS, and the number rated Good in manual vetting, with the "
              "percentage of pairs at each stage. The 2020\u201323 vetting was done in "
              "Labelbox; the 2024\u201326 vetting used vet_planet_chips.py. Right: "
              "distribution of the magnitude of the single rigid AROSICS shift applied "
              "per coregistered scene (density; 1.5 m bins). The dotted line marks one "
              "3 m Planet pixel.\n\n"
              "Also fixed a crash that had discarded the coregistration log after a "
              "2.9 h build, and added a log reconstruction tool.",
    ),
    dict(
        kind="content",
        title="Result: The Extended Training Set Does Not Help Mask R-CNN",
        bullets=[
            (0, "Both models scored on the same 56 2020\u201323 test chips (right halves), "
                "because the two runs\u2019 own test splits differ"),
            (0, "Base (56 chips) peaks at **mAP@50 0.168** (epoch 12); extended (126 chips) "
                "peaks at **0.151** (epoch 5). Binary IoU peaks at 0.308 vs 0.269"),
            (0, "The extended set is ahead only for epochs 1\u20133 (+0.075, +0.054, +0.032). "
                "It is behind from epoch 8 onward, by as much as \u22120.068 at epoch 12"),
            (0, "Both runs collapse late: recall is traded for precision "
                "(recall 0.32 \u2192 0.07, precision 0.13 \u2192 0.60), finishing at about a third "
                "of peak mAP"),
            (0, "Training now keeps the best checkpoint and supports a cosine LR schedule"),
        ],
        figures=["maskrcnn_summary.png"],
        notes="Mask R-CNN mask mAP@50 on the test (right-half) crops of the 56 2020\u201323 "
              "training chips, against training epoch (log scale). The two curves are "
              "models trained on the 56 2020\u201323 chips (blue) and on those plus the 70 "
              "new 2024\u201326 globus chips (red, 126 chips). Twelve epochs between 1 and "
              "200 were scored per run with OCM cloud masks and a 64-pixel minimum "
              "instance size. Stars mark each run's peak. Shading shows which run is "
              "ahead at each epoch: red where the extended set is better, blue where it "
              "is worse.\n\n"
              "Early epochs are the only place the extra data helps. That pattern fits "
              "noisier or differently distributed labels, which regularize early and "
              "hurt once the model fits them closely.",
    ),
    dict(
        kind="content",
        title="Is Drone\u2194Planet Misalignment the Cause?",
        bullets=[
            (0, "**Hypothesis:** the globus mosaics are globally (rigidly) aligned, the STRI "
                "series locally (warped) aligned. One AROSICS shift per scene cannot remove "
                "non-linear drone\u2194Planet warping, so the new labels could be offset from the "
                "crowns in the Planet image"),
            (0, "**Test:**"),
            (1, "phase-correlate each chip\u2019s Planet image against its already-shifted drone "
                "sidecar in 96 m windows"),
            (1, "keep only clear, well-matched windows (NCC \u2265 0.3)"),
            (1, "measure how far each window still has to move"),
            (1, "the metric recovers injected 1.5\u20133 m shifts"),
        ],
        notes="Cross-sensor phase correlation against 3 m data is noisy. Read the "
              "numbers as a comparison between the two sets, not as absolute accuracy.",
    ),
    dict(
        kind="content",
        title="Result: The Globus Chips Are Not Worse",
        layout="below",
        bullets=[
            (0, "Median window offset 0.71 m vs 0.92 m for 2020\u201323; windows off by more than "
                "3 m: 1.9% vs 7.9%; within-chip spread (the non-rigid part): 0.53 m vs 0.79 m"),
            (0, "Same holds with 48 m windows: 0.54 vs 0.62 m median, 4.1% vs 8.8% above 3 m"),
            (0, "**Conclusion:** at scales of about 50\u2013100 m and above, registration error does "
                "not explain the regression. Crown-scale warps smaller than the window are "
                "not measured"),
        ],
        figures=["residual_offsets.png"],
        notes="Residual drone-to-Planet offset remaining in the training chips after the "
              "single per-scene AROSICS shift. It compares the 56 curated 2020\u201323 chips "
              "(blue, STRI locally aligned mosaics) with the 70 Good 2024\u201326 chips (red, "
              "globus globally aligned mosaics). Each chip was divided into 128 \u00d7 128 "
              "pixel windows (96 m at 0.75 m). The Planet chip was phase-correlated "
              "against the shifted drone sidecar downsampled to the chip grid, after a "
              "1.5-pixel Gaussian blur. Windows were kept if they were at least 90% "
              "clear and inside the drone footprint, had a post-shift normalized "
              "cross-correlation of at least 0.3, and needed a shift under 20 m. Left: "
              "distribution of window offset magnitudes, with medians and window counts "
              "in the legend. Centre: per-chip non-rigid spread, the median distance of "
              "the window offset vectors from the chip's median vector (boxes are "
              "interquartile ranges, dots are chips). Right: the window offset vectors, "
              "magnified 8\u00d7, on the chip with the largest spread.\n\n"
              "Both sets have a real tail of badly misregistered windows. The worst ones "
              "are shown next.",
    ),
    dict(
        kind="figrow",
        title="Examples of Residual Misalignment: 2020\u201323 Chips",
        figures=[
            "gifs/blink_1_20230326_145322_83_24bc_zoom.gif",
            "gifs/blink_2_20230410_144554_03_2460_zoom.gif",
            "gifs/blink_4_20230404_154320_37_2413_zoom.gif",
        ],
        captions=[
            "Planet 2023-03-26 / drone 2023-03-28\n12.4 m offset",
            "Planet 2023-04-10 / drone 2023-04-11\n10.3 m offset",
            "Planet 2023-04-04 / drone 2023-04-04\n7.5 m offset",
        ],
        notes="Blink comparison of a Planet training chip (Planet frame) and the drone "
              "orthomosaic after the single AROSICS shift (drone frame). The crown-label "
              "outlines (yellow) are drawn at the same position in both frames. The "
              "labels come from per-pixel drone classification, so they fit the drone "
              "crowns, and any displacement of the matching crowns in the Planet frame "
              "is label misregistration. Each zoom covers 192 \u00d7 192 m around the window "
              "with the largest measured residual offset in that chip. The scale bar is "
              "50 m.\n\n"
              "These are the worst well-matched window in the worst chips of the "
              "2020\u201323 set. Bright (flowering or leafless) crowns appear several metres "
              "away from their outlines in the Planet frame.",
    ),
    dict(
        kind="figrow",
        title="Examples of Residual Misalignment: 2024\u201326 Chips",
        figures=[
            "gifs/blink_3_20250315_161618_73_24fe_zoom.gif",
            "gifs/blink_5_20240304_155459_24_24f6_zoom.gif",
            "gifs/blink_6_20250909_161501_12_253d_zoom.gif",
        ],
        captions=[
            "Planet 2025-03-15 / drone 2025-03-17\n8.6 m offset (Planet frame partly hazy)",
            "Planet 2024-03-04 / drone 2024-03-06\n5.6 m offset",
            "Planet 2025-09-09 / drone 2025-09-11\n4.1 m offset",
        ],
        notes="Same construction as the previous slide, for the worst chips of the "
              "2024\u201326 globus set. Both sets have badly misregistered windows, and the "
              "2020\u201323 set has more.",
    ),
    dict(
        kind="two_content",
        title="Interpretation",
        bullets=[
            (0, "**What we know**"),
            (1, "The extra 70 chips help only in the first 3 epochs and hurt after"),
            (1, "Label geometry in the new chips is not worse than in the old ones"),
            (1, "Class balance (crown fraction) is unchanged"),
        ],
        bullets2=[
            (0, "**Candidate explanations (untested)**"),
            (1, "**Label semantics, not geometry \u2014 the leading candidate.** Globus labels "
                "come from SegFormer models trained on 8-bit STRI mosaics, applied to a "
                "different sensor. Flowering rate runs about 3.5\u00d7 the STRI baseline, and "
                "the spot-check shows yellow-green flush and plain green crowns among the "
                "positives"),
            (1, "**Train/test shift.** The only test set is 2020\u201323 chips"),
            (1, "**Coverage from stale polygons.** New or much-changed crowns far from any "
                "2022 polygon are never classified and read as background"),
        ],
        notes="The early-epoch benefit followed by later harm fits noisy labels. The "
              "false-positive rate has not been measured yet.",
    ),
    dict(
        kind="content",
        title="Next Steps",
        bullets=[
            (0, "Spot-check globus labels against the imagery (the vetting tool and the blink "
                "viewer support this)"),
            (0, "Score both models on the right halves of the new chips too"),
            (0, "Try filtering chips by residual offset or classifier confidence"),
            (0, "Use best-checkpoint selection for all future comparisons"),
            (0, "Start on the whole-island data"),
        ],
    ),
    # ---------------- backup ----------------
    dict(kind="section", title="Backup", backup=True),
    dict(
        kind="figrow",
        title="Mask R-CNN Head-to-Head: All Metrics",
        figures=["maskrcnn_headtohead.png"],
        notes="Full metric sweep behind the summary curve: mask mAP@50, binary IoU, "
              "precision and recall against training epoch for the base (56-chip) and "
              "extended (126-chip) runs.",
        backup=True,
    ),
    dict(
        kind="figrow",
        title="Residual Offsets with 48 m Windows",
        figures=["residual_offsets_w64.png"],
        notes="As the residual-offset figure, but with 64 \u00d7 64 pixel (48 m) windows: "
              "0.54 vs 0.62 m median, and 4.1% vs 8.8% of windows above 3 m. The "
              "comparison between the two sets is unchanged.",
        backup=True,
    ),
    # Static equivalents of the blink GIFs, one per slide so the triptych panels
    # stay legible in a PDF export. Dates and offsets go in the title.
    *[
        dict(
            kind="figrow",
            title=f"Blink Pair, {tag}: Planet {pdate} / Drone {ddate}, {off}",
            figures=[f"gifs/{stem}_pair.png"],
            notes="Planet frame, drone frame after the single AROSICS shift, and the "
                  "red/cyan overlay side by side. Crown-label outlines (yellow) are "
                  "drawn at the same position in both frames; displacement of the "
                  "matching crowns in the Planet frame is label misregistration. "
                  "Static equivalent of the blink GIF on the corresponding main slide.",
            backup=True,
        )
        for tag, stem, pdate, ddate, off in [
            ("2020\u201323", "blink_1_20230326_145322_83_24bc", "2023-03-26", "2023-03-28", "12.4 m"),
            ("2020\u201323", "blink_2_20230410_144554_03_2460", "2023-04-10", "2023-04-11", "10.3 m"),
            ("2020\u201323", "blink_4_20230404_154320_37_2413", "2023-04-04", "2023-04-04", "7.5 m"),
            ("2024\u201326", "blink_3_20250315_161618_73_24fe", "2025-03-15", "2025-03-17", "8.6 m"),
            ("2024\u201326", "blink_5_20240304_155459_24_24f6", "2024-03-04", "2024-03-06", "5.6 m"),
            ("2024\u201326", "blink_6_20250909_161501_12_253d", "2025-09-09", "2025-09-11", "4.1 m"),
        ]
    ],
    dict(
        kind="figrow",
        title="Flowering Spot-Check Contact Sheets: Globus",
        figures=[
            "spotcheck_flowering/globus_flowering_1.png",
            "spotcheck_flowering/globus_flowering_2.png",
        ],
        captions=["Globus positives, sheet 1", "Globus positives, sheet 2"],
        notes="Contact sheets of the 32 sampled globus crown-dates with mean "
              "P(flowering) > 0.5, at reduced resolution.",
        backup=True,
    ),
    dict(
        kind="figrow",
        title="Flowering Spot-Check Contact Sheet: STRI Reference",
        figures=["spotcheck_flowering/stri_flowering_reference.png"],
        notes="Contact sheet of the 16 STRI crown-dates sampled as a reference. These "
              "positives are more often visibly flowering, though several are ambiguous "
              "at that resolution.",
        backup=True,
    ),
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def clear_slides(prs):
    """Remove every slide, keeping the master, layouts, and theme."""
    sldIdLst = prs.slides._sldIdLst
    for sldId in list(sldIdLst):
        prs.part.drop_rel(sldId.rId)
        sldIdLst.remove(sldId)


def add_slide(prs, layout_idx):
    return prs.slides.add_slide(prs.slide_master.slide_layouts[layout_idx])


def drop_unused_placeholders(slide, keep=()):
    """Delete empty placeholders so no "Click to add text" prompt is left."""
    for ph in list(slide.placeholders):
        if ph in keep:
            continue
        if ph.has_text_frame and not ph.text_frame.text.strip():
            ph._element.getparent().remove(ph._element)


def set_title(slide, text, rect=None, size=None):
    title = slide.shapes.title
    frame = title.text_frame
    frame.word_wrap = True
    # "shrink text on overflow", so an over-long title degrades gracefully
    frame.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    frame.text = text
    if size is not None:
        # shrink rather than wrap into the body below
        fitted = fit_size([(0, text)], rect or TITLE_BOX, size, floor=20)
        frame.paragraphs[0].runs[0].font.size = Pt(fitted)
    if rect is not None:
        place(title, rect)
    return title


MARKUP = re.compile(r"\*\*(.+?)\*\*|`(.+?)`", re.S)


def _add_runs(para, text):
    """Add runs to `para`, rendering **bold** and `monospace` spans."""
    pos = 0
    for match in MARKUP.finditer(text):
        if match.start() > pos:
            para.add_run().text = text[pos:match.start()]
        run = para.add_run()
        if match.group(1) is not None:
            run.text = match.group(1)
            run.font.bold = True
        else:
            run.text = match.group(2)
            run.font.name = "Consolas"
        pos = match.end()
    if pos < len(text):
        para.add_run().text = text[pos:]


def estimate_height(items, width, size):
    """Rough rendered height, in inches, of `items` in a box `width` wide.

    python-pptx cannot measure text, so this approximates Aptos at an average
    glyph width of 0.0072 in per point and a line height of 1.22 em. Only used
    to pick a font size that will not overflow, so it errs on the wide side.
    """
    total = 0.0
    for level, text in items:
        pts = size if level == 0 else size - 2
        usable = width - 0.20 - 0.35 * level
        per_line = max(1, int(usable / (0.0072 * pts)))
        plain = re.sub(r"\*\*|`", "", text)
        lines = max(1, -(-len(plain) // per_line))
        total += lines * pts * 1.22 / 72
    return total


def fit_size(items, rect, start, floor=12):
    """Largest size <= `start` whose estimated height fits `rect`."""
    _, _, width, height = rect
    size = start
    while size > floor and estimate_height(items, width, size) > height:
        size -= 1
    return size


def fill_bullets(frame, items, size):
    frame.word_wrap = True
    for i, (level, text) in enumerate(items):
        para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        para.level = level
        _add_runs(para, text)
        for run in para.runs:
            run.font.size = Pt(size if level == 0 else size - 2)


def place(shape, rect):
    """Set all four geometry values.

    Setting only some of them on a placeholder with no explicit xfrm makes
    python-pptx emit x=0/cx=0 for the ones left alone.
    """
    left, top, width, height = rect
    shape.left, shape.top = Inches(left), Inches(top)
    shape.width, shape.height = Inches(width), Inches(height)


def add_picture_fit(slide, path, rect):
    """Place `path` scaled to fit `rect`, centred. Returns the placed rect."""
    left, top, width, height = rect
    pic = slide.shapes.add_picture(path, Inches(left), Inches(top))
    scale = min(Inches(width) / pic.width, Inches(height) / pic.height)
    pic.width, pic.height = int(pic.width * scale), int(pic.height * scale)
    pic.left = Inches(left) + (Inches(width) - pic.width) // 2
    pic.top = Inches(top) + (Inches(height) - pic.height) // 2
    return (pic.left / 914400, pic.top / 914400,
            pic.width / 914400, pic.height / 914400)


def add_caption(slide, text, rect, size=11):
    left, top, width, height = rect
    box = slide.shapes.add_textbox(Inches(left), Inches(top),
                                   Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    for i, line in enumerate(text.split("\n")):
        para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        para.alignment = PP_ALIGN.CENTER
        run = para.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.italic = True
    return box


def set_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# --------------------------------------------------------------------------
# Slide builders
# --------------------------------------------------------------------------
def build_title(prs, spec, date):
    slide = add_slide(prs, LAYOUT_TITLE)
    title = slide.shapes.title
    title.text_frame.word_wrap = True
    # normAutofit = "shrink text on overflow", as the rest of the series uses
    title.text_frame.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    title.text_frame.text = spec["title"]
    title.text_frame.paragraphs[0].runs[0].font.size = Pt(50)
    slide.placeholders[1].text_frame.text = spec.get("subtitle") or date
    return slide


def build_section(prs, spec):
    slide = add_slide(prs, LAYOUT_SECTION)
    set_title(slide, spec["title"])
    drop_unused_placeholders(slide, keep=(slide.shapes.title,))
    return slide


def build_content(prs, spec):
    """Bullets, optionally with one figure to the right or below."""
    slide = add_slide(prs, LAYOUT_CONTENT)
    set_title(slide, spec["title"], TITLE_BOX, TITLE_PT)
    figures = spec.get("figures", [])
    if not figures:
        text_rect, fig_rect, size = FULL_TEXT, None, 18
    elif spec.get("layout") == "below":
        text_rect, fig_rect, size = BELOW_TEXT, BELOW_FIG, 16
    else:
        text_rect, fig_rect, size = SIDE_TEXT, SIDE_FIG, 16

    body = slide.placeholders[1]
    place(body, text_rect)
    fill_bullets(body.text_frame, spec["bullets"],
                 fit_size(spec["bullets"], text_rect, size))

    if fig_rect is not None:
        add_picture_fit(slide, spec["_figpaths"][0], fig_rect)
    drop_unused_placeholders(slide, keep=(slide.shapes.title, body))
    return slide


def build_two_content(prs, spec):
    slide = add_slide(prs, LAYOUT_TWO_CONTENT)
    set_title(slide, spec["title"], TITLE_BOX, TITLE_PT)
    left, right = slide.placeholders[1], slide.placeholders[2]
    for ph, rect, items in ((left, TWO_TEXT, spec["bullets"]),
                            (right, TWO_TEXT2, spec["bullets2"])):
        place(ph, rect)
        fill_bullets(ph.text_frame, items, fit_size(items, rect, 16))
    drop_unused_placeholders(slide, keep=(slide.shapes.title, left, right))
    return slide


def build_figrow(prs, spec):
    """One to three figures side by side (or stacked, if spec['stack'])."""
    slide = add_slide(prs, LAYOUT_CONTENT)
    set_title(slide, spec["title"], TITLE_BOX, TITLE_PT)
    paths = spec["_figpaths"]
    captions = spec.get("captions") or [None] * len(paths)
    n = len(paths)
    top, avail_h = 1.15, 5.45
    cap_h = 0.55

    if spec.get("stack") or n == 1:
        cell_h = avail_h / n
        for path, caption in zip(paths, captions):
            fig_h = cell_h - (cap_h if caption else 0.0)
            rect = add_picture_fit(slide, path, (0.60, top, 12.15, fig_h))
            if caption:
                add_caption(slide, caption,
                            (rect[0], rect[1] + rect[3] + 0.04, rect[2], cap_h - 0.08))
            top += cell_h
    else:
        gap = 0.25
        cell_w = (12.15 - gap * (n - 1)) / n
        fig_h = avail_h - cap_h
        left = 0.60
        for path, caption in zip(paths, captions):
            rect = add_picture_fit(slide, path, (left, top, cell_w, fig_h))
            if caption:
                add_caption(slide, caption,
                            (left, rect[1] + rect[3] + 0.06, cell_w, cap_h))
            left += cell_w + gap

    drop_unused_placeholders(slide, keep=(slide.shapes.title,))
    return slide


BUILDERS = {
    "content": build_content,
    "section": build_section,
    "two_content": build_two_content,
    "figrow": build_figrow,
    "figstack": lambda prs, spec: build_figrow(prs, dict(spec, stack=True)),
}


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def preflight(specs, figdir):
    """Resolve every figure path up front; abort listing all missing files."""
    missing = []
    for spec in specs:
        paths = []
        for rel in spec.get("figures", []):
            path = os.path.join(figdir, rel)
            if not os.path.isfile(path):
                missing.append(rel)
            paths.append(path)
        spec["_figpaths"] = paths
        captions = spec.get("captions")
        if captions and len(captions) != len(paths):
            raise SystemExit(f"slide {spec['title']!r}: {len(captions)} captions "
                             f"for {len(paths)} figures")
    if missing:
        raise SystemExit("missing figures in %s:\n  %s"
                         % (figdir, "\n  ".join(missing)))


def check_geometry(prs):
    """Warn about shapes off-slide and overlapping pictures."""
    problems = []
    sw, sh = prs.slide_width, prs.slide_height
    for i, slide in enumerate(prs.slides, 1):
        pics = []
        for shape in slide.shapes:
            if None in (shape.left, shape.top, shape.width, shape.height):
                continue
            if (shape.left < 0 or shape.top < 0
                    or shape.left + shape.width > sw
                    or shape.top + shape.height > sh):
                problems.append(f"slide {i}: {shape.name!r} extends off-slide")
            if shape.shape_type == 13:  # PICTURE
                pics.append(shape)
        for a, b in [(pics[j], pics[k]) for j in range(len(pics))
                     for k in range(j + 1, len(pics))]:
            if (a.left < b.left + b.width and b.left < a.left + a.width
                    and a.top < b.top + b.height and b.top < a.top + a.height):
                problems.append(f"slide {i}: {a.name!r} overlaps {b.name!r}")
    return problems


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", default=DEFAULT_TEMPLATE,
                    help="deck to clone theme/master/layouts from")
    ap.add_argument("--figdir", default=DEFAULT_FIGDIR,
                    help="directory holding the rendered figures")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output .pptx")
    ap.add_argument("--date", default=DEFAULT_DATE,
                    help="date shown on the title slide")
    ap.add_argument("--no-backup", action="store_true",
                    help="omit the backup section")
    args = ap.parse_args()

    template = os.path.expanduser(args.template)
    out = os.path.expanduser(args.out)
    figdir = os.path.expanduser(args.figdir)

    specs = [s for s in SLIDES if not (args.no_backup and s.get("backup"))]
    preflight(specs, figdir)

    prs = Presentation(template)
    clear_slides(prs)

    for spec in specs:
        kind = spec["kind"]
        if kind == "title":
            slide = build_title(prs, spec, args.date)
        else:
            slide = BUILDERS[kind](prs, spec)
        if spec.get("notes"):
            set_notes(slide, spec["notes"])
        n_fig = len(spec.get("figures", []))
        n_notes = len(spec.get("notes", ""))
        print(f"{len(prs.slides._sldIdLst):3d}  {kind:11s} "
              f"figs={n_fig} notes={n_notes:4d}  {spec['title'][:60]}")

    problems = check_geometry(prs)
    for problem in problems:
        print("WARNING:", problem, file=sys.stderr)

    prs.save(out)
    print(f"\nwrote {out} ({os.path.getsize(out) / 1e6:.1f} MB, "
          f"{len(prs.slides._sldIdLst)} slides)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
