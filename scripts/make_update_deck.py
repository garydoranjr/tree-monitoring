#!/usr/bin/env python
"""Build a monthly TreeFlowering update deck from a rendered figure set.

The deck is assembled by cloning an existing deck in the series (for its
theme, slide master, and layouts), stripping its slides, and rebuilding from
the ``SLIDES`` spec below. Terse bullets go on the slides; the long-form
figure captions and speaker notes go in the notes pane.

Requires ``python-pptx``, which lives in the *base* conda env rather than
``flower``::

    /opt/miniconda3/bin/python scripts/make_update_deck.py

With no arguments this reproduces the 2026-10 deck at
``~/Documents/powerpoint/flower/20261003-TreeFlowering.pptx`` from the figures
in ``/Volumes/Earth03/flower/figs/202610_updates``. For a later month, point
``--figdir``/``--out``/``--date`` at the new material and edit ``SLIDES``.
Earlier months' ``SLIDES`` are in git history.
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
DEFAULT_FIGDIR = "/Volumes/Earth03/flower/figs/202610_updates"
DEFAULT_OUT = "~/Documents/powerpoint/flower/20261003-TreeFlowering.pptx"
DEFAULT_DATE = "October 3, 2026"

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
        title="Summary",
        bullets=[
            (0, "**Gap filled.** Classified the 16 C3KW2X phantom dates (both "
                "alignments, 32 mosaics). The 50ha drone record now runs without a "
                "break from 2018-04-04 to 2026-01-20"),
            (0, "**Deciduous phenology is continuous.** The deciduous share climbs "
                "steadily through the 2023\u201324 dry season, from 0.2% to 4.1%, and "
                "meets the first mavic dates at the same level"),
            (0, "**The mavic flowering excess is a sensor effect.** On the two days "
                "both drones flew, the same classifier calls 2.3% and 1.4% of crowns "
                "flowering on mavic, against 0.5% and 0.4% on phantom. Deciduous "
                "calls agree on both sensors"),
            (0, "**New chips.** Drone labels went onto Planet: 91 pairs and 46 "
                "coregistered (50.5%, the best yield so far), and 21 rated Good, as "
                "well registered as any set we have. The curated sets now hold 77 "
                "chips (phantom only) and 142 (all sources)"),
            (0, "**Mask R-CNN: no extension beats the base peak.** base 0.168 mAP@50, "
                "all three 0.164, +phantom 2023\u201324 0.156 and +mavic 0.151. Clean "
                "same-sensor chips do not rescue mAP, so mavic labels are not the "
                "whole story"),
        ],
        notes="The same-day flights settle the sensor question. The training result "
              "is less clean: it argues against the simple story that mavic labels "
              "alone caused September's regression.\n\n"
              "Runs that include mavic chips do lose about 0.03 binary IoU. These are "
              "single-seed runs, and the gaps are close to the epoch-to-epoch noise.\n\n"
              "Naming: from this month the 50ha drone records are named by drone. "
              "Phantom is the Phantom 4 Pro record, the 24782016 release (90 dates, "
              "2018-04-04 to 2023-10-24) plus the C3KW2X release (16 dates, "
              "2023-10-31 to 2024-03-18). Mavic is the Mavic 3M record (96 dates, "
              "2024-03-06 to 2026-01-20). September's slides called these STRI and "
              "globus.",
    ),
    dict(
        kind="content",
        title="The 50ha Drone Record Is Now Continuous",
        layout="below",
        bullets=[
            (0, "**C3KW2X:** 16 Phantom 4 Pro dates, 2023-10-31 \u2192 2024-03-18, from "
                "the 2024 Smithsonian release (doi:10.60635/C3KW2X). Global and local "
                "alignments, 4-band uint8, on the same grid as 24782016, classified "
                "with no radiometric mapping"),
            (0, "**Overlaps:** two dates, 2024-03-06 and 2024-03-18, were flown by "
                "both drones"),
            (0, "**Caveat at the release boundary.** The 2024 release reprocessed every "
                "date, and RGB correlation on dates present in both releases is only "
                "about 0.27. A step at 2023-10-24 \u2192 2023-10-31 is processing, not "
                "phenology"),
            (0, "**Correction to September.** Every mosaic, from both drones and every "
                "date, was classified with the static 2022-09-29 crown map. Dated "
                "outlines exist through 2024-03-18 but were never used"),
        ],
        figures=["coverage.png"],
        notes="Temporal coverage of the data feeding the 50ha training set. The top "
              "panel shows the number of 50ha Planet scenes per month (grey bars, "
              "2,146 scenes, 2020-01 to 2026-08-11) and the number of whole-island "
              "Planet scenes per month that are at least 50% clear according to "
              "OmniCloudMask (black line). The bottom panel marks the acquisition date "
              "of each drone mosaic with a tick: the 106-date phantom series, 90 dates "
              "from the 24782016 release (blue, 2018-04-04 to 2023-10-24) and 16 from "
              "the C3KW2X release (green, 2023-10-31 to 2024-03-18); the 96-date mavic "
              "series (red, 2024-03-06 to 2026-01-20); and the 13 whole-island "
              "orthomosaics (black, 2024-06-11 to 2025-09-15). The training-chip row "
              "marks the Planet acquisition dates of the 56 phantom 2020\u201323 chips "
              "(blue), the 21 Good phantom 2023\u201324 chips (green) and the 70 mavic "
              "chips (red). The grey bar spans the dates classified with the static "
              "2022-09-29 crown map, the hatched bar spans the dates for which dated "
              "crown outlines exist (2018-04-04 to 2024-03-18), which no classification "
              "used, and the dashed line marks the boundary between the 24782016 and "
              "C3KW2X releases.\n\n"
              "The September caption saying the STRI dates used dated outlines was "
              "wrong; the static crown map was checked against the rasters' footprints.",
    ),
    dict(
        kind="content",
        title="Planet and Cloud Masks for the New Window",
        bullets=[
            (0, "No new Planet ingest this month; holdings already run to 2026-08-11"),
            (0, "No new OmniCloudMask runs were needed: all 175 50ha scenes acquired "
                "2023-10-29 \u2192 2024-03-20 already had readable masks, checked before "
                "the chip build"),
            (0, "Fixed the example figure: the RGB stretch is now computed on OCM-clear "
                "pixels only"),
            (1, "with a stretch over every pixel, bright cloud rendered the clear "
                "forest of a half-clouded scene almost black, so it looked like "
                "unmasked shadow"),
        ],
        figures=["ocm_example.png"],
        notes="OmniCloudMask classification of two PlanetScope scenes over the 50ha "
              "plot from the C3KW2X period. The left column shows each scene's RGB, "
              "stretched to the 2nd\u201398th percentiles of its OCM-clear pixels. The "
              "right column shows the same scene with thick cloud (white), thin cloud "
              "(orange) and cloud shadow (purple) overlaid. Each title gives the "
              "fraction of valid pixels classified clear. OmniCloudMask runs at 10 m "
              "and is upsampled nearest-neighbour to the 3 m Planet grid, so mask "
              "edges are blocky at about 10 m. The scenes are 20240314_150434_31_242b "
              "(2024-03-14, 66% clear) and 20240124_150201_30_2415 (2024-01-24, 50% "
              "clear).",
    ),
    dict(
        kind="content",
        title="Crown Classification Across the Full Record",
        layout="below",
        bullets=[
            (0, "Classified all 32 C3KW2X mosaics with the September HPC pipeline "
                "(`config/pipeline_phantom_ext_{global,local}.sh`). Values for the 186 "
                "mosaics shared with September reproduce exactly"),
            (0, "**Deciduous:** the C3KW2X dates trace the 2023\u201324 dry-season rise, "
                "from 0.2% of crowns (2023-10-31) to 4.1% (2024-03-18), and the first "
                "mavic dates sit on the same curve (2.7%, 3.8%, 4.1%)"),
            (0, "**Flowering:** C3KW2X stays at or below 0.8% (median 0.15%), in line "
                "with the 24782016 phantom record (median 0.5%); mavic jumps to 2.3% on "
                "its first date and never falls below 0.6% (median 1.8%)"),
            (0, "The jump happens at the change of sensor, not the change of season"),
        ],
        figures=["classification_timeseries.png"],
        notes="Share of the 50ha plot's crowns classified as flowering (top) or "
              "deciduous (bottom) at each drone mosaic date. The three series are the "
              "90 phantom mosaics of the 24782016 release (blue circles, locally "
              "aligned, 2018-04-04 to 2023-10-24), the 16 phantom mosaics of the "
              "C3KW2X release (green triangles, locally aligned, 2023-10-31 to "
              "2024-03-18), and the 96 mavic mosaics (red squares, globally aligned, "
              "2024-03-06 to 2026-01-20, converted from uint16 to uint8 with a fixed "
              "per-band gain/offset before classification). A crown counts when the "
              "mean classifier probability over its 2022-09-29 crown-map polygon "
              "exceeds 0.5, and 2,279 of the 2,280 crowns are large enough to score. "
              "Probabilities were averaged from the classification rasters read at 1/16 "
              "resolution (about 0.75 m). Dashed lines are the median of each series "
              "over its own dates, and the dotted line marks the boundary between the "
              "24782016 and C3KW2X releases.\n\n"
              "The C3KW2X medians cover only the dry-season onset, so they are not "
              "annual medians.",
    ),
    dict(
        kind="figrow",
        title="Classification Example: 2024-02-28 Phantom C3KW2X Mosaic",
        figures=["classification_example.png"],
        notes="HPC SegFormer crown classification of the 2024-02-28 phantom C3KW2X "
              "mosaic (locally aligned), the C3KW2X date with the most crowns "
              "classified as flowering. The panels are the RGB mosaic, the per-pixel "
              "flowering probability and the per-pixel deciduous probability, with the "
              "static 2022-09-29 crown map (2,280 crowns) outlined. The top row shows "
              "the whole plot read at 1/16 resolution (about 0.75 m pixels). The bottom "
              "row is a 120 m zoom at 1/2 resolution (about 9 cm), centred on the crown "
              "with the highest mean flowering probability, and the green box marks the "
              "zoom on the top row.",
    ),
    dict(
        kind="content",
        title="Same-Day Flights: the Flowering Excess Is the Sensor",
        bullets=[
            (0, "2024-03-06 and 2024-03-18 were flown by both drones: the same crowns, "
                "the same day, the same classifier"),
            (0, "**Deciduous agrees** (Spearman \u03c1 0.73 and 0.80): 3.2% of crowns on "
                "phantom vs 2.7% on mavic, and 4.1% vs 3.8%"),
            (0, "**Flowering does not** (\u03c1 0.53 and 0.58): 0.5% vs 2.3%, and 0.4% "
                "vs 1.4%"),
            (1, "the extra mavic positives are crowns phantom scores around 0.1 \u2014 a "
                "vertical streak in the scatter, not a scaled copy of the phantom "
                "positives"),
            (0, "Phantom local and global alignments agree with each other, so "
                "alignment is not the cause"),
            (0, "**Conclusion:** the excess comes from the mavic sensor or its uint16 "
                "\u2192 uint8 mapping meeting a flowering model trained on phantom "
                "imagery. It is not phenology, and it confirms last month's leading "
                "explanation for the regression"),
        ],
        figures=["same_date_comparison.png"],
        notes="Per-crown mean classifier probability on the two days flown by both "
              "drones. The x axis is the phantom C3KW2X locally aligned mosaic, the y "
              "axis the mavic mosaic, and each dot is one of the 2,279 scored crowns of "
              "the 2022-09-29 crown map. Rows are 2024-03-06 and 2024-03-18; columns "
              "are P(flowering) and P(deciduous). Probabilities are crown means from "
              "classification rasters read at 1/16 resolution (about 0.75 m). The "
              "dotted lines mark the 0.5 threshold on each axis, and the grey line is "
              "y = x. Each inset gives the share of crowns above 0.5 on phantom local, "
              "phantom global and mavic, and the Spearman rank correlation between "
              "phantom local and mavic.",
    ),
    dict(
        kind="content",
        title="Spot-Check: C3KW2X Flowering Labels Look Like Flowers",
        bullets=[
            (0, "Sampled 16 C3KW2X crown-dates with mean P(flowering) > 0.5 (93 "
                "candidates on 16 dates). The mavic and 24782016 samples are the same "
                "32 and 16 crown-dates as in September"),
            (0, "**C3KW2X positives are mostly real \u2014 about 9 of 16 show visible "
                "flowers**"),
            (1, "three are pink-flowered Tabebuia rosea (crowns 623, 1020 and 427, all "
                "on 2024-02-28), a species that flowers in the dry season, which fits"),
            (1, "others are cream-flowered, such as Nectandra lineata (2101) and "
                "Cordia alliodora (113, 1452)"),
            (1, "about 5 are ambiguous (yellowish crowns, or flowers on a neighbouring "
                "crown); two show no flowers, Luehea seemannii 931 and Prioria "
                "copaifera 66"),
            (0, "Positives per date: about 6 on C3KW2X, about 18 on 24782016 and about "
                "45 on mavic, out of 2,279 crowns"),
            (0, "Several positives fall on a flowering crown that the 2022 polygon only "
                "partly covers, for example 2024-02-28 crown 427 \u2014 real flowers "
                "with stale outlines"),
        ],
        notes="This is a visual spot-check, not blind labelling; the proportions are "
              "rough.",
    ),
    dict(
        kind="figstack",
        title="Phantom C3KW2X Crowns Labelled Flowering, With and Without Flowers",
        figures=[
            "spotcheck_flowering/zoom_phantom_c3kw2x_likely_tp.png",
            "spotcheck_flowering/zoom_phantom_c3kw2x_suspect_fp.png",
        ],
        captions=[
            "Visible flowers (pink Tabebuia rosea, cream Nectandra and Cordia) "
            "\u2014 likely true positives",
            "No visible flowers \u2014 suspected false positives",
        ],
        notes="Phantom C3KW2X drone crowns labelled flowering by the HPC SegFormer "
              "classifier. Each was selected because its mean P(flowering) over the "
              "2022-09-29 crown polygon exceeds 0.5, and the title gives the flight "
              "date, crown ID, mean probability and the polygon's species from the "
              "crown map (Latin name and BCI six-letter code). Crowns are shown at "
              "native mosaic resolution (about 4.5 cm) from the locally aligned uint8 "
              "mosaic, with the crown polygon outlined in cyan. The top figure shows "
              "crowns with visible flowers; the bottom figure shows crowns with no "
              "visible flowers that were nonetheless labelled flowering.",
    ),
    dict(
        kind="figstack",
        title="For Comparison: Mavic Crowns Labelled Flowering (September's Picks)",
        figures=[
            "spotcheck_flowering/zoom_mavic_likely_tp.png",
            "spotcheck_flowering/zoom_mavic_suspect_fp.png",
        ],
        captions=[
            "Visible flowers (lilac, cream, pink, white) \u2014 likely true positives",
            "No visible flowers \u2014 suspected false positives",
        ],
        notes="The same construction for the mavic record, on the 32 crown-dates "
              "sampled in September: mavic drone crowns with mean P(flowering) over the "
              "2022-09-29 crown polygon above 0.5, at native mosaic resolution (about "
              "4.7 cm) after the fixed per-band uint16 to uint8 mapping used for "
              "classification, with the crown polygon outlined in cyan. Roughly half "
              "the mavic positives show no flowers, against about 7 of 16 on C3KW2X.",
    ),
    dict(
        kind="content",
        title="Same-Day Crowns, Side by Side",
        bullets=[
            (0, "**Whole plot:** the mavic P(flowering) map is speckled with moderate "
                "probabilities everywhere, and on 2024-03-06 it also shows ring-shaped "
                "bands that follow no crown pattern in the RGB, most likely an "
                "acquisition or mosaicking artifact. The phantom maps are dark except "
                "for a few crowns"),
            (0, "**Direction of disagreement:** 42 crown-dates are flowering on mavic "
                "only (mavic > 0.5, phantom < 0.2), 1 on phantom only, and 16 on both"),
            (0, "**Some mavic-only crowns may be real early flowering.** Crowns 1386 "
                "and 1766 look plain green at this zoom, but both are Jacaranda "
                "copaia, the species of crown 646, which both sensors call flowering a "
                "few days later. Crown 1836 is red-flowered Symphonia globulifera"),
            (0, "**Some are clear false positives.** Crown 823 is Cecropia insignis, "
                "whose silvery leaves give the grey look the model seems to mistake for "
                "flowers; crown 1150 (Trichilia tuberculata) is reddish-brown, more "
                "likely new leaves or fruit"),
            (0, "**On two the blob sits beside the polygon** \u2014 crowns 86 and 87 are "
                "Astrocaryum standleyanum palms, and the high probability lies on a "
                "neighbouring crown"),
            (0, "**Deciduous:** the only disagreements are two bare-branched crowns "
                "that phantom calls deciduous and mavic does not"),
            (0, "**Caveat on species:** each label belongs to the tree tagged for that "
                "polygon in the 2022-09-29 map. Where the polygon misfits, the visible "
                "crown can be a neighbour"),
        ],
        notes="Agreement: the crowns both sensors call flowering include a flowering "
              "Jacaranda copaia (646, 2024-03-18) and a red-leaved crown next to the "
              "Astrocaryum palm that polygon 471 belongs to. The single "
              "phantom-only crown is Luehea seemannii (1978).\n\n"
              "On the deciduous disagreements: one is Astronium graveolens (2270), "
              "which is dry-season deciduous. The other, crown 908, carries lilac "
              "flowers on bare branches, which mavic shows more clearly; its polygon is "
              "tagged Tabernaemontana arborea, so the visible crown may be a neighbour.",
    ),
    dict(
        kind="figrow",
        title="Same-Day Whole-Plot Flowering Maps",
        figures=["same_date_maps.png"],
        notes="Phantom and mavic drone mosaics of the 50ha plot flown on the same day, "
              "with the per-pixel flowering probability from the same SegFormer model. "
              "The rows are 2024-03-06 and 2024-03-18, and for each date the columns "
              "are the phantom C3KW2X locally aligned RGB mosaic, its P(flowering), the "
              "mavic RGB mosaic (uint16 converted to uint8 with the fixed per-band "
              "gain/offset used for classification), and its P(flowering). All panels "
              "were read at 1/16 resolution (about 0.75 m pixels). Grey marks pixels "
              "outside the classification windows, which the static 2022-09-29 crown "
              "map places.",
    ),
    dict(
        kind="figrow",
        title="Crowns Flowering on Mavic Only",
        figures=["same_date_mavic_only_flowering.png"],
        notes="The eight crowns with the largest mavic minus phantom difference in "
              "P(flowering), among crowns with mavic above 0.5 and phantom below 0.2, "
              "on the two days both drones flew. Each crown takes four panels at native "
              "mosaic resolution (about 4.5 cm phantom, 4.7 cm mavic): phantom C3KW2X "
              "local RGB, phantom probability map, mavic RGB, and mavic probability "
              "map. The probability maps are read at 1/4 resolution, and each panel "
              "title gives the crown-mean probability computed at 1/16 resolution. The "
              "cyan outline is the crown polygon in the static 2022-09-29 crown map, "
              "and each row label gives the case, the date, the crown ID and that "
              "polygon's species as Latin name with BCI six-letter code. Species comes "
              "from the crown map's latin field, or where that is blank from the map's "
              "own code-to-name pairs, the dated crown time series by tag, or AVUELO "
              "(2026) by code.",
    ),
    dict(
        kind="figrow",
        title="Same-Day Disagreements: Other Cases",
        figures=["same_date_other_cases.png"],
        notes="Crowns where the phantom and mavic classifications of the same day "
              "disagree in the other directions, one crown per row: the two crowns most "
              "confidently flowering on both sensors, the single crown flowering on "
              "phantom only (above 0.5 against below 0.3), and the two crowns deciduous "
              "on phantom only (P(deciduous) above 0.5 against below 0.3). Each crown "
              "takes four panels at native mosaic resolution (about 4.5 cm phantom, "
              "4.7 cm mavic): phantom C3KW2X local RGB, phantom probability map, mavic "
              "RGB, and mavic probability map. The probability maps are read at 1/4 "
              "resolution, and each panel title gives the crown-mean probability "
              "computed at 1/16 resolution. The cyan outline is the crown polygon in "
              "the static 2022-09-29 crown map, and each row label gives the case, the "
              "date, the crown ID and that polygon's species.",
    ),
    dict(
        kind="content",
        title="Drone Labels \u2192 Planet Chips for 2023\u201324",
        layout="below",
        bullets=[
            (0, "**Pairing:** each of the 16 C3KW2X flights was paired with Planet "
                "scenes within \u00b12 days, giving 91 pairs on 91 distinct scenes; "
                "2023-12-05 has none"),
            (0, "**Coregistration:** one AROSICS shift per scene; 46 coregistered "
                "(50.5%, against 40.6% for phantom 2020\u201323 and 41.6% for mavic). "
                "Median shift 7.4 m, close to the 8.4 m for phantom 2020\u201323"),
            (0, "**Ratings:** 21 Good, 11 Fair, 14 Poor on image quality. Claude "
                "pre-rated the chips after calibrating on a sample of the mavic "
                "ratings, and the review moved two partly shadowed chips Good "
                "\u2192 Fair"),
            (0, "**Two curated sets:** 77 chips (56 + 21, a same-sensor extension, "
                "3,258 training instances) and 142 (126 + 16, all sources, 7,908). "
                "Five scene-name collisions keep their mavic copy"),
        ],
        figures=["coreg_stats.png"],
        notes="Yield and shift magnitude of the drone-label to Planet-chip transfer for "
              "three builds: phantom 2020\u201323 (blue, 24782016 locally aligned "
              "mosaics, vetted in Labelbox), phantom 2023\u201324 (green, C3KW2X locally "
              "aligned mosaics, pre-rated by Claude and reviewed with "
              "vet_planet_chips.py), and mavic 2024\u201326 (red, globally aligned "
              "mosaics, vetted with vet_planet_chips.py). The left panel shows, for "
              "each build, the number of drone\u2013Planet pairs within \u00b12 days, the "
              "number successfully coregistered by AROSICS, and the number rated Good, "
              "with the percentage of pairs at each stage. The right panel shows the "
              "distribution of the magnitude of the single rigid AROSICS shift applied "
              "per coregistered scene (density, 1.5 m bins), and the dotted line marks "
              "one 3 m Planet pixel.\n\n"
              "Five chips have the same scene name as Good mavic chips (the "
              "2024-03-06/18 flights). copy_good_planet_vetting.py --no-overwrite now "
              "keeps the existing copy and lists them, so the mavic half of the "
              "142-chip set is identical to September's 126-chip set. Both sets pass "
              "the Mask R-CNN loader on both halves with no empty chips.",
    ),
    dict(
        kind="content",
        title="Result: No Training Set Beats the Base Peak",
        bullets=[
            (0, "Four runs scored on the same 56 phantom 2020\u201323 test chips (right "
                "halves). Base and +mavic reproduce September exactly"),
            (0, "**Peak mask mAP@50:** base (56 chips) **0.168** at epoch 12; all three "
                "(142) 0.164 at 8; +phantom 2023\u201324 (77) 0.156 at 5; +mavic (126) "
                "0.151 at 5"),
            (0, "**The same-sensor extension does not rescue mAP** \u2014 ahead of base "
                "for epochs 1\u20135 (by up to +0.045), behind at 8, 12 and 25 "
                "(\u22120.041 at 12)"),
            (0, "**Binary IoU splits on whether mavic chips are included:** 0.308 and "
                "0.299 without, 0.269 and 0.273 with"),
            (0, "**Noise caveat:** single-seed runs, 12 epochs scored. The base peak is "
                "a one-epoch spike (0.148 at 8, 0.168 at 12, 0.136 at 17), and "
                "epoch-to-epoch swings of 0.02\u20130.04 are larger than most gaps "
                "between runs"),
            (0, "Late collapse is unchanged: every run trades recall for precision and "
                "ends at 0.04\u20130.06 mAP@50"),
        ],
        figures=["maskrcnn_summary.png"],
        notes="Mask R-CNN mask mAP@50 on the test (right-half) crops of the 56 phantom "
              "2020\u201323 training chips, against training epoch (log scale). The four "
              "curves are models trained on the 56 phantom 2020\u201323 chips (blue), "
              "those plus the 70 mavic 2024\u201326 chips (red, 126 chips), those plus "
              "the 21 phantom 2023\u201324 C3KW2X chips (green, 77 chips), and all three "
              "sources (purple, 142 chips: 126 plus the 16 C3KW2X chips whose scenes "
              "are not already mavic chips). Twelve epochs between 1 and 200 were "
              "scored per run with OCM cloud masks and a 64-pixel minimum instance "
              "size, and stars mark each run's peak, labelled with its value and "
              "epoch.\n\n"
              "Last month's prediction was that clean, same-sensor extra chips would "
              "help where mavic chips hurt. On mAP they do not: every extension helps "
              "early and loses the peak. The IoU split is the one place where mavic "
              "chips look specifically harmful. With single seeds and a spiky base "
              "curve, I would not rank 0.164 against 0.168.",
    ),
    dict(
        kind="content",
        title="Registration of the New Chips",
        layout="below",
        bullets=[
            (0, "**Same test as September:** phase-correlate each chip's Planet image "
                "against its shifted drone sidecar in 96 m windows, keeping clear, "
                "well-matched windows (NCC \u2265 0.3)"),
            (0, "**The C3KW2X chips are the best registered of the three sets:** median "
                "window offset 0.57 m, against 0.92 m for phantom 2020\u201323 and "
                "0.71 m for mavic; windows off by more than 3 m 1.2%, against 7.9% and "
                "1.9%; within-chip spread 0.44 m, against 0.79 m and 0.53 m"),
            (0, "The ordering holds with 48 m windows: 0.38 / 0.62 / 0.54 m median and "
                "2.5% / 8.8% / 4.1% of windows above 3 m"),
            (0, "**Large global shifts are real.** Several adjacent Planet frames got "
                "AROSICS shifts more than 10 m apart, yet each is aligned to within "
                "about 0.5 m after its own shift"),
        ],
        figures=["residual_offsets.png"],
        notes="Residual drone-to-Planet offset remaining in the training chips after "
              "the single per-scene AROSICS shift. The three sets are the 56 curated "
              "phantom 2020\u201323 chips (blue, 24782016 locally aligned mosaics), the "
              "21 Good phantom 2023\u201324 chips (green, C3KW2X locally aligned "
              "mosaics) and the 70 Good mavic 2024\u201326 chips (red, globally aligned "
              "mosaics). Each chip was divided into 128 \u00d7 128 pixel windows (96 m "
              "at 0.75 m), and the Planet chip was phase-correlated against the shifted "
              "drone sidecar, downsampled to the chip grid, after a 1.5-pixel Gaussian "
              "blur. Windows were kept if they were at least 90% clear and inside the "
              "drone footprint, had a post-shift normalized cross-correlation of at "
              "least 0.3, and needed a shift under 20 m. The left panel shows the "
              "distribution of window offset magnitudes, with medians and window counts "
              "in the legend; the centre panel the per-chip non-rigid spread, the "
              "median distance of the window offset vectors from the chip's median "
              "vector (boxes are interquartile ranges, dots are chips); the right panel "
              "the window offset vectors, magnified 8\u00d7, on the chip with the "
              "largest spread.\n\n"
              "Planet's per-scene georeferencing varies by more than 10 m between "
              "adjacent frames, which is why the applied shifts are large while the "
              "residuals are small.",
    ),
    dict(
        kind="content",
        title="Does Global vs Local Alignment Matter? Phantom Both Ways",
        layout="below",
        bullets=[
            (0, "Rebuilt the phantom chips from the globally aligned mosaics (both "
                "releases, 414 pairs), so phantom is one set per alignment against "
                "globally aligned mavic"),
            (0, "**Yield is similar:** 171 coregistered global, against 177 local and "
                "154 of 370 for mavic"),
            (0, "**AROSICS applies the same shift either way** (151 scenes both ways, "
                "median difference +0.01 m), so the larger phantom shifts are an offset "
                "between phantom mosaics and Planet, not the local warp"),
            (0, "**Residuals on the same 74 Good scenes** (median offset / windows "
                "> 3 m / spread): phantom local 0.75 m / 5.8% / 0.64 m; phantom global "
                "0.86 / 7.6% / 0.78; mavic global 0.71 / 1.9% / 0.53"),
            (0, "**Per chip, local and global are a wash. Alignment is not what limits "
                "label registration**"),
        ],
        figures=["coreg_stats_by_alignment.png"],
        notes="Yield and shift magnitude of the drone-label to Planet-chip transfer, "
              "grouped by drone and alignment. The three groups are phantom locally "
              "aligned (blue, 24782016 and C3KW2X releases pooled, 2020-01 to 2024-03), "
              "the same phantom flights globally aligned (cyan) and mavic globally "
              "aligned (red, 2024-03 to 2026-01). The left panel shows the number of "
              "drone\u2013Planet pairs within \u00b12 days, the number coregistered by "
              "AROSICS, and the number rated Good, with the percentage of pairs at each "
              "stage; phantom global Good counts use the local-build ratings of the "
              "same Planet scenes, and twenty global chips whose scene failed "
              "coregistration in the local build are unrated and not counted. The "
              "centre panel shows the distribution of the single rigid AROSICS shift "
              "applied per coregistered scene (density, 1.5 m bins), with the dotted "
              "line at one 3 m Planet pixel. The right panel shows the shift magnitude "
              "applied to each of the 151 scenes coregistered under both phantom "
              "alignments, local (x) against global (y), with the 1:1 line.\n\n"
              "Seventy-four of the 77 Good phantom scenes also coregistered under "
              "global alignment, which is the set the residuals are measured on. "
              "Paired on the same scene, the median global minus local difference is "
              "+0.01 m in spread (worse in 39 of 74 chips, Wilcoxon p = 0.16) and "
              "0.00 m in rigid residual (p = 0.46).\n\n"
              "With 48 m windows the residuals are 0.53 / 0.60 / 0.54 m median and "
              "6.9% / 9.4% / 4.1% above 3 m. The global set's worse summary comes from "
              "a heavier tail of badly offset windows, not a consistent per-chip "
              "penalty. The mavic chips, globally aligned, are better registered than "
              "phantom in either alignment, which rules out last month's alignment "
              "hypothesis.",
    ),
    dict(
        kind="figrow",
        title="Residual Offsets by Drone and Alignment",
        figures=["residual_offsets_by_alignment.png"],
        notes="Residual drone-to-Planet offset remaining in the training chips after "
              "the single per-scene AROSICS shift, grouped by drone and alignment. The "
              "sets are phantom locally aligned (blue) and phantom globally aligned "
              "(cyan), both restricted to the same 74 Planet scenes (2020\u20132024) "
              "rated Good and coregistered under both alignments, and the 70 Good mavic "
              "globally aligned chips (red, 2024\u20132026). Each chip was divided into "
              "128 \u00d7 128 pixel windows (96 m at 0.75 m), and the Planet chip was "
              "phase-correlated against the shifted drone sidecar, downsampled to the "
              "chip grid, after a 1.5-pixel Gaussian blur. Windows were kept if they "
              "were at least 90% clear and inside the drone footprint, had a post-shift "
              "normalized cross-correlation of at least 0.3, and needed a shift under "
              "20 m. The left panel shows the distribution of window offset magnitudes, "
              "with medians and window counts in the legend; the centre panel the "
              "per-chip non-rigid spread, the median distance of the window offset "
              "vectors from the chip's median vector (boxes are interquartile ranges, "
              "dots are chips); the right panel the window offset vectors, magnified "
              "8\u00d7, on the chip with the largest spread.",
    ),
    # Blink GIFs: the worst well-matched window in the two worst chips of each set.
    dict(
        kind="figrow",
        title="Examples of Residual Misalignment: Phantom 2020\u201323 Chips",
        figures=[
            "gifs/blink_1_20230326_145322_83_24bc_zoom.gif",
            "gifs/blink_2_20230410_144554_03_2460_zoom.gif",
        ],
        captions=[
            "Planet 2023-03-26 / drone 2023-03-28\n12.4 m offset",
            "Planet 2023-04-10 / drone 2023-04-11\n10.3 m offset",
        ],
        notes="Blink comparison of a Planet training chip (Planet frame) and the drone "
              "orthomosaic after the single AROSICS shift (drone frame). The crown-label "
              "outlines (yellow) are drawn at the same position in both frames. The "
              "labels come from per-pixel drone classification, so they fit the drone "
              "crowns, and any displacement of the matching crowns in the Planet frame "
              "is label misregistration. Each zoom covers 192 \u00d7 192 m around the "
              "window with the largest measured residual offset in that chip, and the "
              "scale bar is 50 m.\n\n"
              "These are the worst chips of the phantom 2020\u201323 set, the worst of "
              "the three. Bright (flowering or leafless) crowns appear several metres "
              "away from their outlines in the Planet frame.",
    ),
    dict(
        kind="figrow",
        title="Examples of Residual Misalignment: Phantom 2023\u201324 Chips",
        figures=[
            "gifs/blink_5_20240209_155640_21_24ad_zoom.gif",
            "gifs/blink_6_20240307_150321_14_24a8_zoom.gif",
        ],
        captions=[
            "Planet 2024-02-09 / drone 2024-02-08\n5.0 m offset",
            "Planet 2024-03-07 / drone 2024-03-06\n4.1 m offset",
        ],
        notes="Same construction as the previous slide, for the worst chips of the new "
              "phantom 2023\u201324 C3KW2X set. The worst C3KW2X windows, at 4\u20135 m, "
              "are milder than the worst of either other set.",
    ),
    dict(
        kind="figrow",
        title="Examples of Residual Misalignment: Mavic 2024\u201326 Chips",
        figures=[
            "gifs/blink_3_20250315_161618_73_24fe_zoom.gif",
            "gifs/blink_4_20240304_155459_24_24f6_zoom.gif",
        ],
        captions=[
            "Planet 2025-03-15 / drone 2025-03-17\n8.6 m offset (Planet frame partly hazy)",
            "Planet 2024-03-04 / drone 2024-03-06\n5.6 m offset",
        ],
        notes="Same construction as the previous two slides, for the worst chips of the "
              "mavic 2024\u201326 set. All three sets have a real tail of badly "
              "misregistered windows, and the phantom 2020\u201323 set has the most.",
    ),
    dict(
        kind="two_content",
        title="Interpretation",
        bullets=[
            (0, "**What we know now**"),
            (1, "The 50ha record is continuous from 2018 to 2026, and the deciduous "
                "signal carries across both the release boundary and the change of "
                "sensor"),
            (1, "Mavic flowering labels are inflated by the sensor: three to five times "
                "the phantom rate on the same day, while deciduous labels agree"),
            (1, "The C3KW2X chips are the best registered and best yielding chip set, "
                "and their flowering labels look mostly genuine"),
            (1, "Even so, adding them does not raise peak mAP@50, so mavic label "
                "semantics cannot be the whole story behind September's flat result"),
            (1, "Mavic chips do cost about 0.03 of binary IoU, a gap phantom-only extra "
                "chips do not show"),
            (1, "Alignment is not the problem, and the mavic-only flowering crowns "
                "mostly show no flowers on either sensor"),
        ],
        bullets2=[
            (0, "**Remaining explanations**"),
            (1, "**Evaluation noise.** One seed and a 56-chip test set. The base peak "
                "is a single-epoch spike, and the differences between runs are about "
                "the size of the epoch-to-epoch swings"),
            (1, "**Train/test shift.** The only test set is 2020\u201323. Every "
                "extension adds a different period or season, which this test cannot "
                "reward"),
            (1, "**Data volume is not the bottleneck.** All extensions win early and "
                "lose the peak, which looks like overfitting dynamics rather than a "
                "shortage of labels"),
        ],
        notes="Two of last month's three hypotheses are now ruled out: registration "
              "(slide on phantom both ways) and label semantics alone (the C3KW2X "
              "extension). What is left is mostly about how the comparison is scored, "
              "not about the chips.",
    ),
    dict(
        kind="content",
        title="Next Steps",
        bullets=[
            (0, "Repeat base and +phantom 2023\u201324 with 3 seeds each, so the gap can "
                "be put beside run-to-run variance"),
            (0, "Score all four runs on the right halves of the C3KW2X and mavic chips "
                "as well, by passing `compare_maskrcnn_runs.py --chip-dir` once per set"),
            (0, "Use best-checkpoint selection and the cosine schedule from now on, "
                "since peaks come at epochs 5\u201317 and every run collapses after"),
            (0, "For mavic flowering, try per-date histogram matching to the phantom "
                "radiometry instead of the fixed gain/offset, or fine-tune the "
                "flowering model on mavic imagery"),
            (1, "use the 2024-03-06/18 phantom rates as the acceptance test"),
        ],
    ),
    # ---------------- backup ----------------
    dict(kind="section", title="Backup", backup=True),
    dict(
        kind="figrow",
        title="Mask R-CNN Four-Run Sweep: All Metrics",
        figures=["maskrcnn_headtohead.png"],
        notes="Mask R-CNN training-set comparison over training epochs on the test "
              "(right-half) crops of the 56 phantom 2020\u201323 chips. The runs are "
              "trained on the 56 phantom 2020\u201323 chips (blue), plus the mavic chips "
              "(red, 126), plus the C3KW2X chips (green, 77), and all three sources "
              "(purple, 142). The top-left panel shows mask mAP@50 per epoch, with "
              "stars marking each run's peak; the top-right panel binary IoU per epoch, "
              "with stars marking each run's peak; the bottom-left panel crown-level "
              "precision (solid circles) and recall (dashed squares) per epoch; and the "
              "bottom-right panel the precision\u2013recall trajectory of each run, with "
              "marker fill giving the epoch (log scale, 1\u2013200). All values are from "
              "the same 12 epochs per run, scored with OCM cloud masks and a 64-pixel "
              "minimum instance size.",
        backup=True,
    ),
    dict(
        kind="figrow",
        title="Residual Offsets with 48 m Windows",
        figures=["residual_offsets_w64.png"],
        notes="As the three-set residual-offset figure, but with 64 \u00d7 64 pixel "
              "(48 m) windows: 0.38 m median for phantom 2023\u201324, against 0.62 m "
              "for phantom 2020\u201323 and 0.54 m for mavic, and 2.5% / 8.8% / 4.1% of "
              "windows above 3 m. The ordering between the sets is unchanged.",
        backup=True,
    ),
    dict(
        kind="figrow",
        title="Residual Offsets by Alignment, 48 m Windows",
        figures=["residual_offsets_by_alignment_w64.png"],
        notes="As the by-alignment residual-offset figure, but with 64 \u00d7 64 pixel "
              "(48 m) windows, on the same 74 Good phantom scenes coregistered under "
              "both alignments: 0.53 m median for phantom local, 0.60 m for phantom "
              "global and 0.54 m for mavic global, and 6.9% / 9.4% / 4.1% of windows "
              "above 3 m. Local and global remain a wash.",
        backup=True,
    ),
    # Static equivalents of the blink GIFs, one per slide so the triptych panels
    # stay legible in a PDF export. Dates and offsets go in the title, and come
    # from gifs/blink_examples.csv.
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
            ("phantom 2020\u201323", "blink_1_20230326_145322_83_24bc", "2023-03-26", "2023-03-28", "12.4 m"),
            ("phantom 2020\u201323", "blink_2_20230410_144554_03_2460", "2023-04-10", "2023-04-11", "10.3 m"),
            ("phantom 2023\u201324", "blink_5_20240209_155640_21_24ad", "2024-02-09", "2024-02-08", "5.0 m"),
            ("phantom 2023\u201324", "blink_6_20240307_150321_14_24a8", "2024-03-07", "2024-03-06", "4.1 m"),
            ("mavic 2024\u201326", "blink_3_20250315_161618_73_24fe", "2025-03-15", "2025-03-17", "8.6 m"),
            ("mavic 2024\u201326", "blink_4_20240304_155459_24_24f6", "2024-03-04", "2024-03-06", "5.6 m"),
        ]
    ],
    dict(
        kind="figrow",
        title="Flowering Spot-Check Contact Sheet: Phantom C3KW2X",
        figures=["spotcheck_flowering/phantom_c3kw2x_flowering.png"],
        notes="Contact sheet of the 16 sampled phantom C3KW2X crown-dates with mean "
              "P(flowering) > 0.5, at reduced resolution. About 9 of 16 show visible "
              "flowers.",
        backup=True,
    ),
    dict(
        kind="figrow",
        title="Flowering Spot-Check Contact Sheets: Mavic",
        figures=[
            "spotcheck_flowering/mavic_flowering_1.png",
            "spotcheck_flowering/mavic_flowering_2.png",
        ],
        captions=["Mavic positives, sheet 1", "Mavic positives, sheet 2"],
        notes="Contact sheets of the 32 sampled mavic crown-dates with mean "
              "P(flowering) > 0.5, at reduced resolution. These are September's sample, "
              "unchanged.",
        backup=True,
    ),
    dict(
        kind="figrow",
        title="Flowering Spot-Check Contact Sheet: Phantom 24782016 Reference",
        figures=["spotcheck_flowering/phantom_flowering_reference.png"],
        notes="Contact sheet of the 16 phantom 24782016 crown-dates sampled as a "
              "reference, unchanged from September. These positives are more often "
              "visibly flowering than the mavic ones, though several are ambiguous at "
              "that resolution.",
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
