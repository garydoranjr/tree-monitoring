#!/usr/bin/env python
"""Persistence baseline for the Planet temporal CV: how well does one drone
flight's crown labels predict another's?

The temporal CV holds out dates, not ground, so a model can score partly by
knowing which of the same crowns were events on nearby training dates. This
script measures that persistence directly from the drone crown labels (the
rule of plot_crown_species_comparison.py: event when
1 - (1 - p_flowering)(1 - p_deciduous) > 0.5, crowns of at least
--min-crown-px Planet pixels), without any imagery:

1. Flight pairs: for every ordered pair of flights at most --max-gap days
   apart, the second flight's events are predicted by the first's. Counts are
   pooled by gap bin.
2. CV analogue: for every chip of the fold CSV, its flight's events are
   predicted by the flight of the nearest-dated chip in the other folds (the
   training data that fold's model saw). Counts are pooled by the chip's
   days-to-nearest-training bin, as in eval_temporal_cv.py.

Each reports crown precision and recall, and an area-weighted crown Jaccard
(the summed crown_px of crowns that are events on both dates over the summed
crown_px of crowns that are events on either), an approximation of the binary
IoU the CV reports. With --cv-json, the CV's own pooled IoU per bin is printed
alongside.

Typical usage:
    python scripts/persistence_baseline.py \\
        --folds config/temporal_folds_20261012.csv \\
        --cv-json /Volumes/Earth03/flower/202610_temporal/temporal_cv.json \\
        --out-prefix results/persistence_baseline
"""
import json

import click
import numpy as np
import pandas as pd

from plot_crown_species_comparison import APPLY, NC4S, load_drone

BINS = (('0-3', 0, 3), ('4-7', 4, 7), ('8-14', 8, 14),
        ('15-30', 15, 30), ('>30', 31, 10**6))  # as eval_temporal_cv.DAY_BINS


def day_bin(days):
    for label, lo, hi in BINS:
        if lo <= days <= hi:
            return label
    raise ValueError(f'negative day distance {days}')


def pair_counts(events, area, src, dst):
    """Counts for predicting flight dst's events with flight src's, over the
    crowns observed on both."""
    a, b = events[src], events[dst]
    both = a.notna() & b.notna()
    a, b = a[both].astype(bool), b[both].astype(bool)
    w = area.reindex(a.index)
    return {'crowns': int(both.sum()), 'tp': int((a & b).sum()),
            'n_pred': int(a.sum()), 'n_true': int(b.sum()),
            'inter_px': float(w[a & b].sum()), 'union_px': float(w[a | b].sum())}


def summarize(df, key):
    g = df.groupby(key, sort=False)[['crowns', 'tp', 'n_pred', 'n_true',
                                     'inter_px', 'union_px']].sum()
    g.insert(0, 'pairs', df.groupby(key, sort=False).size())
    g['precision'] = g.tp / g.n_pred
    g['recall'] = g.tp / g.n_true
    g['area_jaccard'] = g.inter_px / g.union_px
    order = [b[0] for b in BINS if b[0] in g.index]
    return g.loc[order, ['pairs', 'crowns', 'n_true', 'precision', 'recall', 'area_jaccard']]


@click.command()
@click.option('--folds', 'folds_csv', required=True, type=click.Path(exists=True))
@click.option('--cv-json', default=None, type=click.Path(exists=True),
              help="A temporal_cv.json, to print the CV's pooled IoU per bin.")
@click.option('--apply-dir', default=APPLY, show_default=True,
              type=click.Path(exists=True, file_okay=False),
              help='Apply run whose crown_timeseries.csv gives crown_px.')
@click.option('--nc4', 'nc4s', multiple=True, default=NC4S, show_default=True)
@click.option('--min-crown-px', default=64, show_default=True)
@click.option('--max-gap', default=60, show_default=True,
              help='Largest flight-pair gap (days) in part 1.')
@click.option('--out-prefix', default=None, help='Write <prefix>_{pairs,cv}.csv.')
def main(folds_csv, cv_json, apply_dir, nc4s, min_crown_px, max_gap, out_prefix):
    area = (pd.read_csv(f'{apply_dir}/crown_timeseries.csv', usecols=['tag', 'crown_px'])
              .groupby('tag')['crown_px'].first())
    area = area[area >= min_crown_px]
    drone = load_drone(nc4s)
    drone = drone[drone['tag'].isin(area.index)]
    events = drone.pivot_table(index='tag', columns='date', values='event',
                               aggfunc='first')
    flights = sorted(events.columns)

    rows = []
    for src in flights:
        for dst in flights:
            gap = abs((dst - src).days)
            if src != dst and gap <= max_gap:
                rows.append({'src': src, 'dst': dst, 'gap': gap, 'bin': day_bin(gap),
                             **pair_counts(events, area, src, dst)})
    pairs = pd.DataFrame(rows)

    folds = pd.read_csv(folds_csv, parse_dates=['planet_date'])
    folds['flight'] = pd.to_datetime(folds['drone_dates'].astype(str).str.split(';').str[0])
    missing = sorted(set(folds['flight']) - set(flights))
    if missing:
        raise click.ClickException(f'fold flights without drone labels: {missing[:5]}')
    rows = []
    for _, chip in folds.iterrows():
        train = folds[folds['fold'] != chip['fold']]
        near = train.loc[(train['planet_date'] - chip['planet_date']).abs().idxmin()]
        rows.append({'chip': chip['chip'], 'fold': chip['fold'],
                     'days_to_nearest_train': chip['days_to_nearest_train'],
                     'bin': day_bin(chip['days_to_nearest_train']),
                     'src': near['flight'], 'dst': chip['flight'],
                     **pair_counts(events, area, near['flight'], chip['flight'])})
    cv = pd.DataFrame(rows)

    p_tab, c_tab = summarize(pairs, 'bin'), summarize(cv, 'bin')
    tot = cv[['tp', 'n_pred', 'n_true', 'inter_px', 'union_px']].sum()
    if cv_json:
        with open(cv_json) as fh:
            d = json.load(fh)
        sel = d['per_epoch'][str(d['best_epoch'])]
        c_tab[f'cv_iou_e{d["best_epoch"]}'] = [sel['by_day_bin'][b]['test_iou']
                                              for b in c_tab.index]
    with pd.option_context('display.float_format', '{:.3f}'.format, 'display.width', 160):
        click.echo(f'{len(flights)} flights, {len(area)} crowns >= {min_crown_px} px')
        click.echo(f'\n1. flight pairs at most {max_gap} d apart, by gap:')
        click.echo(p_tab.to_string())
        click.echo('\n2. each CV chip predicted from its nearest training chip\'s flight, '
                   'by days to nearest training chip:')
        click.echo(c_tab.to_string())
        click.echo(f'pooled: precision {tot.tp / tot.n_pred:.3f}, recall '
                   f'{tot.tp / tot.n_true:.3f}, area Jaccard '
                   f'{tot.inter_px / tot.union_px:.3f}'
                   + (f'; CV pooled IoU {sel["pooled"]["test_iou"]:.3f}' if cv_json else ''))
    if out_prefix:
        pairs.to_csv(f'{out_prefix}_pairs.csv', index=False)
        cv.to_csv(f'{out_prefix}_cv.csv', index=False)
        click.echo(f'wrote {out_prefix}_pairs.csv, {out_prefix}_cv.csv')


if __name__ == '__main__':
    main()
