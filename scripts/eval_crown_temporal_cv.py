#!/usr/bin/env python
"""Score a temporal CV of train_crown_pse_ltae.py against the drone crown
labels, side by side with the persistence baseline on the same crowns.

For each fold run under RUNDIR (any subdirectory holding params.json and
predictions.csv), every withheld flight is scored on the crowns that:
- have a drone label on that flight;
- are at least --min-crown-area;
- have at least one valid (cloud-free) observation within the run's
  --label-days of the flight.

A crown's score is the mean p_event over those observations. Crowns
without a clear view are left out, so the model and persistence are scored
on exactly the same crown-flights. Persistence predicts each crown's event
from its label on the nearest training flight of the fold CSV, as in
persistence_baseline.py; crowns unlabelled there are left out of both.

Rows are pooled by the flight's distance to the nearest training flight of
the fold CSV (the bins of eval_temporal_cv.py) and by season (dry: January
to April). Each row reports:
- for the model at --thresh: precision, recall, F1 and area Jaccard (summed
  crown area that is an event in both / in either, as in
  persistence_baseline.py);
- the model's average precision over crowns;
- the model at the persistence's number of calls (top-k by score within
  the row);
- persistence on the same pairs.

With crown hold-out runs (--crown-folds), only the withheld crowns are
scored, and persistence (which reads the withheld labels) is a reference
only.

Typical usage:
    python scripts/eval_crown_temporal_cv.py RUNDIR \\
        --folds config/temporal_folds_20261012.csv --out-prefix RUNDIR/eval
"""
import json
import os
from glob import glob

import click
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from persistence_baseline import BINS, day_bin
from plot_crown_species_comparison import NC4S, load_drone
from train_crown_pse_ltae import flight_folds


def score_fold(run, fmap, drone, min_area):
    with open(os.path.join(run, 'params.json')) as fh:
        params = json.load(fh)
    pred = pd.read_csv(os.path.join(run, 'predictions.csv'),
                       parse_dates=['date'])
    if params['crown_folds']:
        pred = pred[pred['test_crown']]
    pred = pred[(pred['area_m2'] >= min_area) & (pred['tag'] != -9999)]
    test = [pd.Timestamp(f) for f in params['test_flights']]
    train = [f for f, k in fmap.items() if f not in test]
    events = drone.pivot_table(index='tag', columns='date', values='event',
                               aggfunc='first')
    rows = []
    for fl in test:
        near = pred[(pred['date'] - fl).abs() <= pd.Timedelta(days=params['label_days'])]
        p = near.groupby('tag').agg(p=('p_event', 'mean'), area=('area_m2', 'first'))
        src = min(train, key=lambda f: abs((f - fl).days))
        df = p.join(events[fl].rename('true')).join(events[src].rename('persist'))
        df = df.dropna(subset=['true', 'persist'])
        df['flight'] = fl
        df['days_to_nearest_train'] = abs((src - fl).days)
        df['fold'] = params['fold']
        rows.append(df.reset_index())
    return pd.concat(rows, ignore_index=True), params


def metrics(df, thresh):
    y = df['true'].astype(bool).values
    w = df['area'].values
    out = {'flights': df['flight'].nunique(), 'crowns': len(df),
           'n_true': int(y.sum())}

    def binary(name, yhat):
        tp = (y & yhat).sum()
        prec = tp / yhat.sum() if yhat.sum() else np.nan
        rec = tp / y.sum() if y.sum() else np.nan
        out[f'{name}_n_pred'] = int(yhat.sum())
        out[f'{name}_precision'] = prec
        out[f'{name}_recall'] = rec
        out[f'{name}_f1'] = 2 * prec * rec / (prec + rec) if tp else 0.0
        union = w[y | yhat].sum()
        out[f'{name}_area_jaccard'] = w[y & yhat].sum() / union if union else np.nan

    persist = df['persist'].astype(bool).values
    binary('model', df['p'].values >= thresh)
    out['model_ap'] = (average_precision_score(y, df['p']) if 0 < y.sum() < len(y)
                       else np.nan)
    k = persist.sum()
    topk = np.zeros(len(df), bool)
    topk[np.argsort(-df['p'].values, kind='stable')[:k]] = True
    binary('model_at_k', topk)
    binary('persist', persist)
    return out


@click.command()
@click.argument('rundir', type=click.Path(exists=True, file_okay=False))
@click.option('--folds', 'folds_csv', required=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option('--nc4', 'nc4s', multiple=True, default=NC4S, show_default=True)
@click.option('--min-crown-area', default=36.0, show_default=True)
@click.option('--thresh', default=0.5, show_default=True)
@click.option('--out-prefix', default=None,
              help='Write <prefix>_pairs.csv and <prefix>_summary.csv.')
def main(rundir, folds_csv, nc4s, min_crown_area, thresh, out_prefix):
    """Score every fold run under RUNDIR."""
    runs = sorted(os.path.dirname(p) for p in
                  glob(os.path.join(rundir, '**', 'params.json'), recursive=True))
    fmap = flight_folds(folds_csv)
    drone = load_drone(nc4s)
    parts = []
    for run in runs:
        df, params = score_fold(run, fmap, drone, min_crown_area)
        if params['fold'] == 'all':
            continue
        parts.append(df)
        click.echo(f'{run}: fold {params["fold"]}, {len(df)} crown-flights')
    if not parts:
        raise click.ClickException(f'no fold runs under {rundir}')
    pairs = pd.concat(parts, ignore_index=True)
    pairs['bin'] = pairs['days_to_nearest_train'].map(day_bin)
    pairs['season'] = np.where(pairs['flight'].dt.month <= 4, 'dry', 'wet')

    table = []
    for b, sub in pairs.groupby('bin'):
        table.append({'group': f'bin {b}', **metrics(sub, thresh)})
    for s, sub in pairs.groupby('season'):
        table.append({'group': f'season {s}', **metrics(sub, thresh)})
    table.append({'group': 'pooled', **metrics(pairs, thresh)})
    table = pd.DataFrame(table).set_index('group')
    order = ([f'bin {b[0]}' for b in BINS if f'bin {b[0]}' in table.index]
             + [g for g in table.index if not g.startswith('bin ')])
    table = table.loc[order]

    cols = ['flights', 'crowns', 'n_true',
            'model_precision', 'model_recall', 'model_f1', 'model_area_jaccard',
            'model_ap', 'model_at_k_precision', 'model_at_k_area_jaccard',
            'persist_n_pred', 'persist_precision', 'persist_recall',
            'persist_area_jaccard']
    with pd.option_context('display.float_format', '{:.3f}'.format,
                           'display.width', 250, 'display.max_columns', 30):
        click.echo(table[cols].to_string())
    if out_prefix:
        pairs.to_csv(f'{out_prefix}_pairs.csv', index=False)
        table.to_csv(f'{out_prefix}_summary.csv')
        click.echo(f'wrote {out_prefix}_pairs.csv, {out_prefix}_summary.csv')


if __name__ == '__main__':
    main()
