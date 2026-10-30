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

With crown hold-out runs (--crown-folds or --split-x), only the withheld
crowns are scored, and persistence (which reads the withheld labels) is a
reference only. A spatial run (--fold all with a crown hold-out) is scored
fold by fold as if it were a temporal CV, so its crown-flights, persistence
and day bins are the same as the temporal CV's. --split-x X --side east
restricts any run to the crowns east (or west) of x = X, e.g. to score the
temporal CV on the same crowns as a west-to-east spatial run.
--benchmark-chips adds a row for the flights of those chips (the Mask R-CNN
spatial benchmark: config/spatial_test_chips_56.txt).

The same rows are also split by event type, from the drone probabilities on
the withheld flight:
- flowering: p_flower > 0.5 and p_flower >= p_decid;
- leaf loss: p_decid > 0.5 and p_decid > p_flower;
- mixed: neither is above 0.5, but the combined event is.

Each type is scored on its own events plus every non-event. Precision
therefore counts every false call of either method, so only recall, AP and
the model-persistence gap compare across types. These rows go to
<prefix>_by_type.csv.

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
from train_crown_pse_ltae import crown_centroid_x, flight_folds


def load_run(run, min_area, split_x, side):
    """params.json and the scored predictions of one run."""
    with open(os.path.join(run, 'params.json')) as fh:
        params = json.load(fh)
    pred = pd.read_csv(os.path.join(run, 'predictions.csv'),
                       parse_dates=['date'])
    if params['crown_folds'] or params.get('split_x') is not None:
        pred = pred[pred['test_crown']]
    if split_x is not None:
        east = crown_centroid_x(params['series']) >= split_x
        pred = pred[east[pred['crown']] == (side == 'east')]
    pred = pred[(pred['area_m2'] >= min_area) & (pred['tag'] != -9999)]
    return params, pred


def drone_tables(drone):
    """tag x date tables of the drone event, p_flower and p_decid."""
    return {c: drone.pivot_table(index='tag', columns='date', values=c,
                                 aggfunc='first')
            for c in ('event', 'p_flower', 'p_decid')}


def score_fold(pred, params, fold, test, fmap, tables):
    """Score the test flights, with persistence from the other flights
    of the fold CSV."""
    train = [f for f in fmap if f not in test]
    events = tables['event']
    parts = {c: tables[c] for c in ('p_flower', 'p_decid')}
    rows = []
    for fl in test:
        near = pred[(pred['date'] - fl).abs() <= pd.Timedelta(days=params['label_days'])]
        p = near.groupby('tag').agg(p=('p_event', 'mean'), area=('area_m2', 'first'))
        src = min(train, key=lambda f: abs((f - fl).days))
        df = p.join(events[fl].rename('true')).join(events[src].rename('persist'))
        for c, tab in parts.items():
            df = df.join(tab[fl].rename(c))
        df = df.dropna(subset=['true', 'persist'])
        df['flight'] = fl
        df['days_to_nearest_train'] = abs((src - fl).days)
        df['fold'] = fold
        rows.append(df.reset_index())
    return pd.concat(rows, ignore_index=True)


def event_type(df):
    """flowering, leaf loss or mixed for true events; none otherwise."""
    t = np.where(df['p_flower'] >= df['p_decid'], 'flowering', 'leaf loss')
    t = np.where((df['p_flower'] <= 0.5) & (df['p_decid'] <= 0.5), 'mixed', t)
    return np.where(df['true'].astype(bool), t, 'none')


def grouped(pairs, thresh):
    """Metrics by day bin, by season, on the benchmark flights and pooled,
    in BINS order."""
    table = []
    for b, sub in pairs.groupby('bin'):
        table.append({'group': f'bin {b}', **metrics(sub, thresh)})
    for s, sub in pairs.groupby('season'):
        table.append({'group': f'season {s}', **metrics(sub, thresh)})
    if pairs['benchmark'].any():
        table.append({'group': 'benchmark flights',
                      **metrics(pairs[pairs['benchmark']], thresh)})
    table.append({'group': 'pooled', **metrics(pairs, thresh)})
    table = pd.DataFrame(table).set_index('group')
    order = ([f'bin {b[0]}' for b in BINS if f'bin {b[0]}' in table.index]
             + [g for g in table.index if not g.startswith('bin ')])
    return table.loc[order]


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
@click.option('--split-x', default=None, type=float,
              help='Score only the crowns on --side of this x (m, crown map '
                   'CRS).')
@click.option('--side', default='east', show_default=True,
              type=click.Choice(['east', 'west']))
@click.option('--benchmark-chips', default=None,
              type=click.Path(exists=True, dir_okay=False),
              help='Chip list whose flights get their own row.')
@click.option('--out-prefix', default=None,
              help='Write <prefix>_pairs.csv and <prefix>_summary.csv.')
def main(rundir, folds_csv, nc4s, min_crown_area, thresh, split_x, side,
         benchmark_chips, out_prefix):
    """Score every fold run under RUNDIR."""
    runs = sorted(os.path.dirname(p) for p in
                  glob(os.path.join(rundir, '**', 'params.json'), recursive=True))
    fmap = flight_folds(folds_csv)
    tables = drone_tables(load_drone(nc4s, parts=True))
    parts = []
    for run in runs:
        params, pred = load_run(run, min_crown_area, split_x, side)
        if params['fold'] != 'all':
            folds = {params['fold']: [pd.Timestamp(f)
                                      for f in params['test_flights']]}
        elif params['crown_folds'] or params.get('split_x') is not None:
            folds = {k: [f for f, j in fmap.items() if j == k]
                     for k in sorted(set(fmap.values()))}
        else:
            continue
        df = pd.concat([score_fold(pred, params, k, test, fmap, tables)
                        for k, test in folds.items()], ignore_index=True)
        parts.append(df)
        click.echo(f'{run}: fold {params["fold"]}, {len(df)} crown-flights')
    if not parts:
        raise click.ClickException(f'no fold runs under {rundir}')
    pairs = pd.concat(parts, ignore_index=True)
    pairs['bin'] = pairs['days_to_nearest_train'].map(day_bin)
    pairs['season'] = np.where(pairs['flight'].dt.month <= 4, 'dry', 'wet')
    pairs['event_type'] = event_type(pairs)
    bench = set()
    if benchmark_chips:
        chips = pd.read_csv(folds_csv).set_index('chip')['drone_dates']
        bench = {pd.Timestamp(str(chips[c]).split(';')[0])
                 for c in open(benchmark_chips).read().split()}
    pairs['benchmark'] = pairs['flight'].isin(bench)

    table = grouped(pairs, thresh)
    negatives = pairs['event_type'] == 'none'
    by_type = pd.concat(
        {t: grouped(pairs[negatives | (pairs['event_type'] == t)], thresh)
         for t in ('flowering', 'leaf loss', 'mixed')},
        names=['event_type'])

    cols = ['flights', 'crowns', 'n_true',
            'model_precision', 'model_recall', 'model_f1', 'model_area_jaccard',
            'model_ap', 'model_at_k_precision', 'model_at_k_area_jaccard',
            'persist_n_pred', 'persist_precision', 'persist_recall',
            'persist_area_jaccard']
    with pd.option_context('display.float_format', '{:.3f}'.format,
                           'display.width', 250, 'display.max_columns', 30):
        click.echo(table[cols].to_string())
        type_cols = ['crowns', 'n_true', 'persist_recall', 'model_at_k_recall',
                     'persist_precision', 'model_at_k_precision',
                     'persist_area_jaccard', 'model_at_k_area_jaccard',
                     'model_recall', 'model_ap']
        click.echo('\nby event type (each scored against all non-events):')
        click.echo(by_type[type_cols].to_string())
    if out_prefix:
        pairs.to_csv(f'{out_prefix}_pairs.csv', index=False)
        table.to_csv(f'{out_prefix}_summary.csv')
        by_type.to_csv(f'{out_prefix}_by_type.csv')
        click.echo(f'wrote {out_prefix}_pairs.csv, {out_prefix}_summary.csv, '
                   f'{out_prefix}_by_type.csv')


if __name__ == '__main__':
    main()
