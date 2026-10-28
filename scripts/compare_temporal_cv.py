#!/usr/bin/env python
"""Compare two or more blocked temporal cross-validations scored by
eval_temporal_cv.py, e.g. the TOA chips with the per-image stretch against the
MODIS-calibrated chips with --fixed-stretch.

The runs must share folds, epochs and scoring settings (checked). Prints, per
run: the pooled metrics at every epoch, the selected epoch, and at that epoch
the breakdown by distance to training data and by fold. With --plot, draws the
pooled metrics against epoch (one panel per metric, one line per run) and the
selected-epoch metric by distance bin.

With --windows (each run's temporal_cv_windows.csv, the per-window rows of
eval_temporal_cv.py --chip-csv), it also pools IoU, crown precision and crown
recall at the comparison epoch over strata of chips: overall, by distance bin
and, with --chip-meta (a chips.csv from build_calibrated_chips.py), by the
scene's offset source (own MODIS fit or borrowed). Each stratum gets a paired
chip-bootstrap 95 % interval of the difference between the last and the first
run. mAP is not decomposable from the window counts, so it is left out there.

Typical usage:
    python scripts/compare_temporal_cv.py \\
        --run toa=$HOME/Documents/tree-flowering/202610_temporal/temporal_cv.json \\
        --run cal=$HOME/Documents/tree-flowering/202610_temporal_cal/temporal_cv.json \\
        --plot temporal_cv_toa_vs_cal.png \\
        --windows toa=$HOME/Documents/tree-flowering/202610_temporal/temporal_cv_windows.csv \\
        --windows cal=$HOME/Documents/tree-flowering/202610_temporal_cal/temporal_cv_windows.csv \\
        --chip-meta /Volumes/Earth03/flower/20261012_full_label_extended2_mkl_x4_coreg_4band_cal_curated/chips.csv
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import json

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = {'test_map/map_50': 'mask mAP@50', 'test_iou': 'binary IoU',
           'test_event/precision': 'crown precision',
           'test_event/recall': 'crown recall'}
SHARED = ('folds', 'epochs', 'day_bins', 'score_thresh', 'iou_thresh',
          'min_instance_size', 'use_ocm_masks', 'size', 'select_metric')
INK = '#2b2b2b'
COLORS = ['#2a78d6', '#1baf7a', '#eb6834', '#8a5bd6']  # dataviz categorical order
COUNTS = ['inter_px', 'union_px', 'tp', 'fp', 'fn']


def pooled_counts(c):
    """IoU, precision and recall from summed counts; c is (..., 5) in COUNTS order."""
    inter, union, tp, fp, fn = np.moveaxis(c, -1, 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return inter / union, tp / (tp + fp), tp / (tp + fn)


def strata_table(windows, meta, epoch, n_boot, seed=0):
    """Pooled IoU / precision / recall per run and stratum at `epoch`, with a
    paired chip-bootstrap 95 % interval of last run minus first run."""
    labels = list(windows)
    per = []
    for label, w in windows.items():
        w = w[w.epoch == epoch]
        if w.empty:
            raise click.ClickException(f'{label}: no window rows at epoch {epoch}')
        per.append(w.groupby('chip')[COUNTS].sum().add_suffix(f'_{label}'))
    chips = pd.concat(per, axis=1, join='inner')
    first = next(iter(windows.values()))
    info = first[first.epoch == epoch].groupby('chip')[['day_bin']].first()
    chips = chips.join(info)
    chips['all'] = 'all'
    keys = ['all', 'day_bin']
    if meta is not None:
        chips = chips.join(meta.set_index('chip')[['offset_source']])
        if chips.offset_source.isna().any():
            raise click.ClickException('--chip-meta lacks some evaluated chips')
        keys.append('offset_source')
    rng = np.random.default_rng(seed)
    rows = []
    for key in keys:
        for value, g in chips.groupby(key, sort=False):
            r = {'stratum': key, 'value': value, 'chips': len(g)}
            arrs = {lab: g[[f'{k}_{lab}' for k in COUNTS]].to_numpy(float) for lab in labels}
            for lab in labels:
                iou, p, rec = pooled_counts(arrs[lab].sum(axis=0))
                r.update({f'iou_{lab}': iou, f'precision_{lab}': p, f'recall_{lab}': rec})
            if n_boot and len(labels) > 1:
                idx = rng.integers(0, len(g), size=(n_boot, len(g)))
                a = pooled_counts(arrs[labels[0]][idx].sum(axis=1))
                b = pooled_counts(arrs[labels[-1]][idx].sum(axis=1))
                for name, x, y in zip(('iou', 'precision', 'recall'), a, b):
                    lo, hi = np.nanpercentile(y - x, [2.5, 97.5])
                    r[f'd_{name}_95'] = f'[{lo:+.3f}, {hi:+.3f}]'
            rows.append(r)
    return pd.DataFrame(rows)


@click.command()
@click.option('--run', 'runs', multiple=True, required=True, metavar='LABEL=JSON',
              help='A temporal_cv.json, as label=path. Repeatable.')
@click.option('--plot', type=click.Path(dir_okay=False), default=None)
@click.option('--style', 'styles', multiple=True, metavar='LABEL=COLOR[:LINESTYLE]',
              help='Colour and matplotlib line style (solid, dashed, dotted) of a '
                   'run in --plot, e.g. nojit=#2a78d6:dashed; non-solid runs get '
                   'hatched bars. Default: the categorical order, solid. Repeatable.')
@click.option('--windows', 'window_csvs', multiple=True, metavar='LABEL=CSV',
              help="A run's temporal_cv_windows.csv, as label=path, with the "
                   'same labels as --run. Repeatable.')
@click.option('--chip-meta', type=click.Path(exists=True, dir_okay=False), default=None,
              help='chips.csv from build_calibrated_chips.py, for the '
                   'offset-source strata.')
@click.option('--epoch', 'cmp_epoch', type=int, default=None,
              help='Epoch for the strata table (default: the first run\'s '
                   'selected epoch).')
@click.option('--n-boot', default=2000, show_default=True,
              help='Paired chip-bootstrap resamples for the strata intervals.')
def main(runs, plot, styles, window_csvs, chip_meta, cmp_epoch, n_boot):
    cv = {}
    for spec in runs:
        label, path = spec.split('=', 1)
        with open(path) as fh:
            cv[label] = json.load(fh)
    first = next(iter(cv.values()))
    for label, d in cv.items():
        diff = [k for k in SHARED if d.get(k) != first.get(k)]
        if diff:
            raise click.ClickException(f'{label} differs from the first run in {diff}')
    epochs = [str(e) for e in first['epochs']]

    rows = []
    for label, d in cv.items():
        for e in epochs:
            p = d['per_epoch'][e]['pooled']
            rows.append(dict(run=label, epoch=int(e), **{k: p[k] for k in METRICS}))
    pooled = pd.DataFrame(rows)
    with pd.option_context('display.float_format', '{:.4f}'.format, 'display.width', 160):
        click.echo('pooled held-out metrics by epoch:')
        click.echo(pooled.pivot(index='epoch', columns='run').to_string())
        for label, d in cv.items():
            best = str(d['best_epoch'])
            sel = d['per_epoch'][best]
            click.echo(f'\n{label}: selected epoch {best} by {d["select_metric"]}')
            for part in ('by_day_bin', 'by_fold'):
                t = pd.DataFrame(sel[part]).T
                cols = ['n_chips'] + [k for k in METRICS if k in t]
                click.echo(t[cols].to_string())

    if window_csvs:
        windows = {}
        for spec in window_csvs:
            label, path = spec.split('=', 1)
            windows[label] = pd.read_csv(path)
        epoch = cmp_epoch if cmp_epoch is not None else first['best_epoch']
        meta = pd.read_csv(chip_meta) if chip_meta else None
        t = strata_table(windows, meta, epoch, n_boot)
        labels = list(windows)
        with pd.option_context('display.float_format', '{:.3f}'.format, 'display.width', 200):
            click.echo(f'\npooled over chip strata at epoch {epoch}'
                       + (f'; 95 % intervals are {labels[-1]} - {labels[0]}, '
                          f'paired chip bootstrap ({n_boot})' if len(labels) > 1 else '') + ':')
            click.echo(t.to_string(index=False))

    if plot:
        style = {label: (c, 'solid') for label, c in zip(cv, COLORS)}
        for spec in styles:
            label, val = spec.split('=', 1)
            color, _, ls = val.partition(':')
            style[label] = (color, ls or 'solid')
        missing = [label for label in cv if label not in style]
        if missing:
            raise click.ClickException(f'no colour for {missing}; pass --style')
        hatch = {'solid': None, 'dashed': '///', 'dotted': '...'}
        fig, axes = plt.subplots(1, len(METRICS) + 1, figsize=(4 * (len(METRICS) + 1), 3.4))
        for ax, (k, name) in zip(axes, METRICS.items()):
            for label, d in cv.items():
                c, ls = style[label]
                s = pooled[pooled.run == label]
                ax.plot(s.epoch, s[k], marker='o', ms=4, lw=2, color=c, ls=ls, label=label)
                ax.plot(d['best_epoch'], s.set_index('epoch').loc[d['best_epoch'], k],
                        marker='o', ms=9, mfc='none', mec=c)
            ax.set_xscale('log')
            ax.set_xlabel('epoch', color=INK)
            ax.set_title(name, color=INK, fontsize=10)
        axes[0].legend(frameon=False, fontsize=8)
        ax = axes[-1]
        bins = [b[0] for b in first['day_bins']]
        w = 0.8 / len(cv)
        for i, (label, d) in enumerate(cv.items()):
            c, ls = style[label]
            sel = d['per_epoch'][str(d['best_epoch'])]['by_day_bin']
            vals = [sel[b][first['select_metric']] for b in bins]
            ax.bar([j + (i - (len(cv) - 1) / 2) * w for j in range(len(bins))], vals,
                   width=w, color=c, hatch=hatch.get(ls), edgecolor='white', lw=0,
                   label=f'{label} (e{d["best_epoch"]})')
        ax.set_xticks(range(len(bins)), bins)
        ax.set_xlabel('days to nearest training chip', color=INK)
        ax.set_title(f'{METRICS.get(first["select_metric"], first["select_metric"])} '
                     'at selected epoch', color=INK, fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        for ax in axes:
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', color='#e6e6e3', lw=0.6)
        fig.tight_layout()
        fig.savefig(plot, dpi=150)
        click.echo(f'\nwrote {plot}')


if __name__ == '__main__':
    main()
