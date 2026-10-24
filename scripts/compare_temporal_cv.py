#!/usr/bin/env python
"""Compare two or more blocked temporal cross-validations scored by
eval_temporal_cv.py, e.g. the TOA chips with the per-image stretch against the
MODIS-calibrated chips with --fixed-stretch.

The runs must share folds, epochs and scoring settings (checked). Prints, per
run: the pooled metrics at every epoch, the selected epoch, and at that epoch
the breakdown by distance to training data and by fold. With --plot, draws the
pooled metrics against epoch (one panel per metric, one line per run) and the
selected-epoch metric by distance bin.

Typical usage:
    python scripts/compare_temporal_cv.py \\
        --run toa=$HOME/Documents/tree-flowering/202610_temporal/temporal_cv.json \\
        --run cal=$HOME/Documents/tree-flowering/202610_temporal_cal/temporal_cv.json \\
        --plot temporal_cv_toa_vs_cal.png
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import json

import click
import matplotlib.pyplot as plt
import pandas as pd

METRICS = {'test_map/map_50': 'mask mAP@50', 'test_iou': 'binary IoU',
           'test_event/precision': 'crown precision',
           'test_event/recall': 'crown recall'}
SHARED = ('folds', 'epochs', 'day_bins', 'score_thresh', 'iou_thresh',
          'min_instance_size', 'use_ocm_masks', 'size', 'select_metric')
INK = '#2b2b2b'
COLORS = ['#2a78d6', '#1baf7a', '#eb6834', '#8a5bd6']  # dataviz categorical order


@click.command()
@click.option('--run', 'runs', multiple=True, required=True, metavar='LABEL=JSON',
              help='A temporal_cv.json, as label=path. Repeatable.')
@click.option('--plot', type=click.Path(dir_okay=False), default=None)
def main(runs, plot):
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

    if plot:
        fig, axes = plt.subplots(1, len(METRICS) + 1, figsize=(4 * (len(METRICS) + 1), 3.4))
        for ax, (k, name) in zip(axes, METRICS.items()):
            for (label, d), c in zip(cv.items(), COLORS):
                s = pooled[pooled.run == label]
                ax.plot(s.epoch, s[k], marker='o', ms=4, lw=2, color=c, label=label)
                ax.plot(d['best_epoch'], s.set_index('epoch').loc[d['best_epoch'], k],
                        marker='o', ms=9, mfc='none', mec=c)
            ax.set_xscale('log')
            ax.set_xlabel('epoch', color=INK)
            ax.set_title(name, color=INK, fontsize=10)
        axes[0].legend(frameon=False, fontsize=8)
        ax = axes[-1]
        bins = [b[0] for b in first['day_bins']]
        w = 0.8 / len(cv)
        for i, ((label, d), c) in enumerate(zip(cv.items(), COLORS)):
            sel = d['per_epoch'][str(d['best_epoch'])]['by_day_bin']
            vals = [sel[b][first['select_metric']] for b in bins]
            ax.bar([j + (i - (len(cv) - 1) / 2) * w for j in range(len(bins))], vals,
                   width=w, color=c, label=f'{label} (e{d["best_epoch"]})')
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
