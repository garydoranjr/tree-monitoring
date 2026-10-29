#!/usr/bin/env python
"""Compare temporal-CV runs at matched numbers of detections.

eval_temporal_cv.py keeps every instance the model returns (score >= the
model's own score_thresh, 0.05), so pooled IoU, crown precision and crown
recall depend on how many detections a checkpoint emits, which falls with
training epoch. A run that emits fewer detections looks better on IoU and
precision without ranking crowns any better. This script pools each run's
temporal_cv_windows.csv per epoch and plots IoU, precision and recall against
the total number of detections (one line per run, one marker per epoch), and
prints each metric interpolated (linearly in log detections, between epochs)
at common detection counts.

Typical usage:
    F=/Volumes/Earth03/flower
    python scripts/plot_temporal_cv_detections.py \\
        --windows toa=$F/202610_temporal/temporal_cv_windows.csv \\
        --windows cal=$F/202610_temporal_cal/temporal_cv_windows.csv \\
        --plot figs/planet_temporal/temporal_cv_detections.png
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_temporal_cv import COLORS, INK, pooled_counts

COUNTS = ['n_pred', 'inter_px', 'union_px', 'tp', 'fp', 'fn']
NAMES = {'iou': 'binary IoU', 'precision': 'crown precision', 'recall': 'crown recall'}


def by_epoch(windows, min_epoch):
    """Pooled detections, IoU, precision and recall per epoch."""
    g = windows[windows.epoch >= min_epoch].groupby('epoch')[COUNTS].sum()
    iou, prec, rec = pooled_counts(g[['inter_px', 'union_px', 'tp', 'fp', 'fn']].to_numpy())
    return pd.DataFrame({'n_pred': g.n_pred, 'iou': iou, 'precision': prec,
                         'recall': rec}, index=g.index)


@click.command()
@click.option('--windows', 'window_csvs', multiple=True, required=True, metavar='LABEL=CSV',
              help="A run's temporal_cv_windows.csv, as label=path. Repeatable.")
@click.option('--style', 'styles', multiple=True, metavar='LABEL=COLOR[:LINESTYLE]',
              help='Colour and line style of a run, as in compare_temporal_cv.py.')
@click.option('--at', 'at', default='20000,15000,12000,10000,8000', show_default=True,
              help='Detection counts to interpolate the printed table at.')
@click.option('--min-epoch', default=5, show_default=True,
              help='Leave out earlier epochs, whose detection counts are far apart.')
@click.option('--plot', type=click.Path(dir_okay=False))
def main(window_csvs, styles, at, min_epoch, plot):
    runs = {}
    for spec in window_csvs:
        label, path = spec.split('=', 1)
        runs[label] = by_epoch(pd.read_csv(path), min_epoch)

    targets = [int(t) for t in at.split(',')]
    rows = {}
    for label, d in runs.items():
        x = np.log(d.n_pred.to_numpy())[::-1]
        row = {}
        for t in targets:
            if x.min() <= np.log(t) <= x.max():
                for m in NAMES:
                    row[(t, m)] = np.interp(np.log(t), x, d[m].to_numpy()[::-1])
        rows[label] = row
    with pd.option_context('display.float_format', '{:.3f}'.format, 'display.width', 200):
        click.echo('pooled detections by epoch:')
        click.echo(pd.DataFrame({k: d.n_pred for k, d in runs.items()}).to_string())
        click.echo('\nmetrics at matched detection counts (blank: outside the run\'s range):')
        click.echo(pd.DataFrame(rows).T.to_string(na_rep=''))

    if plot:
        style = {label: (c, 'solid') for label, c in zip(runs, COLORS)}
        for spec in styles:
            label, val = spec.split('=', 1)
            color, _, ls = val.partition(':')
            style[label] = (color, ls or 'solid')
        missing = [label for label in runs if label not in style]
        if missing:
            raise click.ClickException(f'no colour for {missing}; pass --style')
        fig, axes = plt.subplots(1, len(NAMES), figsize=(4 * len(NAMES), 3.4))
        for ax, (m, name) in zip(axes, NAMES.items()):
            for label, d in runs.items():
                c, ls = style[label]
                ax.plot(d.n_pred, d[m], marker='o', ms=4, lw=2, color=c, ls=ls, label=label)
            ax.set_xscale('log')
            ax.xaxis.set_minor_locator(plt.NullLocator())
            ticks = [5000, 7500, 10000, 15000, 20000, 30000, 45000]
            ax.set_xticks(ticks, [f'{t / 1000:g}k' for t in ticks])
            ax.set_xlabel('detections, all held-out windows', color=INK)
            ax.set_title(name, color=INK, fontsize=10)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', color='#e6e6e3', lw=0.6)
        axes[-1].legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(1.0, 1.0))
        fig.tight_layout()
        fig.savefig(plot, dpi=150, bbox_inches='tight')
        click.echo(f'wrote {plot}')


if __name__ == '__main__':
    main()
