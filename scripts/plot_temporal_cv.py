#!/usr/bin/env python
"""Plot a blocked temporal cross-validation scored by eval_temporal_cv.py.

Left: the pooled held-out metric against training epoch, with each fold's
own curve behind it and the selected epoch marked. Right: the same metric at
the selected epoch, broken down by the held-out chip's distance in days to
the nearest training chip, with the number of chips in each bin.
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import json

import click
import matplotlib.pyplot as plt
import numpy as np

INK = '#1a1a1a'
FOLD_GREY = '#b0b0b0'

METRIC_LABELS = {
    'test_map/map_50': 'mask mAP@50',
    'test_map/map': 'mask mAP@[.5:.95]',
    'test_iou': 'binary IoU (clear pixels)',
    'test_event/precision': 'crown precision (IoU >= 0.5)',
    'test_event/recall': 'crown recall (IoU >= 0.5)',
    'test_event/accuracy': 'crown TP / (TP + FP + FN)',
}


@click.command()
@click.argument('cvjson', type=click.Path(exists=True, dir_okay=False))
@click.argument('outputfile', type=click.Path(dir_okay=False))
@click.option('--metric', default=None,
              help="Metric to plot; defaults to the JSON's select_metric.")
def main(cvjson, outputfile, metric):
    with open(cvjson) as fh:
        cv = json.load(fh)
    metric = metric or cv['select_metric']
    label = METRIC_LABELS.get(metric, metric)
    epochs = cv['epochs']
    per_epoch = cv['per_epoch']
    # The selected epoch is defined by select_metric, whatever is plotted.
    best = cv['best_epoch']

    fig, (ax_ep, ax_day) = plt.subplots(
        ncols=2, figsize=(11, 4.2), gridspec_kw={'width_ratios': [1.3, 1]},
    )

    folds = sorted(per_epoch[str(epochs[0])]['by_fold'], key=int)
    for i, k in enumerate(folds):
        ax_ep.plot(epochs, [per_epoch[str(e)]['by_fold'][k][metric]
                            for e in epochs],
                   color=FOLD_GREY, lw=1, label='single fold' if i == 0 else None)
    pooled = [per_epoch[str(e)]['pooled'][metric] for e in epochs]
    ax_ep.plot(epochs, pooled, color=INK, lw=2, marker='o', ms=5,
               label='pooled over folds')
    ax_ep.axvline(best, color=INK, lw=1, ls='--')
    ax_ep.annotate(f'selected: epoch {best}', xy=(best, 0),
                   xycoords=('data', 'axes fraction'), xytext=(4, 4),
                   textcoords='offset points', color=INK, fontsize=9)
    ax_ep.set_xscale('log')
    ax_ep.set_xticks(epochs)
    ax_ep.set_xticklabels([str(e) for e in epochs])
    ax_ep.minorticks_off()
    ax_ep.set_xlabel('training epoch')
    ax_ep.set_ylabel(f'held-out {label}')
    ax_ep.legend(frameon=False, loc='upper left')

    bins = per_epoch[str(best)]['by_day_bin']
    names = [b for b, _, _ in cv['day_bins'] if b in bins]
    x = np.arange(len(names))
    vals = [bins[b][metric] for b in names]
    ax_day.bar(x, vals, width=0.6, color=INK)
    ymax = max([v for v in vals if np.isfinite(v)] + [0]) or 1
    for xi, b, v in zip(x, names, vals):
        ax_day.text(xi, (v if np.isfinite(v) else 0) + 0.02 * ymax,
                    f"n={bins[b]['n_chips']}", ha='center', va='bottom',
                    fontsize=9, color=INK)
    ax_day.set_ylim(0, ymax * 1.15)
    ax_day.set_xticks(x)
    ax_day.set_xticklabels(names)
    ax_day.set_xlabel('days from held-out chip to nearest training chip')
    ax_day.set_ylabel(f'held-out {label} at epoch {best}')

    for ax in (ax_ep, ax_day):
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#e6e6e6', lw=0.8)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(outputfile, dpi=200)
    if outputfile.endswith('.png'):
        fig.savefig(os.path.splitext(outputfile)[0] + '.pdf')
    print(f'Wrote {outputfile}')


if __name__ == '__main__':
    main()
