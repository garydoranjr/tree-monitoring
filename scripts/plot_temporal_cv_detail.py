#!/usr/bin/env python
"""Detail plots for a blocked temporal cross-validation scored by
eval_temporal_cv.py, complementing plot_temporal_cv.py's mAP@50 summary.

Writes three figures with the given output prefix (PNG + PDF each):

- <prefix>_metrics: mAP@50, binary IoU, crown precision and crown recall
  against training epoch, pooled over folds and per fold.
- <prefix>_bins: the same four metrics at the selected epoch, by the held-out
  chip's distance in days to the nearest training chip.
- <prefix>_chips: per-chip IoU, precision and recall at the selected epoch
  (left and right windows summed), against distance to training data and
  against acquisition date, coloured by fold.
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import json

import click
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = '#1a1a1a'
FOLD_GREY = '#b0b0b0'
GRID = '#e6e6e6'
# Fixed categorical order, one slot per fold; marker shape is a second cue.
FOLD_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
FOLD_MARKERS = ['o', 's', '^', 'D', 'v', 'P']
# Bin bar shades for the four metrics, one hue light to dark.
METRIC_SHADES = ['#0d366b', '#256abf', '#5598e7', '#9ec5f4']
ENVELOPE_DAYS = 30

METRICS = [
    ('test_map/map_50', 'mask mAP@50'),
    ('test_iou', 'binary IoU (clear pixels)'),
    ('test_event/precision', 'crown precision (IoU >= 0.5)'),
    ('test_event/recall', 'crown recall (IoU >= 0.5)'),
]
CHIP_METRICS = [
    ('iou', 'test_iou', 'binary IoU (clear pixels)'),
    ('precision', 'test_event/precision', 'crown precision (IoU >= 0.5)'),
    ('recall', 'test_event/recall', 'crown recall (IoU >= 0.5)'),
]


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, prefix, suffix):
    out = f'{prefix}_{suffix}.png'
    fig.savefig(out, dpi=200)
    fig.savefig(f'{prefix}_{suffix}.pdf')
    plt.close(fig)
    print(f'Wrote {out}')


def plot_metrics(cv, prefix):
    epochs = cv['epochs']
    per_epoch = cv['per_epoch']
    best = cv['best_epoch']
    folds = sorted(per_epoch[str(epochs[0])]['by_fold'], key=int)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, (metric, label) in zip(axes.flat, METRICS):
        for i, k in enumerate(folds):
            ax.plot(epochs, [per_epoch[str(e)]['by_fold'][k][metric]
                             for e in epochs],
                    color=FOLD_GREY, lw=1,
                    label='single fold' if i == 0 else None)
        ax.plot(epochs, [per_epoch[str(e)]['pooled'][metric] for e in epochs],
                color=INK, lw=2, marker='o', ms=5, label='pooled over folds')
        ax.axvline(best, color=INK, lw=1, ls='--')
        ax.set_xscale('log')
        ax.set_xticks(epochs)
        ax.set_xticklabels([str(e) for e in epochs])
        ax.minorticks_off()
        ax.set_ylabel(f'held-out {label}')
        style(ax)
    for ax in axes[1]:
        ax.set_xlabel('training epoch')
    axes[0, 0].legend(frameon=False, loc='upper left')
    axes[0, 0].annotate(f'selected: epoch {best}', xy=(best, 0),
                        xycoords=('data', 'axes fraction'), xytext=(4, 4),
                        textcoords='offset points', color=INK, fontsize=9)
    fig.tight_layout()
    save(fig, prefix, 'metrics')


def plot_bins(cv, prefix):
    best = cv['best_epoch']
    bins = cv['per_epoch'][str(best)]['by_day_bin']
    names = [b for b, _, _ in cv['day_bins'] if b in bins]
    outside = [lo > ENVELOPE_DAYS for b, lo, _ in cv['day_bins'] if b in bins]
    x = np.arange(len(names))
    width = 0.8 / len(METRICS)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    top = 0.0
    for j, ((metric, label), shade) in enumerate(zip(METRICS, METRIC_SHADES)):
        vals = np.array([bins[b][metric] for b in names], dtype=float)
        top = max(top, np.nanmax(vals))
        offs = x - 0.4 + width * (j + 0.5)
        # 2 px surface gap between adjacent bars.
        bars = ax.bar(offs, vals, width=width, color=shade, label=label,
                      edgecolor='white', linewidth=1)
        for bar, out in zip(bars, outside):
            if out:
                bar.set_hatch('///')
                bar.set_alpha(0.55)
    ax.set_ylim(0, top * 1.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={bins[b]['n_chips']} chips)" for b in names])
    ax.set_xlabel('days from held-out chip to nearest training chip '
                  f'(hatched: outside the {ENVELOPE_DAYS}-day envelope)')
    ax.set_ylabel(f'held-out value at epoch {best}')
    ax.legend(frameon=False, ncol=2, loc='upper right')
    style(ax)
    fig.tight_layout()
    save(fig, prefix, 'bins')


def chip_table(windows, epoch):
    w = windows[windows['epoch'] == epoch]
    chips = (w.groupby(['chip', 'fold', 'planet_date', 'days_to_nearest_train',
                        'day_bin'], as_index=False)
             [['tp', 'fp', 'fn', 'inter_px', 'union_px', 'n_gt']].sum())
    with np.errstate(invalid='ignore', divide='ignore'):
        chips['iou'] = chips['inter_px'] / chips['union_px']
        chips['precision'] = chips['tp'] / (chips['tp'] + chips['fp'])
        chips['recall'] = chips['tp'] / (chips['tp'] + chips['fn'])
    chips['planet_date'] = pd.to_datetime(chips['planet_date'])
    return chips


def plot_chips(cv, windows, prefix):
    best = cv['best_epoch']
    chips = chip_table(windows, best)
    pooled = cv['per_epoch'][str(best)]['pooled']
    bins = cv['per_epoch'][str(best)]['by_day_bin']
    # Pooled bin values are drawn across each bin's day range; the open-ended
    # bin stops at the furthest chip.
    dmax = chips['days_to_nearest_train'].max()
    spans = [(b, lo, min(hi, dmax)) for b, lo, hi in cv['day_bins'] if b in bins]

    # Totals over chips must reproduce the pooled event counts.
    for col, key in (('tp', 'test_event/tp'), ('fp', 'test_event/fp'),
                     ('fn', 'test_event/fn')):
        assert chips[col].sum() == pooled[key], (col, chips[col].sum(),
                                                 pooled[key])

    fig, axes = plt.subplots(len(CHIP_METRICS), 2, figsize=(12, 9),
                             sharey='row', gridspec_kw={'width_ratios': [1, 1.5]})
    folds = sorted(chips['fold'].unique())
    for row, (col, key, label) in enumerate(CHIP_METRICS):
        ax_d, ax_t = axes[row]
        for k in folds:
            c = chips[chips['fold'] == k]
            kw = dict(color=FOLD_COLORS[k], marker=FOLD_MARKERS[k], s=28,
                      edgecolor='white', linewidth=0.8, label=f'fold {k}')
            ax_d.scatter(c['days_to_nearest_train'], c[col], **kw)
            ax_t.scatter(c['planet_date'], c[col], **kw)
        for i, (b, lo, hi) in enumerate(spans):
            ax_d.hlines(bins[b][key], lo, hi, color=INK, lw=2,
                        label='pooled over bin' if i == 0 else None)
        ax_d.axvline(ENVELOPE_DAYS, color=INK, lw=1, ls='--')
        ax_t.axhline(pooled[key], color=INK, lw=1, ls=':',
                     label='pooled over all chips')
        ax_d.set_ylabel(f'per-chip {label}')
        for ax in (ax_d, ax_t):
            style(ax)
    axes[0, 0].annotate(f'{ENVELOPE_DAYS}-day envelope', xy=(ENVELOPE_DAYS, 1),
                        xycoords=('data', 'axes fraction'), xytext=(4, -12),
                        textcoords='offset points', color=INK, fontsize=9)
    axes[-1, 0].set_xlabel('days from held-out chip to nearest training chip')
    axes[-1, 1].set_xlabel('Planet acquisition date')
    for ax in axes[:, 1]:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    handles, labels = [], []
    for ax in (axes[0, 0], axes[0, 1]):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, frameon=False, loc='upper center',
               ncol=len(labels))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, prefix, 'chips')
    return chips


@click.command()
@click.argument('cvjson', type=click.Path(exists=True, dir_okay=False))
@click.argument('windowscsv', type=click.Path(exists=True, dir_okay=False))
@click.argument('outputprefix', type=click.Path(dir_okay=False))
def main(cvjson, windowscsv, outputprefix):
    with open(cvjson) as fh:
        cv = json.load(fh)
    windows = pd.read_csv(windowscsv)
    plot_metrics(cv, outputprefix)
    plot_bins(cv, outputprefix)
    chips = plot_chips(cv, windows, outputprefix)
    print(chips.groupby('fold')[['iou', 'precision', 'recall']].median()
          .round(3).to_string())


if __name__ == '__main__':
    main()
