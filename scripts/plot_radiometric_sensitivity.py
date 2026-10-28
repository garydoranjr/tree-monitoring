#!/usr/bin/env python
"""Plot the test-time offset sensitivity of one or more temporal-CV models, as
written by radiometric_sensitivity.py (--out-csv).

Draws the pooled held-out metrics against the offset k (in SDs of the fitted
Planet-MODIS offsets; one panel per metric, one line per model), and the
change in IoU from k = 0 by distance to the nearest training chip.

Typical usage:
    python scripts/plot_radiometric_sensitivity.py \\
        --run TOA=/Volumes/Earth03/flower/202610_temporal/sensitivity_toa.csv \\
        --run calibrated=/Volumes/Earth03/flower/202610_temporal_cal/sensitivity_cal.csv \\
        --plot figs/planet_temporal/radiometric_sensitivity.png
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import click
import matplotlib.pyplot as plt
import pandas as pd

from compare_temporal_cv import COLORS, INK, METRICS

BINS = ['0-3', '4-7', '8-14', '15-30', '>30']


@click.command()
@click.option('--run', 'runs', multiple=True, required=True, metavar='LABEL=CSV',
              help='A radiometric_sensitivity.py CSV, as label=path. Repeatable.')
@click.option('--plot', type=click.Path(dir_okay=False), required=True)
def main(runs, plot):
    data = {}
    for spec in runs:
        label, path = spec.split('=', 1)
        data[label] = pd.read_csv(path)
        if 0 not in set(data[label].k):
            raise click.ClickException(f'{label}: no k = 0 row')

    fig, axes = plt.subplots(1, len(METRICS) + 1, figsize=(4 * (len(METRICS) + 1), 3.4))
    for ax, (m, name) in zip(axes, METRICS.items()):
        for (label, d), c in zip(data.items(), COLORS):
            s = d[d.group == 'pooled'].sort_values('k')
            ax.plot(s.k, s[m], marker='o', ms=5, lw=2, color=c, label=label)
        ax.set_xticks(sorted({k for d in data.values() for k in d.k}))
        ax.set_xlabel('offset (SD of fitted offsets)', color=INK)
        ax.set_title(name, color=INK, fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)

    ax = axes[-1]
    ks = [k for k in sorted({k for d in data.values() for k in d.k}) if k != 0]
    bars = [(label, k) for label in data for k in ks]
    w = 0.8 / len(bars)
    for i, (label, k) in enumerate(bars):
        d = data[label]
        iou = d.set_index(['group', 'k'])['test_iou']
        vals = [iou[f'bin:{b}', k] - iou[f'bin:{b}', 0] for b in BINS]
        c = COLORS[list(data).index(label)]
        ax.bar([j + (i - (len(bars) - 1) / 2) * w for j in range(len(BINS))], vals,
               width=w, color=c, hatch='///' if k > 0 else None, edgecolor='white',
               lw=0, label=f'{label}, k = {k:+g}')
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(len(BINS)), BINS)
    ax.set_xlabel('days to nearest training chip', color=INK)
    ax.set_title('change in binary IoU from k = 0', color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#e6e6e3', lw=0.6)
    fig.tight_layout()
    fig.savefig(plot, dpi=150, bbox_inches='tight')
    click.echo(f'wrote {plot}')


if __name__ == '__main__':
    main()
