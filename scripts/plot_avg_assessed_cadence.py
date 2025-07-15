#!/usr/bin/env python
import os
import click
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from scipy.stats import scoreatpercentile
import matplotlib.pyplot as plt

THRESHOLD = 0.5
START = np.datetime64('2020-01-01')
END = np.datetime64('2024-01-01')

WINDOW = np.timedelta64(30, 'D')


def parse_date(f):
    parts = f.split('_')
    dt = '_'.join(parts[:2])
    return np.datetime64(datetime.strptime(dt, '%Y%m%d_%H%M%S'))


def get_rate_pcs(dates, vis, sample, pcs, years=[0, 1, 2]):
    periods = []
    for y in years:
        sy = sample + (y * np.timedelta64(365, 'D'))
        start = sy - WINDOW
        end = sy + WINDOW
        good = np.logical_and(start <= dates, dates <= end)
        vis_sub = vis[:, good]
        n_vis = np.maximum(np.sum(vis_sub, axis=1), 1)
        window_size = (2 * WINDOW).astype('timedelta64[D]').astype(float)
        periods.append(window_size / n_vis)
    return [
        scoreatpercentile(np.hstack(periods), p)
        for p in pcs
    ]


@click.command()
@click.argument('inputfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('plotfile', type=click.Path(
    path_type=Path, exists=False
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
def main(inputfile, plotfile, outputfile):

    data = np.load(inputfile)

    files = data['files']
    dates = np.array(list(map(parse_date, files)))
    V = data['values']
    V[np.isnan(V)] = 0

    good = np.logical_and(START <= dates, dates <= END)
    dates = dates[good]
    V = V[:, good]

    print(V.shape)

    vis = (V > THRESHOLD)

    samples = pd.date_range(
        start='2021-01-01',
        end='2021-12-31',
        periods=365,
    ).values

    pcs = [10., 50., 90.]
    #pcs = np.linspace(0.5, 99.5, 99)
    pc = np.vstack([
        get_rate_pcs(dates, vis, s, pcs)
        for s in tqdm(samples, 'Windows')
    ])

    np.savez_compressed(outputfile, dates=samples, cadence=pc, percentiles=pcs)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(
        samples, pc[:, 0], pc[:, 2], fc='red', ec='none',
        alpha=0.3,
    )
    ax.plot(samples, pc[:, 1], 'k-', lw=2)
    ax.set_ylim(0, 35)
    ax.set_xlim(np.datetime64('2021-01-01'), np.datetime64('2022-01-01'))
    ticks = []
    labels = []
    for month in range(1, 13):
        ticks.append(np.datetime64(f'2021-{month:02d}-01'))
    ax.set_xticks(ticks)
    ax.set_xticklabels([
        'Jan', 'Feb', 'Mar',
        'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep',
        'Oct', 'Nov', 'Dec',
    ])
    ax.set_yticks([0, 7, 14, 21, 28, 35])
    ax.set_xlabel('Date', fontsize=16)
    ax.set_ylabel('Average Days\nBetween Clear Observations', fontsize=16)
    plt.grid(color='gray', linestyle=':', linewidth=2)

    fig.savefig(plotfile)


if __name__ == '__main__':
    main()
