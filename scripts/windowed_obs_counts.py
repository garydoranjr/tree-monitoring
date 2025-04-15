#!/usr/bin/env python
import os
import click
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from scipy.stats import scoreatpercentile

THRESHOLD = 0.5
START = np.datetime64('2020-01-01')
END = np.datetime64('2024-01-01')

WINDOW = np.timedelta64(30, 'D')


def parse_date(f):
    parts = f.split('_')
    dt = '_'.join(parts[:2])
    return np.datetime64(datetime.strptime(dt, '%Y%m%d_%H%M%S'))


def get_rate_pcs(dates, vis, sample, years=[0, 1, 2]):
    counts = []
    for y in years:
        sy = sample + (y * np.timedelta64(365, 'D'))
        start = sy - WINDOW
        end = sy + WINDOW
        good = np.logical_and(start <= dates, dates <= end)
        vis_sub = vis[:, good]
        n_vis = np.maximum(np.sum(vis_sub, axis=1), 1)
        counts.append(n_vis)

    return np.hstack(counts)


@click.command()
@click.argument('inputfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
def main(inputfile, outputfile):

    data = np.load(inputfile)

    files = data['files']
    dates = np.array(list(map(parse_date, files)))
    V = data['values']
    V[np.isnan(V)] = 0

    good = np.logical_and(START <= dates, dates <= END)
    dates = dates[good]
    V = V[:, good]

    vis = (V > THRESHOLD)

    samples = pd.date_range(
        start='2021-01-01',
        end='2021-12-31',
        periods=365,
    ).values

    counts = np.vstack([
        get_rate_pcs(dates, vis, s)
        for s in tqdm(samples, 'Windows')
    ])

    window_size = (2 * WINDOW).astype('timedelta64[D]').astype(float)

    np.savez_compressed(outputfile, dates=samples, counts=counts, window_size=window_size)


if __name__ == '__main__':
    main()
