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



def parse_date(f):
    parts = f.split('_')
    dt = '_'.join(parts[:2])
    return np.datetime64(datetime.strptime(dt, '%Y%m%d_%H%M%S'))


def get_rate_pcs(dates, vis, sample, half_window, years=[1, 2]):
    w = np.timedelta64(half_window, 'D') \
        if type(half_window) == int else half_window
    counts = []
    for y in years:
        sy = sample + (y * np.timedelta64(365, 'D'))
        start = sy - w
        end = sy + w
        good = np.logical_and(start <= dates, dates <= end)
        vis_sub = vis[:, good]
        #n_vis = np.maximum(np.sum(vis_sub, axis=1), 1)
        n_vis = np.sum(vis_sub, axis=1)
        counts.append(n_vis)

    return np.hstack(counts)


def setup_inputs(inputfile):
    data = np.load(inputfile)

    files = data['files']
    dates = np.array(list(map(parse_date, files)))
    V = data['values']
    V[np.isnan(V)] = 0

    good = np.logical_and(START <= dates, dates <= END)
    dates = dates[good]
    V = V[:, good]

    visible = (V > THRESHOLD)

    samples = pd.date_range(
        start='2021-01-01',
        end='2021-12-31',
        periods=365,
    ).values

    return dates, visible, samples



@click.command()
@click.argument('inputfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
@click.option('-w', '--halfwidth', default=5, type=int)
def main(inputfile, outputfile, halfwidth):

    dates, visible, samples = setup_inputs(inputfile)

    counts = np.vstack([
        get_rate_pcs(dates, visible, s, halfwidth)
        for s in tqdm(samples, 'Windows')
    ])

    half_window = np.timedelta64(halfwidth, 'D')
    window_size = (2 * half_window).astype('timedelta64[D]').astype(float)

    np.savez_compressed(outputfile, dates=samples, counts=counts, window_size=window_size)


if __name__ == '__main__':
    main()
