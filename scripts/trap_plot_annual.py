#!/usr/bin/env python
import os
import click
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages

from trap_plot import (
    get_heatmap, FLOWER_CODES, FRUIT_CODES,
    START_DATE, END_DATE,
)
from analyze_windowed_counts import shift_months


YEARS = 4
W_PER_Y = 52


def split_into_years(traps, dates, H, years=YEARS):
    Hs = np.vstack([
        H[:, (i * W_PER_Y):((i + 1) * W_PER_Y)]
        for i in range(years)
    ])
    ystart = int(dates[0].split('-')[0]) + 1
    ts = np.array(sum(
        [[f'{ystart+y} - {t}' for t in traps] for y in range(years)]
        , []
    ))
    ds = dates[:W_PER_Y]

    totals = np.sum(Hs, axis=1)
    idx = np.argsort(-totals)

    ts = ts[idx]
    Hs = Hs[idx]
    totals = totals[idx]

    good = (totals > 0)
    ts = ts[good]
    Hs = Hs[good]

    return ts, ds, Hs


@click.command()
@click.argument('inputfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
@click.option('-f', '--fruit', is_flag=True)
def main(inputfile, outputfile, fruit):

    df = pd.read_csv(inputfile, delimiter='\t')
    df = df.loc[df['fecha'] >= START_DATE]
    df = df.loc[df['fecha'] < END_DATE]
    if fruit:
        df = df.loc[df['part'].isin(FRUIT_CODES)]
    else:
        df = df.loc[df['part'].isin(FLOWER_CODES)]

    species = sorted(np.unique(df['sp']))
    dates = sorted(np.unique(df['fecha']))


    figs = []
    for sp in tqdm(species, 'Plotting'):
        traps, H = get_heatmap(df, sp, dates)
        if H is None: continue

        ts, ds, Hs = split_into_years(traps, dates, H)

        skip = 4
        date_labels = ds[::skip]
        date_ticks = np.arange(len(ds))[::skip]

        fig, ax = plt.subplots(figsize=(12, 8))
        figs.append(fig)
        ax.set_title(sp, fontsize=20)

        im = ax.imshow(
            Hs, cmap='plasma', interpolation='nearest',
            norm=LogNorm(vmin=1.0, vmax=1000.),
            extent=[0, 365, 0, 1],
        )
        ax.set_aspect('auto')
        #ax.set_yticks(np.arange(len(ts)))
        #ax.set_yticklabels(ts)
        ax.set_yticks([])
        ax.set_ylabel('Trap / Year', fontsize=16)
        ax.set_xlabel('Month', fontsize=16)

        #ax.set_xticks(date_ticks)
        #ax.set_xticklabels(date_labels, rotation=20)
        sep = pd.to_datetime(f'2021-09-01').dayofyear
        ticks = []
        for month in range(1, 13):
            ticks.append(pd.to_datetime(f'2021-{month:02d}-01').dayofyear)
        labels = [
            'Jan', 'Feb', 'Mar',
            'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep',
            'Oct', 'Nov', 'Dec',
        ]

        ticks, labels = shift_months(sep, ticks, labels)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)

        cbar = fig.colorbar(im)
        cbar.set_label('Quantity', fontsize=16)


    with PdfPages(outputfile) as pdf:
        for fig in tqdm(figs, 'Saving'):
            pdf.savefig(fig)


if __name__ == '__main__':
    main()
