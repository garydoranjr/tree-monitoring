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


FLOWER_CODES = [6, 9]
FRUIT_CODES = [1, 2, 3, 4, 7, 10]
START_DATE = '2020-09-01'
END_DATE = '2024-09-01'
YEARS = 4
W_PER_Y = 52


def get_heatmap(df, spcode, dates):
    df_s = df.loc[df['sp'] == spcode]

    sums = df_s.groupby('trap')['quantity'].sum().sort_values(ascending=False)
    traps = sums.index.values
    if len(traps) < 10:
        return None, None

    df_s = df_s.loc[df_s['trap'].isin(traps)]

    rows = []
    for t in traps:
        row = []
        for d in dates:
            r = df_s.loc[(df_s['trap'] == t) & (df_s['fecha'] == d)]
            if len(r) == 0:
                value = 0
            else:
                value = np.sum(r['quantity'])
            row.append(value)
        rows.append(row)

    H = np.array(rows)

    return traps, H


@click.command()
@click.argument('inputfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
def main(inputfile, outputfile):

    df = pd.read_csv(inputfile, delimiter='\t')
    df = df.loc[df['fecha'] >= START_DATE]
    df = df.loc[df['fecha'] < END_DATE]
    df = df.loc[df['part'].isin(FLOWER_CODES)]

    species = sorted(np.unique(df['sp']))
    dates = sorted(np.unique(df['fecha']))

    skip = 16
    date_labels = dates[::skip]
    date_ticks = np.arange(len(dates))[::skip]

    figs = []
    for sp in tqdm(species, 'Plotting'):
        traps, H = get_heatmap(df, sp, dates)
        if H is None: continue

        fig, ax = plt.subplots(figsize=(12, 8))
        figs.append(fig)
        ax.set_title(sp, fontsize=20)

        im = ax.imshow(H, cmap='plasma', norm=LogNorm(vmin=1.0, vmax=1000.))
        ax.set_aspect('auto')
        ax.set_yticks(np.arange(len(traps)))
        ax.set_yticklabels(traps)
        ax.set_ylabel('Trap', fontsize=16)
        ax.set_xlabel('Date', fontsize=16)

        ax.set_xticks(date_ticks)
        ax.set_xticklabels(date_labels, rotation=20)

        cbar = fig.colorbar(im)
        cbar.set_label('Quantity', fontsize=16)

    with PdfPages(outputfile) as pdf:
        for fig in tqdm(figs, 'Saving'):
            pdf.savefig(fig)


if __name__ == '__main__':
    main()
