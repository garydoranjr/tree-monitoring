#!/usr/bin/env python
import os
import click
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt


def parse_date(path):
    datestr = '_'.join(os.path.basename(path).split('_')[:2])
    return datetime.strptime(datestr, '%Y%m%d_%H%M%S')


def load_data(inputfile):

    dates = []
    pcs = []

    df = pd.read_csv(inputfile)
    for i, row in df.iterrows():
        dates.append(parse_date(row['File']))
        pcs.append(row['PercentClear'])

    return dates, pcs


@click.command()
@click.argument('inputfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
def main(inputfile, outputfile):

    dates, pcs = load_data(inputfile)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title('Usable Data in 50 ha Plot', fontsize=20)

    for d, p in zip(dates, pcs):
        ax.plot([d, d], [0, 1], '-', color='lightgray')
        ax.plot([d, d], [0, p], 'g-')

    ax.set_facecolor('red')

    ax.set_xlim(np.min(dates), np.max(dates))
    ax.set_ylim(0, 1)
    ax.set_xticks([
        datetime(year, 1, 1)
        for year in range(2020, 2025)
    ])

    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('Clear Fraction', fontsize=18)

    fig.savefig(outputfile, bbox_inches='tight')


if __name__ == '__main__':
    main()
