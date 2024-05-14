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


@click.command()
@click.argument('inputfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
def main(inputfile, outputfile):

    dates = []
    pcs = []

    df = pd.read_csv(inputfile)
    for i, row in df.iterrows():
        dates.append(parse_date(row['File']))
        pcs.append(row['PercentClear'])

    for d, p in zip(dates, pcs):
        plt.plot([d, d], [0, 1], 'k-')
        plt.plot([d, d], [0, p], 'r-')

    plt.show()


if __name__ == '__main__':
    main()
