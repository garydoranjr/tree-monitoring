#!/usr/bin/env python
import os
import click
import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt


def get_peak(dates, counts, n):
    w = np.ones(n)
    conv = convolve1d(counts, w, mode='wrap')
    return dates[np.argmax(conv)]


@click.command()
@click.argument('inputfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
def main(inputfile, outputfile):
    data = np.load(inputfile, allow_pickle=True)
    dates = data['dates']
    all_counts = data['counts']
    all_species = data['species']
    all_years = data['years']
    all_traps = data['traps']

    data = []
    for species, traps, years, counts in zip(all_species, all_traps, all_years, all_counts):

        for trap, year, count in zip(traps, years, counts):
            l = np.sum(count > 0)
            if l > 0:
                peak = get_peak(dates, count, l)
                data.append({
                    'trap': trap,
                    'species': species,
                    'year': year,
                    'event_length': 7*l,
                    'event_peak': peak,
                })


    df_out = pd.DataFrame(data)
    df_out.to_csv(outputfile, index=False)



if __name__ == '__main__':
    main()
