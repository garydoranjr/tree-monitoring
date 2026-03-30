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

from trap_plot import get_heatmap, FLOWER_CODES, FRUIT_CODES, START_DATE, END_DATE
from trap_plot_annual import split_into_years


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


    all_Hs = []
    all_traps = []
    all_years = []
    all_species = []

    for sp in tqdm(species, 'Calculating'):
        traps, H = get_heatmap(df, sp, dates)
        if H is None: continue

        ts, ds, Hs = split_into_years(traps, dates, H)

        years = np.array([ e.split(' - ')[0] for e in ts ], dtype=int)
        tags = np.array([ e.split(' - ')[1] for e in ts ], dtype=int)

        all_Hs.append(Hs)
        all_traps.append(tags)
        all_years.append(years)
        all_species.append(sp)


    results = {
        'counts': np.array(all_Hs, dtype=object),
        'traps': np.array(all_traps, dtype=object),
        'years': np.array(all_years, dtype=object),
        'dates': ds,
        'species': np.array(all_species),
    }

    np.savez_compressed(outputfile, **results)



if __name__ == '__main__':
    main()
