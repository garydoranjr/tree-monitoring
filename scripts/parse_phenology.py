#!/usr/bin/env python
import os
import click
import warnings
import pandas as pd
from glob import glob
from tqdm import tqdm
from tabula import read_pdf


@click.command()
@click.argument('phenologyfile')
@click.argument('outputfile')
def main(phenologyfile, outputfile):

    warnings.simplefilter(action='ignore', category=FutureWarning)

    all_dfs = []
    for page in tqdm(list(range(7, 725, 2)), 'Parsing'):
        if page in (435, 439): continue
        dfs = read_pdf(
            phenologyfile, pages=[page], stream=True, guess=False,
            relative_area=True, area=[63.6, 13.5, 81.8, 82.4],
        )
        if len(dfs) != 1:
            print(page)
            print(dfs)

        all_dfs += dfs

    df = pd.concat(all_dfs, axis=0)
    df.to_csv(outputfile, index=False)


if __name__ == '__main__':
    main()
