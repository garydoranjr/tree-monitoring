#!/usr/bin/env python
import os
import json
import click
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import geopandas as gpd
from shutil import copy
from datetime import datetime


WINDOW = 2


def extract_date_from_path(path: str) -> datetime.date:
    """
    Extracts the YYYYMMDD date from the beginning of a filename in a path.
    """
    filename = os.path.basename(path)
    date_str = filename.split("_")[0]   # take first token before "_"
    return np.datetime64(datetime.strptime(date_str, "%Y%m%d").date(), 'D')


@click.command()
@click.argument('labelfile')
@click.argument('imagedir')
@click.argument('outputdir')
def main(labelfile, imagedir, outputdir):

    labels = gpd.read_file(labelfile, layer='flowering_dataset')
    labels = labels.loc[labels['status'] == 'Done']
    dates = pd.to_datetime(labels['date'].unique(), format='%Y_%m_%d').to_numpy()

    files = glob(os.path.join(imagedir, '*.png'))

    file_dates = [ extract_date_from_path(f) for f in files ]

    to_copy = []
    for f in files:
        fd = extract_date_from_path(f)
        mindiff = np.min(np.abs([
            (d - fd).astype('timedelta64[D]').astype(int) for d in dates
        ]))
        if mindiff <= WINDOW:
            to_copy.append(f)

    for tc in tqdm(to_copy, 'Copying'):
        dst = os.path.join(outputdir, os.path.basename(tc))
        copy(tc, dst)


if __name__ == '__main__':
    main()
