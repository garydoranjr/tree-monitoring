#!/usr/bin/env python
import os
import click
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def hist(x, title=None, yscale='linear'):
    fig, ax = plt.subplots(figsize=(16, 7))

    if title is not None:
        ax.set_title(title, fontsize=18)

    ax.hist(x, bins=np.linspace(0, 100, 11))

    ax.set_yscale(yscale)

    ax.set_ylabel('Count', fontsize=16)

    return fig


def pie(x, title=None):
    fig, ax = plt.subplots(figsize=(16, 7))

    if title is not None:
        ax.set_title(title, fontsize=18)

    counts = x.value_counts(normalize=True) * 100

    # Hide labels for slices below threshold
    threshold = 3  # percent
    labels = [
        name if pct >= threshold else "" 
        for name, pct in zip(counts.index, counts.values)
    ]

    ax.pie(counts, labels=labels, labeldistance=1.1)

    return fig


@click.command()
@click.argument('labelfile')
@click.argument('outputfile')
def main(labelfile, outputfile):

    labels = gpd.read_file(labelfile, layer='flowering_dataset')
    labels['polyid'] = labels['polygon_id'].apply(lambda i: i.split('_')[0])
    labels['easting'] = labels['geometry'].apply(lambda g: g.centroid.x)

    threshold = np.quantile(labels['easting'].values, 0.8)

    labels['split'] = labels['easting'].apply(
        lambda e: 'train' if e < threshold else 'test'
    )

    df_out = labels[['polygon_id', 'split']]
    df_out.to_csv(outputfile, index=False)


if __name__ == '__main__':
    main()
