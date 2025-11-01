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
    threshold = 1  # percent
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

    labels = labels.loc[labels['status'] == 'Done']
    labels = labels.drop([
        'geometry',
        'score',
        'iou',
        'tag',
        'isFlowerin',
        'floweringI',
        'area',
    ], axis=1)
    print(len(labels))

    figs = []

    figs.append(hist(labels['leafing'], title='Leafing', yscale='log'))
    figs.append(hist(labels['floweringIntensity'], title='Flowering Intensity'))

    figs.append(pie(labels['isFlowering'], title='Flowering?'))
    figs.append(pie(labels['isFruiting'], title='Fruiting?'))
    figs.append(pie(labels['newLeaves'], title='New Leaves?'))
    figs.append(pie(labels['latin'], title='Species'))

    with PdfPages(outputfile) as pdf:
        for fig in tqdm(figs, 'Saving'):
            pdf.savefig(fig)


if __name__ == '__main__':
    main()
