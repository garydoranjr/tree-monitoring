#!/usr/bin/env python
import os
import json
import click
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def get_stats(labelfile, crowns, info):
    labels = pd.read_csv(labelfile)

    tags = [
        int(crowns.iloc[int(l['best_idx'])]['tag'])
        for _, l in labels.iterrows()
        if l['best_scores'] != 0.0
    ]

    labeled = info.loc[info['tag'].isin(tags)]
    labeled = labeled.loc[~labeled['Latin'].isna()]

    return labeled['Latin'].values, labeled['crownArea'].values


def eff_radius(area):
    return np.sqrt(area / np.pi)


def plot_counter(counter, top_n=None, title='String Frequency'):
    """
    Plot a horizontal bar chart from a Counter object with string keys.

    Parameters:
    - counter: collections.Counter
    - top_n: int or None, number of top entries to display (if None, show all)
    - title: str, title of the plot
    """
    if top_n is not None:
        most_common = counter.most_common(top_n)
    else:
        most_common = counter.most_common()

    labels, values = zip(*most_common)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, min(8, max(4, len(labels) * 0.4))))  # auto-adjust height

    y_pos = range(len(labels))
    ax.barh(y_pos, values, color='skyblue', edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)

    ax.invert_yaxis()  # Highest frequency on top
    ax.set_xlabel('Frequency')
    ax.set_title(title)

    # Improve layout for long strings
    plt.tight_layout()
    return fig


def plot_radii(radii, title):

    fig, ax = plt.subplots(figsize=(8, 6))

    bins = np.linspace(0, 25, 26)
    ax.hist(radii, bins=bins, histtype='step', ec='k', lw=3)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_xlabel('Radius (m)', fontsize=14)

    return fig


@click.command()
@click.argument('labelfiles', nargs=-1)
@click.argument('shapefile')
@click.argument('infofile')
@click.argument('outputfile')
def main(labelfiles, shapefile, infofile, outputfile):

    crowns = gpd.read_file(shapefile)
    info = gpd.read_file(infofile)

    all_radii = []
    all_species = []

    for labelfile in labelfiles:
        species, areas = get_stats(labelfile, crowns, info)
        radii = eff_radius(areas)
        all_species.append(species)
        all_radii.append(eff_radius(areas))

    all_radii = np.hstack(all_radii)
    all_species = np.hstack(all_species)

    sp_counts = Counter(all_species)

    sp_fig = plot_counter(sp_counts, title='Species Frequency')
    ra_fig = plot_radii(all_radii, title='Radii Frequency')

    with PdfPages(outputfile) as pdf:
        pdf.savefig(sp_fig)
        pdf.savefig(ra_fig)


if __name__ == '__main__':
    main()
