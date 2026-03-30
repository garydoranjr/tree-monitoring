#!/usr/bin/env python
import os
import json
import click
import numpy as np
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
from scipy.stats import percentileofscore
from collections import defaultdict
import matplotlib.pyplot as plt
from werkzeug.security import safe_join

from evaluate_coreg import load_offsets, offset_getter


def effective_radius(crown):
    area = crown['area']
    return np.sqrt(area / np.pi)


def plot_hist(diameters, focal_diameter, title, tag):

    fig, ax = plt.subplots(figsize=(8, 4))

    bins = np.linspace(np.min(diameters), np.max(diameters), 20)

    ax.hist(diameters, bins=bins, fc='k', ec='none')

    ax.set_title(title, fontsize=18)

    if focal_diameter is not None:
        ymin, ymax = ax.get_ylim()
        ax.plot([focal_diameter, focal_diameter], [ymin, ymax], 'r-', lw=2, label=f'Tag {tag}')
        ax.legend(loc='upper right', fontsize=16)
        ax.set_ylim(ymin, ymax)

    ax.set_xlabel('Crown Diameter (pixels)', fontsize=16)
    ax.set_ylabel('Count', fontsize=16)

    return fig


@click.command()
@click.argument('shapefile')
@click.argument('labelfile')
@click.argument('coregfile')
@click.argument('outputfile')
@click.option('-m', '--mperpix', default=3.0)
def main(shapefile, labelfile, coregfile, outputfile, mperpix):

    crowns = gpd.read_file(shapefile)

    radii = defaultdict(list)
    for _, c in crowns.iterrows():
        radii[c['latin']].append(effective_radius(c))

    label_df = pd.read_csv(labelfile)
    offsets = load_offsets(coregfile)

    calc_offset = offset_getter(offsets)

    errors = []

    for i, row in label_df.iterrows():
        try:
            est = calc_offset(row)
        except KeyError: continue
        act = np.array([row['dx'], row['dy']])
        error = (est - act)
        errors.append(error)

    errors = np.array(errors)

    error_mag = mperpix * np.sqrt(np.sum(errors**2, axis=1))

    med_rad = {
        k: np.average(percentileofscore(error_mag, v)) for k, v in radii.items()
        if k is not None
    }

    species, d = zip(*sorted(med_rad.items(), key=lambda p: -p[1]))

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(d))
    ax.bar(x, np.array(d))
    ax.set_xticks(x)
    ax.set_xticklabels(species, rotation=90)
    ax.set_ylabel('Crown Localization Rate (%)', fontsize=14)
    ax.set_ylim(0, 100)
    plt.tight_layout()

    fig.savefig(outputfile, bbox_inches='tight')



if __name__ == '__main__':
    main()
