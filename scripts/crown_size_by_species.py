#!/usr/bin/env python
import os
import json
import click
import numpy as np
from tqdm import tqdm
import geopandas as gpd
from collections import defaultdict
import matplotlib.pyplot as plt
from werkzeug.security import safe_join


def effective_radius(crown):
    poly = crown['geometry']
    return np.sqrt(poly.area / np.pi)


def effective_diam(crown, mperpix):
    return 2.0 * effective_radius(crown) / mperpix


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
@click.argument('outputfile')
@click.option('-m', '--mperpix', default=3.0)
def main(shapefile, outputfile, mperpix):

    crowns = gpd.read_file(shapefile)

    diameters = defaultdict(list)
    for _, c in crowns.iterrows():
        diameters[c['latin']].append(effective_diam(c, mperpix))

    med_diam = {
        k: np.median(v) for k, v in diameters.items()
        if k is not None
    }

    species, d = zip(*sorted(med_diam.items(), key=lambda p: -p[1]))

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(d))
    ax.bar(x, np.array(d) / 2.0)
    ax.set_xticks(x)
    ax.set_xticklabels(species, rotation=90)
    ax.set_ylabel('Crown Radius (m)', fontsize=14)
    plt.tight_layout()

    fig.savefig(outputfile, bbox_inches='tight')



if __name__ == '__main__':
    main()
