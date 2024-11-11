#!/usr/bin/env python
import os
import json
import click
import numpy as np
from tqdm import tqdm
import geopandas as gpd
from PIL import Image, ImageDraw
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
@click.argument('species')
@click.argument('outputfile')
@click.option('-c', '--crownid', default=None)
@click.option('-m', '--mperpix', default=3.0)
def main(shapefile, species, outputfile, crownid, mperpix):

    crowns = gpd.read_file(shapefile)
    relevant = crowns.loc[crowns['latin'] == species]

    diameters = [
        effective_diam(r, mperpix) for _, r in relevant.iterrows()
    ]

    if crownid is not None:
        focal_crown = crowns.loc[crowns['tag'] == crownid]
        if len(focal_crown) != 1:
            raise ValueError(f'{len(focal_crown)} crowns found with id {crownid}')
        focal_diameter = float(effective_diam(focal_crown, mperpix).iloc[0])
    else:
        focal_diameter = None

    fig = plot_hist(diameters, focal_diameter, species, crownid)

    fig.savefig(outputfile, bbox_inches='tight')



if __name__ == '__main__':
    main()
