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

from arosics import COREG
from geoarray import GeoArray


def load_offsets(globalregistration):
    with open(globalregistration, 'r') as f:
        results = json.load(f)

    keys = [e['key'] for e in results]
    offsets = np.array([
        [e['x_offset'], e['y_offset']]
        for e in results
    ])

    return keys, offsets


def normalize(keys, offsets, reference_key):
    if reference_key not in keys:
        raise ValueError(f'Key {reference_key} not in offset list')

    i = keys.index(reference_key)
    return (offsets - offsets[i])


def extract_window(imagedir, outputdir, key, poly, offset, radius):
    imagefile = safe_join(imagedir, key + '.tif')
    outputfile = safe_join(outputdir, key + '.png')

    img = GeoArray(imagefile)

    xscale = 1.0 / img.geotransform[1]
    yscale = 1.0 / img.geotransform[5]
    xoff = -xscale * img.geotransform[0]
    yoff = -yscale * img.geotransform[3]

    # Translate from map to image pixels
    poly = poly.affine_transform([xscale, 0, 0, yscale, xoff, yoff])

    # Apply co-registration shift
    poly = poly.translate(xoff=-offset[0], yoff=-offset[1])

    height = 2*radius + 1
    width = 2*radius + 1

    topleft = poly.centroid.translate(xoff=-radius, yoff=-radius)
    row, col = (
        int(np.round(topleft.y.iloc[0])),
        int(np.round(topleft.x.iloc[0])),
    )

    if min(row, col) < 0:
        return

    # Apply subframe shift
    poly = poly.translate(xoff=-col, yoff=-row)

    subframe = np.array(img[row:row+height, col:col+height], dtype=float)
    maxval = np.max(subframe)
    if maxval == 0:
        return

    subframe *= (255 / maxval)
    subframe = subframe.astype(np.uint8)

    im = Image.fromarray(subframe)
    draw = ImageDraw.Draw(im)

    points = poly.iloc[0]
    points = list(poly.iloc[0].exterior.coords)

    draw.polygon(points, outline='red')

    im.save(outputfile)


@click.command()
@click.argument('droneregistration')
@click.argument('globalregistration')
@click.argument('shapefile')
@click.argument('imagedir')
@click.argument('outputdir')
@click.argument('crownid')
@click.option('-r', '--radius', type=int, default=25)
def main(droneregistration, globalregistration, shapefile, imagedir, outputdir, crownid, radius):

    crowns = gpd.read_file(shapefile)

    focal_crown = crowns.loc[crowns['tag'] == crownid]
    if len(focal_crown) != 1:
        raise ValueError(f'{len(focal_crown)} crowns found with id {crownid}')

    poly = focal_crown['geometry']

    keys, offsets = load_offsets(globalregistration)

    with open(droneregistration, 'r') as f:
        drone = json.load(f)

    offsets = normalize(keys, offsets, drone['planet_map'])
    drone_offset = np.array([
        drone['coreg_info']['corrected_shifts_px']['x'],
        drone['coreg_info']['corrected_shifts_px']['y'],
    ])

    for key, offset in tqdm(list(zip(keys, offsets))):
        total_offset = offset + drone_offset
        extract_window(imagedir, outputdir, key, poly, total_offset, radius)


if __name__ == '__main__':
    main()
