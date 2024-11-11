#!/usr/bin/env python
import os
import json
import click
import numpy as np
from tqdm import tqdm
import geopandas as gpd
from werkzeug.security import safe_join

from geoarray import GeoArray
from coreg_crown_sequence import load_offsets, normalize


def extract_score(imagedir, key, poly, offset, radius):
    ndvi_key = '_'.join(key.split('_')[:-1]) + '_ndvi'
    imagefile = safe_join(imagedir, ndvi_key + '.tif')

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
    subframe[subframe == 0] = np.nan

    if np.all(np.isnan(subframe)):
        return

    return np.nanmean(subframe)


@click.command()
@click.argument('droneregistration')
@click.argument('globalregistration')
@click.argument('shapefile')
@click.argument('imagedir')
@click.argument('crownid')
@click.argument('outputfile')
@click.option('-r', '--radius', type=int, default=25)
def main(droneregistration, globalregistration, shapefile, imagedir, crownid, outputfile, radius):

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

    scores = {}
    for key, offset in tqdm(list(zip(keys, offsets))):
        total_offset = offset + drone_offset
        try:
            score = extract_score(imagedir, key, poly, total_offset, radius)
            if score is None: continue
            scores[key] = score
        except ValueError:
            continue

    with open(outputfile, 'w') as f:
        json.dump(scores, f, indent=2)


if __name__ == '__main__':
    main()
