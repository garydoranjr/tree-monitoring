#!/usr/bin/env python
import os
import json
import click
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from tqdm import tqdm
from shapely.ops import transform
from scipy.optimize import minimize
from shapely.affinity import translate
from shapely.geometry import Polygon as sPolygon
from matplotlib.patches import Rectangle, Polygon
import matplotlib.pyplot as plt
from werkzeug.security import safe_join


def best_match(label, geoms):
    scores = [
        label.intersection(g).area /
        label.union(g).area
        for g in geoms
    ]
    idx = np.argmax(scores)
    return idx, scores[idx]


def match(labels, geoms):
    return zip(*[
        best_match(label, geoms) for label in labels
    ])


def score(labels, crowns, dx, dy):
    offset_labels = [
        translate(label, xoff=dx, yoff=dy)
        for label in labels
    ]
    _, best_scores = match(offset_labels, crowns)
    return np.average(best_scores)


def rectangle_from_top_left(top, left, width, height):
    return sPolygon([
        (left, top),  # top-left
        (left + width, top),  # top-right
        (left + width, top + height),  # bottom-right
        (left, top + height),  # bottom-left
        (left, top)  # close the polygon
    ])


def match_crowns(labels, filename, crowns, imgfile, plotfile, evalfile):
    # Select relevant rows and drop column
    labels = labels.loc[labels['filename'] == filename]
    labels = labels.drop(columns=['filename'])

    with rio.open(imgfile) as f:
        I = f.read()
        tform = ~f.transform

    def map_to_pix(x, y, z=None):
        return tform * (x, y)

    fig, ax = plt.subplots()
    I = plt.imread(imgfile)
    ax.imshow(I)

    poly_labels = []
    for _, row in labels.iterrows():
        rect = Rectangle(
            (row['left'], row['top']),
            row['width'],
            row['height'],
            facecolor='none',
            edgecolor='red', zorder=10,
        )
        ax.add_patch(rect)
        poly_labels.append(
            rectangle_from_top_left(
                row['top'], row['left'],
                row['width'], row['height'],
            )
        )

    poly_crowns = [
        transform(map_to_pix, crown['geometry'])
        for _, crown in crowns.iterrows()
    ]

    def obj(params):
        return -score(poly_labels, poly_crowns, *params)

    window = 15
    result = minimize(
        obj, (0, 0),
        bounds=[(-window, window), (-window, window)]
    )
    print(result)

    dx, dy = result.x
    trans_crowns = [
        translate(pc, xoff=-dx, yoff=-dy)
        for pc in poly_crowns
    ]

    best_idx, best_scores = match(poly_labels, trans_crowns)
    valid_idx = set([
        i for i, s in zip(best_idx, best_scores) if s > 0
    ])

    for c, crn in enumerate(trans_crowns):
        facecolor = 'blue' if c in valid_idx else 'none'
        poly = Polygon(
            crn.exterior.coords,
            facecolor=facecolor,
            edgecolor='blue',
        )
        ax.add_patch(poly)

    fig.savefig(plotfile)

    evaluate(poly_labels, trans_crowns, evalfile)

    #plt.show()
    #exit()


def random_perturbation(crowns, seed=0, w=10):
    np.random.seed(seed)
    dx = np.random.uniform(-w, w)
    dy = np.random.uniform(-w, w)
    new_crowns = [
        translate(pc, xoff=dx, yoff=dy)
        for pc in crowns
    ]
    return (dx, dy), new_crowns


def fit(poly_labels, poly_crowns, window=15):

    def obj(params):
        return -score(poly_labels, poly_crowns, *params)

    result = minimize(
        obj, (0, 0),
        bounds=[(-window, window), (-window, window)]
    )
    if not result.success: raise ValueError('Failure')

    dx, dy = result.x
    trans_crowns = [
        translate(pc, xoff=-dx, yoff=-dy)
        for pc in poly_crowns
    ]
    best_idx, best_scores = match(poly_labels, trans_crowns)

    return result.x, best_idx

def evaluate(poly_labels, trans_crowns, evalfile, n_trials=100):
    best_idx, best_scores = match(poly_labels, trans_crowns)
    filtered = [
        p for p, s in zip(poly_labels, best_scores)
        if s > 0
    ]
    best_idx, best_scores = match(filtered, trans_crowns)

    results = []
    for i in tqdm(range(n_trials), 'Simulating'):
        off_true, new_crowns = random_perturbation(trans_crowns, i)
        try:
            off_est, best_est = fit(filtered, new_crowns)
        except ValueError: continue

        is_match = (np.asarray(best_idx) - np.asarray(best_est)) == 0
        match_frac = float(np.average(is_match))

        results.append({
            'seed': i,
            'match_fraction': match_frac,
            'dx': off_true[0],
            'dy': off_true[1],
            'dx_est': off_est[0],
            'dy_est': off_est[1],
            'xerr': off_est[0] - off_true[0],
            'yerr': off_est[1] - off_true[1],
        })

        df = pd.DataFrame.from_dict(results)
        df.to_csv(evalfile, index=False)


@click.command()
@click.argument('labelfile')
@click.argument('shapefile')
@click.argument('imagedir')
@click.argument('outputdir')
def main(labelfile, shapefile, imagedir, outputdir):

    labels = pd.read_csv(labelfile)

    files = np.unique(labels['filename'])
    crowns = gpd.read_file(shapefile)

    for f in files:
        base = os.path.splitext(f)[0]
        fname = base + '.tif'
        plotfile = safe_join(outputdir, base + '.pdf')
        evalfile = safe_join(outputdir, base + '.csv')
        imgfile = safe_join(imagedir, fname)
        match_crowns(labels, f, crowns, imgfile, plotfile, evalfile)

    exit()


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
        try:
            extract_window(imagedir, outputdir, key, poly, total_offset, radius, draw_poly=drawpoly, ndvi=ndvi)
        except (ValueError, FileNotFoundError):
            continue


if __name__ == '__main__':
    main()
