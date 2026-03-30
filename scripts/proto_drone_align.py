#!/usr/bin/env python
import os
import click
import numpy as np
import pandas as pd
import rioxarray
from tqdm import tqdm
import geopandas as gpd
from scipy.optimize import minimize
from rasterstats import zonal_stats
from shapely.affinity import translate
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def shift(crowns, dx, dy):
    crowns = crowns.copy()
    crowns['geometry'] = crowns.geometry.apply(
        lambda geom: translate(geom, xoff=dx, yoff=dy)
    )
    return crowns


class ObjectiveWithLogging:


    def __init__(self, func, print_every=1):
        self.func = func
        self.nfev = 0
        self.print_every = print_every


    def __call__(self, x):
        self.nfev += 1
        val = self.func(x)
        if self.nfev % self.print_every == 0:
            print(f"eval={self.nfev}, f={val:.6g}, x={x}")
        return val


class ObjFunc:


    def __init__(self, crowns, mask):
        self.crowns = crowns
        self.mask = mask


    def __call__(self, args):
        dx, dy = args
        crowns = shift(self.crowns, dx, dy)

        stats = zonal_stats(
            crowns,
            np.squeeze(self.mask.values),
            affine=self.mask.rio.transform(),
            stats=['std'],
            nodata=np.nan,
        )

        std = np.array([ np.nan if s['std'] is None else s['std'] for s in stats ])
        var = std**2
        return np.nanmean(var)


@click.command()
@click.argument('crownfile')
@click.argument('labelfile')
@click.argument('outputfile')
def main(crownfile, labelfile, outputfile):

    crowns = gpd.read_file(crownfile)
    mask = rioxarray.open_rasterio(labelfile)

    obj = ObjectiveWithLogging(ObjFunc(crowns, mask))

    lim = 4
    #result = minimize(
    #    obj, [0, 0], method='Powell',
    #    bounds = [(-lim, lim), (-lim, lim)],
    #    options={ 'xtol': 1e-2 },
    #)
    #best_x = result.x

    best_x = [ 3.887e+00, -1.270e+00]
    shifted = shift(crowns, *best_x)

    stats = zonal_stats(
        shifted,
        np.squeeze(mask.values),
        affine=mask.rio.transform(),
        stats=['mean'],
        nodata=np.nan,
    )
    shifted['mask_conf'] = np.array([
        np.nan if s['mean'] is None else s['mean']
        for s in stats
    ])

    relevant = shifted.loc[shifted['mask_conf'] > 0.5]
    print(relevant['latin'].value_counts())
    exit()

    shifted.to_file(outputfile)


if __name__ == '__main__':
    main()
