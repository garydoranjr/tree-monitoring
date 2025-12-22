#!/usr/bin/env python
import os
import json
import click
import rioxarray
import numpy as np
import xarray as xr
from rasterio import features
from glob import glob
import geopandas as gpd
from tqdm import tqdm


def update_labels(df):

    cat_f = lambda v: 'no' if v is None else v
    int_f = lambda v: 0.0 if np.isnan(v) else v

    df['isFlowering'] = df['isFlowering'].apply(cat_f)
    df['floweringIntensity'] = df['floweringIntensity'].apply(int_f)
    df['isFruiting'] = df['isFruiting'].apply(cat_f)
    df['newLeaves'] = df['newLeaves'].apply(cat_f)
    
    return df


def generate_mask(imgfile, labels, outputdir, criterion='flower'):
    base = os.path.splitext(os.path.basename(imgfile))[0]
    relevant = labels.loc[labels['polygon_id'] == base]
    if len(relevant) < 1:
        return None
    assert len(relevant) == 1
    label = relevant.iloc[0]

    candidates = labels.loc[labels['date'] == label['date']]

    match criterion:
        case 'flower':
            candidates = candidates.loc[candidates['isFlowering'] == 'yes']
        case 'leaf':
            candidates = candidates.loc[candidates['leafing'] < 50.]
        case 'crown':
            candidates = candidates.loc[candidates['polygon_id'] == label['polygon_id']]
        case _:
            raise ValueError(f'Unknown criterion "{criterion}"')

    rds = rioxarray.open_rasterio(imgfile)
    template = rds.sel(band=1)

    candidates = candidates.to_crs(template.rio.crs)

    mask = features.rasterize(
        [(geom, 255) for geom in candidates.geometry],
        out_shape=template.shape,
        transform=template.rio.transform(),
        fill=0,
        dtype='uint8',
    )

    mask_da = xr.DataArray(
        mask,
        dims=template.dims,
        coords=template.coords,
        name='mask',
    )

    outputfile = os.path.join(outputdir, f'{base}.tif')
    mask_da.rio.to_raster(outputfile)


@click.command()
@click.argument('labelfile')
@click.argument('imagedir')
@click.argument('outputdir')
@click.option('-c', '--criterion', default='leaf', type=click.Choice(['flower', 'leaf', 'crown']))
def main(labelfile, imagedir, outputdir, criterion):

    labels = gpd.read_file(labelfile, layer='flowering_dataset')
    labels = update_labels(labels)

    imagefiles = sorted(glob(os.path.join(imagedir, '*.tif')))[::-1]

    for imgfile in tqdm(imagefiles, 'Generating Masks'):
        mask = generate_mask(imgfile, labels, outputdir, criterion)


if __name__ == '__main__':
    main()
