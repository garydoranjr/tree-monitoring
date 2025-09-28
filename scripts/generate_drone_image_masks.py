#!/usr/bin/env python
import os
import json
import click
import rioxarray
import xarray as xr
from rasterio import features
from glob import glob
import geopandas as gpd
from tqdm import tqdm


def generate_mask(imgfile, labels, outputdir):
    base = os.path.splitext(os.path.basename(imgfile))[0]
    relevant = labels.loc[labels['polygon_id'] == base]
    if len(relevant) < 1:
        return None
    assert len(relevant) == 1
    label = relevant.iloc[0]

    candidates = labels.loc[labels['date'] == label['date']]
    candidates = candidates.loc[candidates['isFlowering'] == 'yes']

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
def main(labelfile, imagedir, outputdir):

    labels = gpd.read_file(labelfile, layer='flowering_dataset')
    labels = labels.loc[labels['status'] == 'Done']

    imagefiles = sorted(glob(os.path.join(imagedir, '*.tif')))[::-1]

    for imgfile in tqdm(imagefiles, 'Generating Masks'):
        mask = generate_mask(imgfile, labels, outputdir)


if __name__ == '__main__':
    main()
