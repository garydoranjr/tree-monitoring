#!/usr/bin/env python
import click
import rasterio
from rasterio.mask import mask
import numpy as np
from pathlib import Path
from shapely.geometry import shape, box

from util import load_config


@click.command()
@click.argument('inputfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('configfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
def main(inputfile, configfile, outputfile):
    clip(inputfile, configfile, outputfile)


def clip(inputfile, configfile, outputfile):

    config = load_config(configfile)
    region = shape(config['region'])
    mask_kwargs = config.get('mask_kwargs', {})

    with rasterio.open(inputfile) as data:
        bounds = box(*data.bounds)
        if not bounds.intersects(region):
            return False

        else:
            out_img, out_trans = mask(
                data, shapes=[region], crop=True, **mask_kwargs
            )
            if np.all(out_img == data.nodata):
                return False

        out_meta = data.meta.copy()

        out_meta.update({
            'transform': out_trans,
            'height': out_img.shape[1],
            'width': out_img.shape[2],
            'driver': 'GTiff',
        })

        with rasterio.open(outputfile, 'w', **out_meta) as out:
            out.write(out_img)

    return True


if __name__ == '__main__':
    main()
