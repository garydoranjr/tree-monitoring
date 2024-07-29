#!/usr/bin/env python
import click
import rasterio
import numpy as np
from pathlib import Path
from xml.dom import minidom


def get_coeff(metadata_file):
    """
    Code taken from https://developers.planet.com/docs/planetschool/calculate-an-ndvi-in-python/
    """
    with open(metadata_file, 'r') as f:
        xmldoc = minidom.parse(f)

    nodes = xmldoc.getElementsByTagName("ps:bandSpecificMetadata")

    # XML parser refers to bands by numbers 1-4
    coeffs = {}
    for node in nodes:
        bn = node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
        if bn in ['1', '2', '3', '4']:
            i = int(bn)
            value = node.getElementsByTagName("ps:reflectanceCoefficient")[0].firstChild.data
            coeffs[i] = float(value)

    return coeffs


@click.command()
@click.argument('inputfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('metafile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
def main(inputfile, metafile, outputfile):
    ndvi(inputfile, metafile, outputfile)


def ndvi(inputfile, metafile, outputfile):

    coeffs = get_coeff(metafile)

    with rasterio.open(inputfile) as src:
        assert src.descriptions[2] == 'red'
        assert src.descriptions[3] == 'nir'
        band_red = (src.read(3) * coeffs[3]).astype(float)
        band_nir = (src.read(4) * coeffs[4]).astype(float)

        out_meta = src.meta.copy()

    out_meta.update({
        'dtype': rasterio.float32,
        'count': 1,
    })

    # Handle division by zero
    total = band_nir + band_red
    total[total == 0] = 1e-9

    # Calculate NDVI
    ndvi = (band_nir - band_red) / total

    with rasterio.open(outputfile, 'w', **out_meta) as out:
        out.write_band(1, ndvi.astype(rasterio.float32))

    return True


if __name__ == '__main__':
    main()
