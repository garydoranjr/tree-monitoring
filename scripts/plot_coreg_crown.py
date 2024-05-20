import click
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

from arosics import COREG
from geoarray import GeoArray


@click.command()
@click.argument('dronefile')
@click.argument('planetfile')
@click.argument('shapefile')
@click.argument('crownid')
def main(dronefile, planetfile, shapefile, crownid):

    crowns = gpd.read_file(shapefile)

    focal_crown = crowns.loc[crowns['tag'] == crownid]
    if len(focal_crown) != 1:
        raise ValueError(f'{len(focal_crown)} crowns found with id {crownid}')

    poly = focal_crown['geometry']

    fig, ax = plt.subplots()

    ref = GeoArray(dronefile)
    tgt = GeoArray(planetfile)

    coreg = COREG(
        ref, tgt, ws=(200, 200),
        align_grids=True, max_shift=10,
        ignore_errors=True, q=True,
    )
    result = coreg.calculate_spatial_shifts()

    result = coreg.coreg_info
    shift = result['corrected_shifts_px']

    xscale = 1.0 / tgt.geotransform[1]
    yscale = 1.0 / tgt.geotransform[5]
    xoff = -xscale * tgt.geotransform[0]
    yoff = -yscale * tgt.geotransform[3]

    # Translate from map to image pixels
    poly = poly.affine_transform([xscale, 0, 0, yscale, xoff, yoff])

    # Apply co-registration shift
    poly = poly.translate(xoff=-shift['x'], yoff=-shift['y'])

    ax.imshow(tgt)

    poly.plot(ax=ax, fc='none', ec='red')

    plt.show()


if __name__ == '__main__':
    main()
