#!/usr/bin/env python
import os
import re
import click
import rasterio
import numpy as np
from tqdm import tqdm
import rioxarray as rxr
import imageio.v3 as iio
from pathlib import Path
from skimage import color
from skimage import exposure
from rasterio.enums import Resampling
import pandas as pd
from glob import glob
from affine import Affine
from arosics import COREG
from geoarray import GeoArray


DATE_PATTERN = r"\d{4}_\d{2}_\d{2}"


def get_cls_date(labelfile):
    match = re.search(DATE_PATTERN, os.path.basename(labelfile))
    assert match is not None
    return match.group()


def get_planet_date(planetfile):
    return os.path.basename(planetfile).split('_')[0]


def load_planet(planetfile, scale_factor):
    img = rxr.open_rasterio(planetfile)

    if scale_factor is None:
        return img

    new_height = int(img.rio.height * scale_factor)
    new_width  = int(img.rio.width * scale_factor)

    return img.rio.reproject(
        img.rio.crs,
        shape=(new_height, new_width),
        resampling=Resampling.bilinear,
    )


def percentile_stretch_global(rgb_array, lower_pct=2, upper_pct=98):
    """
    Apply a global percentile contrast stretch to an RGB image.
    The same scaling is applied to all channels.

    Parameters
    ----------
    rgb_array : np.ndarray (H, W, 3)
        uint8 or uint16 image
    lower_pct : float
    upper_pct : float

    Returns
    -------
    np.ndarray (uint8)
    """

    if rgb_array.dtype not in (np.uint8, np.uint16):
        raise TypeError(
            f"Unsupported dtype {rgb_array.dtype}. "
            "Expected uint8 or uint16."
        )

    # Compute percentiles over entire image (all bands together)
    p_low, p_high = np.percentile(rgb_array, (lower_pct, upper_pct))

    # Rescale intensities
    stretched = exposure.rescale_intensity(
        rgb_array,
        in_range=(p_low, p_high),
        out_range=(0, 255)
    )

    return stretched.astype(np.uint8)


def load_label(labelfile, mode='both'):
    da = rxr.open_rasterio(labelfile, masked=True)

    # Compute union probability across all bands
    if mode == 'both':
        confidence = 1 - np.prod(1 - da, axis=0)
    else:
        # Assign band names as a coordinate
        da = da.assign_coords(band=list(da.attrs['long_name']))
        confidence = da.sel(band=mode)

    # Preserve CRS and transform metadata
    confidence.rio.write_crs(da.rio.crs, inplace=True)
    confidence.rio.write_transform(da.rio.transform(), inplace=True)

    return confidence


def load_as_geoarray(filepath):
    """Load a raster as a GeoArray with metadata stripped to avoid band-count mismatches."""
    with rasterio.open(filepath) as src:
        data = src.read()           # (bands, rows, cols)
        transform = src.transform
        projection = src.crs.to_wkt()

    # GeoArray expects (rows, cols) for single band or (rows, cols, bands) for multi
    if data.shape[0] == 1:
        arr = data[0]
    else:
        arr = np.moveaxis(data, 0, -1)  # -> (rows, cols, bands)

    return GeoArray(arr, geotransform=transform.to_gdal(), projection=projection)


def compute_coreg_shift(dronefile, planetfile):
    drone_ga = load_as_geoarray(dronefile)
    planet_ga = load_as_geoarray(planetfile)

    coreg = COREG(
        drone_ga, planet_ga,
        ws=(200, 200),
        align_grids=True,
        max_shift=10,
        ignore_errors=True,
        q=True,
    )
    coreg.calculate_spatial_shifts()
    info = coreg.coreg_info

    shift = info.get('corrected_shifts_map', {})
    x_shift = shift.get('x', 0.0)
    y_shift = shift.get('y', 0.0)

    return x_shift, y_shift


def find_drone(labelfile, dronedir):
    base = os.path.splitext(os.path.basename(labelfile))[0]
    dronefiles = glob(os.path.join(dronedir, "*.tif"))
    for d in dronefiles:
        if base.startswith(os.path.splitext(os.path.basename(d))[0]):
            return d
    raise ValueError(f"No drone file found for {labelfile}")


def apply_shift_to_label(label_da, x_shift, y_shift):
    """
    Translate a label DataArray's spatial reference by (x_shift, y_shift)
    in its CRS units, returning a new DataArray with the updated transform.
    """
    old_transform = label_da.rio.transform()

    new_transform = Affine(
        old_transform.a,
        old_transform.b,
        old_transform.c + x_shift,
        old_transform.d,
        old_transform.e,
        old_transform.f + y_shift,
    )

    shifted = label_da.assign_coords(
        x=label_da.coords['x'] + x_shift,
        y=label_da.coords['y'] + y_shift,
    )
    shifted.rio.write_transform(new_transform, inplace=True)
    shifted.rio.write_crs(label_da.rio.crs, inplace=True)

    return shifted


def create_mask(dronedir, labelfile, planetfile, outputdir, resize, mode):

    dronefile = find_drone(labelfile, dronedir)

    pimg = load_planet(planetfile, resize)
    conf = load_label(labelfile, mode)

    x_shift, y_shift = compute_coreg_shift(dronefile, planetfile)
    conf = apply_shift_to_label(conf, x_shift, y_shift)

    conf_resampled = conf.rio.reproject_match(
        pimg,
        resampling=Resampling.average,
    )

    mask = (conf_resampled > 0.5)

    outputdir = Path(outputdir)
    outputdir.mkdir(parents=True, exist_ok=True)
    basename = Path(planetfile).stem

    rgb_array = np.moveaxis(pimg.values, 0, -1)

    rgb_array = percentile_stretch_global(rgb_array, lower_pct=0, upper_pct=99.9)

    rgb_path = outputdir / f"{basename}.png"
    iio.imwrite(rgb_path, rgb_array)

    mask_array = (mask.astype(np.uint8) * 255).values
    mask_path = outputdir / f"{basename}.mask.png"
    iio.imwrite(mask_path, mask_array)


def process_label(dronedir, labelfile, planet_df, outputdir, timewindow, resize, mode):
    label_date = pd.to_datetime(
        get_cls_date(labelfile),
        format="%Y_%m_%d",
    )

    mask = (planet_df["date"] - label_date).abs() <= pd.Timedelta(days=timewindow)
    planetfiles = planet_df.loc[mask]["path"]

    for planetfile in planetfiles.tolist():
        create_mask(dronedir, labelfile, planetfile, outputdir, resize, mode)


def filter_files(planetfiles, filterdir):
    if filterdir is None:
        return planetfiles

    goodfiles = glob(os.path.join(filterdir, '*rgb.png'))
    goodkeys = set([
        os.path.splitext(os.path.basename(f))[0]
        for f in goodfiles
    ])

    return [
        p for p in planetfiles
        if os.path.splitext(os.path.basename(p))[0] in goodkeys
    ]


@click.command()
@click.argument('labelfiles', nargs=-1)
@click.argument('dronedir')
@click.argument('planetdir')
@click.argument('outputdir')
@click.option('-t', '--timewindow', default=2, type=int)
@click.option('-r', '--resize', default=None, type=float)
@click.option('-f', '--filterdir', default=None)
@click.option('-m', '--mode', default='both')
def main(labelfiles, dronedir, planetdir, outputdir, timewindow, resize, filterdir, mode):
    planetfiles = glob(os.path.join(planetdir, '*rgb.tif'))

    planetfiles = filter_files(planetfiles, filterdir)

    planet_df = pd.DataFrame({
        "path": planetfiles,
    })
    planet_df["date"] = pd.to_datetime(
        planet_df["path"].apply(get_planet_date),
        format="%Y%m%d",
    )

    for labelfile in tqdm(labelfiles):
        process_label(dronedir, labelfile, planet_df, outputdir, timewindow, resize, mode)


if __name__ == '__main__':
    main()
