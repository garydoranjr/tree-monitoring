#!/usr/bin/env python
import os
import re
import click
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
    confidence = 1 - np.prod(1 - da, axis=0)

    # Preserve CRS and transform metadata
    confidence.rio.write_crs(da.rio.crs, inplace=True)
    confidence.rio.write_transform(da.rio.transform(), inplace=True)

    return confidence


def create_mask(labelfile, planetfile, outputdir, resize):
    pimg = load_planet(planetfile, resize)
    conf = load_label(labelfile)

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


def process_label(labelfile, planet_df, outputdir, timewindow, resize):
    label_date = pd.to_datetime(
        get_cls_date(labelfile),
        format="%Y_%m_%d",
    )

    mask = (planet_df["date"] - label_date).abs() <= pd.Timedelta(days=timewindow)
    planetfiles = planet_df.loc[mask]["path"]

    for planetfile in planetfiles.tolist():
        create_mask(labelfile, planetfile, outputdir, resize)


@click.command()
@click.argument('labelfiles', nargs=-1)
@click.argument('planetdir')
@click.argument('outputdir')
@click.option('-t', '--timewindow', default=2, type=int)
@click.option('-r', '--resize', default=None, type=float)
def main(labelfiles, planetdir, outputdir, timewindow, resize):
    planetfiles = glob(os.path.join(planetdir, '*rgb.tif'))

    planet_df = pd.DataFrame({
        "path": planetfiles,
    })
    planet_df["date"] = pd.to_datetime(
        planet_df["path"].apply(get_planet_date),
        format="%Y%m%d",
    )

    for labelfile in tqdm(labelfiles):
        process_label(labelfile, planet_df, outputdir, timewindow, resize)


if __name__ == '__main__':
    main()
