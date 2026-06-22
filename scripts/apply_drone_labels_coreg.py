#!/usr/bin/env python
import json
import logging
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
from skimage.measure import label as label_components
from rasterio.enums import Resampling
from scipy.ndimage import median_filter
import pandas as pd
from glob import glob
from affine import Affine
from arosics import COREG
from geoarray import GeoArray

log = logging.getLogger(__name__)

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


def compute_coreg_shift(dronefile, planetfile, planet_match_band=1):
    drone_ga = load_as_geoarray(dronefile)
    planet_ga = load_as_geoarray(planetfile)

    try:
        coreg = COREG(
            drone_ga, planet_ga,
            ws=(200, 200),
            align_grids=True,
            max_shift=10,
            ignore_errors=True,
            q=True,
            # Match on the Red band of both images. For 3-band RGB chips Red
            # is band 1; for 4-band (B,G,R,NIR) chips Red is band 3. Keeping
            # Red as the reference makes alignment consistent across modes.
            r_b4match=1,
            s_b4match=planet_match_band,
        )
        coreg.calculate_spatial_shifts()
    except Exception:
        return 0.0, 0.0, False

    info = coreg.coreg_info
    success = bool(info.get('success', False))
    shift = info.get('corrected_shifts_map', {})
    x_shift = shift.get('x', 0.0)
    y_shift = shift.get('y', 0.0)

    return x_shift, y_shift, success


def find_drone(labelfile, dronedir):
    base = os.path.splitext(os.path.basename(labelfile))[0]
    dronefiles = glob(os.path.join(dronedir, "*.tif"))
    for d in dronefiles:
        if base.startswith(os.path.splitext(os.path.basename(d))[0]):
            return d
    raise ValueError(f"No drone file found for {labelfile}")


def find_ocm_mask(planetfile, planetdir, maskdir):
    try:
        rel = Path(planetfile).relative_to(planetdir)
    except ValueError:
        log.warning("Cannot make %s relative to %s; skipping OCM mask", planetfile, planetdir)
        return None
    stem = Path(planetfile).stem
    for suffix in ('_rgb', '_4band'):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    ocm_path = Path(maskdir) / rel.parent / f"{stem}_ocm.tif"
    if not ocm_path.exists():
        log.warning("OCM mask not found: %s", ocm_path)
        return None
    return ocm_path


def compute_clear_fraction(ocm_path):
    with rasterio.open(ocm_path) as src:
        band1 = src.read(1)
    valid = band1 != 255
    total_valid = int(valid.sum())
    if total_valid == 0:
        return None
    return float((band1 == 0).sum() / total_valid)


def reproject_ocm_to_grid(ocm_path, pimg):
    ocm = rxr.open_rasterio(ocm_path).sel(band=1)
    return ocm.rio.reproject_match(pimg, resampling=Resampling.nearest)


def crown_filter_by_ocm(conf_resampled, ocm_rep):
    clear_pixels = (ocm_rep.values == 0)

    binary = (conf_resampled.values > 0.5)
    labeled = label_components(binary)
    for comp_id in range(1, labeled.max() + 1):
        comp_mask = (labeled == comp_id)
        if not clear_pixels[comp_mask].all():
            binary[comp_mask] = False

    return (binary.astype(np.uint8) * 255)


def render_rgb_from_4band(pimg, clear_mask, lower_pct=2.0, upper_pct=98.0):
    """Per-band percentile stretch on a 4-band Planet DataArray. Percentiles
    are computed using only clear (cloud-free, non-nodata) pixels, but the
    stretch is then applied to *every* non-nodata pixel — cloudy regions stay
    visible (they typically saturate near 255 since they exceed the upper
    percentile of clear pixels). Planet AnalyticMS band order is
    [Blue, Green, Red, NIR] (1,2,3,4). Returns a uint8 (H, W, 3) array in
    R, G, B order."""
    nodata = pimg.rio.nodata
    arr = pimg.values  # (bands, H, W)
    if arr.shape[0] < 4:
        raise ValueError(f"Expected 4-band Planet imagery, got {arr.shape[0]} bands")

    red = arr[2].astype(np.float32)
    green = arr[1].astype(np.float32)
    blue = arr[0].astype(np.float32)

    height, width = red.shape
    out = np.zeros((height, width, 3), dtype=np.uint8)
    for i, b in enumerate((red, green, blue)):
        not_nodata = np.ones_like(b, dtype=bool) if nodata is None else (b != nodata)
        clear_valid = clear_mask & not_nodata
        if not clear_valid.any():
            continue
        lo, hi = np.percentile(b[clear_valid], (lower_pct, upper_pct))
        if hi <= lo:
            continue
        scaled = np.clip((b - lo) * (255.0 / (hi - lo)), 0, 255)
        scaled = np.where(not_nodata, scaled, 0)
        out[..., i] = scaled.astype(np.uint8)

    return out


def _round_sig(x, sig=5):
    if x is None or not np.isfinite(x):
        return None
    return float(f"{float(x):.{sig}g}")


def compute_noise_metrics(pimg, clear_mask, min_pixels=200):
    """Per-scene image-quality metrics computed on a 4-band Planet DataArray
    (B, G, R, NIR), restricted to clear OCM pixels.

    Returns a dict with:
      - noise_mad_sigma: per-band robust noise std (uint16 DN units)
      - noise_mad_cv: per-band sigma normalized by mean clear-pixel signal
      - noise_mad_cv_mean: mean of noise_mad_cv across bands
      - band_corr_mean: mean Pearson r across the six band pairs
      - band_decorrelation: 1 - band_corr_mean

    Estimates:
      sigma via 1.4826 * MAD of (band - 3x3 median(band)). The 3x3 median
      preserves crown edges, so the residual is closer to pure noise.
      Inter-channel correlation is computed on raw clear-pixel band values
      (the rainbow-noise failure mode breaks band correlation directly)."""
    nodata = pimg.rio.nodata
    arr = pimg.values  # (bands, H, W)
    n_bands = arr.shape[0]

    valid = clear_mask.copy() if clear_mask is not None else np.ones(arr.shape[1:], dtype=bool)
    if nodata is not None:
        for b in range(n_bands):
            valid &= (arr[b] != nodata)

    none_result = {
        'noise_mad_sigma': None,
        'noise_mad_cv': None,
        'noise_mad_cv_mean': None,
        'band_corr_mean': None,
        'band_decorrelation': None,
    }
    if int(valid.sum()) < min_pixels:
        return none_result

    sigmas = []
    cvs = []
    flat_clear = []
    for b in range(n_bands):
        band = arr[b].astype(np.float32)
        smoothed = median_filter(band, size=3)
        residual = band - smoothed
        r = residual[valid]
        med = np.median(r)
        mad = np.median(np.abs(r - med))
        sigma = float(1.4826 * mad)

        signal_mean = float(np.mean(band[valid]))
        cv = sigma / signal_mean if signal_mean > 0 else float('nan')
        sigmas.append(sigma)
        cvs.append(cv)
        flat_clear.append(band[valid])

    finite_cvs = [c for c in cvs if np.isfinite(c)]
    cv_mean = float(np.mean(finite_cvs)) if finite_cvs else float('nan')

    pair_corrs = []
    for i in range(n_bands):
        for j in range(i + 1, n_bands):
            xi, xj = flat_clear[i], flat_clear[j]
            if xi.std() == 0 or xj.std() == 0:
                continue
            pair_corrs.append(float(np.corrcoef(xi, xj)[0, 1]))

    if pair_corrs:
        corr_mean = float(np.mean(pair_corrs))
        decorr = 1.0 - corr_mean
    else:
        corr_mean = None
        decorr = None

    return {
        'noise_mad_sigma': [_round_sig(s) for s in sigmas],
        'noise_mad_cv': [_round_sig(c) for c in cvs],
        'noise_mad_cv_mean': _round_sig(cv_mean),
        'band_corr_mean': _round_sig(corr_mean),
        'band_decorrelation': _round_sig(decorr),
    }


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


def generate_drone_png(dronefile, pimg, x_shift_m, y_shift_m, drone_scale, outputpath):
    drone_da = rxr.open_rasterio(dronefile)
    drone_shifted = apply_shift_to_label(drone_da, x_shift_m, y_shift_m)
    drone_clipped = drone_shifted.rio.clip_box(*pimg.rio.bounds())

    # Reproject onto a grid that exactly matches the Planet bounds at
    # drone_scale times finer resolution. Specifying bounds via transform+shape
    # (rather than resolution alone) keeps the drone pixel grid an exact
    # integer subdivision of the Planet grid, so downstream fractional crops
    # (e.g. the Mask R-CNN viewer) line up pixel-for-pixel.
    p_transform = pimg.rio.transform()
    p_h, p_w = pimg.rio.height, pimg.rio.width
    target_transform = Affine(
        p_transform.a / drone_scale, p_transform.b, p_transform.c,
        p_transform.d, p_transform.e / drone_scale, p_transform.f,
    )
    drone_reprojected = drone_clipped.rio.reproject(
        pimg.rio.crs,
        transform=target_transform,
        shape=(int(p_h * drone_scale), int(p_w * drone_scale)),
        resampling=Resampling.bilinear,
    )

    rgb_arr = np.moveaxis(drone_reprojected.isel(band=slice(0, 3)).values, 0, -1)
    if rgb_arr.dtype not in (np.uint8, np.uint16):
        rgb_arr = np.clip(rgb_arr, 0, 65535).astype(np.uint16)
    rgb_arr = percentile_stretch_global(rgb_arr, lower_pct=0, upper_pct=99.9)
    iio.imwrite(outputpath, rgb_arr)


def generate_ocm_png(ocm_path, pimg, outputpath):
    ocm = rxr.open_rasterio(ocm_path).sel(band=1)
    ocm_rep = ocm.rio.reproject_match(pimg, resampling=Resampling.nearest)
    vals = ocm_rep.values
    rgba = np.zeros((*vals.shape, 4), dtype=np.uint8)
    cloudy = (vals != 0) & (vals != 255)
    rgba[cloudy] = [255, 80, 0, 150]
    iio.imwrite(outputpath, rgba)


def create_mask(dronedir, labelfile, planetfile, planetdir, outputdir, resize, mode, maskdir=None, drone_scale=None, bands=3):

    dronefile = find_drone(labelfile, dronedir)

    clear_fraction = None
    if maskdir is not None:
        ocm_path = find_ocm_mask(planetfile, planetdir, maskdir)
        if ocm_path is not None:
            clear_fraction = compute_clear_fraction(ocm_path)

    # Red is band 1 in 3-band RGB chips and band 3 in 4-band (B,G,R,NIR) chips.
    planet_match_band = 3 if bands == 4 else 1
    x_shift, y_shift, coreg_ok = compute_coreg_shift(
        dronefile, planetfile, planet_match_band=planet_match_band,
    )

    with rasterio.open(planetfile) as src:
        planet_res_m = src.res[0]
    with rasterio.open(dronefile) as src:
        drone_res_m = src.res[0]

    record = {
        'scene': Path(planetfile).stem,
        'label': Path(labelfile).name,
        'coreg_ok': coreg_ok,
        'clear_fraction': clear_fraction,
        'drone_file': str(Path(dronefile).resolve()),
        'x_shift_m': x_shift,
        'y_shift_m': y_shift,
        'planet_res_m': planet_res_m,
        'drone_res_m': drone_res_m,
    }

    if not coreg_ok:
        log.warning(
            "Coregistration failed for %s (clear_fraction=%s)",
            Path(planetfile).stem,
            f"{clear_fraction:.3f}" if clear_fraction is not None else "n/a",
        )
        return record

    pimg = load_planet(planetfile, resize)
    conf = load_label(labelfile, mode)
    conf = apply_shift_to_label(conf, x_shift, y_shift)

    conf_resampled = conf.rio.reproject_match(
        pimg,
        resampling=Resampling.average,
    )

    ocm_rep = None
    if maskdir is not None and ocm_path is not None:
        ocm_rep = reproject_ocm_to_grid(ocm_path, pimg)

    if ocm_rep is not None:
        mask_array = crown_filter_by_ocm(conf_resampled, ocm_rep)
        clear_mask = (ocm_rep.values == 0)
    else:
        mask_array = (conf_resampled.values > 0.5).astype(np.uint8) * 255
        clear_mask = np.ones(pimg.values.shape[1:], dtype=bool)

    outputdir = Path(outputdir)
    outputdir.mkdir(parents=True, exist_ok=True)
    basename = Path(planetfile).stem
    if basename.endswith('_4band'):
        basename = basename[:-6]
    elif basename.endswith('_rgb'):
        basename = basename[:-4]

    if bands == 4:
        # Write the 4-band uint16 GeoTIFF training chip (preserves CRS,
        # transform, dtype, and nodata), then build an RGB QA PNG with a
        # cloud-aware per-band stretch on the (B,G,R,NIR) image.
        pimg.rio.to_raster(outputdir / f"{basename}.tif")
        rgb_array = render_rgb_from_4band(pimg, clear_mask)
        record.update(compute_noise_metrics(pimg, clear_mask))
    else:
        rgb_array = np.moveaxis(pimg.values, 0, -1)
        rgb_array = percentile_stretch_global(rgb_array, lower_pct=0, upper_pct=99.9)

    iio.imwrite(outputdir / f"{basename}.png", rgb_array)
    iio.imwrite(outputdir / f"{basename}.mask.png", mask_array)

    if drone_scale is not None:
        generate_drone_png(
            dronefile, pimg, x_shift, y_shift, drone_scale,
            outputdir / f"{basename}.drone.png",
        )

    if maskdir is not None and ocm_path is not None:
        generate_ocm_png(ocm_path, pimg, outputdir / f"{basename}.ocm.png")

    return record


def process_label(dronedir, labelfile, planet_df, planetdir, outputdir, timewindow, resize, mode, maskdir=None, drone_scale=None, bands=3):
    label_date = pd.to_datetime(
        get_cls_date(labelfile),
        format="%Y_%m_%d",
    )

    date_mask = (planet_df["date"] - label_date).abs() <= pd.Timedelta(days=timewindow)
    planetfiles = planet_df.loc[date_mask]["path"]

    return [
        create_mask(dronedir, labelfile, planetfile, planetdir, outputdir, resize, mode, maskdir, drone_scale, bands)
        for planetfile in planetfiles.tolist()
    ]


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
@click.option('-k', '--maskdir', default=None, type=click.Path())
@click.option('-d', '--drone-scale', default=None, type=float,
              help='If set, generate {scene}.drone.png at this multiple of the resized Planet pixel size.')
@click.option('-b', '--bands', default=3, type=click.Choice(['3', '4']),
              callback=lambda c, p, v: int(v),
              help='3 = RGB chips from *rgb.tif (default); 4 = RGB+NIR chips '
                   'from *4band.tif, written as a 4-band uint16 GeoTIFF plus '
                   'an RGB QA PNG.')
def main(labelfiles, dronedir, planetdir, outputdir, timewindow, resize, filterdir, mode, maskdir, drone_scale, bands):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    planet_glob = '*4band.tif' if bands == 4 else '*rgb.tif'
    planetfiles = list(Path(planetdir).rglob(planet_glob))
    planetfiles = filter_files(planetfiles, filterdir)

    planet_df = pd.DataFrame({
        "path": [str(p) for p in planetfiles],
    })
    planet_df["date"] = pd.to_datetime(
        planet_df["path"].apply(get_planet_date),
        format="%Y%m%d",
    )

    all_records = []
    for labelfile in tqdm(labelfiles):
        records = process_label(
            dronedir, labelfile, planet_df, planetdir,
            outputdir, timewindow, resize, mode, maskdir, drone_scale, bands,
        )
        all_records.extend(records)

    if all_records:
        out_path = Path(outputdir) / 'coreg_log.json'
        Path(outputdir).mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(all_records, f, indent=2)
        n_failed = sum(1 for r in all_records if not r['coreg_ok'])
        log.info(
            "Wrote %d records (%d succeeded, %d failed) to %s",
            len(all_records), len(all_records) - n_failed, n_failed, out_path,
        )


if __name__ == '__main__':
    main()
