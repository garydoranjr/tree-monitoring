#!/usr/bin/env python
"""Drone and Planet image layers for the per-crown timeseries viewer.

All layers are rendered onto one north-up grid over the crown map
(`grid_bounds`), so the page places each image at the same SVG coordinates
as the crown outlines. Drone mosaics are 1-4 GB without overviews, so the
whole-plot overview of each flight is cached as a WebP under
CACHE_DIR/drone; `prerender` fills that cache ahead of time:

    python scripts/crown_viewer_imagery.py prerender --jobs 3

The server (serve_crown_timeseries.py) also renders zoomed drone crops and
the Planet RGB, detection-score and cloud layers, which are small enough to
render on demand.
"""
import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from affine import Affine
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import from_bounds
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_update_figures as pu  # noqa: E402
from plot_crown_species_comparison import NC4S, PHANTOM_END  # noqa: E402

CROWNMAP = str(pu.CROWNMAP)
CACHE_DIR = pu.FLOWER / 'crown_viewer_cache'
PLANET_DIR = pu.FLOWER / 'planet_clipped/4band'
OVERVIEW_RES = 0.25
PLANET_RES = 0.75
GRID_PAD_M = 20
# Score ramp (light -> dark orange) for the detection-score overlay.
SCORE_RAMP = np.array([[254, 230, 206], [253, 174, 107], [241, 105, 19],
                       [166, 54, 3]], dtype=np.float32)
CLOUD_RGB = np.array([90, 140, 255], dtype=np.uint8)


def grid_bounds(crownmap=CROWNMAP):
    """(x0, y0, x1, y1) of the viewer grid: the crown map's bounds rounded
    out to 10 m and padded by GRID_PAD_M."""
    x0, y0, x1, y1 = gpd.read_file(crownmap).total_bounds
    return (np.floor(x0 / 10) * 10 - GRID_PAD_M, np.floor(y0 / 10) * 10 - GRID_PAD_M,
            np.ceil(x1 / 10) * 10 + GRID_PAD_M, np.ceil(y1 / 10) * 10 + GRID_PAD_M)


def flight_dates(nc4s=NC4S):
    """The flight dates of the viewer, as load_drone() in
    make_crown_timeseries_html.py selects them."""
    dates = set()
    for path in nc4s:
        d = pd.DatetimeIndex(xr.open_dataset(path)['date'].values)
        if 'mavic' in os.path.basename(path):
            d = d[d > PHANTOM_END]
        dates.update(d)
    return sorted(dates)


def drone_file(date, alignment='local'):
    """The orthomosaic of a flight: phantom (24782016 or C3KW2X, in
    `alignment`) up to PHANTOM_END, mavic after."""
    date = pd.Timestamp(date)
    stem = f'BCI_50ha_{date:%Y_%m_%d}'
    if date > PHANTOM_END:
        return pu.MAVIC_RGB / f'{stem}_M3M_aligned_global_RGB.tif'
    release = 'C3KW2X' if date in pu.c3kw2x_dates() else '24782016'
    return pu.phantom_dir(release, alignment) / f'{stem}_{alignment}.tif'


def overview_path(cache_dir, date, res=OVERVIEW_RES):
    return Path(cache_dir) / 'drone' / f'{pd.Timestamp(date):%Y_%m_%d}_{res:g}m.webp'


def encode_webp(rgba, quality=85):
    buf = io.BytesIO()
    Image.fromarray(rgba, 'RGBA').save(buf, 'WEBP', quality=quality, method=4)
    return buf.getvalue()


def render_drone(path, bounds, width, height):
    """RGBA (H, W, 4) uint8 of `bounds` from a drone mosaic, block-averaged
    to width x height, with the classifier's display colours (pu.rgb_display)
    and nodata transparent."""
    with rasterio.open(path) as src:
        win = from_bounds(*bounds, transform=src.transform)
        arr = src.read([1, 2, 3], window=win, out_shape=(3, height, width),
                       resampling=Resampling.average, boundless=True, fill_value=0)
        native = src.res[0]
    rgb = (pu.rgb_display(arr) * 255).round().astype(np.uint8)
    alpha = np.where(arr.sum(axis=0) == 0, 0, 255).astype(np.uint8)
    return np.dstack([rgb, alpha]), native


def render_overview(date, cache_dir, bounds, res=OVERVIEW_RES, alignment='local'):
    """Write the cached overview of a flight (if missing) and return its path."""
    out = overview_path(cache_dir, date, res)
    if out.exists():
        return out
    width = int(round((bounds[2] - bounds[0]) / res))
    height = int(round((bounds[3] - bounds[1]) / res))
    rgba, _ = render_drone(drone_file(date, alignment), bounds, width, height)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    tmp.write_bytes(encode_webp(rgba))
    tmp.rename(out)
    return out


def _grid(bounds, res):
    width = int(round((bounds[2] - bounds[0]) / res))
    height = int(round((bounds[3] - bounds[1]) / res))
    return Affine(res, 0, bounds[0], 0, -res, bounds[3]), width, height


def _to_grid(path, bands, bounds, res, shift=(0.0, 0.0), dtype=np.float32):
    """Bands of a raster translated by -shift and resampled (nearest) onto the
    viewer grid; NaN outside the raster."""
    transform, width, height = _grid(bounds, res)
    with rasterio.open(path) as src:
        src_t = Affine.translation(-shift[0], -shift[1]) @ src.transform
        out = np.full((len(bands), height, width), np.nan, dtype=np.float32)
        for k, b in enumerate(bands):
            reproject(src.read(b).astype(np.float32), out[k], src_transform=src_t,
                      src_crs=src.crs, dst_transform=transform, dst_crs=src.crs,
                      src_nodata=src.nodata, dst_nodata=np.nan,
                      resampling=Resampling.nearest)
    return out


def render_planet_rgb(planet_path, clear_path, clear_shift, bounds, shift,
                      res=PLANET_RES, lower_pct=2.0, upper_pct=98.0):
    """Planet RGB on the viewer grid with the per-band 2/98 % stretch of
    render_rgb_from_4band (percentiles over clear pixels, applied to all)."""
    bgr = _to_grid(planet_path, [1, 2, 3], bounds, res, shift)
    clear = _to_grid(clear_path, [3], bounds, res, clear_shift)[0] > 0.5
    valid = np.isfinite(bgr).all(axis=0) & (np.nan_to_num(bgr).sum(axis=0) > 0)
    rgb = np.zeros(bgr.shape[1:] + (3,), dtype=np.uint8)
    for i, band in enumerate(bgr[::-1]):
        ref = band[clear & valid] if (clear & valid).any() else band[valid]
        if ref.size == 0:
            continue
        lo, hi = np.percentile(ref, (lower_pct, upper_pct))
        scaled = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)
        rgb[..., i] = np.nan_to_num(scaled * 255).astype(np.uint8)
    alpha = np.where(valid, 255, 0).astype(np.uint8)
    return np.dstack([rgb, alpha])


def render_planet_score(pred_path, bounds, shift=(0.0, 0.0), res=PLANET_RES):
    """The highest instance score per pixel (pred band 2) as a translucent
    orange ramp; score 0 is transparent."""
    score = np.nan_to_num(_to_grid(pred_path, [2], bounds, res, shift)[0])
    x = np.clip(score, 0, 1) * (len(SCORE_RAMP) - 1)
    i = np.minimum(np.floor(x).astype(int), len(SCORE_RAMP) - 2)
    t = (x - i)[..., None]
    rgb = SCORE_RAMP[i] + (SCORE_RAMP[i + 1] - SCORE_RAMP[i]) * t
    alpha = np.where(score > 0, 90 + 140 * np.clip(score, 0, 1), 0)
    return np.dstack([rgb, alpha]).round().astype(np.uint8)


def render_planet_cloud(pred_path, bounds, shift=(0.0, 0.0), res=PLANET_RES):
    """Pixels that are not clear (pred band 3 == 0) tinted blue, as in the QA
    PNGs; clear pixels and those outside the scene are transparent."""
    clear = _to_grid(pred_path, [3], bounds, res, shift)[0]
    rgba = np.zeros(clear.shape + (4,), dtype=np.uint8)
    cloud = np.isfinite(clear) & (clear < 0.5)
    rgba[cloud, :3] = CLOUD_RGB
    rgba[cloud, 3] = 140
    return rgba


def _prerender_one(date, cache_dir, bounds, res, alignment):
    return render_overview(date, cache_dir, bounds, res, alignment)


@click.group()
def cli():
    pass


@cli.command()
@click.option('--cache-dir', default=str(CACHE_DIR), show_default=True,
              type=click.Path(file_okay=False))
@click.option('--crownmap', default=CROWNMAP, show_default=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option('--nc4', 'nc4s', multiple=True, default=NC4S, show_default=True)
@click.option('--res', default=OVERVIEW_RES, show_default=True,
              help='Overview resolution (m).')
@click.option('--alignment', default='local', show_default=True,
              type=click.Choice(['local', 'global']),
              help='Phantom mosaic alignment (crown_classifications.nc4 is local).')
@click.option('--jobs', default=3, show_default=True)
def prerender(cache_dir, crownmap, nc4s, res, alignment, jobs):
    """Cache the whole-plot overview of every viewer flight."""
    bounds = grid_bounds(crownmap)
    dates = flight_dates(nc4s)
    missing = [d for d in dates if not drone_file(d, alignment).exists()]
    if missing:
        raise click.ClickException(
            f'{len(missing)} flights have no mosaic, e.g. {drone_file(missing[0], alignment)}')
    todo = [d for d in dates if not overview_path(cache_dir, d, res).exists()]
    click.echo(f'{len(dates)} flights, {len(dates) - len(todo)} cached, '
               f'{len(todo)} to render at {res:g} m over {bounds}')
    failed = []
    with ProcessPoolExecutor(jobs) as pool:
        futs = {pool.submit(_prerender_one, d, cache_dir, bounds, res, alignment): d
                for d in todo}
        for f in tqdm(as_completed(futs), total=len(futs), mininterval=30):
            d = futs[f]
            try:
                out = f.result()
                tqdm.write(f'{d:%Y-%m-%d} -> {out.name} '
                           f'({out.stat().st_size / 1e6:.1f} MB)')
            except Exception as e:  # keep going; report at the end
                failed.append(d)
                tqdm.write(f'{d:%Y-%m-%d} FAILED: {e!r}')
    if failed:
        raise click.ClickException(f'{len(failed)} flights failed: '
                                   + ', '.join(f'{d:%Y-%m-%d}' for d in failed))
    click.echo('done')


if __name__ == '__main__':
    cli()
