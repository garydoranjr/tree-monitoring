#!/usr/bin/env python
"""Cross-calibrate PlanetScope SR to BRDF-adjusted MODIS and cut 50ha clips.

Implements the histogram-matching cross-calibration of Wang et al. (2020, RSE
246:111865; Wu et al. 2021, ISPRS J. 171:36-48, Sect. 3.3.3) on the BCNM order
box (~12 x 14 km) around Barro Colorado Island:

1. Scenes are grouped by UTC acquisition date and satellite, so one fit never
   mixes two sensors, and each group is mosaicked onto the shared 3 m box grid
   (scripts/planet_box_grid.py). Pixels that are not OCM-clear (cloud, thin
   cloud and shadow dilated by --dilate pixels), water in the static mask
   (scripts/build_bcnm_water_mask.py), or nodata are dropped.
2. Clear land reflectance is aggregated onto the MODIS 500 m sinusoidal grid of
   the reference (scripts/build_modis_reference.py) as sums over each MODIS
   pixel, giving the mean reflectance and the clear-land fraction per pixel.
3. MODIS pixels are used when they are at least --min-land land, their 3 x 3
   neighbourhood is at least --min-land-nbhd land on average (a one-pixel
   buffer against MODIS geolocation error and point-spread blur from the
   lake), and they are at least --min-clear clear in the Planet mosaic.
4. Per band, Gaussian fits (mean, standard deviation) of the upscaled Planet
   and the same-day MODIS reference give offset = mu_M - gain * mu_P with
   gain = 1 (--gain-mode fixed, the default) or gain = sd_M / sd_P as in Wang
   et al. (--gain-mode sd). Over the near-uniform forest of the box the
   standard deviations are a few thousandths of reflectance, so the sd ratio
   is dominated by noise and MODIS/Planet misregistration (Feb-2024 pilot:
   blue gains 1.5-2.6, bootstrap 95% CI width 0.3-0.6) while the mean is
   pinned to ~0.001 (visible) and ~0.003 (NIR). Groups with fewer than
   --min-pixels usable pixels are left uncalibrated.
5. Every scene is clipped to the 50ha plot; the raw SR clip goes to
   <sr-root>/4band/<YYYY>/<scene>_4band.tif and, for calibrated groups, the
   calibrated clip (uint16 reflectance x 10000, nodata 0) to
   <cal-root>/4band/<YYYY>/<scene>_4band.tif, the layout of planet_clipped/.

Fit diagnostics (results CSV, one row per group and band): pixel count, Planet
and MODIS mean/sd, the applied gain and offset, Pearson r, RMSE against MODIS
before and after, and an out-of-sample RMSE from a checkerboard split (fit on
one parity of MODIS pixels, score on the other, averaged over both directions).
Uncertainty is tracked with a pixel bootstrap (--n-boot resamples): 95%
intervals of the applied offset and of the sd-ratio gain, which is reported
(gain_sd_ratio) even when not applied. The bootstrap treats 500 m pixels as
independent, so with spatial autocorrelation its intervals are lower bounds.

Typical usage:
    python scripts/calibrate_planet_modis.py
    python scripts/calibrate_planet_modis.py --start 2024-02-01 --end 2024-02-29
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import click
import numpy as np
import rasterio as rio
import xarray as xr
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy.ndimage import binary_dilation, uniform_filter
from shapely.geometry import shape
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_planet_clipped_50ha import _clip_to  # noqa: E402
from planet_box_grid import UTM, box_grid, warp_to_box  # noqa: E402
from util import load_config  # noqa: E402

BANDS = ['blue', 'green', 'red', 'nir']
SCALE = 10000.0
MODIS_SINU = CRS.from_proj4('+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 '
                            '+units=m +no_defs')
FLOWER = Path('/Volumes/Earth03/flower')
REPO = Path(__file__).resolve().parent.parent


def modis_grid(ref: xr.Dataset):
    x, y = ref['x'].values, ref['y'].values
    dx, dy = float(x[1] - x[0]), float(y[0] - y[1])
    return from_origin(x[0] - dx / 2, y[0] + dy / 2, dx, dy), len(x), len(y), dx * dy


def to_modis(arr: np.ndarray, box_tf, m_tf, m_w, m_h) -> np.ndarray:
    """Sum a box-grid float array over each MODIS pixel."""
    out = np.zeros((m_h, m_w), np.float64)
    reproject(arr.astype(np.float32), out, src_transform=box_tf, src_crs=UTM,
              dst_transform=m_tf, dst_crs=MODIS_SINU, resampling=Resampling.sum,
              src_nodata=None, dst_nodata=None)
    return out


def load_group_mosaic(scenes, stage_dir, water, box_tf, w, h, dilate):
    """Return (refl (4, h, w) float32 reflectance, valid (h, w) bool)."""
    refl = np.zeros((4, h, w), np.float32)
    valid = np.zeros((h, w), bool)
    for r in scenes:
        sid = r['scene_id']
        sr = warp_to_box(Path(r['sr_path']), [1, 2, 3, 4], box_tf, w, h, 0, np.uint16)
        cls = warp_to_box(stage_dir / 'ocm' / sid[:4] / f'{sid}_ocm.tif', [1],
                          box_tf, w, h, 255, np.uint8)[0]
        cloud = binary_dilation(np.isin(cls, (1, 2, 3)), iterations=dilate) if dilate else np.isin(cls, (1, 2, 3))
        ok = (cls == 0) & ~cloud & np.all(sr > 0, axis=0) & (water == 0) & ~valid
        refl[:, ok] = sr[:, ok] / SCALE
        valid |= ok
    return refl, valid


def sd_ratio(p: np.ndarray, m: np.ndarray) -> float:
    return np.std(m, ddof=1) / np.std(p, ddof=1)


def fit(p: np.ndarray, m: np.ndarray, gain_mode: str = 'fixed'):
    gain = sd_ratio(p, m) if gain_mode == 'sd' else 1.0
    return gain, np.mean(m) - gain * np.mean(p)


def bootstrap(p, m, gain_mode, n_boot, rng):
    """95% intervals (lo, hi) of the applied offset and of the sd-ratio gain."""
    offs, ratios = np.empty(n_boot), np.empty(n_boot)
    for k in range(n_boot):
        i = rng.integers(0, len(p), len(p))
        offs[k] = fit(p[i], m[i], gain_mode)[1]
        ratios[k] = sd_ratio(p[i], m[i])
    return np.percentile(offs, [2.5, 97.5]), np.percentile(ratios, [2.5, 97.5])


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def write_calibrated(raw_clip: Path, out: Path, gains, offsets):
    with rio.open(raw_clip) as src:
        sr = src.read().astype(np.float32)
        meta = src.meta.copy()
    nodata = np.any(sr == 0, axis=0)
    cal = np.empty_like(sr)
    for b in range(4):
        cal[b] = (gains[b] * sr[b] / SCALE + offsets[b]) * SCALE
    cal = np.clip(np.rint(cal), 1, 65535).astype(np.uint16)
    cal[:, nodata] = 0
    meta.update(dtype='uint16', nodata=0, compress='lzw')
    out.parent.mkdir(parents=True, exist_ok=True)
    with rio.open(out, 'w', **meta) as dst:
        dst.write(cal)


@click.command()
@click.option('--stage-dir', type=click.Path(path_type=Path, exists=True),
              default=FLOWER / 'planet_bcnm_sr', show_default=True)
@click.option('--reference', type=click.Path(path_type=Path, exists=True),
              default=FLOWER / 'modis/mcd43a1_bcnm/reference_nadir_sza45.nc',
              show_default=True)
@click.option('--water-mask', type=click.Path(path_type=Path), default=None,
              help='Default: <stage-dir>/aux/bcnm_water_mask.tif')
@click.option('--clip-config', type=click.Path(path_type=Path, exists=True),
              default=REPO / 'config/clip_50ha_plot.yml', show_default=True)
@click.option('--sr-root', type=click.Path(path_type=Path),
              default=FLOWER / 'planet_clipped_sr', show_default=True)
@click.option('--cal-root', type=click.Path(path_type=Path),
              default=FLOWER / 'planet_clipped_cal', show_default=True)
@click.option('--output', type=click.Path(path_type=Path),
              default=REPO / 'results/planet_modis_calibration.csv', show_default=True)
@click.option('--start', default=None, help='YYYY-MM-DD (inclusive)')
@click.option('--end', default=None, help='YYYY-MM-DD (inclusive)')
@click.option('--min-land', type=float, default=0.99, show_default=True)
@click.option('--min-land-nbhd', type=float, default=0.9, show_default=True)
@click.option('--min-clear', type=float, default=0.95, show_default=True)
@click.option('--min-pixels', type=int, default=30, show_default=True)
@click.option('--dilate', type=int, default=2, show_default=True,
              help='Cloud/shadow dilation in 3 m pixels.')
@click.option('--pixels-dir', type=click.Path(path_type=Path),
              default=REPO / 'results/planet_modis_calibration_pixels', show_default=True,
              help='Per-group npz of the matched 500 m Planet/MODIS values.')
@click.option('--no-clips', is_flag=True, help='Fit only; do not write 50ha clips.')
@click.option('--gain-mode', type=click.Choice(['fixed', 'sd']), default='fixed',
              show_default=True,
              help='fixed: gain 1, match means only; sd: gain = sd_M / sd_P (Wang et al.).')
@click.option('--n-boot', type=int, default=500, show_default=True,
              help='Bootstrap resamples for the offset / sd-ratio intervals.')
def main(stage_dir, reference, water_mask, clip_config, sr_root, cal_root, output,
         start, end, min_land, min_land_nbhd, min_clear, min_pixels, dilate,
         pixels_dir, no_clips, gain_mode, n_boot):
    water_mask = water_mask or stage_dir / 'aux' / 'bcnm_water_mask.tif'
    box_tf, w, h = box_grid()
    with rio.open(water_mask) as src:
        water = src.read(1)
    ref = xr.open_dataset(reference)
    m_tf, m_w, m_h, m_area = modis_grid(ref)
    expected = m_area / (box_tf.a * -box_tf.e)
    land_frac = to_modis((water == 0).astype(np.float32), box_tf, m_tf, m_w, m_h) / expected
    land_ok = (land_frac >= min_land) & (
        uniform_filter(land_frac, size=3, mode='constant') >= min_land_nbhd)
    click.echo(f'{land_ok.sum()} MODIS pixels pass the land criteria')
    region = shape(load_config(clip_config)['region'])
    ref_times = ref['time'].values.astype('datetime64[D]')

    groups = defaultdict(list)
    for r in csv.DictReader(open(stage_dir / 'index.csv')):
        d = r['acquired'][:10]
        if (start and d < start) or (end and d > end):
            continue
        if not (stage_dir / 'ocm' / d[:4] / f'{r["scene_id"]}_ocm.tif').exists():
            continue
        groups[(d, r['satellite_id'])].append(r)

    rows = []
    rng = np.random.default_rng(0)
    parity = (np.add.outer(np.arange(m_h), np.arange(m_w)) % 2).astype(bool)
    for (d, sat), scenes in tqdm(sorted(groups.items()), desc='groups'):
        refl, valid = load_group_mosaic(scenes, stage_dir, water, box_tf, w, h, dilate)
        n_valid = to_modis(valid.astype(np.float32), box_tf, m_tf, m_w, m_h)
        use = land_ok & (n_valid / expected >= min_clear)
        ti = np.searchsorted(ref_times, np.datetime64(d))
        has_ref = ti < len(ref_times) and ref_times[ti] == np.datetime64(d)
        n = int(use.sum()) if has_ref else 0
        base = {'date': d, 'satellite_id': sat, 'instrument': scenes[0]['instrument'],
                'scene_ids': ';'.join(s['scene_id'] for s in scenes),
                'sun_elevation': scenes[0]['sun_elevation'],
                'view_angle': scenes[0]['view_angle'], 'n_pixels': n}
        calibrated = n >= min_pixels
        gains, offsets = [np.nan] * 4, [np.nan] * 4
        matched = {}
        for b, name in enumerate(BANDS):
            row = dict(base, band=name, calibrated=calibrated)
            if calibrated:
                p = to_modis(refl[b] * valid, box_tf, m_tf, m_w, m_h)[use] / n_valid[use]
                m = ref['reflectance'].isel(time=ti).sel(band=name).values[use]
                ff = ref['fill_flag'].isel(time=ti).sel(band=name).values[use]
                ok = np.isfinite(m)
                p, m, par, ff = p[ok], m[ok], parity[use][ok], ff[ok]
                g, o = fit(p, m, gain_mode)
                gains[b], offsets[b] = g, o
                off_ci, ratio_ci = bootstrap(p, m, gain_mode, n_boot, rng)
                matched[f'planet_{name}'], matched[f'modis_{name}'] = p, m
                oos, oos_raw = [], []
                for side in (False, True):
                    tr, te = par == side, par != side
                    if tr.sum() >= min_pixels // 2 and te.sum() >= min_pixels // 2:
                        gs, os_ = fit(p[tr], m[tr], gain_mode)
                        oos.append(rmse(gs * p[te] + os_, m[te]))
                        oos_raw.append(rmse(p[te], m[te]))
                row.update(
                    planet_mean=p.mean(), planet_sd=p.std(ddof=1),
                    modis_mean=m.mean(), modis_sd=m.std(ddof=1),
                    modis_observed_frac=float(np.mean(ff == 0)), gain=g, offset=o,
                    offset_ci_lo=off_ci[0], offset_ci_hi=off_ci[1],
                    gain_sd_ratio=sd_ratio(p, m),
                    gain_sd_ratio_ci_lo=ratio_ci[0], gain_sd_ratio_ci_hi=ratio_ci[1],
                    r=float(np.corrcoef(p, m)[0, 1]), rmse_before=rmse(p, m),
                    rmse_after=rmse(g * p + o, m),
                    rmse_oos_before=np.mean(oos_raw) if oos else np.nan,
                    rmse_oos_after=np.mean(oos) if oos else np.nan)
            rows.append(row)
        if matched:
            pixels_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(pixels_dir / f'{d}_{sat}.npz', gain=gains,
                                offset=offsets, **matched)
        if no_clips:
            continue
        for s in scenes:
            sid = s['scene_id']
            raw = sr_root / '4band' / sid[:4] / f'{sid}_4band.tif'
            status = _clip_to(region, Path(s['sr_path']), raw)
            if status == 'ok' and calibrated:
                write_calibrated(raw, cal_root / '4band' / sid[:4] / f'{sid}_4band.tif',
                                 gains, offsets)

    fields = ['date', 'satellite_id', 'instrument', 'band', 'calibrated', 'n_pixels',
              'planet_mean', 'planet_sd', 'modis_mean', 'modis_sd',
              'modis_observed_frac', 'gain', 'offset',
              'offset_ci_lo', 'offset_ci_hi', 'gain_sd_ratio', 'gain_sd_ratio_ci_lo',
              'gain_sd_ratio_ci_hi', 'r', 'rmse_before', 'rmse_after', 'rmse_oos_before', 'rmse_oos_after',
              'sun_elevation', 'view_angle', 'scene_ids']
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        wr.writeheader()
        wr.writerows(rows)
    n_groups = len(rows) // 4
    n_cal = sum(r['calibrated'] for r in rows) // 4
    click.echo(f'{n_cal}/{n_groups} date-satellite groups calibrated '
               f'(gain mode {gain_mode}); wrote {output}')


if __name__ == '__main__':
    main()
