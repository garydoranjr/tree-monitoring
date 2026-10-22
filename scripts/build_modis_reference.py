#!/usr/bin/env python
"""Daily gap-filled BRDF-adjusted MODIS reflectance over BCNM (Wang et al. 2020).

Builds the reference that scripts/calibrate_planet_modis.py matches PlanetScope
to, following Wang et al. (2020, RSE 246:111865) as described in Wu et al.
(2021, ISPRS J. 171:36-48, Sect. 3.3.2):

1. BRDF adjustment. MCD43A1 kernel weights (fiso, fvol, fgeo) are evaluated
   with the RossThick-LiSparse-Reciprocal model at nadir view and a 45 deg
   solar zenith angle: rho = fiso + fvol*Kvol + fgeo*Kgeo.
2. Quality control, per day, band and pixel. Keep BRDF mandatory quality 0
   (full inversion) or 1 (magnitude inversion), then require QA30: more than
   30% of the pixels in the surrounding ~5 x 5 km (11 x 11 pixel) window pass.
3. Gap filling, per pixel, band and calendar year, by the number N of good
   days in that year:
     N >= 250       shape-preserving (PCHIP) cubic through the good days of
                    the pixel;
     50 <= N < 250  the 5 km neighbourhood seasonality (that year, smoothed
                    with a Gaussian kernel, if the neighbourhood has >= 250
                    good days, else its multi-year day-of-year mean), shifted
                    by the mean pixel-minus-reference difference on the good
                    days;
     N < 50         the neighbourhood multi-year day-of-year mean, unshifted.
   Good days keep their observed value.

   Deviations from HKU, forced by Gatun Lake (~40% of each 5 km window):
   the neighbourhood series is the pixel's own long-term median plus the mean
   anomaly (value minus that pixel's long-term median) of good neighbours of
   the same cover class (land: long-term median NIR >= 0.15), so the
   seasonality no longer follows the day-to-day land/water mix of the good
   pixels (in July 59% of the good neighbours of the 50ha pixel are water vs
   ~40% otherwise). The interpolating cubic and smoothing splines are
   replaced by PCHIP and a Gaussian-kernel mean, which cannot overshoot across
   long wet-season gaps (the splines gave 0.08-1.19 NIR in Jun-Sep 2022 at the
   50ha pixel, observed 0.29-0.33).

Output netCDF (sinusoidal grid of the input) holds `reflectance`
(band, time, y, x) for MODIS bands blue (B3), green (B4), red (B1) and NIR (B2),
i.e. in PlanetScope 4-band order, and `fill_flag` with
0 = observed, 1 = pixel spline, 2 = shifted current-year neighbourhood,
3 = shifted multi-year neighbourhood, 4 = multi-year neighbourhood unshifted.

Typical usage:
    python scripts/build_modis_reference.py \\
        /Volumes/Earth03/flower/modis/mcd43a1_bcnm/opendap/yearly/*.nc
"""
from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import PchipInterpolator, make_smoothing_spline
from scipy.ndimage import gaussian_filter1d, uniform_filter
from tqdm import tqdm

# PlanetScope band order B, G, R, N -> MODIS bands 3, 4, 1, 2
PLANET_BANDS = ['blue', 'green', 'red', 'nir']
MODIS_BAND_FOR = {'blue': 3, 'green': 4, 'red': 1, 'nir': 2}
WINDOW = 11          # ~5 km at 463 m
QA30 = 0.3
N_SPLINE, N_NEIGHBOUR = 250, 50
EDGE_DAYS = 30       # neighbouring-year days used to stabilise spline ends
SMOOTH_DAYS = 8.0    # Gaussian sigma of the current-year neighbourhood series
LAND_NIR = 0.15      # long-term median NIR separating land from water
DEFAULT_OUT = Path('/Volumes/Earth03/flower/modis/mcd43a1_bcnm/reference_nadir_sza45.nc')


# ---------------------------------------------------------------------------
# RossThick-LiSparse-Reciprocal kernels (Lucht et al. 2000; MODIS ATBD)

def ross_thick(sza, vza, raa):
    cos_xi = np.cos(sza) * np.cos(vza) + np.sin(sza) * np.sin(vza) * np.cos(raa)
    xi = np.arccos(np.clip(cos_xi, -1, 1))
    return (((np.pi / 2 - xi) * np.cos(xi) + np.sin(xi))
            / (np.cos(sza) + np.cos(vza)) - np.pi / 4)


def li_sparse_r(sza, vza, raa, h_b=2.0, b_r=1.0):
    ts = np.arctan(b_r * np.tan(sza))
    tv = np.arctan(b_r * np.tan(vza))
    d = np.sqrt(np.tan(ts) ** 2 + np.tan(tv) ** 2
                - 2 * np.tan(ts) * np.tan(tv) * np.cos(raa))
    sec_s, sec_v = 1 / np.cos(ts), 1 / np.cos(tv)
    cos_t = h_b * np.sqrt(d ** 2 + (np.tan(ts) * np.tan(tv) * np.sin(raa)) ** 2) / (sec_s + sec_v)
    t = np.arccos(np.clip(cos_t, -1, 1))
    overlap = (t - np.sin(t) * np.cos(t)) * (sec_s + sec_v) / np.pi
    cos_xi = np.cos(ts) * np.cos(tv) + np.sin(ts) * np.sin(tv) * np.cos(raa)
    return overlap - sec_s - sec_v + 0.5 * (1 + cos_xi) * sec_s * sec_v


def brdf_kernels(sza_deg=45.0, vza_deg=0.0, raa_deg=0.0):
    sza, vza, raa = np.radians([sza_deg, vza_deg, raa_deg])
    return float(ross_thick(sza, vza, raa)), float(li_sparse_r(sza, vza, raa))


# ---------------------------------------------------------------------------
# Input

def load_mcd43a1(path: Path) -> tuple[xr.Dataset, dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Return (dataset, params[band] (t, y, x, 3), quality[band] (t, y, x))."""
    ds = xr.open_dataset(path, mask_and_scale=True)
    params, quality = {}, {}
    for b in MODIS_BAND_FOR.values():
        p = ds[f'BRDF_Albedo_Parameters_Band{b}']
        par_dim = [d for d in p.dims if d not in ('time', 'ydim', 'xdim', 'y', 'x')][0]
        params[b] = p.transpose('time', ..., par_dim).values.astype(np.float32)
        quality[b] = ds[f'BRDF_Albedo_Band_Mandatory_Quality_Band{b}'].values
    return ds, params, quality


# ---------------------------------------------------------------------------
# QC and gap filling

def neighbourhood_mean(values: np.ndarray, good: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-day WINDOW x WINDOW mean of good values and good fraction; (t, y, x)."""
    v = np.where(good, values, 0).astype(np.float64)
    s = uniform_filter(v, size=(1, WINDOW, WINDOW), mode='nearest')
    frac = uniform_filter(good.astype(np.float64), size=(1, WINDOW, WINDOW), mode='nearest')
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(frac > 0, s / frac, np.nan)
    return mean, frac


def neighbourhood_anomaly(values: np.ndarray, good: np.ndarray, land: np.ndarray
                          ) -> np.ndarray:
    """Per-day neighbourhood series that ignores the land/water mix: each pixel's
    long-term median plus the WINDOW x WINDOW mean anomaly of good neighbours
    of the same class (`land`, (y, x) bool); (t, y, x)."""
    with np.errstate(invalid='ignore'):
        med = np.nanmedian(np.where(good, values, np.nan), axis=0)
    anom = values - med
    out = np.full(values.shape, np.nan)
    for cls in (True, False):
        g = good & (land == cls)
        m, frac = neighbourhood_mean(anom, g)
        out[:, land == cls] = m[:, land == cls]
    return med + out


def kernel_smooth(series: np.ndarray, sigma: float = SMOOTH_DAYS) -> np.ndarray:
    """Gaussian-kernel weighted mean along time of a (t, y, x) series with NaN
    gaps; NaN where no data lies within ~2 sigma."""
    ok = np.isfinite(series)
    num = gaussian_filter1d(np.where(ok, series, 0.0), sigma, axis=0, mode='nearest')
    den = gaussian_filter1d(ok.astype(np.float64), sigma, axis=0, mode='nearest')
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0.05, num / den, np.nan)


def doy_climatology(series: np.ndarray, doy: np.ndarray) -> np.ndarray:
    """Multi-year mean by day of year (1..365; 366 folded into 365); (365, y, x)."""
    d = np.minimum(doy, 365) - 1
    out = np.full((365,) + series.shape[1:], np.nan)
    for k in range(365):
        sel = series[d == k]
        if len(sel):
            with np.errstate(invalid='ignore'):
                out[k] = np.nanmean(sel, axis=0)
    return out


def smooth_periodic(clim: np.ndarray) -> np.ndarray:
    """Smoothing spline of a 365-day climatology with periodic wrap."""
    ok = np.isfinite(clim)
    if ok.sum() < 10:
        return np.full(365, np.nanmean(clim) if ok.any() else np.nan)
    x = np.arange(365)
    xx = np.concatenate([x[ok] - 365, x[ok], x[ok] + 365])
    yy = np.tile(clim[ok], 3)
    return make_smoothing_spline(xx, yy)(x)


def gap_fill_band(refl: np.ndarray, good: np.ndarray, times: pd.DatetimeIndex,
                  land: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gap-fill one band; refl/good are (t, y, x), land (y, x). Returns (filled, flag)."""
    nb_mean = neighbourhood_anomaly(refl, good, land)
    nb_smooth = kernel_smooth(nb_mean)
    doy = times.dayofyear.values
    clim = doy_climatology(nb_mean, doy)
    nt, ny, nx = refl.shape
    clim_fit = np.empty((365, ny, nx))
    for i in range(ny):
        for j in range(nx):
            clim_fit[:, i, j] = smooth_periodic(clim[:, i, j])

    filled = np.where(good, refl, np.nan).astype(np.float32)
    flag = np.where(good, 0, 255).astype(np.uint8)
    tnum = (times - times[0]).days.values.astype(float)
    years = times.year.values
    for yr in tqdm(np.unique(years), desc='years', leave=False):
        in_yr = years == yr
        win = (tnum >= tnum[in_yr][0] - EDGE_DAYS) & (tnum <= tnum[in_yr][-1] + EDGE_DAYS)
        clim_yr = clim_fit[np.minimum(doy[in_yr], 365) - 1]          # (nyr, y, x)
        for i in range(ny):
            for j in range(nx):
                g_yr = good[in_yr, i, j]
                n = int(g_yr.sum())
                miss = ~g_yr
                if not miss.any():
                    continue
                if n >= N_SPLINE:
                    gw = good[win, i, j]
                    cs = PchipInterpolator(tnum[win][gw], refl[win, i, j][gw])
                    est, code = cs(tnum[in_yr]), 1
                elif n >= N_NEIGHBOUR:
                    nb_yr = nb_mean[in_yr, i, j]
                    if np.isfinite(nb_yr).sum() >= N_SPLINE:
                        ref = nb_smooth[in_yr, i, j]
                        ref = np.where(np.isfinite(ref), ref, clim_yr[:, i, j])
                        code = 2
                    else:
                        ref, code = clim_yr[:, i, j], 3
                    shift = np.mean(refl[in_yr, i, j][g_yr] - ref[g_yr])
                    est = ref + shift
                else:
                    est, code = clim_yr[:, i, j], 4
                idx = np.flatnonzero(in_yr)[miss]
                filled[idx, i, j] = est[miss]
                flag[idx, i, j] = code
    return filled, flag


@click.command()
@click.argument('mcd43a1', nargs=-1, required=True,
                type=click.Path(path_type=Path, exists=True))
@click.option('--output', type=click.Path(path_type=Path), default=DEFAULT_OUT,
              show_default=True)
@click.option('--sza', type=float, default=45.0, show_default=True)
def main(mcd43a1, output, sza):
    kvol, kgeo = brdf_kernels(sza_deg=sza)
    click.echo(f'kernels at SZA {sza}, nadir: Kvol={kvol:.4f} Kgeo={kgeo:.4f}')
    parts = [load_mcd43a1(p) for p in mcd43a1]
    ds0 = parts[0][0]
    times = pd.DatetimeIndex(np.concatenate([p[0]['time'].values for p in parts]))
    order = np.argsort(times)
    times = times[order]
    def reflectance_qc(name):
        b = MODIS_BAND_FOR[name]
        par = np.concatenate([p[1][b] for p in parts])[order]
        qa = np.concatenate([p[2][b] for p in parts])[order]
        refl = par[..., 0] + kvol * par[..., 1] + kgeo * par[..., 2]
        good = np.isfinite(refl) & (qa <= 1)
        _, frac = neighbourhood_mean(refl, good)
        return refl, good & (frac > QA30)

    refl, good = reflectance_qc('nir')
    with np.errstate(invalid='ignore'):
        land = np.nanmedian(np.where(good, refl, np.nan), axis=0) >= LAND_NIR
    click.echo(f'{land.mean():.1%} of pixels are land (median NIR >= {LAND_NIR})')
    refl_out, flag_out = [], []
    for name in PLANET_BANDS:
        refl, good = reflectance_qc(name)
        click.echo(f'{name} (MODIS B{MODIS_BAND_FOR[name]}): '
                   f'{good.mean():.1%} of pixel-days pass QC')
        filled, flag = gap_fill_band(refl, good, times, land)
        refl_out.append(filled)
        flag_out.append(flag)
    ydim = [d for d in ds0[f'BRDF_Albedo_Band_Mandatory_Quality_Band1'].dims if d != 'time']
    out = xr.Dataset(
        {'reflectance': (('band', 'time', 'y', 'x'), np.stack(refl_out)),
         'fill_flag': (('band', 'time', 'y', 'x'), np.stack(flag_out)),
         'land': (('y', 'x'), land.astype(np.uint8))},
        coords={'band': PLANET_BANDS, 'time': times,
                'y': ds0[ydim[0]].values, 'x': ds0[ydim[1]].values},
        attrs={'source': 'MCD43A1.061', 'sza_deg': sza, 'vza_deg': 0.0,
               'Kvol': kvol, 'Kgeo': kgeo,
               'modis_bands': 'blue=B3 green=B4 red=B1 nir=B2',
               'fill_flag': '0 observed, 1 pixel spline, 2 shifted current-year '
                            'neighbourhood, 3 shifted multi-year neighbourhood, '
                            '4 multi-year neighbourhood'},
    )
    if 'crs' in ds0:
        out['crs'] = ds0['crs']
        out['reflectance'].attrs['grid_mapping'] = 'crs'
    output.parent.mkdir(parents=True, exist_ok=True)
    enc = {v: {'zlib': True, 'complevel': 4} for v in ('reflectance', 'fill_flag')}
    out.to_netcdf(output, encoding=enc)
    for k, name in enumerate(PLANET_BANDS):
        f = np.stack(flag_out)[k]
        shares = {c: float(np.mean(f[f != 0] == c)) for c in (1, 2, 3, 4)}
        click.echo(f'{name}: filled-day share by scenario {shares}')
    click.echo(f'wrote {output}')


if __name__ == '__main__':
    main()
