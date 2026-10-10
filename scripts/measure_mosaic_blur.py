#!/usr/bin/env python
"""Effective resolution and motion blur of the 50ha global-alignment drone mosaics.

Blur multiplies an image's power spectrum by |MTF|^2, so on one patch of
ground the ratio of two mosaics' spectra is their relative |MTF|^2 with the
scene cancelled; spectra ignore shifts, so the ~0.5 m offsets between global
mosaics do not matter. For a Gaussian-equivalent PSF with covariance Sigma
(m^2) and spatial frequency f = (f_east, f_north) in cycles/m,

    ln[P_i(f) / P_ref(f)] = c - 4 pi^2 f^T (Sigma_i - Sigma_ref) f,

which is linear in the three entries of dSigma and a free offset c that
absorbs contrast, illumination and sensor-gain differences. A linear motion
of length L adds L^2/12 of variance along the motion only.

Two references are used. The isotropic part of dSigma (its trace) is taken
against the per-bin median of all mosaics, which is anchored to an absolute
width by fitting P(f) = A f^-alpha exp(-4 pi^2 s^2 f^2) + N to it. The
anisotropic part (blur direction and motion length) is taken against the
median of mosaics within +/-30 days of the same day of year: against the
all-dates reference the fitted blur direction tracks the solar azimuth
(perpendicular to it, within ~15 deg), because shading changes the canopy
texture's anisotropy through the year. See docs/mosaic_blur.md.

The spectra come from a fixed grid of spot-check tiles (60 x 20.48 m) inside
the phantom 50ha grid, read at native resolution from every phantom
(24782016, C3KW2X) and mavic global mosaic and binned on common
physical-frequency polar bins, so the variable mavic GSD (4.1-5.2 cm) needs
no resampling. Local-alignment mosaics are not used: their warping can look
like blur.

Subcommands:

  tiles     write the tile grid (blur_tiles.geojson in the cache directory)
  spectra   per-mosaic binned spectra + sharpness indices -> CACHE/*.npz
            (restartable; skips mosaics already done)
  validate  blur one mosaic's tiles with known Gaussian / motion kernels and
            check the fit recovers them (blur_validation figure)
  exif      per-mission exposure, flight speed and flight-line azimuth from
            the phantom 24782016 EXIF table (blur_exif.csv)
  fit       dSigma per mosaic, the absolute anchor, mosaic_blur.csv
  plot      blur_resolution_timeseries, blur_anisotropy, blur_spectra_examples,
            blur_anchor_check, blur_exif_crosscheck, blur_mavic_metadata

Mavic has no EXIF here; stri/globus/metadata_all.csv supplies the GNSS
correction (antenna), processing comments and native GSD per flight, which
`fit` joins into mosaic_blur.csv.

Directions are azimuths in degrees clockwise from north, modulo 180.

Typical usage:
    python scripts/measure_mosaic_blur.py tiles
    python scripts/measure_mosaic_blur.py spectra
    python scripts/measure_mosaic_blur.py validate
    python scripts/measure_mosaic_blur.py exif
    python scripts/measure_mosaic_blur.py fit
    python scripts/measure_mosaic_blur.py plot
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import ast
import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import plot_update_figures as pu

CACHE = pu.FLOWER / 'results' / 'mosaic_blur'
EXIF_CSV = pu.PHANTOM_RELEASES['24782016'] / 'raw_images_metadata.csv'
MAVIC_META = pu.MAVIC_RGB.parent / 'metadata_all.csv'

TILE_M = 20.48           # tile side (m)
TILE_GRID = (10, 6)      # columns x rows of tile centres
INSET_M = 30.0           # keep tiles this far inside the phantom grid
F_EDGES = np.arange(0.25, 9.5 + 1e-9, 0.25)   # cycles/m; mavic Nyquist >= 9.63
N_ANG = 12               # orientation bins over 180 deg
F_NORM = (0.25, 1.0)     # band used to normalize spectra before taking the median
F_LO = 0.75              # lowest frequency in the dSigma fit
SNR = 4.0                # bins must exceed SNR x the noise floor in both spectra
F_ANISO_LO, F_ANISO_HI = 0.5, 6.0   # anisotropy fit band (cycles/m); fixed so it does not shrink with blur
MAX_ATTEN = 3.0          # keep bins where |4 pi^2 f^T dSigma f| < this (Gaussian regime)
FWHM = 2 * np.sqrt(2 * np.log(2))


# --------------------------------------------------------------------------
# mosaics and tiles
# --------------------------------------------------------------------------
def global_mosaics():
    """(path, source dict) for every phantom and mavic global mosaic, by date."""
    paths = [p for r in pu.PHANTOM_RELEASES
             for p in pu.phantom_dir(r, 'global').glob('*_global.tif')]
    paths += list(pu.MAVIC_RGB.glob('*_M3M_aligned_global_RGB.tif'))
    out = [(p, pu.classification_source(p)) for p in paths]
    return sorted(out, key=lambda t: (t[1]['date'], t[1]['series']))


def tile_grid():
    """DataFrame of tile id, centre x/y (UTM 17N) and bounds."""
    import rasterio
    ref = next(pu.phantom_dir('24782016', 'global').glob('*_global.tif'))
    with rasterio.open(ref) as src:
        b = src.bounds
    h = TILE_M / 2
    xs = np.linspace(b.left + INSET_M + h, b.right - INSET_M - h, TILE_GRID[0])
    ys = np.linspace(b.top - INSET_M - h, b.bottom + INSET_M + h, TILE_GRID[1])
    rows = [dict(tile=i * len(xs) + j, row=i, x=x, y=y,
                 left=x - h, bottom=y - h, right=x + h, top=y + h)
            for i, y in enumerate(ys) for j, x in enumerate(xs)]
    return pd.DataFrame(rows)


def luminance(arr, mavic):
    """Float luminance and a validity flag.

    Mavic tiles must have no alpha = 0 pixel. Phantom has no alpha (band 4
    is height) and deep shadows clip to RGB = 0, so isolated black pixels
    are data; a tile is dropped only when over 0.5 % of it is black.
    """
    rgb = arr[:3].astype(np.float64)
    if mavic:
        ok = not (arr[3] == 0).any()
    else:
        ok = (rgb.sum(axis=0) == 0).mean() <= 0.005
    y = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return y, ok


def read_tiles(path, tiles, mavic, pad=0):
    """Read each tile at native resolution, with `pad` extra pixels per side.

    Tiles sharing a grid row are read as one window, since the phantom
    mosaics are uncompressed one-row strips. Returns (list of luminance
    arrays or None, pixel size in m).
    """
    import rasterio
    from rasterio.windows import Window
    out = [None] * len(tiles)
    with rasterio.open(path) as src:
        res = src.res[0]
        n = int(round(TILE_M / res))
        inv = ~src.transform
        cols, rows = [], []
        for t in tiles.itertuples():
            c, r = inv * (t.x, t.y)
            cols.append(int(round(c)) - n // 2 - pad)
            rows.append(int(round(r)) - n // 2 - pad)
        cols, rows = np.array(cols), np.array(rows)
        m = n + 2 * pad
        for gr in sorted(tiles.row.unique()):
            idx = np.flatnonzero(tiles.row.to_numpy() == gr)
            r0, r1 = rows[idx].min(), rows[idx].max() + m
            c0, c1 = cols[idx].min(), cols[idx].max() + m
            if r0 < 0 or c0 < 0 or r1 > src.height or c1 > src.width:
                continue
            bands = [1, 2, 3, 4] if mavic else [1, 2, 3]
            block = src.read(bands, window=Window(c0, r0, c1 - c0, r1 - r0))
            for k in idx:
                a = block[:, rows[k] - r0:rows[k] - r0 + m, cols[k] - c0:cols[k] - c0 + m]
                y, ok = luminance(a, mavic)
                out[k] = y if ok else None
    return out, res


# --------------------------------------------------------------------------
# spectra
# --------------------------------------------------------------------------
def tile_spectrum(y, res):
    """Binned power spectrum of one tile.

    Returns (power [n_f, n_ang], counts [n_f, n_ang], noise floor). The
    noise floor is the mean power beyond 0.9 x Nyquist (the FFT corners).
    """
    n = y.shape[0]
    w = np.hanning(n)
    w2 = np.outer(w, w)
    z = (y - y.mean()) * w2
    p = np.abs(np.fft.fft2(z)) ** 2 * res ** 2 / (w2 ** 2).sum()
    fc = np.fft.fftfreq(n, d=res)               # along columns (east)
    fr = -np.fft.fftfreq(n, d=res)              # along rows, flipped to north
    fx, fy = np.meshgrid(fc, fr)
    f = np.hypot(fx, fy)
    ang = np.mod(np.arctan2(fx, fy), np.pi)     # azimuth of the frequency vector
    abin = np.floor(np.mod(ang + np.pi / N_ANG / 2, np.pi) / (np.pi / N_ANG)).astype(int)
    fbin = np.digitize(f, F_EDGES) - 1
    ok = (fbin >= 0) & (fbin < len(F_EDGES) - 1)
    flat = fbin[ok] * N_ANG + abin[ok]
    nb = (len(F_EDGES) - 1) * N_ANG
    cnt = np.bincount(flat, minlength=nb).reshape(-1, N_ANG)
    tot = np.bincount(flat, weights=p[ok], minlength=nb).reshape(-1, N_ANG)
    power = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    noise = p[f > 0.9 / (2 * res)].mean()
    return power, cnt, noise


def sharpness_indices(y):
    """Laplacian variance and Tenengrad, both divided by the tile variance."""
    from scipy import ndimage
    v = y.var()
    lap = ndimage.laplace(y)[1:-1, 1:-1].var() / v
    gx, gy = ndimage.sobel(y, 1), ndimage.sobel(y, 0)
    ten = (gx ** 2 + gy ** 2)[1:-1, 1:-1].mean() / v
    return lap, ten


def mosaic_spectra(lums, res):
    nf, nt = len(F_EDGES) - 1, len(lums)
    power = np.full((nt, nf, N_ANG), np.nan)
    counts = np.zeros((nt, nf, N_ANG), int)
    noise = np.full(nt, np.nan)
    lap = np.full(nt, np.nan)
    ten = np.full(nt, np.nan)
    for k, y in enumerate(lums):
        if y is None:
            continue
        power[k], counts[k], noise[k] = tile_spectrum(y, res)
        lap[k], ten[k] = sharpness_indices(y)
    return dict(power=power, counts=counts, noise=noise, lap=lap, ten=ten,
                valid=~np.isnan(noise))


def f_centres():
    return (F_EDGES[:-1] + F_EDGES[1:]) / 2


def ang_centres():
    return np.arange(N_ANG) * np.pi / N_ANG


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------
def design():
    """Columns [1, f_E^2, 2 f_E f_N, f_N^2] x -4 pi^2 on the (f, angle) bins."""
    f = f_centres()[:, None]
    a = ang_centres()[None, :]
    fe, fn = f * np.sin(a), f * np.cos(a)
    k = -4 * np.pi ** 2
    cols = [np.ones_like(fe * fn), k * fe ** 2, k * 2 * fe * fn, k * fn ** 2]
    return np.stack([np.broadcast_to(c, fe.shape) for c in cols], axis=-1)


DESIGN = design()


def fit_dsigma(p_i, p_ref, n_i, n_ref, counts):
    """Weighted LS fit of ln(p_i/p_ref) -> (c, s_EE, s_EN, s_NN) in m^2, or None."""
    f = np.broadcast_to(f_centres()[:, None], p_i.shape)
    mask = ((f >= F_LO) & (p_i > SNR * n_i) & (p_ref > SNR * n_ref)
            & (counts > 0) & np.isfinite(p_i) & np.isfinite(p_ref))
    y = np.log(p_i / p_ref)
    X = DESIGN
    beta = None
    for _ in range(3):
        if mask.sum() < 12:
            return None
        w = np.sqrt(counts[mask])
        beta, *_ = np.linalg.lstsq(X[mask] * w[:, None], y[mask] * w, rcond=None)
        atten = np.abs(X[..., 1:] @ beta[1:])
        new = mask & (atten < MAX_ATTEN)
        if (new == mask).all():
            break
        mask = new
    return beta


def sigma_summary(s_ee, s_en, s_nn, base=0.0):
    """Major/minor sigma (m), motion length (m) and azimuth (deg) of Sigma."""
    S = np.array([[s_ee + base, s_en], [s_en, s_nn + base]])
    lam, vec = np.linalg.eigh(S)
    lmin, lmax = lam
    v = vec[:, 1]
    az = np.degrees(np.arctan2(v[0], v[1])) % 180
    return dict(sigma_major=np.sqrt(max(lmax, 0)), sigma_minor=np.sqrt(max(lmin, 0)),
                motion_len=np.sqrt(12 * max(lmax - lmin, 0)), azimuth=az,
                var_major=lmax, var_minor=lmin)


def fit_aniso(p, n, counts):
    """Split one tile's own spectral anisotropy into scene and blur parts.

    At each radial bin the second angular harmonic of ln P is
    h(f) = h0 + 4 pi^2 f^2 (a, -b) for a deviatoric blur covariance
    D = [[a, b], [b, -a]] (m^2): blur anisotropy grows as f^2, while the
    texture anisotropy that shading gives the canopy is taken as constant
    across f (h0). Returns (h0_cos, h0_sin, a, b), or None.
    """
    f = f_centres()
    th = ang_centres()
    ok = (p > SNR * n) & (counts > 0) & np.isfinite(p)
    rows = (f >= F_ANISO_LO) & (f <= F_ANISO_HI) & ok.all(axis=1)
    if rows.sum() < 6:
        return None
    lp = np.log(p[rows])
    hc = 2 * (lp * np.cos(2 * th)).mean(axis=1)
    hs = 2 * (lp * np.sin(2 * th)).mean(axis=1)
    X = np.c_[np.ones(rows.sum()), 4 * np.pi ** 2 * f[rows] ** 2]
    w = np.sqrt(counts[rows].sum(axis=1))
    bc = np.linalg.lstsq(X * w[:, None], hc * w, rcond=None)[0]
    bs = np.linalg.lstsq(X * w[:, None], hs * w, rcond=None)[0]
    return np.array([bc[0], bs[0], bc[1], -bs[1]])


def blur_summary(d_glob, dev, base):
    """Blur widths from the isotropic dSigma against the all-dates reference
    (anchored at `base`) and the deviatoric part (a, b) from fit_aniso."""
    iso = base + (d_glob[0] + d_glob[2]) / 2
    a, b = dev
    r = np.hypot(a, b)
    return dict(sigma_iso=np.sqrt(max(iso, 0)), sigma_major=np.sqrt(max(iso + r, 0)),
                sigma_minor=np.sqrt(max(iso - r, 0)), motion_len=np.sqrt(24 * r),
                azimuth=sigma_summary(a, b, -a)['azimuth'])


def scene_azimuth(h0c, h0s):
    """Azimuth (deg) of frequency vectors with the most texture power."""
    return np.degrees(np.arctan2(h0s, h0c)) / 2 % 180


def combine_tiles(betas, n_boot=500, seed=0):
    """Median of the per-tile dSigma entries, with a bootstrap over tiles."""
    b = np.array(betas)[:, 1:]
    med = np.median(b, axis=0)
    rng = np.random.default_rng(seed)
    boots = np.array([np.median(b[rng.integers(0, len(b), len(b))], axis=0)
                      for _ in range(n_boot)])
    return med, boots


def normalized(power):
    """Each tile's spectrum over its mean power in the F_NORM band, and that mean."""
    f = f_centres()
    band = (f >= F_NORM[0]) & (f < F_NORM[1])
    norm = np.nanmean(power[:, band], axis=(1, 2))
    return power / norm[:, None, None], norm


def anchor_model(f, log_a, alpha, sigma, log_n):
    return np.log(np.exp(log_a) * f ** -alpha * np.exp(-4 * np.pi ** 2 * sigma ** 2 * f ** 2)
                  + np.exp(log_n))


def fit_anchor(radial, f=None):
    """Fit the power-law x Gaussian MTF + noise model to a radial spectrum."""
    from scipy.optimize import curve_fit
    f = f_centres() if f is None else f
    ok = np.isfinite(radial) & (radial > 0)
    p0 = [np.log(radial[ok][0]) + 2 * np.log(f[ok][0]), 2.0, 0.03, np.log(radial[ok][-1]) - 1]
    popt, pcov = curve_fit(anchor_model, f[ok], np.log(radial[ok]), p0=p0,
                           bounds=([-50, 0, 0, -60], [50, 6, 0.5, 50]), maxfev=20000)
    return dict(log_a=popt[0], alpha=popt[1], sigma=popt[2], log_n=popt[3],
                sigma_err=np.sqrt(pcov[2, 2]))


def radial(power):
    """Angle-averaged spectrum of a [n_f, n_ang] array."""
    return np.nanmean(power, axis=-1)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
@click.group()
def cli():
    pass


cache_option = click.option('--cache', type=click.Path(file_okay=False, path_type=Path),
                            default=CACHE, show_default=True)
out_option = click.option('--outdir', type=click.Path(file_okay=False, path_type=Path),
                          default=pu.DEFAULT_OUT, show_default=True)


@cli.command()
@cache_option
def tiles(cache):
    """Write the spot-check tile grid."""
    import geopandas as gpd
    from shapely.geometry import box
    cache.mkdir(parents=True, exist_ok=True)
    t = tile_grid()
    gdf = gpd.GeoDataFrame(t, geometry=[box(r.left, r.bottom, r.right, r.top)
                                        for r in t.itertuples()], crs='EPSG:32617')
    gdf.to_file(cache / 'blur_tiles.geojson', driver='GeoJSON')
    click.echo(f'Wrote {len(t)} tiles to {cache / "blur_tiles.geojson"}')


@cli.command()
@cache_option
@click.option('--limit', type=int, default=None, help='At most this many mosaics per series.')
def spectra(cache, limit):
    """Per-mosaic binned spectra of the spot-check tiles."""
    import time
    cache.mkdir(parents=True, exist_ok=True)
    t = tile_grid()
    mosaics = global_mosaics()
    if limit:
        by = {}
        for p, s in mosaics:
            by.setdefault(s['series'], []).append((p, s))
        mosaics = [m for v in by.values()
                   for m in [v[int(i)] for i in np.linspace(0, len(v) - 1, min(limit, len(v)))]]
    for p, s in mosaics:
        out = cache / f'{p.stem}.npz'
        if out.exists():
            continue
        t0 = time.time()
        lums, res = read_tiles(p, t, s['series'] == 'mavic')
        d = mosaic_spectra(lums, res)
        np.savez_compressed(out, res=res, **d)
        click.echo(f'{p.name}: {d["valid"].sum()}/{len(t)} tiles, '
                   f'GSD {res * 100:.2f} cm, {time.time() - t0:.0f} s')


def mavic_meta():
    """Per-mission GNSS correction, processing comments and native GSD of the mavic flights.

    metadata_all.csv has no flight time, exposure or speed, so these are the
    only flight covariates available for mavic.
    """
    meta = pd.read_csv(MAVIC_META, keep_default_na=False)
    g = meta.groupby('mission')
    ortho = meta[(meta.product_family == 'raw') & (meta.type == 'orthomosaic')]
    out = pd.DataFrame({
        'antenna': g.antenna.first().replace('', 'none'),
        'comment_processing': g.comment_processing.agg(lambda c: '; '.join(sorted(set(c) - {''}))),
        'native_gsd_cm': ortho.set_index('mission').scale_x.astype(float) * 100,
    })
    out['date'] = [pu.path_date(m + '.tif') for m in out.index]
    return out.reset_index(drop=True)


def load_cache(cache):
    rows, arrs = [], []
    for p, s in global_mosaics():
        f = cache / f'{p.stem}.npz'
        if not f.exists():
            continue
        d = dict(np.load(f))
        rows.append(dict(stem=p.stem, path=str(p), date=s['date'], series=s['series'],
                         release=s['release'], gsd_cm=float(d['res']) * 100))
        arrs.append(d)
    return pd.DataFrame(rows), arrs


@cli.command()
@cache_option
@out_option
def fit(cache, outdir):
    """Fit dSigma per mosaic and anchor it; write mosaic_blur.csv."""
    meta, arrs = load_cache(cache)
    if meta.empty:
        raise click.ClickException(f'no spectra in {cache}; run `spectra` first')
    P, norm = map(np.stack, zip(*(normalized(a['power']) for a in arrs)))  # [m, t, f, a]
    N = np.stack([a['noise'] for a in arrs]) / norm               # [m, t]
    C = np.stack([a['counts'] for a in arrs])
    valid = np.stack([a['valid'] for a in arrs])
    with np.errstate(all='ignore'):
        ref = np.nanmedian(P, axis=0)                              # [t, f, a]
        n_ref = np.nanmedian(N, axis=0)
    # absolute anchor on the tile-median radial reference spectrum
    ref_radial = np.nanmedian(np.stack([radial(r) for r in ref]), axis=0)
    anchor = fit_anchor(ref_radial)
    s_ref = anchor['sigma']
    click.echo(f'reference spectrum: sigma {s_ref * 100:.2f} +/- '
               f'{anchor["sigma_err"] * 100:.2f} cm, alpha {anchor["alpha"]:.2f}')

    rows, per_tile = [], []
    for i, m in meta.iterrows():
        bg, ba = [], []
        for k in np.flatnonzero(valid[i]):
            g = fit_dsigma(P[i, k], ref[k], N[i, k], n_ref[k], C[i, k])
            q = fit_aniso(P[i, k], N[i, k], C[i, k])
            if g is None or q is None:
                continue
            bg.append(g)
            ba.append(q)
            per_tile.append(dict(stem=m.stem, tile=k, c=g[0], d_ee=g[1], d_en=g[2], d_nn=g[3],
                                 scene_cos=q[0], scene_sin=q[1], dev_a=q[2], dev_b=q[3]))
        row = m.to_dict()
        row['n_tiles'] = len(bg)
        a = arrs[i]
        row['lap_var'] = np.nanmedian(a['lap'])
        row['tenengrad'] = np.nanmedian(a['ten'])
        try:
            own = fit_anchor(np.nanmedian(np.stack([radial(p) for p in P[i][valid[i]]]), axis=0))
            row['sigma_param_cm'] = own['sigma'] * 100
        except Exception:
            row['sigma_param_cm'] = np.nan
        if len(bg) >= 5:
            both = np.hstack([np.array(bg)[:, 1:], np.array(ba)])
            med = np.median(both, axis=0)
            rng = np.random.default_rng(0)
            boots = np.array([np.median(both[rng.integers(0, len(both), len(both))], axis=0)
                              for _ in range(500)])
            ss = blur_summary(med[:3], med[5:], s_ref ** 2)
            bt = pd.DataFrame([blur_summary(b[:3], b[5:], s_ref ** 2) for b in boots])
            row['d_ee'], row['d_en'], row['d_nn'] = med[:3]
            row['dvar_iso_m2'] = (med[0] + med[2]) / 2
            row['dev_a'], row['dev_b'] = med[5:]
            row['scene_aniso'] = np.hypot(med[3], med[4])
            row['scene_azimuth_deg'] = scene_azimuth(med[3], med[4])
            for key in ('sigma_iso', 'sigma_major', 'sigma_minor', 'motion_len'):
                row[f'{key}_cm'] = ss[key] * 100
                row[f'{key}_lo_cm'], row[f'{key}_hi_cm'] = np.percentile(bt[key], [2.5, 97.5]) * 100
            for key in ('iso', 'major', 'minor'):
                for suf in ('', '_lo', '_hi'):
                    row[f'fwhm_{key}{suf}_cm'] = row[f'sigma_{key}{suf}_cm'] * FWHM
            row['fwhm_iso_px'] = row['fwhm_iso_cm'] / row['gsd_cm']
            row['anisotropy'] = ss['sigma_major'] / max(ss['sigma_minor'], 1e-6)
            row['azimuth_deg'] = ss['azimuth']
            z = np.exp(2j * np.radians(bt.azimuth))
            row['azimuth_spread_deg'] = np.degrees(np.sqrt(-2 * np.log(abs(z.mean())))) / 2
        rows.append(row)
    out = pd.DataFrame(rows)
    mm = mavic_meta()
    mm['series'] = 'mavic'
    out = out.merge(mm, on=['date', 'series'], how='left')
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / 'mosaic_blur.csv', index=False)
    pd.DataFrame(per_tile).to_csv(cache / 'mosaic_blur_tiles.csv', index=False)
    np.savez(cache / 'reference.npz', ref=ref, n_ref=n_ref, ref_radial=ref_radial,
             **{k: v for k, v in anchor.items()})
    (cache / 'anchor.json').write_text(json.dumps(anchor, indent=1))
    click.echo(f'Wrote {outdir / "mosaic_blur.csv"} ({len(out)} mosaics)')
    cols = ['fwhm_iso_cm', 'fwhm_major_cm', 'fwhm_minor_cm', 'motion_len_cm', 'anisotropy']
    click.echo(out.groupby('series')[cols].median().round(2).to_string())


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------
def motion_kernel(length_px, azimuth_deg):
    """Linear motion PSF of `length_px` along `azimuth_deg` (bilinear splat)."""
    k = int(np.ceil(length_px)) + 3
    k += 1 - k % 2
    c = k // 2
    a = np.radians(azimuth_deg)
    s = np.linspace(-length_px / 2, length_px / 2, 401)
    cols, rows = c + s * np.sin(a), c - s * np.cos(a)
    ker = np.zeros((k, k))
    c0, r0 = np.floor(cols).astype(int), np.floor(rows).astype(int)
    fc, fr = cols - c0, rows - r0
    for dr, dc, w in ((0, 0, (1 - fr) * (1 - fc)), (0, 1, (1 - fr) * fc),
                      (1, 0, fr * (1 - fc)), (1, 1, fr * fc)):
        np.add.at(ker, (r0 + dr, c0 + dc), w)
    return ker / ker.sum()


def kernel_sigma(ker, res):
    """(s_EE, s_EN, s_NN) in m^2 of a PSF array (rows run south)."""
    r, c = np.indices(ker.shape)
    e, n = c - (c * ker).sum(), -(r - (r * ker).sum())
    return tuple((a * b * ker).sum() * res ** 2 for a, b in ((e, e), (e, n), (n, n)))


@cli.command()
@cache_option
@out_option
@click.option('--mosaic', default='BCI_50ha_2019_04_24_global',
              show_default=True, help='Stem of the mosaic whose tiles are blurred.')
@click.option('--pad', default=16, show_default=True)
def validate(cache, outdir, mosaic, pad):
    """Blur one mosaic's tiles with known kernels and recover them."""
    from scipy import ndimage
    match = [(p, s) for p, s in global_mosaics() if p.stem == mosaic]
    if not match:
        raise click.ClickException(f'no global mosaic {mosaic}')
    path, src = match[0]
    mavic = src['series'] == 'mavic'
    t = tile_grid()
    lums, res = read_tiles(path, t, mavic, pad=pad)
    keep = [k for k, y in enumerate(lums) if y is not None]
    crop = lambda y: y[pad:-pad, pad:-pad]
    base = mosaic_spectra([crop(lums[k]) for k in keep], res)
    nb, norm_b = normalized(base['power'])
    aniso_b = [fit_aniso(nb[j], base['noise'][j] / norm_b[j], base['counts'][j])
               for j in range(len(keep))]
    kernels = [('gaussian', s, np.nan) for s in (0.5, 1, 1.5, 2, 2.5, 3)]
    kernels += [('motion', L, az) for L in (1, 2, 3, 4, 6, 8) for az in (0, 30, 90, 135)]
    rows = []
    for kind, size, az in kernels:
        blurred = []
        if kind == 'gaussian':
            true = sigma_summary((size * res) ** 2, 0, (size * res) ** 2)
        else:
            ker = motion_kernel(size, az)
            true = sigma_summary(*kernel_sigma(ker, res))
        for k in keep:
            y = lums[k]
            if kind == 'gaussian':
                b = ndimage.gaussian_filter(y, size)
            else:
                b = ndimage.convolve(y, ker)
            blurred.append(np.round(crop(b)))      # re-quantize like the mosaics
        d = mosaic_spectra(blurred, res)
        pb, norm = normalized(d['power'])
        betas = [fit_dsigma(pb[j], nb[j], d['noise'][j] / norm[j],
                            base['noise'][j] / norm_b[j], d['counts'][j])
                 for j in range(len(keep))]
        betas = [b for b in betas if b is not None]
        med, _ = combine_tiles(betas, n_boot=1)
        # anisotropy as `fit` measures it: the f^2 term of each tile's own
        # spectrum, here differenced against the unblurred tile
        dev = []
        for j in range(len(keep)):
            q = fit_aniso(pb[j], d['noise'][j] / norm[j], d['counts'][j])
            if q is not None and aniso_b[j] is not None:
                dev.append(q[2:] - aniso_b[j][2:])
        ss = blur_summary((0, 0, 0), np.median(dev, axis=0), 0.0)
        rows.append(dict(kind=kind, size_px=size, azimuth_in=az,
                         size_cm=size * res * 100, n_tiles=len(betas),
                         # the discrete kernel's own second moments; bilinear
                         # splatting widens short motion kernels along the motion
                         true_sigma_iso_cm=np.sqrt((true['var_major'] + true['var_minor']) / 2) * 100,
                         true_motion_len_cm=true['motion_len'] * 100,
                         dsigma_iso_cm=np.sqrt(max((med[0] + med[2]) / 2, 0)) * 100,
                         motion_len_cm=ss['motion_len'] * 100,
                         azimuth_out=ss['azimuth']))
    v = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    v.to_csv(outdir / 'blur_validation.csv', index=False)
    click.echo(v.round(2).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    g = v[v.kind == 'gaussian']
    ax = axes[0]
    ax.plot(g.true_sigma_iso_cm, g.dsigma_iso_cm, 'ko-', ms=4)
    lim = [0, g.true_sigma_iso_cm.max() * 1.1]
    ax.plot(lim, lim, color=pu.C_GREY, lw=0.8, ls='--')
    ax.set(xlabel='injected Gaussian $\\sigma$ (cm)', ylabel='recovered $\\Delta\\sigma$ (cm)',
           title='(a) isotropic blur')
    m = v[v.kind == 'motion']
    cmap = plt.get_cmap('viridis')
    ax = axes[1]
    for i, az in enumerate(sorted(m.azimuth_in.unique())):
        s = m[m.azimuth_in == az]
        ax.plot(s.true_motion_len_cm, s.motion_len_cm, 'o-', ms=4, color=cmap(i / 3),
                label=f'{az:.0f}°')
    lim = [0, m.true_motion_len_cm.max() * 1.1]
    ax.plot(lim, lim, color=pu.C_GREY, lw=0.8, ls='--')
    # beyond ~4 px the box blur's first spectral zero (f = 1/L) falls inside
    # the 0.5-6 cycles/m anisotropy band and the Gaussian model breaks down
    for a in axes[1:]:
        a.axvspan(4.5 * res * 100, lim[1], color=pu.C_GREY, alpha=0.12, lw=0)
    ax.set(xlabel='injected motion length (cm)', ylabel='recovered motion length (cm)',
           title='(b) linear motion')
    ax.legend(title='azimuth', fontsize=7, title_fontsize=7)
    ax = axes[2]
    for i, az in enumerate(sorted(m.azimuth_in.unique())):
        s = m[m.azimuth_in == az]
        ax.scatter(s.true_motion_len_cm, (s.azimuth_out - az + 90) % 180 - 90, s=14,
                   color=cmap(i / 3))
    ax.axhline(0, color=pu.C_GREY, lw=0.8, ls='--')
    ax.set(xlabel='injected motion length (cm)', ylabel='azimuth error (°)',
           title='(c) motion direction', ylim=(-90, 90))
    for a in axes:
        a.tick_params(labelsize=8)
    fig.suptitle(f'Synthetic blur on {mosaic} ({len(keep)} tiles, GSD {res * 100:.1f} cm)',
                 fontsize=9)
    fig.tight_layout()
    pu.save(fig, outdir, 'blur_validation')


# --------------------------------------------------------------------------
# exif
# --------------------------------------------------------------------------
def sun_position(ts, lat=9.152, lon=-79.846, utc_offset=-5):
    """Solar (azimuth, elevation) in degrees at local time `ts` (NOAA approximation)."""
    hr = ts.hour + ts.minute / 60 + ts.second / 3600
    g = 2 * np.pi / 365 * (ts.dayofyear - 1 + (hr - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
           + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    ha = np.radians((hr * 60 + eqt + 4 * lon - 60 * utc_offset) / 4 - 180)
    la = np.radians(lat)
    zen = np.arccos(np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha))
    az = np.degrees(np.arctan2(np.sin(ha), np.cos(ha) * np.sin(la) - np.tan(dec) * np.cos(la))) + 180
    return az % 360, 90 - np.degrees(zen)


def gps_xy(gps):
    """UTM 17N (x, y) from the stringified EXIF GPS dict."""
    d = ast.literal_eval(gps)
    dms = lambda t: t[0] + t[1] / 60 + t[2] / 3600
    lat = dms(d[2]) * (1 if d[1] == 'N' else -1)
    lon = dms(d[4]) * (1 if d[3] == 'E' else -1)
    return lon, lat


@cli.command()
@out_option
def exif(outdir):
    """Exposure, flight speed and flight-line azimuth per phantom mission."""
    from pyproj import Transformer
    df = pd.read_csv(EXIF_CSV, low_memory=False,
                     usecols=['Mission', 'GPSInfo', 'DateTimeOriginal', 'ExposureTime',
                              'ISOSpeedRatings', 'FNumber'])
    tr = Transformer.from_crs('EPSG:4326', 'EPSG:32617', always_xy=True)
    lonlat = np.array([gps_xy(g) for g in df.GPSInfo])
    df['x'], df['y'] = tr.transform(lonlat[:, 0], lonlat[:, 1])
    df['t'] = pd.to_datetime(df.DateTimeOriginal, format='%Y:%m:%d %H:%M:%S')
    rows = []
    for mission, g in df.groupby('Mission'):
        g = g.sort_values('t')
        dx, dy = np.diff(g.x.to_numpy()), np.diff(g.y.to_numpy())
        dt = np.diff(g.t.to_numpy()).astype('timedelta64[s]').astype(float)
        dist = np.hypot(dx, dy)
        head = np.degrees(np.arctan2(dx, dy)) % 360
        # straight-line pairs: short gap, moving, heading within 15 deg of the next
        dh = np.abs((np.diff(head) + 180) % 360 - 180)
        straight = np.r_[dh < 15, False] & (dt > 0) & (dt <= 15) & (dist > 5)
        # speed over straight runs (sum dist / sum dt) to beat 1 s timestamps
        run_id = np.cumsum(~straight)
        runs = pd.DataFrame(dict(run=run_id[straight], dist=dist[straight], dt=dt[straight]))
        rs = runs.groupby('run').sum()
        rs = rs[rs.dt >= 15]
        speed = (rs.dist.sum() / rs.dt.sum()) if len(rs) else np.nan
        z = np.exp(2j * np.radians(head[straight]))
        line_az = np.degrees(np.angle(z.mean())) / 2 % 180
        exp = g.ExposureTime.astype(float)
        sun_az, sun_el = sun_position(g.t.iloc[len(g) // 2])  # EXIF clock is local (UTC-5)
        rows.append(dict(date=pu.path_date(mission + '.tif'), mission=mission, n_images=len(g),
                         time_mid=g.t.iloc[len(g) // 2], sun_azimuth_deg=sun_az,
                         sun_elevation_deg=sun_el,
                         exposure_median_s=exp.median(), exposure_p90_s=exp.quantile(0.9),
                         iso_median=g.ISOSpeedRatings.median(), fnumber_median=g.FNumber.median(),
                         speed_m_s=speed, line_azimuth_deg=line_az,
                         line_axial_r=abs(z.mean()),
                         motion_pred_cm=speed * exp.median() * 100,
                         motion_pred_p90_cm=speed * exp.quantile(0.9) * 100))
    out = pd.DataFrame(rows).sort_values('date')
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / 'blur_exif.csv', index=False)
    click.echo(out.describe().round(4).to_string())
    click.echo(f'Wrote {outdir / "blur_exif.csv"}')


# --------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------
def date_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.axvline(pu.release_boundary(), color=pu.C_GREY, lw=0.7, ls=':')
    ax.tick_params(labelsize=8)


def series_rows(df):
    for key, (label, color, marker) in pu.SERIES.items():
        s = df[df.series == key].sort_values('date')
        if len(s):
            yield s, label, color, marker


def plot_timeseries(df, outdir):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for s, label, color, marker in series_rows(df):
        ax.vlines(s.date, s.fwhm_minor_cm, s.fwhm_major_cm, color=color, lw=0.8, alpha=0.6)
        ax.scatter(s.date, s.fwhm_major_cm, marker=marker, s=16, color=color,
                   label=f'{label} major axis')
        ax.scatter(s.date, s.fwhm_minor_cm, marker=marker, s=16, facecolor='none',
                   edgecolor=color, lw=0.8, label=f'{label} minor axis')
        ax.plot(s.date, 2 * s.gsd_cm, color=color, lw=0.8, ls='--', alpha=0.7)
    ax.plot([], [], color=pu.C_GREY, ls='--', lw=0.8, label='2 × GSD (Nyquist limit)')
    ax.set_ylabel('PSF FWHM (cm)', fontsize=9)
    ax.set_ylim(bottom=0)
    date_axis(ax)
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    pu.save(fig, outdir, 'blur_resolution_timeseries')


def plot_anisotropy(df, ex, outdir):
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True)
    ax = axes[0]
    for s, label, color, marker in series_rows(df):
        ax.errorbar(s.date, s.motion_len_cm,
                    yerr=[(s.motion_len_cm - s.motion_len_lo_cm).clip(lower=0),
                          (s.motion_len_hi_cm - s.motion_len_cm).clip(lower=0)],
                    fmt=marker, ms=3.5, color=color, ecolor=color, elinewidth=0.6, alpha=0.9,
                    label=label)
    if ex is not None:
        ax.plot(ex.date, ex.motion_pred_cm, color='k', lw=0.8,
                label='phantom predicted forward motion (EXIF speed × exposure)')
    ax.set_ylabel('motion length (cm)', fontsize=9)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc='upper left')
    ax = axes[1]
    for s, label, color, marker in series_rows(df):
        sig = s.motion_len_lo_cm > 0.5 * s.gsd_cm
        ax.scatter(s.date[sig], s.azimuth_deg[sig], marker=marker, s=14, color=color)
        ax.scatter(s.date[~sig], s.azimuth_deg[~sig], marker=marker, s=14, facecolor='none',
                   edgecolor=color, lw=0.6)
    if ex is not None:
        ax.plot(ex.date, ex.line_azimuth_deg, 'k_', ms=6, label='phantom flight-line azimuth (EXIF)')
        ax.legend(fontsize=7, loc='upper left')
    ax.set_ylabel('blur azimuth (° from N)', fontsize=9)
    ax.set_ylim(0, 180)
    ax.set_yticks([0, 45, 90, 135, 180])
    for a in axes:
        date_axis(a)
    fig.tight_layout()
    pu.save(fig, outdir, 'blur_anisotropy')


def plot_spectra_examples(df, cache, outdir):
    """Spectral-ratio profiles and native-resolution crops, sharpest vs blurriest."""
    meta, arrs = load_cache(cache)
    r = np.load(cache / 'reference.npz')
    ref = r['ref']
    tiles_df = tile_grid()
    picks = []
    for key in ('phantom-24782016', 'mavic'):
        s = df[(df.series == key) & df.fwhm_iso_cm.notna()].sort_values('fwhm_iso_cm')
        if len(s):
            picks += [(key, 'sharpest', s.iloc[0]), (key, 'blurriest', s.iloc[-1])]
    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, 1 + len(picks) // 2, width_ratios=[1.6] + [1] * (len(picks) // 2))
    ax = fig.add_subplot(gs[:, 0])
    f = f_centres()
    for key, which, row in picks:
        i = meta.index[meta.stem == row.stem][0]
        P, _ = normalized(arrs[i]['power'])
        v = arrs[i]['valid']
        with np.errstate(all='ignore'):
            prof = np.nanmedian(np.log(radial(P[v]) / radial(ref[v])), axis=0)
        color = pu.SERIES[key][1]
        ls = '-' if which == 'sharpest' else '--'
        k0 = np.flatnonzero(f >= F_LO)[0]
        slope = -4 * np.pi ** 2 * row.dvar_iso_m2
        ax.plot(f ** 2, prof - prof[k0], color=color, ls=ls, lw=1.2,
                label=f'{pu.SERIES[key][0]} {row.date:%Y-%m-%d} ({which})')
        ax.plot(f ** 2, slope * (f ** 2 - f[k0] ** 2), color=color, ls=ls, lw=0.6, alpha=0.6)
    ax.set_xlabel('$f^2$ (cycles$^2$ m$^{-2}$)', fontsize=9)
    ax.set_ylabel('ln(P / P$_{ref}$), offset to 0 at f = %.2g cyc/m' % F_LO, fontsize=9)
    ax.axhline(0, color=pu.C_GREY, lw=0.6)
    ax.set_xlim(0, F_EDGES[-1] ** 2)
    ax.legend(fontsize=7, title='thin lines: fitted Gaussian $\\Delta\\sigma^2$', title_fontsize=7)
    ax.tick_params(labelsize=8)
    # crops: the most textured valid tile shared by all picks, 8 m square
    half = 4.0
    t = tiles_df.iloc[len(tiles_df) // 2 + 3]
    bounds = (t.x - half, t.y - half, t.x + half, t.y + half)
    for j, (key, which, row) in enumerate(picks):
        a = fig.add_subplot(gs[j % 2, 1 + j // 2])
        arr, tr = pu.read_bounds(row.path, bounds, [1, 2, 3])
        rgb = np.moveaxis(arr, 0, -1).astype(float)
        lo, hi = np.percentile(rgb, [1, 99])        # per-crop stretch; dates differ in brightness
        a.imshow(np.clip((rgb - lo) / (hi - lo), 0, 1), extent=pu.extent(arr, tr),
                 interpolation='nearest')
        a.set_title(f'{pu.SERIES[key][0]} {row.date:%Y-%m-%d}\n{which}: FWHM '
                    f'{row.fwhm_iso_cm:.1f} cm', fontsize=8)
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    pu.save(fig, outdir, 'blur_spectra_examples')


def plot_anchor_check(df, cache, outdir):
    r = np.load(cache / 'reference.npz')
    f = f_centres()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    ax = axes[0]
    ax.loglog(f, r['ref_radial'], 'k.', ms=4, label='reference (median over mosaics, tiles)')
    model = np.exp(anchor_model(f, r['log_a'], r['alpha'], r['sigma'], r['log_n']))
    ax.loglog(f, model, color=pu.C_MAVIC, lw=1,
              label=f'fit: $\\alpha$={float(r["alpha"]):.2f}, $\\sigma$={float(r["sigma"]) * 100:.1f} cm')
    ax.set(xlabel='f (cycles/m)', ylabel='normalized power')
    ax.legend(fontsize=7)
    ax = axes[1]
    for s, label, color, marker in series_rows(df):
        ax.scatter(s.sigma_iso_cm, s.sigma_param_cm, marker=marker, s=12, color=color, label=label)
    lim = [0, np.nanmax([df.sigma_iso_cm.max(), df.sigma_param_cm.max()]) * 1.05]
    ax.plot(lim, lim, color=pu.C_GREY, lw=0.8, ls='--')
    ax.set(xlabel='$\\sigma$ from spectral ratio (cm)', ylabel='$\\sigma$ from own-spectrum fit (cm)')
    ax.legend(fontsize=7)
    ax = axes[2]
    for s, label, color, marker in series_rows(df):
        ax.scatter(s.fwhm_iso_px, s.tenengrad, marker=marker, s=12, color=color, label=label)
    ax.set(xlabel='PSF FWHM (pixels)', ylabel='Tenengrad / tile variance')
    for a in axes:
        a.tick_params(labelsize=8)
        a.xaxis.label.set_size(9)
        a.yaxis.label.set_size(9)
    fig.tight_layout()
    pu.save(fig, outdir, 'blur_anchor_check')


def plot_exif(df, ex, outdir):
    m = df[df.series == 'phantom-24782016'].merge(ex, on='date')
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    years = m.date.dt.year
    norm = plt.Normalize(years.min(), years.max())
    ax = axes[0]
    sc = ax.scatter(m.motion_pred_cm, m.motion_len_cm, c=years, cmap='viridis', norm=norm, s=16)
    lim = [0, np.nanmax([m.motion_pred_cm.max(), m.motion_len_cm.max()]) * 1.05]
    ax.plot(lim, lim, color=pu.C_GREY, lw=0.8, ls='--')
    ax.set(xlabel='predicted forward motion, speed × exposure (cm)',
           ylabel='measured motion length (cm)')
    ax = axes[1]
    ax.scatter(m.exposure_median_s * 1000, m.fwhm_major_cm, c=years, cmap='viridis', norm=norm, s=16)
    ax.set(xlabel='median exposure time (ms)', ylabel='PSF FWHM, major axis (cm)')
    ax = axes[2]
    rel = lambda az, ref: (az - ref + 90) % 180 - 90
    bins = np.arange(-90, 91, 10)
    for vals, label, color in (
            (rel(m.scene_azimuth_deg, m.sun_azimuth_deg), 'texture term − solar azimuth', '#ff7f0e'),
            (rel(m.azimuth_deg, m.sun_azimuth_deg), 'blur term − solar azimuth', pu.C_GREY),
            (rel(m.azimuth_deg, m.line_azimuth_deg), 'blur term − flight-line azimuth', pu.C_PHANTOM)):
        ax.hist(vals, bins=bins, histtype='step', lw=1.4, color=color, label=label)
    ax.set(xlabel='azimuth difference (°)', ylabel='phantom mosaics', xlim=(-90, 90))
    ax.set_xticks([-90, -45, 0, 45, 90])
    ax.legend(fontsize=7)
    for a in axes:
        a.tick_params(labelsize=8)
        a.xaxis.label.set_size(9)
        a.yaxis.label.set_size(9)
    cb = fig.colorbar(sc, ax=axes[1], shrink=0.9)
    cb.set_label('year', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    pu.save(fig, outdir, 'blur_exif_crosscheck')


def plot_mavic_meta(df, outdir):
    m = df[df.series == 'mavic'].copy()
    colors = {'PPK_Emlid': '#d62728', 'RTK': '#ff7f0e', 'none': '#7f7f7f'}
    flagged = m.comment_processing.fillna('').str.contains('artifact|incomplete|distorted')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ant, c in colors.items():
        s = m[m.antenna == ant]
        f = flagged[s.index]
        for ax, y in zip(axes, ('fwhm_iso_cm', 'motion_len_cm')):
            ax.scatter(s.gsd_cm[~f], s[y][~f], s=16, color=c, label=f'{ant} ({len(s)})')
            ax.scatter(s.gsd_cm[f], s[y][f], s=40, facecolor='none', edgecolor=c, lw=1)
    for ax in axes:
        ax.scatter([], [], s=40, facecolor='none', edgecolor='k', label='flagged in comments')
        ax.set_xlabel('mosaic GSD (cm)', fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel('PSF FWHM, isotropic (cm)', fontsize=9)
    axes[1].set_ylabel('motion length (cm)', fontsize=9)
    axes[0].legend(fontsize=7, title='GNSS correction', title_fontsize=7)
    fig.tight_layout()
    pu.save(fig, outdir, 'blur_mavic_metadata')


@cli.command()
@cache_option
@out_option
def plot(cache, outdir):
    """Figures from mosaic_blur.csv (and blur_exif.csv if present)."""
    df = pd.read_csv(outdir / 'mosaic_blur.csv', parse_dates=['date'])
    ex_path = outdir / 'blur_exif.csv'
    ex = pd.read_csv(ex_path, parse_dates=['date']) if ex_path.exists() else None
    plot_timeseries(df, outdir)
    plot_anisotropy(df, ex, outdir)
    plot_spectra_examples(df, cache, outdir)
    plot_anchor_check(df, cache, outdir)
    if 'antenna' in df:
        plot_mavic_meta(df, outdir)
    if ex is not None:
        plot_exif(df, ex, outdir)


if __name__ == '__main__':
    cli()
