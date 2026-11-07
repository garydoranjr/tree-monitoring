#!/usr/bin/env python
"""Coregister every clear PlanetScope scene over the whole of Barro Colorado
Island to one master scene, and measure whether a local (non-rigid) warp is
needed on top of a translation or an affine.

Planet-to-drone registration so far (apply_drone_labels_coreg.py) fits one
rigid shift per scene over the 50ha plot. Over the 5.9 x 5.4 km island,
orthorectification with Planet's DEM, the view angle (0-7.4 deg here) and the
strip geometry could leave residuals that one shift cannot remove. This script
measures them against a measurement-noise floor.

Pipeline (subcommands, in order):

  select     scene list: island clear fraction >= --min-clear with an SR
             product (`use_measure`), and >= --min-clear-video with MODIS
             offsets (`use_video`).
  measure    per scene: a whole-island NCC search for the integer global shift
             against the master, then a tile displacement field (64 px tiles,
             32 px stride, +-6 px search, masked Lucas-Kanade sub-pixel
             refinement) in red and NIR after that shift is removed.
  fit        per scene and band: translation (T, median tile vector), affine
             (A, robust least squares) and local (L, affine plus a normalized
             Gaussian smoothing of the affine residual on the tile lattice)
             models, their residuals, an honest hold-out residual for L (each
             tile predicted from tiles that share no pixels with it), and the
             red-NIR noise floor.
  dem        Copernicus GLO-30 DEM on the island grid (for the terrain test).
  remeasure  warp the video scenes with each model and re-run the tile matcher
             against the master: the residual the viewer sees.
  warp       write the aligned SR scenes (uint16, uncalibrated) for the video.
  video      the whole-island MP4 and the three-panel comparison MP4.
  figures    the figures for the update, with captions.
  selftest   synthetic affine + bump field on the master; pins the sign
             convention and the recovery of each model.

Displacement convention. A displacement d = (dy, dx) in pixels means that the
content at master pixel p appears at p + d in the scene. The correction samples
the scene at p + d (scipy.ndimage.map_coordinates); a pure translation moves the
scene's transform by -d in map units. This is the same convention as
apply_drone_labels_coreg (a reference point p appears at p + shift in the
Planet frame). Pixel rows increase southward, so dy > 0 is a shift to the
south and dx > 0 to the east.

Data: planet_bcnm_sr (SR scenes, OCM masks, water mask, index.csv),
whole_island/bci_clear_fraction.csv, results/planet_calibration/
planet_scene_offsets.csv. Outputs: results/island_coreg/ (tables),
planet_bci_aligned/<model>/ (rasters), figs/20261110_updates/ (figures, video).
See docs/island_coregistration.md.
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import click  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from rasterio.transform import Affine  # noqa: E402
from rasterio.windows import from_bounds  # noqa: E402
from scipy import ndimage  # noqa: E402
from skimage.feature import match_template  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import util  # noqa: E402

FLOWER = util.data_root()
SR = FLOWER / 'planet_bcnm_sr'
RESULTS = FLOWER / 'results' / 'island_coreg'
ALIGNED = FLOWER / 'planet_bci_aligned'
DEM_DIR = FLOWER / 'dem'
DEFAULT_OUT = FLOWER / 'figs' / '20261110_updates'
CLEAR_CSV = FLOWER / 'whole_island' / 'bci_clear_fraction.csv'
ISLAND_GEOJSON = FLOWER / 'whole_island' / 'Barro_Colorado_Island.geojson'
REPO = Path(__file__).resolve().parent.parent
STRETCH_YML = REPO / 'config' / 'planet_fixed_stretch.yml'

# The island grid is the TOA clip grid of planet_bci_clipped (config/clip_bci_margin.yml).
RES = 3.0
ISLAND_BOUNDS = (623979.0, 1009614.0, 629853.0, 1015020.0)
ISLAND_TRANSFORM = Affine(RES, 0, ISLAND_BOUNDS[0], 0, -RES, ISLAND_BOUNDS[3])
ISLAND_SHAPE = (1802, 1958)
# Every SR scene, OCM mask and the water mask sit on this 3 m lattice (checked on
# 311 of the 2,177 files), with varying extents: the island is a boundless window read.
LATTICE = (620943.0, 1018860.0)
CRS = 'EPSG:32617'

# Master: 100% island-clear SuperDove scene, view angle 1.8 deg, Feb 2023 (the middle
# of the 2020-2026 archive, dry season). Chosen by hand, not optimized.
MASTER = '20230227_145542_89_24be'
OCM_CLEAR, OCM_NODATA = 0, 255
BANDS = {'red': 3, 'nir': 4}
SCALE = 1e-4  # SR uint16 -> reflectance

# Matching parameters (see docs/island_coregistration.md for the probes behind them).
GLOBAL_SEARCH = 30         # px; +-90 m
GLOBAL_MIN_PEAK = 0.2
TILE, STRIDE, TILE_SEARCH = 64, 32, 6
TILE_MIN_VALID, TILE_MIN_PEAK = 0.9, 0.35
LAND_ERODE = 2             # px; the shoreline moves with the lake level
OUTLIER_PX = 2.0           # vs the 3x3 tile-neighbor median
SIGMA_TILES = 1.5          # local field smoothing, in tiles (x STRIDE px)
HOLDOUT_SPACING = 4        # test tiles every 4th lattice point; exclude their 3x3 ring
MODELS = ('T', 'A', 'L')
MODEL_NAMES = {'raw': 'uncorrected', 'T': 'translation', 'A': 'affine', 'L': 'local warp',
               'Lh': 'local warp (hold-out)'}
INSTRUMENTS = {'PS2': ('Dove Classic', '#1f77b4'), 'PS2.SD': ('Dove-R', '#ff7f0e'),
               'PSB.SD': ('SuperDove', '#2ca02c')}

DEM_NAME = 'Copernicus_DSM_COG_10_N09_00_W080_00_DEM'
DEM_URL = f'https://copernicus-dem-30m.s3.amazonaws.com/{DEM_NAME}/{DEM_NAME}.tif'


# --------------------------------------------------------------------------- I/O

def scene_paths(scene_id):
    y = scene_id[:4]
    return (SR / '4band' / y / f'{scene_id}_4band.tif',
            SR / 'ocm' / y / f'{scene_id}_ocm.tif')


def read_island(path, bands, fill=0):
    """Read `bands` (1-based list) of a lattice-aligned raster on the island grid."""
    with rasterio.open(path) as src:
        t = src.transform
        if ((t.c - LATTICE[0]) % RES or (t.f - LATTICE[1]) % RES
                or t.a != RES or t.e != -RES):
            raise ValueError(f'{path} is not on the 3 m lattice')
        win = from_bounds(*ISLAND_BOUNDS, transform=t).round_offsets().round_lengths()
        arr = src.read(bands, window=win, boundless=True, fill_value=fill)
    if arr.shape[-2:] != ISLAND_SHAPE:
        raise ValueError(f'{path}: island window is {arr.shape[-2:]}')
    return arr


def land_mask():
    water = read_island(SR / 'aux' / 'bcnm_water_mask.tif', [1], fill=OCM_NODATA)[0]
    return ndimage.binary_erosion(water == 0, iterations=LAND_ERODE)


def read_scene(scene_id, bands=('red', 'nir')):
    """Float32 bands (dict by name), the OCM-clear mask and the data mask."""
    p4, pocm = scene_paths(scene_id)
    arr = read_island(p4, [BANDS[b] for b in bands]).astype(np.float32)
    ocm = read_island(pocm, [1], fill=OCM_NODATA)[0]
    data = arr[0] > 0
    return dict(zip(bands, arr)), ocm == OCM_CLEAR, data


_MASTER = {}


def master_data(master=MASTER):
    """The master's bands and valid mask, cached per process."""
    if master not in _MASTER:
        bands, clear, data = read_scene(master)
        _MASTER.clear()
        _MASTER[master] = dict(bands=bands, valid=clear & data & land_mask())
    return _MASTER[master]


def write_tif(path, arr, dtype, nodata, transform=ISLAND_TRANSFORM, **extra):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None]
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = dict(driver='GTiff', height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                dtype=dtype, crs=CRS, transform=transform, nodata=nodata, compress='lzw',
                tiled=True, blockxsize=256, blockysize=256, **extra)
    with rasterio.open(path, 'w', **prof) as dst:
        dst.write(arr.astype(dtype))


def scene_offsets():
    """Per-scene MODIS offsets (reflectance) as a wide table: offset_blue..offset_nir, source."""
    off = pd.read_csv(util.calibration_results() / 'planet_scene_offsets.csv')
    wide = off.pivot(index='scene_id', columns='band', values='offset')
    wide.columns = [f'offset_{b}' for b in wide.columns]
    src = off.groupby('scene_id').source.first()
    return wide.join(src.rename('offset_source'))


def load_scenes(results):
    return pd.read_csv(results / 'scenes.csv', parse_dates=['acquired'])


# --------------------------------------------------------------------------- matching

def zscore(a, valid):
    out = np.zeros_like(a, dtype=np.float32)
    v = a[valid]
    if v.size > 1:
        out[valid] = (v - v.mean()) / (v.std() + 1e-6)
    return out


def translate(arr, dy, dx, fill=0):
    """out[y, x] = arr[y + dy, x + dx] for integer dy, dx; `fill` outside."""
    out = np.full_like(arr, fill)
    h, w = arr.shape[-2:]
    ys, yd = (slice(dy, h), slice(0, h - dy)) if dy >= 0 else (slice(0, h + dy), slice(-dy, h))
    xs, xd = (slice(dx, w), slice(0, w - dx)) if dx >= 0 else (slice(0, w + dx), slice(-dx, w))
    out[..., yd, xd] = arr[..., ys, xs]
    return out


def ncc_search(image, template, search):
    """Integer displacement of `template` (a scene array, same shape as the master
    `image`) by NCC of its interior, trimmed by `search`, over the image.

    Both inputs are z-scored with invalid pixels at 0. Returns (dy, dx, peak,
    second, on_edge, surface): the best NCC more than `search`/2 px from the peak
    is `second` (a diagnostic; at 3 m the surface is smooth and the ratio does
    not discriminate)."""
    m = search
    tpl = template[m:-m, m:-m]
    r = match_template(image, tpl)
    i, j = np.unravel_index(np.argmax(r), r.shape)
    yy, xx = np.ogrid[:r.shape[0], :r.shape[1]]
    far = (yy - i) ** 2 + (xx - j) ** 2 > (m / 2) ** 2
    second = float(r[far].max()) if far.any() else np.nan
    on_edge = i in (0, r.shape[0] - 1) or j in (0, r.shape[1] - 1)
    # The template (scene interior from scene pixel m) matched the image at (i, j):
    # scene pixel m holds master pixel (i, j), so master p appears at p + (m - i, m - j).
    return int(m - i), int(m - j), float(r[i, j]), second, on_edge, r


def parabolic(r, i, j):
    """Sub-pixel offset of the peak of surface r at (i, j), per axis."""
    def axis(a, b, c):
        den = a - 2 * b + c
        return 0.5 * (a - c) / den if den < 0 else 0.0
    di = axis(r[i - 1, j], r[i, j], r[i + 1, j]) if 0 < i < r.shape[0] - 1 else 0.0
    dj = axis(r[i, j - 1], r[i, j], r[i, j + 1]) if 0 < j < r.shape[1] - 1 else 0.0
    return di, dj


def lk_refine(a, b, valid, d0, n_iter=10, tol=1e-2):
    """Refine displacement d (content at a's pixel p appears at b's p + d) from d0
    by masked Lucas-Kanade: least squares on the gradients of b resampled at
    p + d, over valid pixels away from the edge. Returns NaN if it diverges."""
    d = np.array(d0, dtype=np.float64)
    for _ in range(n_iter):
        inner = ndimage.binary_erosion(valid, iterations=int(np.ceil(np.abs(d).max())) + 2)
        if inner.sum() < 50:
            return np.full(2, np.nan)
        bw = ndimage.shift(b, -d, order=3, mode='nearest')
        A, B = zscore(a, inner), zscore(bw, inner)
        gy, gx = np.gradient(B)
        J = np.stack([gy[inner], gx[inner]], axis=1)
        delta, *_ = np.linalg.lstsq(J, (A - B)[inner], rcond=None)
        d = d + delta
        if not np.all(np.isfinite(d)) or np.abs(d - d0).max() > 1.5:
            return np.full(2, np.nan)
        if np.abs(delta).max() < tol:
            break
    return d


def tile_lattice(shape=ISLAND_SHAPE, tile=TILE, stride=STRIDE):
    rows = np.arange(0, shape[0] - tile + 1, stride)
    cols = np.arange(0, shape[1] - tile + 1, stride)
    return rows, cols


def tile_field(master_z, master_valid, scene, scene_valid, refine='lk', search=TILE_SEARCH,
               tile=TILE, stride=STRIDE):
    """Displacement of every tile of `scene` relative to the master (both on the
    island grid, scene already corrected for the integer global shift).
    Yields dicts with the tile center (row, col), dy_px, dx_px, peak, valid_frac,
    method."""
    scene_z = zscore(scene, scene_valid)
    rows, cols = tile_lattice(scene.shape, tile, stride)
    for r0 in rows:
        for c0 in cols:
            sl = (slice(r0, r0 + tile), slice(c0, c0 + tile))
            valid = master_valid[sl] & scene_valid[sl]
            vf = valid.mean()
            if vf < TILE_MIN_VALID:
                continue
            dy, dx, peak, _, on_edge, r = ncc_search(master_z[sl], scene_z[sl], search)
            if peak < TILE_MIN_PEAK or on_edge:
                continue
            i, j = search - dy, search - dx
            di, dj = parabolic(r, i, j)
            d_par = (dy - di, dx - dj)
            method = 'parabolic'
            d = d_par
            if refine == 'lk':
                d_lk = lk_refine(master_z[sl], scene_z[sl], valid, (dy, dx))
                if np.all(np.isfinite(d_lk)) and np.abs(d_lk - np.array(d_par)).max() < 1.0:
                    d, method = tuple(d_lk), 'lk'
            yield dict(row=r0 + tile / 2, col=c0 + tile / 2, dy_px=float(d[0]),
                       dx_px=float(d[1]), peak=peak, valid_frac=float(vf), method=method)


def measure_arrays(bands, clear, data, master=MASTER, refine='lk', global_search=GLOBAL_SEARCH,
                   skip_global=False):
    """Global shift (red) and the tile fields (all bands) of one scene given its
    arrays. Returns (global record, list of tile dicts)."""
    md = master_data(master)
    land = md['valid']  # master clear & land
    scene_valid = clear & data & land_mask()
    rec = dict(global_dy_px=0, global_dx_px=0, global_peak=np.nan, global_second=np.nan,
               global_edge=False, global_ok=True,
               valid_frac=float((scene_valid & land).sum() / land.sum()))
    tiles = []
    if not skip_global:
        mz = zscore(md['bands']['red'], md['valid'])
        sz = zscore(bands['red'], scene_valid)
        dy, dx, peak, second, on_edge, _ = ncc_search(mz, sz, global_search)
        rec.update(global_dy_px=dy, global_dx_px=dx, global_peak=peak, global_second=second,
                   global_edge=bool(on_edge), global_ok=bool(peak >= GLOBAL_MIN_PEAK and not on_edge))
        if not rec['global_ok']:
            return rec, tiles
    dy, dx = rec['global_dy_px'], rec['global_dx_px']
    shifted_valid = translate(scene_valid, dy, dx, False)
    for name, arr in bands.items():
        mz = zscore(md['bands'][name], md['valid'])
        s = translate(arr, dy, dx, 0)
        for t in tile_field(mz, md['valid'], s, shifted_valid, refine=refine):
            t.update(band=name, dy_px=t['dy_px'] + dy, dx_px=t['dx_px'] + dx)
            tiles.append(t)
    return rec, tiles


def measure_scene(args):
    scene_id, results, master, refine = args
    out_csv = results / 'tiles' / f'{scene_id}.csv'
    out_json = results / 'global' / f'{scene_id}.json'
    if out_csv.exists() and out_json.exists():
        return scene_id, 'cached'
    t0 = time.time()
    try:
        bands, clear, data = read_scene(scene_id)
        rec, tiles = measure_arrays(bands, clear, data, master, refine)
    except Exception as e:  # noqa: BLE001 - record and continue the batch
        return scene_id, f'error: {e}'
    rec.update(scene_id=scene_id, n_tiles_red=sum(t['band'] == 'red' for t in tiles),
               n_tiles_nir=sum(t['band'] == 'nir' for t in tiles), seconds=time.time() - t0)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tiles, columns=['band', 'row', 'col', 'dy_px', 'dx_px', 'peak', 'valid_frac',
                                 'method']).assign(scene_id=scene_id).to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(rec))
    return scene_id, f"ok {rec['n_tiles_red']}+{rec['n_tiles_nir']} tiles, {rec['seconds']:.0f} s"


# --------------------------------------------------------------------------- models

def lattice_index(df, tile=TILE, stride=STRIDE):
    ri = np.round((df.row.values - tile / 2) / stride).astype(int)
    ci = np.round((df.col.values - tile / 2) / stride).astype(int)
    return ri, ci


def grid_shape(tile=TILE, stride=STRIDE):
    rows, cols = tile_lattice(ISLAND_SHAPE, tile, stride)
    return len(rows), len(cols)


def to_grid(ri, ci, values, shape):
    g = np.full(shape, np.nan)
    g[ri, ci] = values
    return g


def neighbor_outliers(ri, ci, dy, dx, shape, thresh=OUTLIER_PX):
    """Tiles whose vector differs from the median of their 3x3 lattice
    neighbors by more than `thresh` px (or that have no neighbor)."""
    out = np.zeros(len(ri), bool)
    for comp in (dy, dx):
        g = to_grid(ri, ci, comp, shape)
        med = ndimage.generic_filter(g, np.nanmedian, size=3, mode='constant', cval=np.nan)
        d = np.abs(comp - med[ri, ci])
        out |= ~np.isfinite(d) | (d > thresh)
    return out


def fit_affine(ri, ci, dy, dx, rows, cols, passes=2):
    """Least squares d = c0 + c_r * row + c_c * col per component (row, col in
    pixels of the island grid), with `passes` of 3-MAD clipping. Returns the
    (2, 3) coefficient array and the keep mask."""
    y, x = rows[ri], cols[ci]
    X = np.stack([np.ones_like(y, float), y, x], axis=1)
    keep = np.ones(len(y), bool)
    coef = None
    for _ in range(passes + 1):
        coef = np.stack([np.linalg.lstsq(X[keep], v[keep], rcond=None)[0] for v in (dy, dx)])
        res = np.hypot(dy - X @ coef[0], dx - X @ coef[1])
        mad = np.median(np.abs(res[keep] - np.median(res[keep]))) * 1.4826 + 1e-9
        keep = res <= np.median(res[keep]) + 3 * mad
    return coef, keep


def affine_predict(coef, y, x):
    return coef[0, 0] + coef[0, 1] * y + coef[0, 2] * x, coef[1, 0] + coef[1, 1] * y + coef[1, 2] * x


def affine_terms(coef):
    """Jacobian of the affine displacement scaled to meters across the island
    (height for the row derivatives, width for the column ones), and the
    largest displacement difference between two island points it implies."""
    h, w = ISLAND_SHAPE
    j = dict(jac_yy_m=coef[0, 1] * h * RES, jac_yx_m=coef[0, 2] * w * RES,
             jac_xy_m=coef[1, 1] * h * RES, jac_xx_m=coef[1, 2] * w * RES)
    J = np.array([[coef[0, 1], coef[0, 2]], [coef[1, 1], coef[1, 2]]])
    corners = np.array([[-h / 2, -w / 2], [-h / 2, w / 2], [h / 2, -w / 2], [h / 2, w / 2]])
    disp = corners @ J.T
    j['affine_amp_m'] = float(np.max(np.hypot(*(disp[:, None] - disp[None]).transpose(2, 0, 1))) * RES)
    return j


def smooth_field(resid_grid, weight, sigma):
    """Normalized Gaussian smoothing of a gridded residual (NaN where absent)
    using `weight` (0/1) as the support; 0 where the support is too thin."""
    r = np.where(weight > 0, np.nan_to_num(resid_grid), 0.0)
    num = ndimage.gaussian_filter(r, sigma, mode='constant')
    den = ndimage.gaussian_filter(weight.astype(float), sigma, mode='constant')
    out = np.zeros_like(num)
    ok = den > 0.05
    out[ok] = num[ok] / den[ok]
    return out


def holdout_predict(ri, ci, rdy, rdx, shape, sigma, spacing=HOLDOUT_SPACING):
    """Hold-out prediction of the smoothed residual at every tile: tiles on every
    `spacing`-th lattice point are predicted from a field fitted without them
    and without their 3x3 ring (tiles that overlap them), for each of the
    spacing^2 lattice phases."""
    gy, gx = to_grid(ri, ci, rdy, shape), to_grid(ri, ci, rdx, shape)
    present = np.isfinite(gy)
    pred_y, pred_x = np.full(shape, np.nan), np.full(shape, np.nan)
    for a in range(spacing):
        for b in range(spacing):
            test = np.zeros(shape, bool)
            test[a::spacing, b::spacing] = True
            ring = ndimage.binary_dilation(test, structure=np.ones((3, 3), bool))
            w = present & ~ring
            fy, fx = smooth_field(gy, w, sigma), smooth_field(gx, w, sigma)
            sel = test & present
            pred_y[sel], pred_x[sel] = fy[sel], fx[sel]
    return pred_y[ri, ci], pred_x[ri, ci]


def fit_scene(df, rec, sigma=SIGMA_TILES, sigmas_eval=(1.0, 1.5, 2.0, 3.0)):
    """Fit T, A, L to the tiles of one scene and band. Returns (summary dict,
    per-tile DataFrame with residuals, model dict for warping)."""
    shape = grid_shape()
    rows, cols = tile_lattice()
    rows_c, cols_c = rows + TILE / 2, cols + TILE / 2
    ri, ci = lattice_index(df)
    dy, dx = df.dy_px.values.astype(float), df.dx_px.values.astype(float)
    out = dict(n_tiles_all=len(df))
    bad = neighbor_outliers(ri, ci, dy, dx, shape)
    out['n_outliers'] = int(bad.sum())
    ri, ci, dy, dx = ri[~bad], ci[~bad], dy[~bad], dx[~bad]
    df = df[~bad].copy()
    out['n_tiles'] = len(df)
    if len(df) < 30:
        return out, None, None
    raw = np.hypot(dy, dx)
    # T
    t = np.array([np.median(dy), np.median(dx)])
    rT = np.hypot(dy - t[0], dx - t[1])
    # A
    coef, keep = fit_affine(ri, ci, dy, dx, rows_c, cols_c)
    ay, ax = affine_predict(coef, rows_c[ri], cols_c[ci])
    rdy, rdx = dy - ay, dx - ax
    rA = np.hypot(rdy, rdx)
    # L: smoothed affine residual on the lattice
    gy, gx = to_grid(ri, ci, rdy, shape), to_grid(ri, ci, rdx, shape)
    present = np.isfinite(gy)
    fy, fx = smooth_field(gy, present, sigma), smooth_field(gx, present, sigma)
    rL = np.hypot(rdy - fy[ri, ci], rdx - fx[ri, ci])
    # L hold-out, at the working sigma and over a small sigma grid
    hy, hx = holdout_predict(ri, ci, rdy, rdx, shape, sigma)
    rLh = np.hypot(rdy - hy, rdx - hx)
    for s in sigmas_eval:
        if s == sigma:
            out[f'med_Lh_s{s:g}'] = float(np.nanmedian(rLh))
        else:
            py, px = holdout_predict(ri, ci, rdy, rdx, shape, s)
            out[f'med_Lh_s{s:g}'] = float(np.nanmedian(np.hypot(rdy - py, rdx - px)))
    for name, r in (('raw', raw), ('T', rT), ('A', rA), ('L', rL), ('Lh', rLh)):
        out[f'med_{name}'] = float(np.nanmedian(r))
        out[f'p90_{name}'] = float(np.nanpercentile(r, 90))
        out[f'rms_{name}'] = float(np.sqrt(np.nanmean(r ** 2)))
    out.update(t_dy_px=float(t[0]), t_dx_px=float(t[1]), n_affine_kept=int(keep.sum()),
               **affine_terms(coef), **{f'a_{k}': float(v) for k, v in
                                        zip(['c0_y', 'cr_y', 'cc_y', 'c0_x', 'cr_x', 'cc_x'],
                                            coef.ravel())})
    tiles = df.assign(ri=ri, ci=ci, rT_dy=dy - t[0], rT_dx=dx - t[1], rA_dy=rdy, rA_dx=rdx,
                      rL_dy=rdy - fy[ri, ci], rL_dx=rdx - fx[ri, ci], rLh_dy=rdy - hy, rLh_dx=rdx - hx)
    model = dict(t=t, coef=coef, field_dy=ay_grid(coef, rows_c, cols_c)[0] + fy,
                 field_dx=ay_grid(coef, rows_c, cols_c)[1] + fx, support=present)
    return out, tiles, model


def ay_grid(coef, rows_c, cols_c):
    Y, X = np.meshgrid(rows_c, cols_c, indexing='ij')
    return affine_predict(coef, Y, X)


def displacement_maps(model, kind, shape=ISLAND_SHAPE, window=None):
    """Pixel displacement (dy, dx) arrays for model kind T, A or L over `shape`
    (or the (r0, r1, c0, c1) `window` of it)."""
    r0, r1, c0, c1 = window or (0, shape[0], 0, shape[1])
    Y, X = np.mgrid[r0:r1, c0:c1].astype(np.float64)
    if kind == 'raw':
        return np.zeros_like(Y), np.zeros_like(X)
    if kind == 'T':
        return np.full_like(Y, model['t'][0]), np.full_like(X, model['t'][1])
    if kind == 'A':
        return affine_predict(model['coef'], Y, X)
    if kind == 'L':
        fy = (Y - TILE / 2) / STRIDE
        fx = (X - TILE / 2) / STRIDE
        dy = ndimage.map_coordinates(model['field_dy'], [fy, fx], order=1, mode='nearest')
        dx = ndimage.map_coordinates(model['field_dx'], [fy, fx], order=1, mode='nearest')
        return dy, dx
    raise ValueError(kind)


def warp_arrays(arrays, dy, dx, orders, cvals, window=None):
    """Resample each array at p + d. `window` = (r0, r1, c0, c1) of the output."""
    r0, r1, c0, c1 = window or (0, arrays[0].shape[0], 0, arrays[0].shape[1])
    Y, X = np.mgrid[r0:r1, c0:c1].astype(np.float64)
    coords = [Y + dy, X + dx]
    return [ndimage.map_coordinates(a, coords, order=o, cval=c, mode='constant')
            for a, o, c in zip(arrays, orders, cvals)]


def save_model(path, model):
    np.savez_compressed(path, t=model['t'], coef=model['coef'], field_dy=model['field_dy'],
                        field_dx=model['field_dx'], support=model['support'])


def load_model(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- CLI

@click.group()
def cli():
    """Whole-island Planet coregistration and local-warp study."""


results_option = click.option('--results', type=click.Path(path_type=Path), default=RESULTS,
                              show_default=True, help='Table directory.')
master_option = click.option('--master', default=MASTER, show_default=True)


@cli.command()
@results_option
@click.option('--min-clear', default=0.5, show_default=True)
@click.option('--min-clear-video', default=0.8, show_default=True)
def select(results, min_clear, min_clear_video):
    """Scene list: island clear fraction joined to SR metadata and MODIS offsets."""
    results.mkdir(parents=True, exist_ok=True)
    cf = pd.read_csv(CLEAR_CSV).rename(columns={'image_name': 'scene_id'})
    idx = pd.read_csv(SR / 'index.csv', parse_dates=['acquired'])
    idx['has_sr'] = idx.staged_path.map(lambda p: Path(p).exists())
    cols = ['scene_id', 'acquired', 'instrument', 'satellite_id', 'strip_id', 'sun_elevation',
            'sun_azimuth', 'view_angle', 'satellite_azimuth', 'has_sr']
    df = cf.merge(idx[cols], on='scene_id', how='left')
    df['has_sr'] = df.has_sr.fillna(False).astype(bool)
    df = df.merge(scene_offsets(), left_on='scene_id', right_index=True, how='left')
    df['has_offsets'] = df.offset_red.notna()
    df['use_measure'] = (df.fraction_clear >= min_clear) & df.has_sr
    df['use_video'] = (df.fraction_clear >= min_clear_video) & df.has_sr & df.has_offsets
    df = df.sort_values('datetime_utc')
    df.to_csv(results / 'scenes.csv', index=False)
    m, v = df[df.use_measure], df[df.use_video]
    click.echo(f'{len(df)} scenes with a clear fraction; {m.shape[0]} to measure '
               f'(>= {min_clear:.0%} clear with SR), {v.shape[0]} for the video '
               f'(>= {min_clear_video:.0%} with offsets)\n'
               f'measure by instrument: {m.instrument.value_counts().to_dict()}\n'
               f'video by instrument: {v.instrument.value_counts().to_dict()}\n'
               f'video offsets by source: {v.offset_source.value_counts().to_dict()}')
    (results / 'master.json').write_text(json.dumps(dict(
        master=MASTER, reason='100% island-clear SuperDove scene, view angle 1.8 deg, '
        'February 2023 (middle of the 2020-2026 archive, dry season); chosen by hand.',
        grid=dict(crs=CRS, transform=list(ISLAND_TRANSFORM)[:6], shape=ISLAND_SHAPE),
        tile=TILE, stride=STRIDE, tile_search=TILE_SEARCH, global_search=GLOBAL_SEARCH),
        indent=1))


@cli.command()
@results_option
@master_option
@click.option('--jobs', default=5, show_default=True)
@click.option('--scenes', default=None, help='Comma-separated scene ids (default: use_measure).')
@click.option('--limit', default=None, type=int, help='First N scenes only.')
@click.option('--refine', type=click.Choice(['lk', 'parabolic']), default='lk', show_default=True)
def measure(results, master, jobs, scenes, limit, refine):
    """Global shift and tile displacement field of every scene against the master."""
    df = load_scenes(results)
    ids = scenes.split(',') if scenes else df[df.use_measure].scene_id.tolist()
    ids = [s for s in ids if s != master][:limit]
    args = [(s, results, master, refine) for s in ids]
    t0 = time.time()
    n_ok = 0
    with Pool(jobs) as pool:
        for k, (sid, msg) in enumerate(pool.imap_unordered(measure_scene, args), 1):
            n_ok += msg.startswith('ok') or msg == 'cached'
            click.echo(f'[{k}/{len(args)} {time.time() - t0:5.0f}s] {sid}: {msg}')
    # concatenate the per-scene caches
    recs = [json.loads(p.read_text()) for p in sorted((results / 'global').glob('*.json'))]
    g = pd.DataFrame(recs).merge(df, on='scene_id', how='left')
    g.to_csv(results / 'global_shifts.csv', index=False)
    tiles = pd.concat([pd.read_csv(p) for p in sorted((results / 'tiles').glob('*.csv'))],
                      ignore_index=True)
    tiles.to_csv(results / 'tiles.csv.gz', index=False)
    click.echo(f'{n_ok}/{len(args)} scenes done; {g.global_ok.sum()} of {len(g)} pass the global '
               f'gate; {len(tiles)} tiles -> {results / "tiles.csv.gz"}')


@cli.command()
@results_option
@click.option('--sigma-tiles', default=SIGMA_TILES, show_default=True,
              help='Local-field Gaussian sigma in tiles (x stride px).')
def fit(results, sigma_tiles):
    """Translation, affine and local models per scene and band; residuals and noise floor."""
    tiles = pd.read_csv(results / 'tiles.csv.gz')
    g = pd.read_csv(results / 'global_shifts.csv')
    g = g[g.global_ok].set_index('scene_id')
    (results / 'models').mkdir(exist_ok=True)
    rows, resid = [], []
    for sid, sdf in tiles.groupby('scene_id'):
        if sid not in g.index:
            continue
        rec = g.loc[sid]
        bands = {b: d for b, d in sdf.groupby('band')}
        per_band, outs = {}, {}
        for band, d in bands.items():
            out, tdf, model = fit_scene(d, rec, sigma_tiles)
            out.update(scene_id=sid, band=band)
            outs[band] = out
            if tdf is not None:
                per_band[band] = tdf
                resid.append(tdf.assign(band=band))
        # Band-to-band offset (Dove Classic NIR sits more than a pixel from red on
        # some scenes), the noise floor from the translation residuals of the two
        # bands, and the coherence of their affine residuals.
        extra = {}
        both = sdf
        if 'red' in per_band and 'nir' in per_band:
            r, n = per_band['red'].set_index(['ri', 'ci']), per_band['nir'].set_index(['ri', 'ci'])
            j = r.join(n, lsuffix='_r', rsuffix='_n', how='inner')
            off = (outs['nir']['t_dy_px'] - outs['red']['t_dy_px'],
                   outs['nir']['t_dx_px'] - outs['red']['t_dx_px'])
            if len(j) >= 30:
                diff = np.hypot(j.rT_dy_r - j.rT_dy_n, j.rT_dx_r - j.rT_dx_n)
                extra = dict(noise_px=float(np.median(diff) / np.sqrt(2)),
                             coherence=float(np.corrcoef(np.r_[j.rA_dy_r, j.rA_dx_r],
                                                         np.r_[j.rA_dy_n, j.rA_dx_n])[0, 1]),
                             n_paired=len(j), band_offset_dy_px=off[0], band_offset_dx_px=off[1],
                             band_offset_px=float(np.hypot(*off)))
            # the pooled model refers to the red band's geometry
            nir = sdf[sdf.band == 'nir'].copy()
            nir['dy_px'] -= off[0]
            nir['dx_px'] -= off[1]
            both = pd.concat([sdf[sdf.band == 'red'], nir], ignore_index=True)
        out, tdf, model = fit_scene(both, rec, sigma_tiles)
        out.update(scene_id=sid, band='both')
        outs['both'] = out
        if tdf is not None:
            resid.append(tdf.assign(band='both'))
            save_model(results / 'models' / f'{sid}.npz', model)
        for out in outs.values():
            out.update(extra)
            rows.append(out)
    sm = pd.DataFrame(rows).merge(g.reset_index(), on='scene_id', how='left')
    sm.to_csv(results / 'scene_models.csv', index=False)
    pd.concat(resid, ignore_index=True).to_csv(results / 'tile_residuals.csv.gz', index=False)
    b = sm[sm.band == 'both']
    click.echo(f'{len(b)} scenes fitted. Median over scenes of the median tile residual (px):\n'
               + '\n'.join(f'  {MODEL_NAMES[k]:26s} {b[f"med_{k}"].median():.3f}'
                           for k in ('raw', 'T', 'A', 'L', 'Lh')))
    r = sm[sm.band == 'red']
    click.echo(f'noise floor (red-NIR, median over scenes): {r.noise_px.median():.3f} px; '
               f'coherence median {r.coherence.median():.2f}\n'
               f'hold-out median by sigma: ' +
               ', '.join(f'{c[7:]}: {b[c].median():.3f}' for c in b.columns if c.startswith('med_Lh_s')))


@cli.command()
@results_option
@master_option
@click.option('--jobs', default=5, show_default=True)
def selftest(results, master, jobs):
    """Recover a known affine + bump field applied to the master itself."""
    rng = np.random.default_rng(0)
    md = master_data(master)
    h, w = ISLAND_SHAPE
    Y, X = np.mgrid[:h, :w].astype(np.float64)
    # truth: content at master p appears at p + d in the synthetic scene
    cy, cx = h / 2, w / 2
    true_dy = 0.5 + 1e-3 * (Y - cy) + 2.0 * np.exp(-(((Y - 700) ** 2 + (X - 1200) ** 2) / (2 * 150 ** 2)))
    true_dx = -1.3 - 1e-3 * (X - cx) + 0.5e-3 * (Y - cy)
    # the scene s satisfies s(p + d) = m(p); build it by sampling m at q - d(q) (d is smooth)
    bands = {}
    for name, m in md['bands'].items():
        bands[name] = ndimage.map_coordinates(m, [Y - true_dy, X - true_dx], order=3, cval=0).astype(np.float32)
    clear = np.ones(ISLAND_SHAPE, bool)
    # a synthetic cloud and a missing strip
    clear[300:500, 200:600] = False
    clear[:, 1700:] = False
    data = bands['red'] > 0
    rec, tiles = measure_arrays(bands, clear, data, master)
    tdf = pd.DataFrame(tiles)
    msgs = [f"global integer shift {rec['global_dy_px']}, {rec['global_dx_px']} px (truth ~ 0.5, -1.3 "
            f"+ affine); peak {rec['global_peak']:.2f}; {len(tdf)} tiles"]
    out, resid, model = fit_scene(tdf, rec)
    # compare the fitted L field with the truth on the tile lattice
    rows, cols = tile_lattice()
    rc, cc = rows + TILE / 2, cols + TILE / 2
    Yg, Xg = np.meshgrid(rc, cc, indexing='ij')
    ty = ndimage.map_coordinates(true_dy, [Yg, Xg], order=1)
    tx = ndimage.map_coordinates(true_dx, [Yg, Xg], order=1)
    sup = model['support']
    err = np.hypot(model['field_dy'] - ty, model['field_dx'] - tx)[sup]
    msgs.append(f"tile vectors vs truth: median |err| {np.median(err):.3f} px, p90 {np.percentile(err, 90):.3f}")
    msgs.append(f"affine terms (truth jac_yy {1e-3 * h * RES:.1f} m, jac_xx {-1e-3 * w * RES:.1f} m, "
                f"jac_xy {0.5e-3 * h * RES:.1f} m): {out['jac_yy_m']:.1f}, {out['jac_xx_m']:.1f}, {out['jac_xy_m']:.1f}")
    msgs.append('residual medians (px): ' + ', '.join(f"{k} {out[f'med_{k}']:.3f}" for k in ('raw', 'T', 'A', 'L', 'Lh')))
    # warp with L and re-measure
    dy, dx = displacement_maps(model, 'L')
    wred, = warp_arrays([bands['red']], dy, dx, [1], [0])
    wclear, = warp_arrays([clear.astype(np.float32)], dy, dx, [1], [0])
    rec2, tiles2 = measure_arrays({'red': wred}, wclear > 0.99, wred > 0, master, skip_global=True)
    t2 = pd.DataFrame(tiles2)
    msgs.append(f're-measured after the L warp: median |d| {np.hypot(t2.dy_px, t2.dx_px).median():.3f} px '
                f'over {len(t2)} tiles (want < 0.1)')
    click.echo('\n'.join(msgs))
    ok = (np.median(err) < 0.15 and np.hypot(t2.dy_px, t2.dx_px).median() < 0.1
          and abs(out['jac_yy_m'] - 1e-3 * h * RES) < 0.1 * abs(1e-3 * h * RES))
    click.echo('SELFTEST ' + ('PASSED' if ok else 'FAILED'))
    if not ok:
        sys.exit(1)


# --------------------------------------------------------------------------- DEM

DEM_3M = DEM_DIR / 'bci_glo30_3m.tif'


@cli.command()
def dem():
    """Copernicus GLO-30 elevation, slope and aspect on the island grid."""
    from rasterio.warp import Resampling, reproject
    src_path = DEM_DIR / f'{DEM_NAME}.tif'
    if not src_path.exists():
        DEM_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(['curl', '-sS', '-o', str(src_path), DEM_URL], check=True)
    out = np.zeros(ISLAND_SHAPE, np.float32)
    with rasterio.open(src_path) as src:
        reproject(rasterio.band(src, 1), out, dst_transform=ISLAND_TRANSFORM, dst_crs=CRS,
                  resampling=Resampling.bilinear)
    gy, gx = np.gradient(out, RES)
    slope = np.degrees(np.arctan(np.hypot(gy, gx)))
    aspect = (np.degrees(np.arctan2(gx, -gy)) + 360) % 360  # downslope direction, from north
    write_tif(DEM_3M, np.stack([out, slope, aspect]), 'float32', -9999)
    land = land_mask()
    click.echo(f'{DEM_3M}: land elevation {out[land].min():.0f}-{out[land].max():.0f} m '
               f'(median {np.median(out[land]):.0f}), slope median {np.median(slope[land]):.1f} deg')


def dem_arrays():
    with rasterio.open(DEM_3M) as src:
        return src.read()


def hillshade(elev, az=315, alt=45):
    gy, gx = np.gradient(elev, RES)
    slope = np.arctan(np.hypot(gy, gx))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(az), np.radians(alt)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


# --------------------------------------------------------------------------- warp and remeasure

def model_path(results, scene_id):
    return results / 'models' / f'{scene_id}.npz'


def remeasure_scene(args):
    scene_id, results, master, kinds = args
    try:
        model = load_model(model_path(results, scene_id))
        bands, clear, data = read_scene(scene_id, bands=('red',))
    except Exception as e:  # noqa: BLE001
        return scene_id, [], f'error: {e}'
    rows = []
    for kind in kinds:
        dy, dx = displacement_maps(model, kind)
        red, clr, dat = warp_arrays([bands['red'], clear.astype(np.float32), data.astype(np.float32)],
                                    dy, dx, [1, 1, 1], [0, 0, 0])
        _, tiles = measure_arrays({'red': red}, clr > 0.99, dat > 0.99, master, skip_global=True)
        mag = np.hypot([t['dy_px'] for t in tiles], [t['dx_px'] for t in tiles])
        rows.append(dict(scene_id=scene_id, model=kind, n_tiles=len(tiles),
                         med_px=float(np.median(mag)) if len(mag) else np.nan,
                         p90_px=float(np.percentile(mag, 90)) if len(mag) else np.nan,
                         med_dy_px=float(np.median([t['dy_px'] for t in tiles])) if len(mag) else np.nan,
                         med_dx_px=float(np.median([t['dx_px'] for t in tiles])) if len(mag) else np.nan))
    return scene_id, rows, 'ok ' + ', '.join(f"{r['model']} {r['med_px']:.2f}" for r in rows)


@cli.command()
@results_option
@master_option
@click.option('--jobs', default=5, show_default=True)
@click.option('--models', default='T,A,L', show_default=True)
@click.option('--limit', default=None, type=int)
def remeasure(results, master, jobs, models, limit):
    """Warp the video scenes with each model and re-run the tile matcher: the residual
    a viewer of the aligned sequence sees."""
    df = load_scenes(results)
    ids = [s for s in df[df.use_video].scene_id if model_path(results, s).exists()][:limit]
    kinds = models.split(',')
    rows = []
    t0 = time.time()
    with Pool(jobs) as pool:
        for k, (sid, r, msg) in enumerate(pool.imap_unordered(
                remeasure_scene, [(s, results, master, kinds) for s in ids]), 1):
            rows += r
            click.echo(f'[{k}/{len(ids)} {time.time() - t0:5.0f}s] {sid}: {msg}')
    out = pd.DataFrame(rows)
    out.to_csv(results / 'remeasure.csv', index=False)
    click.echo('median over scenes of the re-measured median residual (px): '
               + out.groupby('model').med_px.median().round(3).to_dict().__repr__())


def warp_scene(args):
    scene_id, results, kind, outdir = args
    p4, pocm = scene_paths(scene_id)
    out4 = outdir / scene_id[:4] / f'{scene_id}_4band.tif'
    outo = outdir / scene_id[:4] / f'{scene_id}_ocm.tif'
    if out4.exists() and outo.exists():
        return scene_id, 'cached'
    try:
        model = load_model(model_path(results, scene_id))
        sr = read_island(p4, [1, 2, 3, 4]).astype(np.float32)
        ocm = read_island(pocm, [1], fill=OCM_NODATA)[0]
        dy, dx = displacement_maps(model, kind)
        data = (sr[0] > 0).astype(np.float32)
        warped = warp_arrays(list(sr) + [data, ocm], dy, dx, [1] * 5 + [0], [0] * 5 + [OCM_NODATA])
        valid = warped[4] > 0.99
        arr = np.stack(warped[:4])
        arr[:, ~valid] = 0
        write_tif(out4, np.rint(arr), 'uint16', 0)
        o = warped[5].astype(np.uint8)
        o[~valid] = OCM_NODATA
        write_tif(outo, o, 'uint8', OCM_NODATA)
    except Exception as e:  # noqa: BLE001
        return scene_id, f'error: {e}'
    return scene_id, 'ok'


@cli.command()
@results_option
@click.option('--model', 'kind', type=click.Choice(MODELS), default='L', show_default=True)
@click.option('--jobs', default=5, show_default=True)
@click.option('--limit', default=None, type=int)
def warp(results, kind, jobs, limit):
    """Aligned SR scenes (uint16 reflectance x 1e4, uncalibrated, nodata 0) and OCM masks
    for the video scenes, on the island grid, under planet_bci_aligned/<model>/."""
    df = load_scenes(results)
    sm = pd.read_csv(results / 'scene_models.csv')
    sm = sm[sm.band == 'both'].set_index('scene_id')
    ids = [s for s in df[df.use_video].scene_id if model_path(results, s).exists()][:limit]
    outdir = ALIGNED / kind
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Pool(jobs) as pool:
        for k, (sid, msg) in enumerate(pool.imap_unordered(
                warp_scene, [(s, results, kind, outdir) for s in ids]), 1):
            if msg != 'ok' and msg != 'cached':
                click.echo(f'{sid}: {msg}')
            if k % 25 == 0 or k == len(ids):
                click.echo(f'[{k}/{len(ids)} {time.time() - t0:5.0f}s]')
    cols = ['scene_id', 'acquired', 'instrument', 'satellite_id', 'view_angle', 'satellite_azimuth',
            'fraction_clear', 'offset_blue', 'offset_green', 'offset_red', 'offset_nir', 'offset_source']
    idx = df[df.scene_id.isin(ids)][cols].copy()
    idx['model'] = kind
    for c in ('med_raw', 'med_T', 'med_A', 'med_Lh', 'noise_px'):
        idx[f'{c}_px'] = idx.scene_id.map(sm[c])
    idx['path'] = idx.scene_id.map(lambda s: str(outdir / s[:4] / f'{s}_4band.tif'))
    idx.to_csv(outdir / 'index.csv', index=False)
    click.echo(f'{len(ids)} scenes -> {outdir}; index.csv written')


# --------------------------------------------------------------------------- rendering

def fixed_stretch():
    import yaml
    s = yaml.safe_load(STRETCH_YML.read_text())
    lo, hi = np.array(s['lo']), np.array(s['hi'])
    return lo[[2, 1, 0]], hi[[2, 1, 0]]  # R, G, B


def to_rgb(cal_bgrn, lo, hi, valid=None, gamma=0.8, nodata=(0.25, 0.25, 0.25)):
    """Calibrated B,G,R,NIR reflectance -> display RGB with a fixed stretch."""
    rgb = np.stack([cal_bgrn[2], cal_bgrn[1], cal_bgrn[0]], axis=-1)
    out = np.clip((rgb - lo) / (hi - lo), 0, 1) ** gamma
    if valid is None:
        valid = cal_bgrn[0] != 0
    out[~valid] = nodata
    return out


def calibrate(sr_uint16, offsets):
    """SR uint16 x 1e-4 plus the per-band MODIS offsets (B, G, R, NIR), nodata kept at 0."""
    cal = sr_uint16.astype(np.float32) * SCALE + np.asarray(offsets, np.float32)[:, None, None]
    cal[:, sr_uint16[0] == 0] = 0
    return cal


def island_outline():
    import geopandas as gpd
    return gpd.read_file(ISLAND_GEOJSON).to_crs(CRS)


def extent(window=None):
    r0, r1, c0, c1 = window or (0, ISLAND_SHAPE[0], 0, ISLAND_SHAPE[1])
    l, t = ISLAND_TRANSFORM * (c0, r0)
    r, b = ISLAND_TRANSFORM * (c1, r1)
    return (l, r, b, t)


def add_scalebar(ax, length_m, label, loc=(0.05, 0.05), color='white', fontsize=8, lw=3):
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=lw + 1.5, foreground='black')]
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bx, by = x0 + loc[0] * (x1 - x0), y0 + loc[1] * (y1 - y0)
    ax.plot([bx, bx + length_m], [by, by], color=color, lw=lw, solid_capstyle='butt',
            path_effects=halo, zorder=20)
    ax.text(bx + length_m / 2, by + 0.02 * (y1 - y0), label, ha='center', va='bottom', color=color,
            fontsize=fontsize, path_effects=[pe.withStroke(linewidth=1.8, foreground='black')], zorder=20)


def bare(ax):
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)


def ffmpeg_pipe(path, w, h, fps):
    return subprocess.Popen(
        ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{w}x{h}',
         '-r', str(fps), '-i', '-', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264',
         '-pix_fmt', 'yuv420p', '-crf', '18', '-movflags', '+faststart', str(path)],
        stdin=subprocess.PIPE)


def scene_label(row):
    name = INSTRUMENTS.get(row.instrument, (row.instrument,))[0]
    return f"{pd.Timestamp(row.acquired).strftime('%Y-%m-%d')}   {name} {row.satellite_id}"


def video_scenes(results, kind):
    """Video scenes in date order with their model residuals (remeasured if available)."""
    df = load_scenes(results)
    df = df[df.use_video & df.scene_id.map(lambda s: model_path(results, s).exists())].copy()
    sm = pd.read_csv(results / 'scene_models.csv')
    sm = sm[sm.band == 'both'].set_index('scene_id')
    key = {'T': 'med_T', 'A': 'med_A', 'L': 'med_Lh'}[kind]
    df['resid_px'] = df.scene_id.map(sm[key])
    df['resid_src'] = 'hold-out'
    rm = results / 'remeasure.csv'
    if rm.exists():
        r = pd.read_csv(rm)
        r = r[r.model == kind].set_index('scene_id').med_px
        has = df.scene_id.isin(r.index)
        df.loc[has, 'resid_px'] = df.loc[has, 'scene_id'].map(r)
        df.loc[has, 'resid_src'] = 'remeasured'
    return df.sort_values('acquired').reset_index(drop=True)


def comparison_window(results, ids, size=300):
    """(r0, r1, c0, c1) of the size x size block where the local and translation
    models disagree most on average over `ids`."""
    acc, n = np.zeros(grid_shape()), 0
    for s in ids:
        p = model_path(results, s)
        if not p.exists():
            continue
        m = load_model(p)
        d = np.hypot(m['field_dy'] - m['t'][0], m['field_dx'] - m['t'][1])
        d[~m['support']] = np.nan
        acc += np.nan_to_num(d)
        n += 1
    mean = acc / max(n, 1)
    # average over the block footprint (size px = size/STRIDE lattice cells)
    k = max(1, size // STRIDE)
    sm = ndimage.uniform_filter(mean, k, mode='constant')
    ri, ci = np.unravel_index(np.argmax(sm), sm.shape)
    rows, cols = tile_lattice()
    rc, cc = rows[ri] + TILE / 2, cols[ci] + TILE / 2
    r0 = int(np.clip(rc - size / 2, 0, ISLAND_SHAPE[0] - size))
    c0 = int(np.clip(cc - size / 2, 0, ISLAND_SHAPE[1] - size))
    return (r0, r0 + size, c0, c0 + size), float(sm[ri, ci])


@cli.command()
@results_option
@click.option('--outdir', type=click.Path(path_type=Path), default=DEFAULT_OUT, show_default=True)
@click.option('--model', 'kind', type=click.Choice(MODELS), default='L', show_default=True)
@click.option('--fps', default=6, show_default=True)
@click.option('--dpi', default=150, show_default=True)
@click.option('--min-clear', default=0.8, show_default=True)
@click.option('--which', type=click.Choice(['sequence', 'comparison', 'both']), default='both',
              show_default=True)
@click.option('--limit', default=None, type=int)
def video(results, outdir, kind, fps, dpi, min_clear, which, limit):
    """The aligned whole-island sequence and the three-panel comparison video."""
    import matplotlib.pyplot as plt
    outdir.mkdir(parents=True, exist_ok=True)
    df = video_scenes(results, kind)
    df = df[df.fraction_clear >= min_clear][:limit]
    lo, hi = fixed_stretch()
    outline = island_outline()
    ext = extent()
    offsets_cols = ['offset_blue', 'offset_green', 'offset_red', 'offset_nir']
    tag = '' if min_clear == 0.8 else f'_clear{int(min_clear * 100)}'

    if which in ('sequence', 'both'):
        name = f'island_coreg_sequence{tag}'
        fig = plt.figure(figsize=(8.0, 7.6))
        ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
        bare(ax)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect('equal')
        ax.set_facecolor('0.25')
        outline.boundary.plot(ax=ax, color='white', lw=0.6, alpha=0.7, zorder=5)
        bare(ax)
        im = ax.imshow(np.zeros((*ISLAND_SHAPE, 3)), extent=ext, zorder=1, interpolation='nearest')
        add_scalebar(ax, 1000, '1 km', fontsize=9)
        t1 = ax.text(0.02, 0.975, '', transform=ax.transAxes, color='white', fontsize=12, va='top',
                     fontweight='bold', path_effects=[__import__('matplotlib.patheffects', fromlist=['x'])
                                                      .withStroke(linewidth=2.5, foreground='black')])
        t2 = ax.text(0.98, 0.975, '', transform=ax.transAxes, color='white', fontsize=9.5, va='top',
                     ha='right', path_effects=[__import__('matplotlib.patheffects', fromlist=['x'])
                                               .withStroke(linewidth=2.0, foreground='black')])
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        w, h = int(w * dpi / fig.dpi), int(h * dpi / fig.dpi)
        fig.set_dpi(dpi)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        ff = ffmpeg_pipe(outdir / f'{name}.mp4', w, h, fps)
        t0 = time.time()
        for k, row in df.iterrows():
            p4 = ALIGNED / kind / row.scene_id[:4] / f'{row.scene_id}_4band.tif'
            with rasterio.open(p4) as src:
                sr = src.read()
            cal = calibrate(sr, row[offsets_cols].values.astype(float))
            im.set_data(to_rgb(cal, lo, hi, valid=sr[0] > 0))
            t1.set_text(scene_label(row))
            t2.set_text(f'{row.fraction_clear:.0%} clear   aligned: {MODEL_NAMES[kind]}, '
                        f'residual {row.resid_px * RES:.1f} m ({row.resid_src})')
            fig.canvas.draw()
            frame = bytes(fig.canvas.buffer_rgba())
            ff.stdin.write(frame)
            if k in (0, len(df) // 2):
                plt.imsave(outdir / f'{name}_frame{k}.png', np.frombuffer(frame, np.uint8).reshape(h, w, 4))
            if k % 25 == 0:
                click.echo(f'[{k}/{len(df)} {time.time() - t0:4.0f}s] {row.scene_id}')
        ff.stdin.close()
        if ff.wait():
            raise click.ClickException('ffmpeg failed')
        plt.close(fig)
        click.echo(f'Wrote {outdir / name}.mp4: {len(df)} frames at {fps} fps, {w}x{h}')

    if which in ('comparison', 'both'):
        name = f'island_coreg_comparison{tag}'
        win, score = comparison_window(results, df.scene_id.tolist())
        r0, r1, c0, c1 = win
        wext = extent(win)
        json.dump(dict(window=win, extent=wext, mean_local_minus_translation_px=score),
                  open(outdir / f'{name}_window.json', 'w'))
        kinds = ['raw', 'T', kind]
        titles = ['as delivered', 'translation', MODEL_NAMES[kind]]
        md = master_data()
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.5), dpi=dpi)
        fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.90, wspace=0.02)
        ims = []
        for ax, title in zip(axes, titles):
            bare(ax)
            ax.set_title(title, fontsize=11)
            ims.append(ax.imshow(np.zeros((r1 - r0, c1 - c0, 3)), extent=wext, interpolation='nearest'))
            # fixed marks every 150 m so the eye can judge motion
            for v in np.arange(wext[0], wext[1], 150)[1:]:
                ax.axvline(v, color='yellow', lw=0.5, alpha=0.5)
            for v in np.arange(wext[2], wext[3], 150)[1:]:
                ax.axhline(v, color='yellow', lw=0.5, alpha=0.5)
            ax.set_xlim(wext[0], wext[1])
            ax.set_ylim(wext[2], wext[3])
        add_scalebar(axes[0], 300, '300 m', fontsize=8)
        sup = fig.suptitle('', fontsize=11, y=0.985)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        ff = ffmpeg_pipe(outdir / f'{name}.mp4', w, h, max(2, fps // 2))
        t0 = time.time()
        for k, row in df.iterrows():
            p4, _ = scene_paths(row.scene_id)
            sr = read_island(p4, [1, 2, 3, 4]).astype(np.float32)
            model = load_model(model_path(results, row.scene_id))
            for im, kd in zip(ims, kinds):
                dy, dx = displacement_maps(model, kd, window=win)
                arrs = warp_arrays(list(sr) + [(sr[0] > 0).astype(np.float32)], dy, dx, [1] * 5, [0] * 5,
                                   window=win)
                cal = calibrate(np.rint(np.stack(arrs[:4])).astype(np.uint16),
                                row[offsets_cols].values.astype(float))
                im.set_data(to_rgb(cal, lo, hi, valid=arrs[4] > 0.99))
            sup.set_text(f'{scene_label(row)}      {row.fraction_clear:.0%} clear      '
                         f'residual after translation {sm_lookup(results, row.scene_id, "med_T") * RES:.1f} m, '
                         f'after {MODEL_NAMES[kind]} {row.resid_px * RES:.1f} m')
            fig.canvas.draw()
            frame = bytes(fig.canvas.buffer_rgba())
            ff.stdin.write(frame)
            if k == 0:
                plt.imsave(outdir / f'{name}_frame0.png', np.frombuffer(frame, np.uint8).reshape(h, w, 4))
            if k % 25 == 0:
                click.echo(f'[{k}/{len(df)} {time.time() - t0:4.0f}s] {row.scene_id}')
        ff.stdin.close()
        if ff.wait():
            raise click.ClickException('ffmpeg failed')
        plt.close(fig)
        click.echo(f'Wrote {outdir / name}.mp4: window rows {r0}-{r1}, cols {c0}-{c1} '
                   f'(mean |local - translation| {score:.2f} px)')


_SM = {}


def sm_lookup(results, scene_id, col):
    if 'df' not in _SM:
        sm = pd.read_csv(results / 'scene_models.csv')
        _SM['df'] = sm[sm.band == 'both'].set_index('scene_id')
    return _SM['df'].loc[scene_id, col]


# --------------------------------------------------------------------------- figures

def save_fig(fig, outdir, name, dpi):
    import matplotlib.pyplot as plt
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'{name}.{ext}', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    click.echo(f'Wrote {outdir / name}.png/.pdf')


def write_caption(outdir, name, caption, notes, made_by='figures', exts='{png,pdf}'):
    text = (f'# {name}.{exts}\n\nMade by `scripts/island_coreg.py {made_by}`.\n\n## Caption\n\n'
            f'{caption.strip()}\n\n## Notes\n\n' + '\n'.join(f'- {n}' for n in notes) + '\n')
    (outdir / f'{name}_caption.md').write_text(text)


COVERAGE = '2020-01 to 2026-08'


def load_tables(results):
    sm = pd.read_csv(results / 'scene_models.csv', parse_dates=['acquired'])
    tr = pd.read_csv(results / 'tile_residuals.csv.gz')
    meta = sm[sm.band == 'both'].set_index('scene_id')
    return sm, tr, meta


def fig_global_shifts(sm, outdir, dpi):
    import matplotlib.pyplot as plt
    b = sm[sm.band == 'both'].copy()
    b['east_m'] = b.t_dx_px * RES
    b['north_m'] = -b.t_dy_px * RES
    b['mag_m'] = np.hypot(b.east_m, b.north_m)
    fig = plt.figure(figsize=(11, 4.2))
    ax = fig.add_axes([0.05, 0.13, 0.30, 0.80])
    lim = max(12, np.ceil(b.mag_m.quantile(0.99) / 5) * 5)
    for r, lab in ((3, '1 px'), (10, '10 m')):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color='0.45', lw=0.8, ls=':'))
        ax.text(-r * 0.71 - 0.3, r * 0.71 + 0.3, lab, fontsize=7.5, color='0.35', ha='right', va='bottom')
    for key, (label, color) in reversed(list(INSTRUMENTS.items())):
        g = b[b.instrument == key]
        ax.scatter(g.east_m, g.north_m, s=8, lw=0, alpha=0.7, color=color, label=f'{label} ({len(g)})')
    ax.axhline(0, color='0.75', lw=0.6)
    ax.axvline(0, color='0.75', lw=0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('scene offset east of master (m)')
    ax.set_ylabel('scene offset north of master (m)')
    hd, lb = ax.get_legend_handles_labels()
    ax.legend(hd[::-1], lb[::-1], loc='upper left', fontsize=7.5, markerscale=2, framealpha=0.9)
    ax.set_title('(a) translation per scene', fontsize=10)
    ax2 = fig.add_axes([0.42, 0.13, 0.36, 0.80])
    for key, (label, color) in INSTRUMENTS.items():
        g = b[b.instrument == key]
        ax2.scatter(g.acquired, g.mag_m, s=8, lw=0, alpha=0.7, color=color)
    ax2.axhline(3, color='0.4', lw=0.8, ls=':')
    ax2.set_ylabel('offset magnitude (m)')
    ax2.set_title('(b) offset magnitude by acquisition date', fontsize=10)
    import matplotlib.dates as mdates
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.tick_params(axis='x', labelsize=8)
    ax3 = fig.add_axes([0.83, 0.13, 0.16, 0.80])
    bins = np.arange(0, min(b.mag_m.max(), 60) + 1.5, 1.5)
    ax3.hist(b.mag_m.clip(upper=bins[-1]), bins=bins, color='0.55', ec='white', lw=0.3)
    med = b.mag_m.median()
    ax3.axvline(med, color='k', lw=1.2)
    ax3.text(med + 0.5, ax3.get_ylim()[1] * 0.95, f'median {med:.1f} m', fontsize=8, va='top')
    ax3.set_xlabel('offset magnitude (m)')
    ax3.set_ylabel('scenes')
    ax3.set_title('(c)', fontsize=10)
    name = 'island_coreg_global_shifts'
    save_fig(fig, outdir, name, dpi)
    b[['scene_id', 'acquired', 'instrument', 'satellite_id', 'view_angle', 'satellite_azimuth',
       'fraction_clear', 'east_m', 'north_m', 'mag_m', 'global_peak', 'n_tiles']].to_csv(
        outdir / f'{name}.csv', index=False)
    q = b.mag_m.quantile([0.25, 0.5, 0.75, 0.9, 0.99]).round(1).to_dict()
    by = b.groupby('instrument').mag_m.median().round(1).to_dict()
    caption = (
        f'Whole-scene georeferencing offset of each PlanetScope scene over Barro Colorado Island relative '
        f'to the master scene {MASTER} (SuperDove, 27 February 2023): (a) the offset vector (east and north '
        f'components in meters; the median of the tile displacements after a whole-island normalized '
        f'cross-correlation search, so a point in the master appears displaced by this vector in the scene '
        f'as delivered), one point per scene colored by Dove generation (blue, Dove Classic PS2; orange, '
        f'Dove-R PS2.SD; green, SuperDove PSB.SD), with dotted circles at 1 pixel (3 m) and 10 m; (b) the '
        f'offset magnitude (m) against acquisition date, same colors, dotted line at 1 pixel; (c) its '
        f'histogram in 1.5 m bins with the median marked. All {len(b)} scenes at least 50% island-clear '
        f'with a surface-reflectance product that pass the global match gate, {COVERAGE}.')
    notes = [f'{len(b)} scenes; magnitude quantiles (m) {q}', f'median by instrument (m): {by}',
             f'> 3 m (1 px): {(b.mag_m > 3).mean():.0%}; > 10 m: {(b.mag_m > 10).mean():.0%}; '
             f'> 30 m: {(b.mag_m > 30).mean():.0%}',
             'Offsets are relative to the master, whose own georeferencing error is unknown; the median '
             f'offset over all scenes is ({b.east_m.median():.1f}, {b.north_m.median():.1f}) m east, north.',
             'Data: results/island_coreg/scene_models.csv (band both, t_dx_px, t_dy_px).']
    write_caption(outdir, name, caption, notes)


def fig_residuals(sm, tr, outdir, dpi):
    import matplotlib.pyplot as plt
    b = sm[sm.band == 'both'].copy()
    t = tr[tr.band == 'both']
    mags = {'raw': np.hypot(t.dy_px, t.dx_px), 'T': np.hypot(t.rT_dy, t.rT_dx),
            'A': np.hypot(t.rA_dy, t.rA_dx), 'L': np.hypot(t.rL_dy, t.rL_dx),
            'Lh': np.hypot(t.rLh_dy, t.rLh_dx)}
    colors = {'raw': '0.3', 'T': '#1f77b4', 'A': '#ff7f0e', 'L': '#2ca02c', 'Lh': '#2ca02c'}
    styles = {'raw': ':', 'T': '-', 'A': '-', 'L': '--', 'Lh': '-'}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.15, top=0.90, wspace=0.28)
    ax = axes[0]
    xs = np.linspace(0, 6, 400)
    for k, m in mags.items():
        v = m.dropna().values * RES
        ax.plot(xs, np.searchsorted(np.sort(v), xs) / len(v), color=colors[k], ls=styles[k], lw=1.6,
                label=f'{MODEL_NAMES[k]} (median {np.median(v):.2f} m)')
    noise = b.noise_px.median() * RES
    ax.axvline(noise, color='0.5', lw=0.8, ls='-.')
    ax.text(noise + 0.05, 0.97, f'red-NIR noise floor {noise:.2f} m', fontsize=7.5, color='0.35',
            va='top', rotation=90)
    ax.axvline(3, color='0.7', lw=0.6)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    ax.set_xlabel('tile residual displacement (m)')
    ax.set_ylabel('cumulative share of tiles')
    ax.legend(fontsize=7.5, loc='lower right')
    ax.set_title('(a) residual after each model, all tiles', fontsize=10)
    ax = axes[1]
    order = ['T', 'A', 'Lh']
    pos = 0
    ticks, labels = [], []
    for key, (label, color) in INSTRUMENTS.items():
        g = b[b.instrument == key]
        data = [g[f'med_{k}'].dropna().values * RES for k in order]
        bp = ax.boxplot(data, positions=pos + np.arange(3) * 0.8, widths=0.6, patch_artist=True,
                        showfliers=False, medianprops=dict(color='k'))
        for patch, k in zip(bp['boxes'], order):
            patch.set_facecolor(colors[k])
            patch.set_alpha(0.6)
        ticks.append(pos + 0.8)
        labels.append(f'{label}\n({len(g)})')
        pos += 3.2
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('per-scene median residual (m)')
    ax.axhline(noise, color='0.5', lw=0.8, ls='-.')
    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor=colors[k], alpha=0.6) for k in order],
              [MODEL_NAMES[k] for k in order], fontsize=7.5, loc='upper right')
    ax.set_title('(b) per-scene median by Dove generation', fontsize=10)
    ax = axes[2]
    for key, (label, color) in INSTRUMENTS.items():
        g = b[b.instrument == key]
        ax.scatter(g.med_T * RES, g.med_Lh * RES, s=9, lw=0, alpha=0.7, color=color, label=label)
    lim = max(2.5, (b.med_T * RES).quantile(0.995) * 1.05)
    ax.plot([0, lim], [0, lim], color='0.5', lw=0.8)
    ax.plot([0, lim], [noise, noise], color='0.5', lw=0.8, ls='-.')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('median residual after translation (m)')
    ax.set_ylabel('median residual after local warp, hold-out (m)')
    ax.legend(fontsize=7.5, loc='upper left')
    ax.set_title('(c) gain from the local warp per scene', fontsize=10)
    name = 'island_coreg_residuals'
    save_fig(fig, outdir, name, dpi)
    gain = (b.med_T - b.med_Lh) * RES
    share = {k: f"{(b[f'p90_{k}'] * RES > 3).mean():.0%}" for k in ('T', 'A', 'Lh')}
    share_half = {k: f"{(b[f'p90_{k}'] * RES > 1.5).mean():.0%}" for k in ('T', 'A', 'Lh')}
    med = {k: round(float(np.nanmedian(mags[k]) * RES), 2) for k in mags}
    caption = (
        f'Residual misalignment of PlanetScope scenes over Barro Colorado Island relative to the master '
        f'scene under three alignment models. (a) Cumulative distribution over all tiles (64 px = 192 m '
        f'tiles on a 32 px stride, red and NIR pooled) of the residual displacement magnitude in meters '
        f'before any correction (dotted gray) and after a per-scene translation (blue), affine (orange), '
        f'local warp evaluated in-sample (dashed green) and local warp evaluated by hold-out (solid green: '
        f'each tile predicted from a Gaussian-smoothed field, sigma 144 m, fitted to tiles that share no '
        f'pixels with it); the dash-dotted line is the measurement-noise floor, the median over scenes of '
        f'the red-versus-NIR disagreement of the tile vectors divided by sqrt(2). (b) Per-scene median '
        f'residual (m) under translation, affine and hold-out local warp, as box plots (median, '
        f'quartiles, 1.5 IQR whiskers) by Dove generation with scene counts. (c) Per-scene median residual '
        f'after the hold-out local warp against that after translation (m), one point per scene colored by '
        f'generation as in (b), with the 1:1 line and the noise floor; the vertical distance to the 1:1 '
        f'line is what the local warp buys. {len(b)} scenes, {COVERAGE}.')
    notes = [f'median residual over all tiles (m): {med}',
             f'median over scenes of the per-scene median (m): ' +
             str({k: round(float(b[f"med_{k}"].median() * RES), 2) for k in ('raw', 'T', 'A', 'L', 'Lh')}),
             f'noise floor median {noise:.2f} m; coherence (red-NIR correlation of affine residuals) '
             f'median {b.coherence.median():.2f}',
             f'share of scenes with p90 residual > 3 m (1 px): {share}; > 1.5 m: {share_half}',
             f'local-warp gain (median after T minus after hold-out L): median {gain.median():.2f} m, '
             f'p90 {gain.quantile(0.9):.2f} m; gain > 1 m in {(gain > 1).mean():.0%} of scenes, '
             f'> 0.5 m in {(gain > 0.5).mean():.0%}',
             f'hold-out median by smoothing sigma (tiles): ' +
             str({c[7:]: round(float(b[c].median() * RES), 2) for c in b.columns if c.startswith('med_Lh_s')}),
             'Data: results/island_coreg/scene_models.csv, tile_residuals.csv.gz (band both).']
    write_caption(outdir, name, caption, notes)


def tile_centers():
    rows, cols = tile_lattice()
    return rows + TILE / 2, cols + TILE / 2


def fig_residual_map(sm, tr, outdir, dpi):
    import matplotlib.pyplot as plt
    meta = sm[sm.band == 'both'].set_index('scene_id')
    t = tr[tr.band == 'both'].copy()
    t['az'] = t.scene_id.map(meta.satellite_azimuth)
    t['side'] = np.where(t.az < 180, 'east', 'west')
    dem3 = dem_arrays() if DEM_3M.exists() else None
    outline = island_outline()
    ext = extent()
    rc, cc = tile_centers()
    fig, axes = plt.subplots(2, 2, figsize=(10, 9.2))
    fig.subplots_adjust(left=0.01, right=0.90, bottom=0.01, top=0.95, wspace=0.03, hspace=0.08)
    stats = {}
    vmax = None
    for i, (mk, lab) in enumerate((('rT', 'after translation'), ('rA', 'after affine'))):
        for j, side in enumerate(('east', 'west')):
            ax = axes[i, j]
            g = t[t.side == side].groupby(['ri', 'ci']).agg(dy=(f'{mk}_dy', 'mean'), dx=(f'{mk}_dx', 'mean'),
                                                         n=(f'{mk}_dy', 'size')).reset_index()
            ns = t[t.side == side].scene_id.nunique()
            g = g[g.n >= min(20, max(1, ns // 3))]
            mag = np.hypot(g.dy, g.dx) * RES
            if vmax is None:
                vmax = float(np.percentile(mag, 98))
            x = cc[g.ci] * RES + ISLAND_BOUNDS[0]
            y = ISLAND_BOUNDS[3] - rc[g.ri] * RES
            if dem3 is not None:
                ax.imshow(hillshade(dem3[0]), extent=ext, cmap='gray', vmin=0, vmax=1.3, alpha=0.9)
            outline.boundary.plot(ax=ax, color='k', lw=0.5)
            # east is +dx, north is -dy
            q = ax.quiver(x[::2], y[::2], g.dx.values[::2] * RES, -g.dy.values[::2] * RES, mag.values[::2],
                          cmap='viridis', clim=(0, vmax), scale=60, scale_units='width', width=0.0025,
                          pivot='tail')
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
            ax.set_aspect('equal')
            bare(ax)
            ax.set_title(f'{lab}, satellite to the {side} ({ns} scenes)', fontsize=10)
            stats[(mk, side)] = dict(n_scenes=ns, mean_field_median_m=float(np.median(mag)),
                                     mean_field_p90_m=float(np.percentile(mag, 90)))
    add_scalebar(axes[1, 0], 1000, '1 km', color='k', fontsize=8)
    cax = fig.add_axes([0.915, 0.25, 0.018, 0.5])
    fig.colorbar(q, cax=cax, label='mean residual magnitude (m)')
    name = 'island_coreg_residual_map'
    save_fig(fig, outdir, name, dpi)
    rows = []
    for (mk, side), s in stats.items():
        rows.append(dict(model=mk, side=side, **s))
    pd.DataFrame(rows).to_csv(outdir / f'{name}.csv', index=False)
    caption = (
        f'Spatially coherent part of the misalignment of PlanetScope scenes over Barro Colorado Island: '
        f'arrows show the mean residual displacement vector per tile position (64 px = 192 m tiles on a '
        f'32 px stride, every second tile drawn, red and NIR pooled, averaged over scenes, tiles with fewer '
        f'than 20 scenes omitted), after each scene\'s own translation (top row) and after its own affine '
        f'(bottom row), for scenes imaged with the satellite to the east (left column, satellite azimuth '
        f'below 180 degrees) and to the west (right column). Arrow length and color give the mean residual '
        f'magnitude in meters (one color scale, capped at the 98th percentile); arrows point from where the '
        f'master content is to where it appears in the scenes on average, east positive to the right and '
        f'north up. The background is a hillshade of the Copernicus GLO-30 elevation model and the black '
        f'line the island shoreline. A pattern that reverses between the two columns is the signature of '
        f'orthorectification (elevation) error; a pattern common to both is in the master or in all scenes. '
        f'{meta.shape[0]} scenes, {COVERAGE}.')
    notes = [f'{k[0]} {k[1]}: {v}' for k, v in stats.items()] + [
        'The per-scene random residual is larger than the mean field (see island_coreg_residuals), so the '
        'mean field is the part a fixed local correction could remove for every scene.',
        'Data: results/island_coreg/tile_residuals.csv.gz (band both), scene_models.csv (satellite_azimuth), '
        'dem/bci_glo30_3m.tif.']
    write_caption(outdir, name, caption, notes)


def fig_view_geometry(sm, outdir, dpi):
    import matplotlib.pyplot as plt
    b = sm[sm.band == 'both'].dropna(subset=['view_angle']).copy()
    b['gain_m'] = (b.med_T - b.med_Lh) * RES
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.16, top=0.90, wspace=0.3)
    for ax, col, lab, title in ((axes[0], 'affine_amp_m', 'affine non-translational amplitude (m)',
                                 '(a) affine stretch/shear across the island'),
                                (axes[1], 'gain_m', 'local-warp gain, median T minus hold-out L (m)',
                                 '(b) what the local warp removes'),
                                (axes[2], 'coherence', 'red-NIR coherence of affine residuals',
                                 '(c) is the remaining residual real?')):
        for key, (label, color) in INSTRUMENTS.items():
            g = b[b.instrument == key]
            ax.scatter(g.view_angle, g[col], s=9, lw=0, alpha=0.7, color=color, label=label)
        # binned median
        bins = np.arange(0, 8.1, 1)
        idx = np.digitize(b.view_angle, bins)
        med = [b[col][idx == i].median() for i in range(1, len(bins))]
        ax.plot(bins[:-1] + 0.5, med, color='k', lw=1.4, marker='o', ms=3, label='median per 1 deg bin')
        ax.set_xlabel('view angle (degrees)')
        ax.set_ylabel(lab, fontsize=8.5)
        ax.set_title(title, fontsize=10)
        if col != 'coherence':
            ax.set_ylim(bottom=min(0, b[col].quantile(0.005)))
        ax.set_ylim(top=b[col].quantile(0.995) * 1.1 if col != 'coherence' else 1)
    axes[0].legend(fontsize=7.5)
    from scipy.stats import spearmanr
    rho = {c: spearmanr(b.view_angle, b[c], nan_policy='omit')[0] for c in ('affine_amp_m', 'gain_m', 'coherence')}
    name = 'island_coreg_view_geometry'
    save_fig(fig, outdir, name, dpi)
    caption = (
        f'Dependence of the non-rigid part of PlanetScope misalignment over Barro Colorado Island on the '
        f'satellite view angle (degrees off nadir, from the scene metadata). (a) Amplitude of the affine '
        f'model\'s non-translational part, the largest difference in displacement between two points of the '
        f'island implied by its scale, shear and rotation terms (m). (b) Gain from the local warp, the '
        f'per-scene median tile residual after translation minus that after the hold-out local warp (m). '
        f'(c) Coherence, the Pearson correlation between the red-band and NIR-band affine-residual fields '
        f'of the same scene (1 means the residual is the same in both bands and so is geometric, 0 means '
        f'it is matching noise). One point per scene colored by Dove generation (blue, Dove Classic; '
        f'orange, Dove-R; green, SuperDove), with the median per 1 degree bin as a black line. '
        f'{len(b)} scenes, {COVERAGE}.')
    notes = [f'Spearman rho with view angle: {dict((k, round(float(v), 2)) for k, v in rho.items())}',
             f'affine amplitude median {b.affine_amp_m.median():.1f} m, p90 {b.affine_amp_m.quantile(0.9):.1f} m',
             f'view angle range {b.view_angle.min():.1f}-{b.view_angle.max():.1f} deg, median {b.view_angle.median():.1f}',
             f'A DEM error of h meters displaces a pixel by h tan(view angle): at the median view angle '
             f'{b.view_angle.median():.1f} deg, 1 m of displacement needs {1 / np.tan(np.radians(b.view_angle.median())):.0f} m of DEM error.',
             'Data: results/island_coreg/scene_models.csv (band both).']
    write_caption(outdir, name, caption, notes)


def fig_terrain(sm, tr, outdir, dpi):
    import matplotlib.pyplot as plt
    if not DEM_3M.exists():
        click.echo('no DEM; skipping island_coreg_terrain')
        return
    dem3 = dem_arrays()
    meta = sm[sm.band == 'both'].set_index('scene_id')
    t = tr[tr.band == 'both'].copy()
    rc, cc = tile_centers()
    ri, ci = t.ri.values, t.ci.values
    ys, xs = rc[ri].astype(int), cc[ci].astype(int)
    # tile-mean elevation and slope (64 px windows via a uniform filter on the DEM)
    k = TILE
    elev = ndimage.uniform_filter(dem3[0], k)[ys, xs]
    slope = ndimage.uniform_filter(dem3[1], k)[ys, xs]
    t['elev'], t['slope'] = elev, slope
    az = np.radians(t.scene_id.map(meta.satellite_azimuth).values)
    # along-view component of the residual: unit vector toward the satellite (east = sin az, north = cos az)
    t['along_T'] = t.rT_dx * np.sin(az) + (-t.rT_dy) * np.cos(az)
    t['along_A'] = t.rA_dx * np.sin(az) + (-t.rA_dy) * np.cos(az)
    t['magT'] = np.hypot(t.rT_dy, t.rT_dx) * RES
    t['magA'] = np.hypot(t.rA_dy, t.rA_dx) * RES
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.16, top=0.90, wspace=0.3)
    for ax, col, lab, nb in ((axes[0], 'elev', 'tile mean elevation (m a.s.l.)', 10),
                             (axes[1], 'slope', 'tile mean slope (degrees)', 10)):
        edges = np.quantile(t[col], np.linspace(0, 1, nb + 1))
        idx = np.clip(np.digitize(t[col], edges) - 1, 0, nb - 1)
        mid = 0.5 * (edges[1:] + edges[:-1])
        for mk, color, label in (('magT', '#1f77b4', 'after translation'), ('magA', '#ff7f0e', 'after affine')):
            m = [t[mk][idx == i].mean() for i in range(nb)]
            se = [t[mk][idx == i].std() / np.sqrt((idx == i).sum()) for i in range(nb)]
            ax.errorbar(mid, m, yerr=np.array(se) * 1.96, color=color, marker='o', ms=3, lw=1.2, label=label)
        ax.set_xlabel(lab)
        ax.set_ylabel('mean residual magnitude (m)')
    axes[0].legend(fontsize=7.5)
    axes[0].set_title('(a) residual vs elevation', fontsize=10)
    axes[1].set_title('(b) residual vs slope', fontsize=10)
    # per-scene slope of along-view residual (after translation) against elevation, vs tan(view angle)
    rows = []
    for sid, g in t.groupby('scene_id'):
        if len(g) < 100 or sid not in meta.index:
            continue
        p = np.polyfit(g.elev, g.along_T, 1)
        r = np.corrcoef(g.elev, g.along_T)[0, 1]
        rows.append(dict(scene_id=sid, slope_px_per_m=p[0], r=r, view_angle=meta.loc[sid, 'view_angle'],
                         instrument=meta.loc[sid, 'instrument'], side=meta.loc[sid, 'satellite_azimuth'] < 180))
    pr = pd.DataFrame(rows)
    pr['implied_dem_err_m'] = pr.slope_px_per_m * RES / np.tan(np.radians(pr.view_angle.clip(lower=0.2)))
    ax = axes[2]
    for key, (label, color) in INSTRUMENTS.items():
        g = pr[pr.instrument == key]
        ax.scatter(np.tan(np.radians(g.view_angle)), g.slope_px_per_m * RES * 100, s=9, lw=0, alpha=0.7,
                   color=color, label=label)
    ax.axhline(0, color='0.5', lw=0.8)
    ax.set_xlabel('tan(view angle)')
    ax.set_ylabel('along-view residual per 100 m of elevation (m)')
    ax.set_title('(c) elevation-dependent along-view residual', fontsize=10)
    ax.legend(fontsize=7.5)
    from scipy.stats import spearmanr
    rho_e = spearmanr(t.elev, t.magT)[0]
    rho_s = spearmanr(t.slope, t.magT)[0]
    name = 'island_coreg_terrain'
    save_fig(fig, outdir, name, dpi)
    pr.to_csv(outdir / f'{name}.csv', index=False)
    med_r = pr.r.abs().median()
    caption = (
        f'Test for a terrain (orthorectification) signature in the PlanetScope misalignment over Barro '
        f'Colorado Island. (a) Mean tile residual displacement magnitude (m) after each scene\'s translation '
        f'(blue) and affine (orange) in ten equal-count bins of the tile\'s mean Copernicus GLO-30 elevation '
        f'(m above sea level), with 95% intervals of the mean; (b) the same against the tile\'s mean slope '
        f'(degrees). (c) Per scene, the slope of a straight-line fit of the along-view component of the '
        f'residual after translation (positive toward the satellite) against tile elevation, expressed as '
        f'meters of displacement per 100 m of elevation, plotted against the tangent of the view angle; an '
        f'elevation error of h meters in the orthorectification DEM would give a slope of h tan(view angle) '
        f'per unit elevation error, so points far from zero that grow with tan(view angle) indicate '
        f'terrain-driven residuals. Points are scenes colored by Dove generation (blue, Dove Classic; '
        f'orange, Dove-R; green, SuperDove). 64 px (192 m) tiles on a 32 px stride, red and NIR pooled, '
        f'{meta.shape[0]} scenes, {COVERAGE}.')
    notes = [f'Spearman rho of tile residual magnitude after translation with elevation {rho_e:.3f}, '
             f'with slope {rho_s:.3f} ({len(t)} tiles)',
             f'per-scene |r| of along-view residual vs elevation: median {med_r:.2f}; slope median '
             f'{pr.slope_px_per_m.median() * RES * 100:.2f} m per 100 m, p10/p90 '
             f'{pr.slope_px_per_m.quantile(0.1) * RES * 100:.2f}/{pr.slope_px_per_m.quantile(0.9) * RES * 100:.2f}',
             f'implied DEM error (slope x 3 m / tan view angle, per unit elevation): median '
             f'{pr.implied_dem_err_m.median():.2f} m per m, p10/p90 {pr.implied_dem_err_m.quantile(0.1):.2f}/'
             f'{pr.implied_dem_err_m.quantile(0.9):.2f}',
             f'island land elevation {dem3[0][land_mask()].min():.0f}-{dem3[0][land_mask()].max():.0f} m',
             'Data: results/island_coreg/tile_residuals.csv.gz, scene_models.csv, dem/bci_glo30_3m.tif.']
    write_caption(outdir, name, caption, notes)


def fig_worst_cases(sm, tr, results, outdir, dpi, n=3, zoom=100):
    import matplotlib.pyplot as plt
    from PIL import Image
    df = load_scenes(results).set_index('scene_id')
    b = sm[(sm.band == 'both') & sm.scene_id.isin(df[df.use_video].index)].copy()
    b['gain'] = b.med_T - b.med_Lh
    worst = b.sort_values('gain', ascending=False).head(n)
    lo, hi = fixed_stretch()
    md = master_data()
    master_sr = read_island(scene_paths(MASTER)[0], [1, 2, 3, 4])
    master_cal = calibrate(master_sr, df.loc[MASTER, ['offset_blue', 'offset_green', 'offset_red', 'offset_nir']].values.astype(float))
    outline = island_outline()
    ext = extent()
    rc, cc = tile_centers()
    fig, axes = plt.subplots(n, 4, figsize=(13, 3.3 * n))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.95, wspace=0.03, hspace=0.12)
    (outdir / 'gifs').mkdir(exist_ok=True)
    notes = []
    for i, (_, row) in enumerate(worst.iterrows()):
        sid = row.scene_id
        t = tr[(tr.band == 'both') & (tr.scene_id == sid)]
        model = load_model(model_path(results, sid))
        ax = axes[i, 0]
        outline.boundary.plot(ax=ax, color='k', lw=0.5)
        mag = np.hypot(t.rT_dy, t.rT_dx) * RES
        ax.quiver(cc[t.ci] * RES + ISLAND_BOUNDS[0], ISLAND_BOUNDS[3] - rc[t.ri] * RES,
                  t.rT_dx * RES, -t.rT_dy * RES, mag, cmap='viridis', clim=(0, 4), scale=80,
                  scale_units='width', width=0.003)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect('equal')
        bare(ax)
        ax.set_title(f'{scene_label(df.loc[sid])}: residual after translation', fontsize=9)
        # zoom where |L - T| is largest on the lattice
        d = np.hypot(model['field_dy'] - model['t'][0], model['field_dx'] - model['t'][1])
        d[~model['support']] = -1
        ri, ci = np.unravel_index(np.argmax(d), d.shape)
        r0 = int(np.clip(rc[ri] - zoom / 2, 0, ISLAND_SHAPE[0] - zoom))
        c0 = int(np.clip(cc[ci] - zoom / 2, 0, ISLAND_SHAPE[1] - zoom))
        win = (r0, r0 + zoom, c0, c0 + zoom)
        wext = extent(win)
        ax.add_patch(plt.Rectangle((wext[0], wext[2]), wext[1] - wext[0], wext[3] - wext[2], fill=False,
                                   color='red', lw=1.2))
        sr = read_island(scene_paths(sid)[0], [1, 2, 3, 4]).astype(np.float32)
        offs = df.loc[sid, ['offset_blue', 'offset_green', 'offset_red', 'offset_nir']].values.astype(float)
        panels = [('master', to_rgb(master_cal[:, r0:r0 + zoom, c0:c0 + zoom], lo, hi))]
        frames = [panels[0][1]]
        for kd, lab in (('T', 'translation'), ('L', 'local warp')):
            dy, dx = displacement_maps(model, kd, window=win)
            arrs = warp_arrays(list(sr) + [(sr[0] > 0).astype(np.float32)], dy, dx, [1] * 5, [0] * 5, window=win)
            cal = calibrate(np.rint(np.stack(arrs[:4])).astype(np.uint16), offs)
            rgb = to_rgb(cal, lo, hi, valid=arrs[4] > 0.99)
            panels.append((lab, rgb))
            frames.append(rgb)
        # master edges to overlay
        mz = md['bands']['red'][r0:r0 + zoom, c0:c0 + zoom]
        edges = ndimage.sobel(mz, 0) ** 2 + ndimage.sobel(mz, 1) ** 2
        edge_mask = edges > np.percentile(edges, 90)
        for j, (lab, rgb) in enumerate(panels):
            ax = axes[i, j + 1]
            ax.imshow(rgb, extent=wext, interpolation='nearest')
            ov = np.zeros((*edge_mask.shape, 4))
            ov[edge_mask] = (0, 1, 1, 0.45)
            ax.imshow(ov, extent=wext, interpolation='nearest')
            bare(ax)
            ax.set_title(lab, fontsize=9)
        add_scalebar(axes[i, 1], 100, '100 m', fontsize=7)
        # blink GIF of the zoom, 4x nearest
        imgs = [Image.fromarray((f * 255).astype(np.uint8)).resize((zoom * 4, zoom * 4), Image.NEAREST)
                for f in frames]
        gif = outdir / 'gifs' / f'island_worst_{i + 1}_{sid}_blink.gif'
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=700, loop=0)
        notes.append(f'{i + 1}. {sid} ({df.loc[sid, "instrument"]}, view {df.loc[sid, "view_angle"]} deg, '
                     f'{df.loc[sid, "fraction_clear"]:.0%} clear): median residual T {row.med_T * RES:.2f} m, '
                     f'A {row.med_A * RES:.2f} m, hold-out L {row.med_Lh * RES:.2f} m; affine amplitude '
                     f'{row.affine_amp_m:.1f} m; coherence {row.coherence:.2f}; zoom at rows {r0}-{r0 + zoom}, '
                     f'cols {c0}-{c0 + zoom} where |L - T| = {d[ri, ci]:.2f} px; GIF {gif.name} '
                     f'(frames: master, translation, local warp)')
    name = 'island_coreg_worst_cases'
    save_fig(fig, outdir, name, dpi)
    caption = (
        f'The {n} video scenes (at least 80% island-clear) that gain most from a local warp over a '
        f'translation. Left column: the residual displacement of every tile (64 px = 192 m tiles on a 32 px '
        f'stride, red and NIR pooled) after the scene\'s own translation, as arrows pointing from the '
        f'master position to where the content appears in the scene, colored by magnitude in meters (0-4 m '
        f'scale), over the island shoreline; the red square is the 300 x 300 m zoom shown to its right, '
        f'placed where the local and translation models disagree most. Right three columns: the zoom in the '
        f'master scene, in the scene after its translation, and after its local warp, all as true color '
        f'(R, G, B) of MODIS-calibrated surface reflectance on the fixed stretch of '
        f'config/planet_fixed_stretch.yml, with the strongest 10% of the master\'s red-band edges overlaid '
        f'in cyan on every panel as a fixed reference; the forest should sit under the cyan edges when '
        f'aligned. Scene date, Dove generation and satellite are in each row title; {COVERAGE}.')
    write_caption(outdir, name, caption, notes + [
        'Blink GIFs of each zoom (master, translation, local warp; 4x nearest-neighbor upscaling, 0.7 s '
        'per frame) are in gifs/.', 'Data: results/island_coreg/scene_models.csv, tile_residuals.csv.gz, '
        'models/<scene>.npz; planet_bcnm_sr/4band.'])


def fig_master(results, outdir, dpi):
    import matplotlib.pyplot as plt
    df = load_scenes(results).set_index('scene_id')
    lo, hi = fixed_stretch()
    sr = read_island(scene_paths(MASTER)[0], [1, 2, 3, 4])
    cal = calibrate(sr, df.loc[MASTER, ['offset_blue', 'offset_green', 'offset_red', 'offset_nir']].values.astype(float))
    rgb = to_rgb(cal, lo, hi)
    land = land_mask()
    ext = extent()
    fig = plt.figure(figsize=(7.5, 7.2))
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.94])
    ax.imshow(rgb, extent=ext, interpolation='nearest')
    ov = np.zeros((*land.shape, 4))
    ov[~land] = (0.1, 0.3, 0.8, 0.35)
    ax.imshow(ov, extent=ext, interpolation='nearest')
    rows, cols = tile_lattice()
    for r in rows[::8]:
        ax.axhline(ISLAND_BOUNDS[3] - r * RES, color='white', lw=0.3, alpha=0.5)
    for c in cols[::8]:
        ax.axvline(ISLAND_BOUNDS[0] + c * RES, color='white', lw=0.3, alpha=0.5)
    island_outline().boundary.plot(ax=ax, color='yellow', lw=0.6)
    add_scalebar(ax, 1000, '1 km', fontsize=9)
    bare(ax)
    ax.set_title(f'Master scene {MASTER}: SuperDove {df.loc[MASTER, "satellite_id"]}, '
                 f'{pd.Timestamp(df.loc[MASTER, "acquired"]).strftime("%Y-%m-%d")}, view angle '
                 f'{df.loc[MASTER, "view_angle"]} deg', fontsize=10)
    name = 'island_coreg_master'
    save_fig(fig, outdir, name, dpi)
    caption = (
        f'The master scene to which every PlanetScope scene over Barro Colorado Island is aligned: '
        f'{MASTER} (SuperDove PSB.SD, satellite {df.loc[MASTER, "satellite_id"]}, 27 February 2023, view '
        f'angle {df.loc[MASTER, "view_angle"]} degrees, 100% island-clear), in true color (R, G, B) of '
        f'MODIS-calibrated surface reflectance on the fixed stretch of config/planet_fixed_stretch.yml, on '
        f'the 3 m island grid (1958 x 1802 pixels, UTM 17N). The blue tint marks pixels excluded from '
        f'matching (the static water mask eroded by 2 pixels), the yellow line is the island shoreline and '
        f'the thin white grid shows every eighth tile boundary (256 pixels, 768 m; the tiles themselves are '
        f'64 pixels on a 32 pixel stride).')
    write_caption(outdir, name, caption, [
        'Chosen by hand: the archive midpoint, dry season, fully clear, near-nadir; not optimized.',
        f'Land pixels available for matching: {land.sum():,} of {land.size:,}.',
        'Alignment is relative to this scene; its own georeferencing error is inherited by the sequence '
        '(see island_coreg_global_shifts notes for the median offset of the archive relative to it).'])


def write_video_captions(results, outdir, kind):
    df = video_scenes(results, kind)
    df = df[df.fraction_clear >= 0.8]
    seq = 'island_coreg_sequence'
    if (outdir / f'{seq}.mp4').exists():
        caption = (
            f'Time-lapse of Barro Colorado Island in PlanetScope imagery, {len(df)} scenes at least 80% '
            f'island-clear from {df.acquired.min():%Y-%m-%d} to {df.acquired.max():%Y-%m-%d}, each aligned to '
            f'the master scene {MASTER} by a {MODEL_NAMES[kind]} fitted to its tile displacement field, '
            f'rendered as true color (R, G, B) of MODIS-calibrated surface reflectance (Planet SR plus the '
            f'per-scene MODIS offset of planet_scene_offsets.csv) on one fixed stretch for every frame '
            f'(config/planet_fixed_stretch.yml), so brightness and color changes between frames are '
            f'reflectance changes, not display changes. Clouds and their shadows are shown as they are; '
            f'pixels without data are dark gray; the white line is the island shoreline. Each frame is '
            f'labeled with the acquisition date, Dove generation and satellite, the island-clear fraction '
            f'and the median tile residual after alignment in meters (re-measured against the master after '
            f'warping where available, otherwise the hold-out estimate). 6 frames per second.')
        write_caption(outdir, seq, caption, [
            f'{len(df)} frames; by instrument {df.instrument.value_counts().to_dict()}',
            f'median residual after alignment {df.resid_px.median() * RES:.2f} m '
            f'(p90 {df.resid_px.quantile(0.9) * RES:.2f} m)',
            f'offset source: {df.offset_source.value_counts().to_dict()}',
            f'Aligned rasters: planet_bci_aligned/{kind}/ (uint16 SR x 1e4, uncalibrated; index.csv has the offsets).'],
            made_by='video', exts='mp4')
    cmp_ = 'island_coreg_comparison'
    if (outdir / f'{cmp_}.mp4').exists():
        w = json.load(open(outdir / f'{cmp_}_window.json'))
        caption = (
            f'The same {len(df)} scenes as island_coreg_sequence.mp4 in a 900 x 900 m window of Barro '
            f'Colorado Island (island-grid rows {w["window"][0]}-{w["window"][1]}, columns '
            f'{w["window"][2]}-{w["window"][3]}), chosen as the window where the local-warp and translation '
            f'models disagree most on average over the scenes ({w["mean_local_minus_translation_px"]:.2f} px), '
            f'shown as delivered by Planet (left), after each scene\'s translation (center) and after its '
            f'{MODEL_NAMES[kind]} (right), all as true color MODIS-calibrated reflectance on the fixed stretch. '
            f'The yellow lines are fixed map positions every 150 m, so canopy features should stay still '
            f'relative to them when alignment is good. The title gives the date, Dove generation, satellite, '
            f'island-clear fraction and the median tile residual after translation and after the chosen model '
            f'in meters. 3 frames per second.')
        write_caption(outdir, cmp_, caption, ['Window chosen by comparison_window() from the fitted models.'],
                      made_by='video', exts='mp4')


@cli.command()
@results_option
@click.option('--outdir', type=click.Path(path_type=Path), default=DEFAULT_OUT, show_default=True)
@click.option('--dpi', default=200, show_default=True)
@click.option('--only', default=None, help='Comma-separated subset: shifts,residuals,map,view,terrain,worst,master,captions')
@click.option('--model', 'kind', type=click.Choice(MODELS), default='L', show_default=True)
def figures(results, outdir, dpi, only, kind):
    """Figures (PNG + PDF + caption) for the update."""
    import warnings
    warnings.filterwarnings('ignore')
    outdir.mkdir(parents=True, exist_ok=True)
    sel = set(only.split(',')) if only else None
    sm, tr, meta = load_tables(results)

    def want(k):
        return sel is None or k in sel
    if want('shifts'):
        fig_global_shifts(sm, outdir, dpi)
    if want('residuals'):
        fig_residuals(sm, tr, outdir, dpi)
    if want('map'):
        fig_residual_map(sm, tr, outdir, dpi)
    if want('view'):
        fig_view_geometry(sm, outdir, dpi)
    if want('terrain'):
        fig_terrain(sm, tr, outdir, dpi)
    if want('worst'):
        fig_worst_cases(sm, tr, results, outdir, dpi)
    if want('master'):
        fig_master(results, outdir, dpi)
    if want('captions'):
        write_video_captions(results, outdir, kind)


if __name__ == '__main__':
    cli()
