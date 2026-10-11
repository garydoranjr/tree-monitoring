#!/usr/bin/env python
"""Is the phantom -> mavic jump in flowering calls a colour-calibration problem?

On the two days both drones flew (2024-03-06, 2024-03-18) the drone SegFormer
flowering model calls ~5x more crowns flowering on mavic than on phantom. The
mavic mosaics are uint16 and reach the model through the fixed per-band
gain/offset in config/crown_classification_mavic.yml, which was fitted by
matching p2/p98 marginals pooled over *different* dates of each camera. This
script checks that mapping on the same scene and asks whether re-normalizing
mavic to phantom removes the disagreement.

Subcommands write into OUTDIR (default
/Volumes/Earth03/flower/figs/202610_updates/radiometry):

  distributions  same-day RGB distributions of phantom and mavic in the uint8
                 space the model sees: per-band quantiles and CDF distances
                 (against the phantom/mavic day-to-day spread), paired 1 m
                 block regressions, chromaticity, and per-crown colour shifts
  fit-renorm     candidate mavic uint16 -> uint8 transforms fitted to the
                 same-day phantom (per-band linear, histogram match, 3x3
                 colour matrix), fitted cross-date and in-sample
  reclassify     re-runs the flowering/deciduous models on the crowns where
                 the cameras disagree (plus controls) under each transform,
                 and phantom made to look like mavic (the reverse test)

Typical usage:
    python scripts/compare_drone_radiometry.py distributions
    python scripts/compare_drone_radiometry.py fit-renorm
    python scripts/compare_drone_radiometry.py reclassify
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_update_figures as pu  # noqa: E402

DEFAULT_OUT = pu.DEFAULT_OUT / 'radiometry'
BANDS = ('R', 'G', 'B')
BAND_COLORS = ('#d62728', '#2ca02c', '#1f77b4')
# Day-to-day reference pairs: same camera, adjacent flights. The C3KW2X
# phantom flights are roughly weekly; the mavic pair is the two same dates.
PHANTOM_DAY_PAIRS = (('2024_02_21', '2024_02_28'), ('2024_02_28', '2024_03_06'),
                     ('2024_03_06', '2024_03_18'))
SAMPLE_CACHE_N = 2_000_000


# --------------------------------------------------------------------------
# mosaics and grids
# --------------------------------------------------------------------------
def phantom_path(d, alignment='global'):
    return pu.phantom_dir('C3KW2X', alignment) / f'BCI_50ha_{d}_{alignment}.tif'


def mavic_path(d):
    return pu.MAVIC_RGB / f'BCI_50ha_{d}_M3M_aligned_global_RGB.tif'


def grid_bounds(crowns, pad=5.0):
    """Crown-map extent snapped outward to whole metres."""
    b = crowns.total_bounds
    return (np.floor(b[0] - pad), np.floor(b[1] - pad),
            np.ceil(b[2] + pad), np.ceil(b[3] + pad))


def read_on_grid(path, bounds, res, nodata=0):
    """Area-averaged RGB on a res-metre grid over `bounds`.

    Returns (float32 array (3, H, W) in source DN, valid mask, transform).
    Pixels where every RGB band equals `nodata` are excluded from the average
    (mavic declares nodata=0; the phantom global mosaics leave it unset but use
    0 outside the flight). A target cell is valid only if it is fully covered.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    W = int(round((bounds[2] - bounds[0]) / res))
    H = int(round((bounds[3] - bounds[1]) / res))
    t = from_origin(bounds[0], bounds[3], res, res)
    with rasterio.open(path) as src:
        with WarpedVRT(src, crs=src.crs, transform=t, width=W, height=H,
                       resampling=Resampling.average, src_nodata=nodata,
                       nodata=nodata) as vrt:
            arr = vrt.read([1, 2, 3], out_dtype='float32')
            # Coverage: fraction of each cell with data, from a 0/1 mask.
            mask = vrt.read_masks(1)
    valid = (mask == 255) & (arr.sum(axis=0) > 0)
    return arr, valid, t


def read_native_sample(path, bounds, step):
    """Every `step`-th native pixel (nearest, no averaging) inside `bounds`.

    Returns (uint (3, h, w) array, transform). Subsampling with nearest keeps
    the native-resolution value distribution, which area averaging would not.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.windows import from_bounds
    with rasterio.open(path) as src:
        win = from_bounds(*bounds, transform=src.transform)
        win = win.round_offsets().round_lengths().intersection(
            rasterio.windows.Window(0, 0, src.width, src.height))
        shape = (3, int(win.height // step), int(win.width // step))
        arr = src.read([1, 2, 3], window=win, out_shape=shape,
                       resampling=Resampling.nearest)
        t = src.window_transform(win) * Affine.scale(win.width / shape[2],
                                                     win.height / shape[1])
    return arr, t


def sample_on_mask(arr, t, mask, mask_t):
    """Pixels of arr (3, h, w) whose centres fall on True cells of mask."""
    h, w = arr.shape[1:]
    cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    x, y = t * (cols, rows)
    mc, mr = ~mask_t * (x, y)
    mr, mc = np.floor(mr).astype(int), np.floor(mc).astype(int)
    ok = (mr >= 0) & (mr < mask.shape[0]) & (mc >= 0) & (mc < mask.shape[1])
    sel = np.zeros((h, w), bool)
    sel[ok] = mask[mr[ok], mc[ok]]
    px = arr[:, sel].T
    return px[px.sum(axis=1) > 0]


def to_uint8(px, scaling):
    """Apply crown_classification.to_uint8 to an (N, 3) pixel array."""
    import crown_classification as cc
    return cc.to_uint8(px.T[:, :, None], scaling)[:, :, 0].T


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def cdf(v):
    h = np.bincount(v.astype(np.int64), minlength=256)[:256]
    return np.cumsum(h) / max(h.sum(), 1)


def cdf_distance(a, b):
    """Per-band mean |CDF difference| over 0..255 (the config's metric)."""
    return np.array([np.abs(cdf(a[:, k]) - cdf(b[:, k])).mean() for k in range(3)])


def chroma(px):
    s = px.astype(np.float64).sum(axis=1, keepdims=True)
    return px / np.maximum(s, 1)


def band_stats(px):
    q = np.percentile(px, [2, 25, 50, 75, 98], axis=0)
    return {f'{b}_{name}': q[i, k] for k, b in enumerate(BANDS)
            for i, name in enumerate(('p2', 'p25', 'p50', 'p75', 'p98'))} | \
           {f'{b}_mean': px[:, k].mean() for k, b in enumerate(BANDS)} | \
           {f'{b}_std': px[:, k].std() for k, b in enumerate(BANDS)} | \
           {f'{c}_chroma': chroma(px)[:, k].mean() for k, c in enumerate('rgb')}


def r2(y, yhat):
    return 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def fit_paired(mav, pha, rng, n_theilsen=20000):
    """Fits of phantom uint8 on mavic uint8, per 1 m block.

    Returns per-band OLS and Theil-Sen (gain, offset, R^2) and a 3x3 colour
    matrix + offset with its per-band R^2.
    """
    from scipy.stats import theilslopes
    out = {}
    sub = rng.choice(len(mav), min(n_theilsen, len(mav)), replace=False)
    for k, b in enumerate(BANDS):
        x, y = mav[:, k], pha[:, k]
        a, c = np.polyfit(x, y, 1)
        ts = theilslopes(y[sub], x[sub])
        out |= {f'{b}_ols_gain': a, f'{b}_ols_offset': c, f'{b}_ols_r2': r2(y, a * x + c),
                f'{b}_ts_gain': ts.slope, f'{b}_ts_offset': ts.intercept,
                f'{b}_ts_r2': r2(y, ts.slope * x + ts.intercept),
                f'{b}_identity_rmse': np.sqrt(((y - x) ** 2).mean()),
                f'{b}_ols_rmse': np.sqrt(((y - (a * x + c)) ** 2).mean())}
    X = np.c_[mav, np.ones(len(mav))]
    M, *_ = np.linalg.lstsq(X, pha, rcond=None)
    pred = X @ M
    for k, b in enumerate(BANDS):
        out[f'{b}_matrix_r2'] = r2(pha[:, k], pred[:, k])
        out[f'{b}_matrix_rmse'] = np.sqrt(((pha[:, k] - pred[:, k]) ** 2).mean())
    out['matrix'] = M.T.tolist()  # rows: R, G, B outputs; cols: R, G, B, 1
    return out


@click.group()
def cli():
    pass


out_option = click.option('--outdir', type=click.Path(file_okay=False, path_type=Path),
                          default=DEFAULT_OUT, show_default=True)


# --------------------------------------------------------------------------
# distributions
# --------------------------------------------------------------------------
class SameDay:
    """Phantom and mavic RGB for one date on a shared 1 m grid and as samples."""

    def __init__(self, d, crowns, bounds, res, step, scaling):
        import crown_classification  # noqa: F401  (fail early if transformers is missing)
        from rasterio.features import rasterize
        self.d = d
        self.pha_g, v_p, self.t = read_on_grid(phantom_path(d), bounds, res)
        mav16, v_m, _ = read_on_grid(mavic_path(d), bounds, res)
        # Averaging before the linear map equals mapping before averaging
        # except where the clip at 0/255 bites, which the block means rarely do.
        g, o = scaling
        self.mav_g = np.clip(mav16 * g[:, None, None] + o[:, None, None], 0, 255)
        hull = rasterize([(crowns.union_all().convex_hull, 1)], out_shape=v_p.shape,
                         transform=self.t, fill=0).astype(bool)
        self.ids = rasterize(((geom, i + 1) for i, geom in enumerate(crowns.geometry)),
                             out_shape=v_p.shape, transform=self.t, fill=0)
        self.valid = v_p & v_m & hull
        # Native-resolution samples inside the joint footprint, all and crowns.
        crown_cells = self.valid & (self.ids > 0)
        a, ta = read_native_sample(phantom_path(d), bounds, step)
        self.pha_px = sample_on_mask(a, ta, self.valid, self.t)
        self.pha_crown_px = sample_on_mask(a, ta, crown_cells, self.t)
        a, ta = read_native_sample(mavic_path(d), bounds, step)
        self.mav16_px = sample_on_mask(a, ta, self.valid, self.t)
        self.mav_px = to_uint8(self.mav16_px, scaling)
        self.mav_crown_px = to_uint8(sample_on_mask(a, ta, crown_cells, self.t), scaling)

    def blocks(self):
        """(mavic, phantom) uint8-scale block means, (N, 3) each."""
        return self.mav_g[:, self.valid].T, self.pha_g[:, self.valid].T

    def crown_means(self, n):
        """Per-crown mean RGB of each camera on the 1 m grid, (n, 3) each."""
        ids = np.where(self.valid, self.ids, 0).ravel()
        cnt = np.bincount(ids, minlength=n + 1)[1:]
        out = []
        for arr in (self.mav_g, self.pha_g):
            m = np.stack([np.bincount(ids, weights=arr[k].ravel(), minlength=n + 1)[1:]
                          for k in range(3)], axis=1)
            out.append(np.where(cnt[:, None] >= 10, m / np.maximum(cnt[:, None], 1), np.nan))
        return out


def native_pixels(path, bounds, step, scaling, valid, t):
    a, ta = read_native_sample(path, bounds, step)
    px = sample_on_mask(a, ta, valid, t)
    return px if px.dtype == np.uint8 else to_uint8(px, scaling)


@cli.command('distributions')
@out_option
@click.option('--res', default=1.0, show_default=True,
              help='Paired-block grid spacing (m); absorbs the ~0.5 m mosaic offset.')
@click.option('--step', default=4, show_default=True,
              help='Native-resolution sample stride (1/step^2 of pixels).')
@click.option('--seed', default=0, show_default=True)
def distributions(outdir, res, step, seed):
    """Same-day phantom vs mavic RGB distributions in the model's uint8 space."""
    import geopandas as gpd
    import crown_classification as cc
    from scipy.stats import mannwhitneyu
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    crowns = gpd.read_file(pu.CROWNMAP)
    bounds = grid_bounds(crowns)
    scaling = cc.load_scaling(pu.MAVIC_SCALING)

    days = {d: SameDay(d, crowns, bounds, res, step, scaling) for d in pu.SAME_DATES}
    stats, dist, fits, crown_rows = [], [], {}, []
    for d, s in days.items():
        click.echo(f'{d}: {s.valid.sum():,} joint 1 m cells, '
                   f'{len(s.pha_px):,} phantom / {len(s.mav_px):,} mavic native samples')
        for cam, px, cpx in (('phantom', s.pha_px, s.pha_crown_px),
                             ('mavic', s.mav_px, s.mav_crown_px)):
            stats.append(dict(date=d, camera=cam, pixels='all', n=len(px), **band_stats(px)))
            stats.append(dict(date=d, camera=cam, pixels='crowns', n=len(cpx), **band_stats(cpx)))
        for pixels, a, b in (('all', s.pha_px, s.mav_px),
                             ('crowns', s.pha_crown_px, s.mav_crown_px)):
            cd = cdf_distance(a, b)
            dist.append(dict(kind='phantom vs mavic, same day', a=d, b=d, pixels=pixels,
                             **dict(zip(BANDS, cd)), mean=cd.mean()))
        mav, pha = s.blocks()
        fits[d] = fit_paired(mav, pha, rng)

        # Per-crown colour shift, split by how the classifiers disagree.
        n = len(crowns)
        cm, cp = s.crown_means(n)
        cls = pd.read_csv(pu.DEFAULT_OUT / 'same_date_crowns.csv')
        w = cls[cls.date == d.replace('_', '-')].pivot_table(
            index='crown', columns='mosaic', values='p_flower')
        for i in range(n):
            if np.isnan(cm[i]).any() or i not in w.index:
                continue
            fm, fp = w.loc[i, 'mavic'], w.loc[i, 'phantom local']
            case = ('mavic only' if fm > 0.5 and fp < 0.2 else
                    'both' if fm > 0.5 and fp > 0.5 else
                    'phantom only' if fp > 0.5 and fm < 0.3 else
                    'neither' if fm < 0.5 and fp < 0.5 else 'other')
            rec = dict(date=d, crown=i, case=case, p_flower_mavic=fm, p_flower_phantom=fp)
            for k, b in enumerate(BANDS):
                rec[f'{b}_mavic'], rec[f'{b}_phantom'] = cm[i, k], cp[i, k]
                rec[f'd{b}'] = cm[i, k] - cp[i, k]
            chm, chp = chroma(cm[i:i + 1])[0], chroma(cp[i:i + 1])[0]
            for k, c in enumerate('rgb'):
                rec[f'd{c}_chroma'] = chm[k] - chp[k]
            crown_rows.append(rec)

    # Day-to-day spread within each camera, same joint footprint as 2024-03-06.
    ref = days[pu.SAME_DATES[0]]
    for a, b in PHANTOM_DAY_PAIRS:
        pa = native_pixels(phantom_path(a), bounds, step, None, ref.valid, ref.t)
        pb = native_pixels(phantom_path(b), bounds, step, None, ref.valid, ref.t)
        cd = cdf_distance(pa, pb)
        dist.append(dict(kind='phantom day-to-day', a=a, b=b, pixels='all',
                         **dict(zip(BANDS, cd)), mean=cd.mean()))
    d0, d1 = pu.SAME_DATES
    cd = cdf_distance(days[d0].mav_px, days[d1].mav_px)
    dist.append(dict(kind='mavic day-to-day', a=d0, b=d1, pixels='all',
                     **dict(zip(BANDS, cd)), mean=cd.mean()))
    for d in pu.SAME_DATES:
        pl = native_pixels(phantom_path(d, 'local'), bounds, step, None, ref.valid, ref.t)
        cd = cdf_distance(days[d].pha_px, pl)
        dist.append(dict(kind='phantom global vs local (sanity)', a=d, b=d, pixels='all',
                         **dict(zip(BANDS, cd)), mean=cd.mean()))

    # Native-pixel samples for fit-renorm, so it need not re-read the mosaics.
    for d, s in days.items():
        k = min(SAMPLE_CACHE_N, len(s.pha_px), len(s.mav16_px))
        np.savez_compressed(
            outdir / f'samples_{d}.npz',
            phantom=s.pha_px[rng.choice(len(s.pha_px), k, replace=False)],
            mavic16=s.mav16_px[rng.choice(len(s.mav16_px), k, replace=False)])

    stats = pd.DataFrame(stats)
    dist = pd.DataFrame(dist)
    crown_df = pd.DataFrame(crown_rows)
    stats.to_csv(outdir / 'radiometry_stats.csv', index=False)
    dist.to_csv(outdir / 'radiometry_cdf_distances.csv', index=False)
    crown_df.to_csv(outdir / 'radiometry_crowns.csv', index=False)
    (outdir / 'radiometry_paired_fits.json').write_text(json.dumps(fits, indent=1))

    pd.set_option('display.width', 200)
    click.echo('\nPer-band quantiles (uint8, as the model sees them):')
    cols = ['date', 'camera', 'pixels'] + [f'{b}_{q}' for b in BANDS for q in ('p2', 'p50', 'p98')] \
        + ['r_chroma', 'g_chroma', 'b_chroma']
    click.echo(stats[cols].round(3).to_string(index=False))
    click.echo('\nMean |CDF difference|:')
    click.echo(dist.round(3).to_string(index=False))
    click.echo('\nPaired 1 m block fits, phantom ~ mavic:')
    for d, f in fits.items():
        for b in BANDS:
            click.echo(f'  {d} {b}: OLS gain {f[f"{b}_ols_gain"]:.3f} offset {f[f"{b}_ols_offset"]:6.1f} '
                       f'R2 {f[f"{b}_ols_r2"]:.3f} | Theil-Sen gain {f[f"{b}_ts_gain"]:.3f} '
                       f'offset {f[f"{b}_ts_offset"]:6.1f} | 3x3 R2 {f[f"{b}_matrix_r2"]:.3f} | '
                       f'RMSE identity {f[f"{b}_identity_rmse"]:5.1f} OLS {f[f"{b}_ols_rmse"]:5.1f} '
                       f'3x3 {f[f"{b}_matrix_rmse"]:5.1f}')
    click.echo('\nPer-crown mavic - phantom colour shift by classifier case (median):')
    g = crown_df.groupby(['date', 'case'])
    summ = g[['dR', 'dG', 'dB', 'dr_chroma', 'dg_chroma', 'db_chroma']].median()
    summ['n'] = g.size()
    click.echo(summ.round(4).to_string())
    for col in ('dr_chroma', 'dg_chroma', 'dB'):
        a = crown_df.loc[crown_df.case == 'mavic only', col]
        b = crown_df.loc[crown_df.case == 'neither', col]
        click.echo(f'  {col}: mavic-only vs neither, Mann-Whitney p = '
                   f'{mannwhitneyu(a, b).pvalue:.3g}')

    plot_cdfs(days, outdir)
    plot_hexbin(days, fits, outdir)
    plot_chroma(days, crown_df, outdir)


def plot_cdfs(days, outdir):
    fig, axes = plt.subplots(len(days), 3, figsize=(13, 3.6 * len(days)), squeeze=False)
    x = np.arange(256)
    for r, (d, s) in enumerate(days.items()):
        for k, b in enumerate(BANDS):
            ax = axes[r, k]
            ax.plot(x, cdf(s.pha_px[:, k]), color=pu.C_PHANTOM_EXT, label='phantom (global)')
            ax.plot(x, cdf(s.mav_px[:, k]), color=pu.C_MAVIC, label='mavic (scaled to uint8)')
            ax.plot(x, cdf(s.pha_crown_px[:, k]), color=pu.C_PHANTOM_EXT, ls=':', lw=1,
                    label='phantom, crown pixels')
            ax.plot(x, cdf(s.mav_crown_px[:, k]), color=pu.C_MAVIC, ls=':', lw=1,
                    label='mavic, crown pixels')
            dist = np.abs(cdf(s.pha_px[:, k]) - cdf(s.mav_px[:, k])).mean()
            ax.set_title(f'{d.replace("_", "-")} {b}: mean |dCDF| = {dist:.3f}', fontsize=10)
            ax.set_xlim(0, 255)
            ax.set_xlabel(f'{b} (uint8 model input)')
            ax.set_ylabel('cumulative fraction')
    axes[0, 0].legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    pu.save(fig, outdir, 'rgb_cdfs')


def plot_hexbin(days, fits, outdir):
    fig, axes = plt.subplots(len(days), 3, figsize=(13, 4.2 * len(days)), squeeze=False)
    for r, (d, s) in enumerate(days.items()):
        mav, pha = s.blocks()
        f = fits[d]
        for k, b in enumerate(BANDS):
            ax = axes[r, k]
            ax.hexbin(mav[:, k], pha[:, k], gridsize=80, bins='log', cmap='Greys',
                      extent=(0, 255, 0, 255), mincnt=1)
            x = np.array([0, 255])
            ax.plot(x, x, color=pu.C_GREY, lw=0.8, label='1:1')
            ax.plot(x, f[f'{b}_ols_gain'] * x + f[f'{b}_ols_offset'], color=BAND_COLORS[k],
                    label=f'OLS: {f[f"{b}_ols_gain"]:.2f}x {f[f"{b}_ols_offset"]:+.1f}, '
                          f'R$^2$={f[f"{b}_ols_r2"]:.2f}')
            ax.set_xlim(0, 255)
            ax.set_ylim(0, 255)
            ax.set_aspect('equal')
            ax.set_xlabel(f'mavic {b} (uint8, 1 m block mean)')
            ax.set_ylabel(f'phantom {b} (uint8, 1 m block mean)')
            ax.set_title(d.replace('_', '-'), fontsize=10)
            ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    pu.save(fig, outdir, 'rgb_paired_hexbin')


def plot_chroma(days, crown_df, outdir):
    fig, axes = plt.subplots(1, len(days) + 1, figsize=(5 * (len(days) + 1), 4.6))
    for ax, (d, s) in zip(axes, days.items()):
        for px, color, label in ((s.pha_px, pu.C_PHANTOM_EXT, 'phantom'),
                                 (s.mav_px, pu.C_MAVIC, 'mavic')):
            c = chroma(px)
            h, xe, ye = np.histogram2d(c[:, 0], c[:, 1], bins=120,
                                       range=((0.15, 0.55), (0.25, 0.55)))
            ax.contour((xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2, h.T / h.sum(),
                       levels=np.quantile(h[h > 0] / h.sum(), [0.5, 0.9, 0.99]),
                       colors=color, linewidths=(0.6, 1.0, 1.4))
            ax.plot([], [], color=color, label=label)
        ax.set_xlabel('r = R / (R+G+B)')
        ax.set_ylabel('g = G / (R+G+B)')
        ax.set_title(d.replace('_', '-'), fontsize=10)
        ax.legend(fontsize=8)
    ax = axes[-1]
    for case, color in (('neither', '#bbbbbb'), ('both', '#000000'),
                        ('mavic only', pu.C_MAVIC), ('phantom only', pu.C_PHANTOM_EXT)):
        sub = crown_df[crown_df.case == case]
        ax.scatter(sub.dr_chroma, sub.dg_chroma, s=6 if case == 'neither' else 18,
                   color=color, alpha=0.5 if case == 'neither' else 0.9, lw=0,
                   label=f'{case} (n={len(sub)})')
    ax.axhline(0, color=pu.C_GREY, lw=0.6)
    ax.axvline(0, color=pu.C_GREY, lw=0.6)
    ax.set_xlabel('crown mean r, mavic - phantom')
    ax.set_ylabel('crown mean g, mavic - phantom')
    ax.set_title('Per-crown chromaticity shift by classifier case', fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    pu.save(fig, outdir, 'chromaticity')


# --------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------
KINDS = ('linear', 'histmatch', 'mkl')


def baseline_transform():
    import crown_classification as cc
    g, o = cc.load_scaling(pu.MAVIC_SCALING)
    return dict(kind='linear', gain=g.tolist(), offset=o.tolist())


def _sqrtm(C):
    w, V = np.linalg.eigh(C)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def fit_transform(kind, src, dst):
    """Fit a map taking the distribution of src pixels (N, 3) onto dst (N, 3).

    linear     per-band gain/offset matching mean and std
    histmatch  per-band quantile mapping (1001 quantiles)
    mkl        linear Monge-Kantorovich colour transfer (Pitie & Kokaram 2007):
               matches the mean and full 3x3 covariance, so it can rotate hue,
               which no per-band map can
    These are distribution matches on the same footprint, not paired
    regressions, so they are not attenuated by misregistration or blur.
    """
    src, dst = src.astype(np.float64), dst.astype(np.float64)
    if kind == 'linear':
        g = dst.std(axis=0) / src.std(axis=0)
        return dict(kind=kind, gain=g.tolist(),
                    offset=(dst.mean(axis=0) - g * src.mean(axis=0)).tolist())
    if kind == 'histmatch':
        q = np.linspace(0, 1, 1001)
        sq, dq = np.quantile(src, q, axis=0).T, np.quantile(dst, q, axis=0).T
        # np.interp needs strictly increasing knots; uint8 sources repeat
        # quantiles, so average the targets over each run of ties.
        knots = []
        for s, t in zip(sq, dq):
            u, inv = np.unique(s, return_inverse=True)
            knots.append((u.tolist(), (np.bincount(inv, t) / np.bincount(inv)).tolist()))
        return dict(kind=kind, src_q=[k[0] for k in knots], dst_q=[k[1] for k in knots])
    if kind == 'mkl':
        Cs, Cd = np.cov(src.T), np.cov(dst.T)
        Rs = _sqrtm(Cs)
        Rs_inv = np.linalg.inv(Rs)
        M = Rs_inv @ _sqrtm(Rs @ Cd @ Rs) @ Rs_inv
        return dict(kind=kind, matrix=M.tolist(), src_mean=src.mean(axis=0).tolist(),
                    dst_mean=dst.mean(axis=0).tolist())
    raise ValueError(kind)


def apply_transform(tf, data):
    """Apply a transform to (3, ...) DN and return uint8 as the model sees it."""
    x = data.astype(np.float64)
    if tf['kind'] == 'identity':
        y = x
    elif tf['kind'] == 'linear':
        g, o = np.array(tf['gain']), np.array(tf['offset'])
        y = x * g.reshape(3, *[1] * (x.ndim - 1)) + o.reshape(3, *[1] * (x.ndim - 1))
    elif tf['kind'] == 'histmatch':
        y = np.stack([np.interp(x[k], tf['src_q'][k], tf['dst_q'][k]) for k in range(3)])
    elif tf['kind'] == 'mkl':
        flat = x.reshape(3, -1)
        mu_s = np.array(tf['src_mean'])[:, None]
        mu_d = np.array(tf['dst_mean'])[:, None]
        y = (np.array(tf['matrix']) @ (flat - mu_s) + mu_d).reshape(x.shape)
    else:
        raise ValueError(tf['kind'])
    return np.clip(np.rint(y), 0, 255).astype(np.uint8)


def load_samples(outdir, d):
    z = np.load(outdir / f'samples_{d}.npz')
    return z['phantom'], z['mavic16']


@cli.command('fit-renorm')
@out_option
def fit_renorm(outdir):
    """Fit mavic -> phantom (and reverse) colour transforms on the same-day samples.

    Reads OUTDIR/samples_<date>.npz (from `distributions`). Writes
    transforms.json, mavic_linear_<date>.yml (a drop-in --scaling-config for
    crown_classification.py), renorm_cdf_distances.csv, which scores each
    transform in-sample and on the other date, and
    crown_classification_mavic_mkl.yml, the production colour transfer fitted
    on both dates pooled (copied to config/).
    """
    import yaml
    base = baseline_transform()
    samples = {d: load_samples(outdir, d) for d in pu.SAME_DATES}
    tfs = {}
    for d, (pha, mav16) in samples.items():
        mav_a = apply_transform(base, mav16.T).T
        for kind in KINDS:
            tfs[f'forward:{kind}:{d}'] = fit_transform(kind, mav16, pha)
            tfs[f'reverse:{kind}:{d}'] = fit_transform(kind, pha, mav_a)
        lin = tfs[f'forward:linear:{d}']
        (outdir / f'mavic_linear_{d}.yml').write_text(
            f'# Per-band mean/std match of mavic uint16 to the same-day phantom\n'
            f'# (C3KW2X global) uint8 mosaic on {d}, from compare_drone_radiometry.py.\n'
            + yaml.safe_dump({'uint16_to_uint8': [dict(gain=float(g), offset=float(o))
                                                  for g, o in zip(lin['gain'], lin['offset'])]}))
    # Production transform: one fixed colour transfer fitted on both dates.
    pooled = fit_transform('mkl', np.concatenate([m for _, m in samples.values()]),
                           np.concatenate([p for p, _ in samples.values()]))
    tfs['forward:mkl:pooled'] = pooled
    (outdir / 'crown_classification_mavic_mkl.yml').write_text(mkl_config_yaml(pooled))
    (outdir / 'transforms.json').write_text(json.dumps(tfs))

    rows = []
    for d, (pha, mav16) in samples.items():
        mav_a = apply_transform(base, mav16.T).T
        rows.append(dict(direction='forward', kind='baseline', fit='-', date=d,
                         **dict(zip(BANDS, cdf_distance(pha, mav_a)))))
        for fd in pu.SAME_DATES:
            fit = 'insample' if fd == d else 'cross'
            for kind in KINDS:
                out = apply_transform(tfs[f'forward:{kind}:{fd}'], mav16.T).T
                rows.append(dict(direction='forward', kind=kind, fit=fit, date=d,
                                 **dict(zip(BANDS, cdf_distance(pha, out)))))
                out = apply_transform(tfs[f'reverse:{kind}:{fd}'], pha.T).T
                rows.append(dict(direction='reverse', kind=kind, fit=fit, date=d,
                                 **dict(zip(BANDS, cdf_distance(mav_a, out)))))
        out = apply_transform(pooled, mav16.T).T
        rows.append(dict(direction='forward', kind='mkl', fit='pooled', date=d,
                         **dict(zip(BANDS, cdf_distance(pha, out)))))
    df = pd.DataFrame(rows)
    df['mean'] = df[list(BANDS)].mean(axis=1)
    df.to_csv(outdir / 'renorm_cdf_distances.csv', index=False)
    click.echo('Mean |CDF difference| to the target camera after each transform:')
    click.echo(df.round(3).to_string(index=False))
    for d in pu.SAME_DATES:
        click.echo(f'\n{d} transforms:')
        for k in ('linear', 'mkl'):
            tf = tfs[f'forward:{k}:{d}']
            click.echo(f'  {k}: ' + json.dumps({a: np.round(b, 5).tolist()
                                               for a, b in tf.items() if a != 'kind'}))


def mkl_config_yaml(tf):
    """crown_classification.py --scaling-config text for a pooled mkl transform."""
    import yaml
    M = np.array(tf['matrix'])
    body = yaml.safe_dump({'color_transfer': {
        'matrix': [[float(f'{v:.6g}') for v in row] for row in M],
        'src_mean': [round(v, 2) for v in tf['src_mean']],
        'dst_mean': [round(v, 3) for v in tf['dst_mean']],
    }}, default_flow_style=None, sort_keys=False)
    dates = ' and '.join(d.replace('_', '-') for d in pu.SAME_DATES)
    return f"""\
# Colour transfer for the mavic 50ha RGB mosaics
# (/scratch/tree-monitoring/stri/globus/RGB): uint16 DN -> uint8 model input.
#
# Written by `python scripts/compare_drone_radiometry.py fit-renorm`; see
# docs/drone_radiometry.md for the analysis.
#
# The per-band gains/offsets in crown_classification_mavic.yml match each
# band's p2/p98 to the phantom training mosaics, but on the days both drones
# flew ({dates}) mavic still reaches the model greener, with a darker
# blue band, more contrast and more saturated colour than phantom. That made
# the flowering model call ~5x more crowns flowering on mavic.
#
# This file replaces the per-band map with a linear Monge-Kantorovich colour
# transfer (Pitie & Kokaram 2007). The transfer maps the mean and full 3x3
# covariance of mavic RGB onto those of the same-day phantom (C3KW2X global)
# mosaics. It is fitted on native-resolution pixels sampled over the two
# drones' joint footprint inside the 50ha crown map, pooled over both dates.
# Fitted on one date and applied to the other, the transfer removed 33 of the
# 42 mavic-only flowering crown-dates and kept 13 of the 16 that flower on both
# cameras, without creating positives among 60 control crowns.
#
# Like the per-band map, the transform is FIXED rather than refitted per image,
# so the mavic time series stays mutually comparable.
#
# Applied as: uint8 = clip(round(matrix @ (DN - src_mean) + dst_mean), 0, 255)
# per pixel, with DN the (R, G, B) uint16 vector. Rows of `matrix` give the
# R, G, B outputs. Band 4 (alpha) is ignored, as before.
{body}"""


# --------------------------------------------------------------------------
# reclassify
# --------------------------------------------------------------------------
def select_crowns(n_controls, seed):
    """Crown-dates to re-run, labelled by how the archived classifiers disagree.

    Cases are defined against phantom local, as in plot_update_figures.py
    `same-date-examples`.
    """
    df = pd.read_csv(pu.DEFAULT_OUT / 'same_date_crowns.csv')
    w = df.pivot_table(index=['date', 'crown'], columns='mosaic',
                       values=['p_flower', 'p_decid']).dropna()
    f, dc = w['p_flower'], w['p_decid']
    cases = {
        'mavic only': (f['mavic'] > 0.5) & (f['phantom local'] < 0.2),
        'both': (f['mavic'] > 0.5) & (f['phantom local'] > 0.5),
        'phantom only': (f['phantom local'] > 0.5) & (f['mavic'] < 0.3),
        'deciduous, phantom only': (dc['phantom local'] > 0.5) & (dc['mavic'] < 0.3),
        'deciduous, mavic only': (dc['mavic'] > 0.5) & (dc['phantom local'] < 0.3),
    }
    rows = [dict(date=d, crown=c, case=case) for case, m in cases.items()
            for d, c in m[m].index]
    neither = w[(f['mavic'] < 0.1) & (f['phantom local'] < 0.1)
                & (dc['mavic'] < 0.5) & (dc['phantom local'] < 0.5)]
    ctrl = neither.sample(n=min(n_controls, len(neither)), random_state=seed)
    rows += [dict(date=d, crown=c, case='control') for d, c in ctrl.index]
    out = pd.DataFrame(rows)
    arch = w.reset_index()
    arch.columns = ['_'.join(c).strip('_').replace(' ', '_') for c in arch.columns]
    return out.merge(arch, on=['date', 'crown'], how='left')


def run_models(models, imgs, device):
    """Masked-mean-ready probability maps for a list of PIL images.

    Same as crown_classification.preprocess + apply_model, per image.
    """
    import torch
    import torch.nn.functional as F
    import crown_classification as cc
    out = {}
    for key, model in models.items():
        confs = []
        for img in imgs:
            x = cc.preprocess(img).to(device)
            with torch.no_grad():
                logits = model(x).logits
            logits = F.interpolate(logits, size=img.size[::-1], mode='bilinear',
                                   align_corners=False)
            confs.append(torch.sigmoid(logits.squeeze()[1]).cpu().numpy())
        out[key] = confs
    return out


@cli.command('reclassify')
@out_option
@click.option('--flower-model', type=click.Path(exists=True, dir_okay=False),
              default=str(pu.FLOWER / 'results/models/drone_flower_geo_models/epoch_020.pth'),
              show_default=True)
@click.option('--decid-model', type=click.Path(exists=True, dir_okay=False),
              default=str(pu.FLOWER / 'results/models/drone_decid_geo_models/epoch_020.pth'),
              show_default=True)
@click.option('--n-controls', default=60, show_default=True,
              help='Crown-dates where both cameras agree on not flowering.')
@click.option('--device', default='cpu', show_default=True,
              help='cpu reproduces the production run; mps is faster.')
@click.option('--example-kind', default='mkl', show_default=True,
              type=click.Choice(KINDS), help='Transform shown in the example figure.')
@click.option('--n-examples', default=8, show_default=True)
@click.option('--seed', default=0, show_default=True)
def reclassify(outdir, flower_model, decid_model, n_controls, device, example_kind,
               n_examples, seed):
    """Re-run the drone models on disagreeing crowns under each colour transform.

    Reads OUTDIR/transforms.json (from `fit-renorm`). For every selected
    crown-date the production window (crown_classification.centered_window) is
    read from the mavic mosaic and passed through the baseline scaling and each
    forward transform, and from the phantom local mosaic as is and through each
    reverse transform (phantom made to look like mavic). Writes
    reclassify_crowns.csv, reclassify_summary.csv and reclassify_examples.png/.pdf.
    """
    import geopandas as gpd
    import rasterio
    import torch
    from PIL import Image
    from scipy.stats import spearmanr
    import crown_classification as cc

    tfs = json.loads((outdir / 'transforms.json').read_text())
    base = baseline_transform()
    ident = dict(kind='identity')
    models = {}
    for key, path in (('flower', flower_model), ('decid', decid_model)):
        m = torch.load(path, weights_only=False, map_location=torch.device('cpu'))
        models[key] = m.eval().to(device)
    crowns = gpd.read_file(pu.CROWNMAP)
    species = pu.crown_species()
    sel = select_crowns(n_controls, seed)
    click.echo(sel.groupby('case').size().to_string())
    examples = set(map(tuple, sel[sel.case == 'mavic only']
                       .sort_values('p_flower_mavic', ascending=False)
                       .drop_duplicates('crown')[['date', 'crown']].values[:n_examples]))
    ex_data = {}

    rows = []
    for d, g in sel.groupby('date'):
        dd = d.replace('-', '_')
        other = [x for x in pu.SAME_DATES if x != dd][0]
        variants = {'mavic': [('baseline', '-', base)], 'phantom': [('as is', '-', ident)]}
        for kind in KINDS:
            for fit, fd in (('cross', other), ('insample', dd)):
                variants['mavic'].append((kind, fit, tfs[f'forward:{kind}:{fd}']))
                variants['phantom'].append((f'reverse {kind}', fit, tfs[f'reverse:{kind}:{fd}']))
        paths = {'mavic': mavic_path(dd), 'phantom': pu.phantom_dir('C3KW2X', 'local')
                 / f'BCI_50ha_{dd}_local.tif'}
        with rasterio.open(paths['mavic']) as s_m, rasterio.open(paths['phantom']) as s_p:
            srcs = {'mavic': s_m, 'phantom': s_p}
            for _, r in g.iterrows():
                poly = crowns.geometry.iloc[r.crown]
                for cam, src in srcs.items():
                    win = cc.centered_window(src, poly)
                    raw = src.read([1, 2, 3], window=win)
                    mask = cc.polygon_mask(src, win, poly) > 0
                    imgs = [Image.fromarray(np.moveaxis(apply_transform(tf, raw), 0, -1))
                            for _, _, tf in variants[cam]]
                    confs = run_models(models, imgs, device)
                    for i, (kind, fit, _) in enumerate(variants[cam]):
                        rows.append(dict(date=d, crown=r.crown, species=species[r.crown],
                                         case=r.case, camera=cam, variant=kind, fit=fit,
                                         p_flower=confs['flower'][i][mask].mean(),
                                         p_decid=confs['decid'][i][mask].mean()))
                    if (d, r.crown) in examples:
                        keep = [i for i, (k, f, _) in enumerate(variants[cam])
                                if k in ('baseline', 'as is')
                                or (k == example_kind and f == 'cross')]
                        ex_data[(d, r.crown, cam)] = dict(
                            transform=src.window_transform(win), mask=mask,
                            imgs=[(variants[cam][i][:2], np.asarray(imgs[i]),
                                   confs['flower'][i]) for i in keep])
            click.echo(f'{d}: {len(g)} crowns done')

    df = pd.DataFrame(rows)
    df.to_csv(outdir / 'reclassify_crowns.csv', index=False)
    summarize_reclassify(df, sel, outdir)
    plot_reclassify_examples(ex_data, crowns, species, example_kind, outdir)


def summarize_reclassify(df, sel, outdir):
    from scipy.stats import spearmanr
    key = ['date', 'crown']
    ph = df[(df.camera == 'phantom') & (df.variant == 'as is')].set_index(key)
    mv = df[(df.camera == 'mavic') & (df.variant == 'baseline')].set_index(key)
    arch = sel.set_index(key)

    # Reproduction check against the archived production crown means.
    click.echo('\nReproduction of the archived crown means (local run vs HPC mosaics):')
    for cam, local, col in (('phantom', ph, 'phantom_local'), ('mavic', mv, 'mavic')):
        for var in ('flower', 'decid'):
            a, b = local[f'p_{var}'], arch.loc[local.index, f'p_{var}_{col}']
            click.echo(f'  {cam} {var}: Spearman {spearmanr(a, b).statistic:.3f}, '
                       f'median |diff| {np.median(np.abs(a - b)):.3f}')

    rows = []
    for (cam, var, fit), g in df.groupby(['camera', 'variant', 'fit'], sort=False):
        g = g.set_index(key)
        ref = ph if cam == 'mavic' else mv
        rec = dict(camera=cam, variant=var, fit=fit)
        for case in ('mavic only', 'both', 'phantom only', 'control'):
            c = g[g.case == case]
            rec[f'{case}: flower>0.5'] = f'{(c.p_flower > 0.5).sum()}/{len(c)}'
        for case in ('deciduous, phantom only', 'deciduous, mavic only'):
            c = g[g.case == case]
            rec[f'{case}: decid>0.5'] = f'{(c.p_decid > 0.5).sum()}/{len(c)}'
        rec['rho flower vs other camera'] = spearmanr(g.p_flower, ref.loc[g.index, 'p_flower']).statistic
        rec['rho decid vs other camera'] = spearmanr(g.p_decid, ref.loc[g.index, 'p_decid']).statistic
        rows.append(rec)
    summ = pd.DataFrame(rows)
    summ.to_csv(outdir / 'reclassify_summary.csv', index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 20)
    click.echo('\nCrowns called positive under each variant (local model run):')
    click.echo(summ.round(2).to_string(index=False))


def plot_reclassify_examples(ex_data, crowns, species, example_kind, outdir):
    import geopandas as gpd
    keys = sorted({(d, c) for d, c, _ in ex_data})
    if not keys:
        return
    fig, axes = plt.subplots(len(keys), 6, figsize=(17, 2.9 * len(keys)), squeeze=False,
                             layout='constrained')
    for row, (d, c) in zip(axes, keys):
        panels = [('phantom', ex_data[(d, c, 'phantom')]['imgs'][0], ex_data[(d, c, 'phantom')]),
                  ('mavic', ex_data[(d, c, 'mavic')]['imgs'][0], ex_data[(d, c, 'mavic')]),
                  ('mavic', ex_data[(d, c, 'mavic')]['imgs'][1], ex_data[(d, c, 'mavic')])]
        geom = crowns.geometry.iloc[c]
        for k, (cam, ((var, fit), img, conf), ex) in enumerate(panels):
            ext = pu.extent(img.transpose(2, 0, 1), ex['transform'])
            a_rgb, a_p = row[2 * k], row[2 * k + 1]
            a_rgb.imshow(img, extent=ext)
            a_p.imshow(conf, cmap='magma', vmin=0, vmax=1, extent=ext, interpolation='nearest')
            p = conf[ex['mask']].mean()
            label = f'{cam}' + ('' if var in ('as is', 'baseline') else f' {var} ({fit})')
            a_rgb.set_title(f'{label} RGB', fontsize=8)
            a_p.set_title(f'{label} P(flowering) = {p:.2f}', fontsize=8)
            for a in (a_rgb, a_p):
                gpd.GeoSeries([geom]).boundary.plot(ax=a, color='cyan', lw=0.9)
                x0, y0, x1, y1 = geom.bounds
                half = max(x1 - x0, y1 - y0) / 2 + 6
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                a.set_xlim(cx - half, cx + half)
                a.set_ylim(cy - half, cy + half)
                a.set_xticks([])
                a.set_yticks([])
        row[0].set_ylabel(f'{d} crown {c}\n{species[c]}', fontsize=8)
    fig.suptitle(f'Mavic-only flowering crowns re-classified after colour transfer to the '
                 f'same-day phantom ({example_kind}, fitted on the other date)', fontsize=11)
    pu.save(fig, outdir, 'reclassify_examples')


if __name__ == '__main__':
    cli()
