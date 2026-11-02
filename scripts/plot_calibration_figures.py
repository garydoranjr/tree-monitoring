#!/usr/bin/env python
"""Figures explaining the Planet -> MODIS cross-calibration (first written for
the 2026-11-03 weekly update). See docs/planet_radiometric_calibration.md.

Each subcommand writes <name>.png and <name>.pdf into OUTDIR (default
/Volumes/Earth03/flower/figs/20261103_updates) and prints the numbers the
caption needs; the captions (<name>_caption.md) are written by hand next to
the figures.

  schematic          calibration_schematic: the five steps of the method on
                     one date-satellite group, with real data in each panel
  sr-vs-toa          sr_vs_toa: one 50ha clip as TOA and surface reflectance,
                     per-band SR vs TOA pixels, and the SR - TOA plot-mean
                     difference over all clear clips
  modis-example      modis_example: the MODIS reference for one group, the
                     Planet mosaic with its cloud and water masks, and which
                     500 m pixels the fit used
  before-after       calibration_before_after: four clips from three Dove
                     generations within 2 days, as TOA, raw SR and calibrated
                     SR on one shared stretch
  training-metrics   calibration_training_metrics: Mask R-CNN skill trained
                     on TOA vs calibrated chips (temporal CV, two seeds; and
                     the spatial east/west benchmark against epoch)
  augmentation       augmentation_colorjitter: random draws of the training
                     ColorJitter on one TOA training chip

Typical usage:
    python scripts/plot_calibration_figures.py schematic
    python scripts/plot_calibration_figures.py modis-example --group 2021-03-05_227a
    python scripts/plot_calibration_figures.py --help
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import csv
import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio as rio
import xarray as xr
from matplotlib.patches import FancyArrowPatch, Patch
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject
from rasterio.windows import from_bounds
from scipy.ndimage import binary_dilation, uniform_filter
from shapely.geometry import shape

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_planet_modis import (MODIS_SINU, SCALE, modis_grid,  # noqa: E402
                                    to_modis)
from evaluate_planet_calibration import toa_xml  # noqa: E402
from calculate_ndvi import get_coeff  # noqa: E402
from planet_box_grid import UTM, box_grid, warp_to_box  # noqa: E402
from plot_study_area_figure import CAL_EXAMPLE  # noqa: E402
from util import calibration_results, load_config  # noqa: E402

FLOWER = Path('/Volumes/Earth03/flower')
REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = FLOWER / 'figs' / '20261103_updates'
CAL = calibration_results()
STAGE = FLOWER / 'planet_bcnm_sr'
REFERENCE = FLOWER / 'modis/mcd43a1_bcnm/reference_nadir_sza45.nc'
REGION = shape(load_config(REPO / 'config/clip_50ha_plot.yml')['region'])
BANDS = ['blue', 'green', 'red', 'nir']
BAND_LABEL = dict(zip(BANDS, ['blue', 'green', 'red', 'NIR']))
RGB = [2, 1, 0]  # band indices of R, G, B in (B, G, R, NIR)

# Calibration settings of calibrate_planet_modis.py (defaults of the full run).
MIN_LAND, MIN_LAND_NBHD, MIN_CLEAR, DILATE = 0.99, 0.9, 0.95, 2

# dataviz reference categorical slots, as in evaluate_planet_calibration.py:
# TOA blue, SR orange, CAL green.
C_TOA, C_SR, C_CAL = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED, GRID = '#2b2b2b', '#8a8a85', '#e4e4e0'
C_WATER, C_CLOUD, C_USED = '#9cc3e6', '#c9c9c4', '#1baf7a'
C_MASK = '#8a5bd6'  # cloud-mask overlay, dataviz slot 4

SCHEMATIC_GROUP = '2021-03-05_227a'  # most usable pixels (165), as example_histograms.png
TEMPORAL_RUNS = {'TOA': ['202610_temporal', '202610_temporal_seed1'],
                 'calibrated': ['202610_temporal_cal', '202610_temporal_cal_seed1']}
SPATIAL_SWEEP = FLOWER / 'figs/202610_cal_updates/headtohead_sweep_202610_mkl_cal.json'
SPATIAL_RUNS = {'TOA': 'full_mkl', 'calibrated': 'full_mkl_cal'}
TOA_CHIPS = FLOWER / '20261012_full_label_extended2_mkl_x4_coreg_4band_stretch_stats_curated'
EPOCH = 12

out_option = click.option('--outdir', type=click.Path(file_okay=False, path_type=Path),
                          default=DEFAULT_OUT, show_default=True)


@click.group()
def cli():
    pass


def save(fig, outdir, name, dpi=300):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'{name}.{ext}', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    click.echo(f'Wrote {outdir / name}.png/.pdf')


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def style(ax):
    ax.tick_params(colors=INK, labelsize=8, length=3)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def stretch_rgb(a, lo, hi, idx=RGB):
    """(4, H, W) reflectance -> (H, W, 3) in [0, 1]; NaN (masked) -> NaN."""
    idx = list(idx)
    rgb = (a[idx] - lo[idx, None, None]) / (hi[idx] - lo[idx])[:, None, None]
    return np.clip(rgb, 0, 1).transpose(1, 2, 0)


def pooled_stretch(arrays, q=(1, 99)):
    stack = np.concatenate([a.reshape(4, -1) for a in arrays], axis=1)
    return np.nanpercentile(stack, q, axis=1)


def outline_region(ax, tf, **kw):
    """Draw the 50ha plot (UTM) on an image with affine `tf` (UTM, pixel coords)."""
    x, y = REGION.exterior.xy
    inv = ~tf
    cols, rows = zip(*(inv * (a, b) for a, b in zip(x, y)))
    ax.plot(np.array(cols) - 0.5, np.array(rows) - 0.5, **{'color': 'white', 'lw': 1.2, **kw})


# ---------------------------------------------------------------- group data

class Group:
    """One date-satellite group on the BCNM box: the 3 m Planet SR mosaic with
    its masks and the 500 m quantities calibrate_planet_modis.py fits on."""

    def __init__(self, key, decimate=8):
        date, sat = key.split('_', 1)
        self.date, self.sat = date, sat
        self.scenes = [r for r in csv.DictReader(open(STAGE / 'index.csv'))
                       if r['acquired'][:10] == date and r['satellite_id'] == sat]
        if not self.scenes:
            raise click.ClickException(f'no scenes for {key} in {STAGE / "index.csv"}')
        self.instrument = self.scenes[0]['instrument']
        box_tf, w, h = box_grid()
        with rio.open(STAGE / 'aux' / 'bcnm_water_mask.tif') as src:
            water = src.read(1) > 0
        # Mosaic as load_group_mosaic, but keep the cloudy reflectance for display.
        refl = np.zeros((4, h, w), np.float32)
        cloud = np.zeros((h, w), bool)
        valid = np.zeros((h, w), bool)
        have = np.zeros((h, w), bool)
        for r in self.scenes:
            sid = r['scene_id']
            sr = warp_to_box(Path(r['sr_path']), [1, 2, 3, 4], box_tf, w, h, 0, np.uint16)
            cls = warp_to_box(STAGE / 'ocm' / sid[:4] / f'{sid}_ocm.tif', [1],
                              box_tf, w, h, 255, np.uint8)[0]
            data = np.all(sr > 0, axis=0) & ~have
            cl = binary_dilation(np.isin(cls, (1, 2, 3)), iterations=DILATE)
            ok = (cls == 0) & ~cl & data & ~water & ~valid
            refl[:, data] = sr[:, data] / SCALE
            cloud |= data & cl
            valid |= ok
            have |= data
        ref = xr.open_dataset(REFERENCE)
        m_tf, m_w, m_h, m_area = modis_grid(ref)
        expected = m_area / (box_tf.a * -box_tf.e)
        land_frac = to_modis((~water).astype(np.float32), box_tf, m_tf, m_w, m_h) / expected
        self.land_ok = (land_frac >= MIN_LAND) & (
            uniform_filter(land_frac, size=3, mode='constant') >= MIN_LAND_NBHD)
        n_valid = to_modis(valid.astype(np.float32), box_tf, m_tf, m_w, m_h)
        self.clear_frac = n_valid / expected
        self.use = self.land_ok & (self.clear_frac >= MIN_CLEAR)
        with np.errstate(invalid='ignore', divide='ignore'):
            self.planet500 = np.stack([to_modis(refl[b] * valid, box_tf, m_tf, m_w, m_h)
                                       for b in range(4)]) / n_valid
        ti = int(np.searchsorted(ref['time'].values.astype('datetime64[D]'),
                                 np.datetime64(date)))
        self.modis = ref['reflectance'].isel(time=ti).sel(band=BANDS).values  # (4, y, x)
        self.fill = ref['fill_flag'].isel(time=ti).sel(band=BANDS).values
        self.m_tf, self.m_shape = m_tf, (m_h, m_w)
        fits = pd.read_csv(CAL / 'planet_modis_calibration.csv', dtype={'satellite_id': str})
        fits = fits[(fits.date == date) & (fits.satellite_id == sat)].set_index('band')
        self.offset = fits.loc[BANDS, 'offset'].to_numpy(float)
        self.n_pixels = int(fits['n_pixels'].iloc[0])
        # Display grid: the box at 3 m x decimate.
        d = decimate
        self.tf = box_tf * Affine.scale(d)
        self.shape = (h // d, w // d)
        sl = (slice(0, self.shape[0] * d), slice(0, self.shape[1] * d))
        blk = lambda a: a[sl].reshape(self.shape[0], d, self.shape[1], d)  # noqa: E731
        self.refl = np.stack([blk(refl[b]).mean(axis=(1, 3)) for b in range(4)])
        self.have = blk(have).mean(axis=(1, 3)) > 0.5
        self.cloud = blk(cloud).mean(axis=(1, 3)) > 0.5
        self.water = blk(water).mean(axis=(1, 3)) > 0.5
        self.refl[:, ~self.have] = np.nan
        self.box_tf, self.box_shape = box_tf, (h, w)

    def to_display(self, a, nodata=np.nan):
        """Nearest-neighbor warp of a MODIS-grid (y, x) array to the display grid,
        so the sinusoidal pixels appear as they lie on the ground."""
        out = np.full(self.shape, nodata, np.float32)
        reproject(a.astype(np.float32), out, src_transform=self.m_tf, src_crs=MODIS_SINU,
                  dst_transform=self.tf, dst_crs=UTM, resampling=Resampling.nearest,
                  src_nodata=None, dst_nodata=nodata)
        return out

    def to_display_rgb(self, a):
        return np.stack([self.to_display(np.where(np.isfinite(a[b]), a[b], np.nan))
                         for b in range(4)])

    def summary(self):
        return (f'{self.date} {self.instrument} {self.sat}: {len(self.scenes)} scenes, '
                f'{int(self.land_ok.sum())} land-ok pixels, {int(self.use.sum())} used '
                f'(fit table: {self.n_pixels}); land-ok but < {MIN_CLEAR:.0%} clear: '
                f'{int((self.land_ok & ~self.use).sum())}; offsets (B,G,R,N) '
                + ' '.join(f'{o:+.4f}' for o in self.offset))


def plot_clip(sid, root_cal=FLOWER / 'planet_clipped_cal'):
    """The scene's 50ha clips as reflectance (4, H, W), NaN = nodata, all read on
    the OCM mask's grid as evaluate_planet_calibration.py does: TOA from
    planet_clipped DN x metadata reflectance coefficients, raw SR and calibrated
    SR. Returns ({version: array}, OCM-clear (H, W) bool)."""
    inv = {r['scene_id']: r for r in csv.DictReader(open(FLOWER / 'planet_clipped/inventory.csv'))}
    coeff = get_coeff(str(toa_xml(inv[sid]['four_band_src'])))
    with rio.open(FLOWER / 'planet_clipped/ocm' / sid[:4] / f'{sid}_ocm.tif') as src:
        cls, bounds = src.read(1), src.bounds
    out = {}
    for v, root, scale in (('TOA', FLOWER / 'planet_clipped', [coeff[i] for i in (1, 2, 3, 4)]),
                           ('SR', FLOWER / 'planet_clipped_sr', [1e-4] * 4),
                           ('CAL', root_cal, [1e-4] * 4)):
        with rio.open(root / '4band' / sid[:4] / f'{sid}_4band.tif') as src:
            a = src.read(window=from_bounds(*bounds, src.transform), out_shape=(4,) + cls.shape,
                         boundless=True, fill_value=0).astype(np.float32)
        out[v] = np.where(a == 0, np.nan, a * np.asarray(scale, np.float32)[:, None, None])
    clear = (cls == 0) & np.all([np.isfinite(a).all(axis=0) for a in out.values()], axis=0)
    return out, clear


# ------------------------------------------------------------------ figures

@cli.command()
@out_option
@click.option('--group', 'key', default=SCHEMATIC_GROUP, show_default=True,
              help='Date-satellite group, YYYY-MM-DD_<satellite>.')
def schematic(outdir, key):
    """The method in five panels on one group: Planet SR (3 m) -> 500 m means on
    usable land -> MODIS reference -> per-band histograms and offset -> the
    offset applied to the 3 m 50ha clip."""
    g = Group(key)
    click.echo(g.summary())
    usable = np.where(g.use, 1.0, np.nan)
    # Panels 2-3 show NIR, the band with the largest offset, on one color scale.
    p500 = g.to_display(g.planet500[3] * usable)
    m500 = g.to_display(g.modis[3] * usable)
    vmin, vmax = np.nanpercentile(np.concatenate([p500.ravel(), m500.ravel()]), [1, 99])
    box_lo, box_hi = pooled_stretch([np.where(g.water | g.cloud, np.nan, g.refl)])
    masked = np.where((g.water | g.cloud)[None], np.nan, g.refl)
    sid = next(s['scene_id'] for s in g.scenes
               if (FLOWER / 'planet_clipped_cal/4band' / s['scene_id'][:4]
                   / f'{s["scene_id"]}_4band.tif').exists())
    clips, _ = plot_clip(sid)
    clip_lo, clip_hi = pooled_stretch([clips['SR'], clips['CAL']])

    fig = plt.figure(figsize=(13, 3.6))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.95, 1.15], height_ratios=[1, 1],
                          wspace=0.28, hspace=0.25)
    titles = ['Planet SR, 3 m', 'Planet NIR, 500 m', 'MODIS NIR, 500 m',
              'match means', 'apply to 3 m']
    axes = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1]), fig.add_subplot(gs[:, 2])]
    paper = np.array([0.93, 0.93, 0.91])
    img = stretch_rgb(masked, box_lo, box_hi)
    axes[0].imshow(np.where(np.isnan(img).any(axis=-1, keepdims=True), paper, img),
                   interpolation='nearest')
    # Water and cloud in the 3 m panel: masks drawn as flat colors.
    axes[0].imshow(np.where(g.water, 1.0, np.nan), cmap=_flat(C_WATER), vmin=0, vmax=1,
                   interpolation='nearest')
    axes[0].imshow(np.where(g.cloud & ~g.water, 1.0, np.nan), cmap=_flat('white'),
                   vmin=0, vmax=1, interpolation='nearest')
    for ax, a in zip(axes[1:], (p500, m500)):
        ax.imshow(np.ones(a.shape + (3,)) * paper, interpolation='nearest')
        im = ax.imshow(a, cmap='Greens', vmin=vmin, vmax=vmax, interpolation='nearest')
    for ax in axes:
        outline_region(ax, g.tf, color=INK, lw=1.0)
        bare(ax)
    cax = axes[2].inset_axes([0.2, -0.07, 0.6, 0.035])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.tick_params(labelsize=6.5, colors=INK, length=2)
    cb.outline.set_visible(False)

    hist_axes = [fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[1, 3])]
    # Histograms of two bands (red, NIR): Planet 500 m before / after, MODIS.
    for ax, b in zip(hist_axes, (2, 3)):
        p = g.planet500[b][g.use]
        m = g.modis[b][g.use]
        lo_, hi_ = np.percentile(np.concatenate([p, m, p + g.offset[b]]), [1, 99])
        bins = np.linspace(lo_, hi_, 26)
        ax.hist(m, bins, histtype='step', color=INK, lw=1.5)
        ax.hist(p, bins, color=C_SR, alpha=0.55)
        ax.hist(p + g.offset[b], bins, color=C_CAL, alpha=0.55)
        top = ax.get_ylim()[1]
        ax.set_ylim(0, top * 1.25)
        ax.add_patch(FancyArrowPatch((p.mean(), top * 1.12), (p.mean() + g.offset[b], top * 1.12),
                                     arrowstyle='-|>', mutation_scale=8, color=INK, lw=1,
                                     shrinkA=0, shrinkB=0))
        ax.set_yticks([])
        ax.tick_params(colors=INK, labelsize=7, length=2)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.text(0.0, 0.98, BAND_LABEL[BANDS[b]], transform=ax.transAxes, fontsize=8,
                color=INK, va='top')
    hist_axes[0].legend(handles=[Patch(fc='none', ec=INK, label='MODIS'),
                                 Patch(fc=C_SR, alpha=0.55, label='Planet'),
                                 Patch(fc=C_CAL, alpha=0.55, label='Planet + offset')],
                        fontsize=6.5, frameon=False, loc='upper right',
                        bbox_to_anchor=(1.05, 0.9), handlelength=1.0)

    clip_axes = [fig.add_subplot(gs[0, 4]), fig.add_subplot(gs[1, 4])]
    for ax, v, lab in zip(clip_axes, ('SR', 'CAL'), ('raw', 'calibrated')):
        ax.imshow(np.nan_to_num(stretch_rgb(clips[v], clip_lo, clip_hi), nan=1.0),
                  interpolation='nearest')
        bare(ax)
        ax.text(0.02, 0.05, lab, transform=ax.transAxes, fontsize=7.5, color=INK,
                bbox=dict(facecolor='white', alpha=0.85, lw=0, pad=1.5))

    fig.canvas.draw()
    cols = [axes[0], axes[1], axes[2], hist_axes[0], clip_axes[0]]
    top_y = max(ax.get_position().y1 for ax in cols) + 0.04
    for i, (ax, t) in enumerate(zip(cols, titles)):
        fig.text(ax.get_position().x0, top_y, f'{i + 1}. {t}', fontsize=9.5, color=INK)
    # Arrows between the panels, in figure coordinates.
    for a, b in zip(cols[:-1], cols[1:]):
        x0 = a.get_position().x1 + 0.004
        x1 = b.get_position().x0 - 0.004
        fig.add_artist(FancyArrowPatch((x0, 0.5), (x1, 0.5), transform=fig.transFigure,
                                       arrowstyle='-|>', mutation_scale=12, color=MUTED, lw=1.5))
    click.echo(f'clip shown: {sid}; offsets red {g.offset[2]:+.4f}, NIR {g.offset[3]:+.4f}; '
               f'NIR color scale {vmin:.3f}-{vmax:.3f}; clip stretch lo/hi '
               + ' '.join(f'{a:.3f}/{b:.3f}' for a, b in zip(clip_lo, clip_hi)))
    save(fig, outdir, 'calibration_schematic')


@cli.command('modis-example')
@out_option
@click.option('--group', 'key', default='2025-03-25_2538', show_default=True,
              help='Date-satellite group, YYYY-MM-DD_<satellite>. The default has full '
                   'Planet coverage of the box and clouds over part of the land.')
def modis_example(outdir, key):
    """The MODIS reference for one group, the Planet mosaic with the water and
    cloud masks, and the 500 m pixels the fit used."""
    g = Group(key)
    click.echo(g.summary())
    # MODIS NIR: water is near 0 and forest near 0.3, so the island reads at 500 m.
    nir = g.to_display(g.modis[3])
    vmax = float(np.nanpercentile(nir, 99.5))
    filled = g.to_display((g.fill[3] > 0).astype(float))
    p_lo, p_hi = pooled_stretch([np.where((g.water | g.cloud)[None], np.nan, g.refl)])
    cat = np.where(~g.land_ok, 0, np.where(g.use, 2, 1)).astype(float)
    cat_d = g.to_display(cat)
    click.echo(f'gap-filled NIR pixels: {int((g.fill[3] > 0).sum())} of {g.fill[3].size}, '
               f'{int(((g.fill[3] > 0) & g.use).sum())} of them used; fill_flag counts '
               + str(dict(zip(*np.unique(g.fill[3], return_counts=True)))))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.0))
    paper = np.array([0.93, 0.93, 0.91])
    axes[0].imshow(np.ones(nir.shape + (3,)) * paper, interpolation='nearest')
    im = axes[0].imshow(nir, cmap='Greens', vmin=0, vmax=vmax, interpolation='nearest')
    cax = axes[0].inset_axes([0.55, -0.075, 0.45, 0.03])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.tick_params(labelsize=7, colors=INK, length=2)
    cb.outline.set_visible(False)
    if np.nansum(filled) > 0:
        axes[0].contourf(np.nan_to_num(filled), levels=[0.5, 1.5], colors='none',
                         hatches=['////'])
    img = stretch_rgb(g.refl, p_lo, p_hi)
    axes[1].imshow(np.where(np.isnan(img).any(axis=-1, keepdims=True), paper, img),
                   interpolation='nearest')
    axes[1].imshow(np.where(g.water, 1.0, np.nan), cmap=_flat(C_WATER), vmin=0, vmax=1,
                   interpolation='nearest')
    axes[1].imshow(np.where(g.cloud & ~g.water, 1.0, np.nan), cmap=_flat(C_MASK), vmin=0,
                   vmax=1, alpha=0.6, interpolation='nearest')
    from matplotlib.colors import ListedColormap
    axes[2].imshow(cat_d, cmap=ListedColormap([C_WATER, C_CLOUD, C_USED]), vmin=-0.5,
                   vmax=2.5, interpolation='nearest')
    for ax, t in zip(axes, ('MODIS NIR, 500 m', 'Planet SR, 3 m', 'MODIS pixels')):
        outline_region(ax, g.tf, color=INK, lw=1.2)
        bare(ax)
        ax.set_title(t, fontsize=10, color=INK, loc='left')
    kw = dict(fontsize=8, frameon=False, loc='upper left', bbox_to_anchor=(0, -0.01),
              ncol=1, handlelength=1.2)
    if np.nansum(filled) > 0:
        axes[0].legend(handles=[Patch(fc='white', ec=INK, hatch='////', label='gap-filled')], **kw)
    axes[1].legend(handles=[Patch(fc=C_WATER, label='water mask'),
                            Patch(fc=C_MASK, alpha=0.6, label='cloud mask')], **kw)
    axes[2].legend(handles=[Patch(fc=C_WATER, label='water or shore'),
                            Patch(fc=C_CLOUD, label='not clear in Planet'),
                            Patch(fc=C_USED, label=f'used ({int(g.use.sum())})')], **kw)
    fig.subplots_adjust(wspace=0.05)
    click.echo(f'MODIS NIR color scale 0-{vmax:.3f}')
    save(fig, outdir, 'modis_example')


def example_scenes():
    """The four clips of the presentation's planet_calibration figure
    (plot_study_area_figure.CAL_EXAMPLE: all three Dove generations within 2
    days, each >= 97% clear with its own MODIS fit), as rows of the plot-means
    table, sorted by time; and the table."""
    means = pd.read_csv(CAL / 'planet_calibration_plot_means.csv', parse_dates=['time'],
                        dtype={'satellite_id': str})
    ex = means[means.scene_id.isin(CAL_EXAMPLE)].drop_duplicates('scene_id')
    return [r for _, r in ex.sort_values('time').iterrows()], means


@cli.command('sr-vs-toa')
@out_option
@click.option('--scene', default=None,
              help='Scene id; default the SuperDove clip of the before-after examples.')
def sr_vs_toa(outdir, scene):
    """One clip as TOA and SR, per-band SR vs TOA pixels, and SR - TOA plot
    means over all clear clips."""
    scenes, means = example_scenes()
    if scene is None:
        scene = next(s.scene_id for s in scenes if s.instrument == 'PSB.SD')
    clips, clear = plot_clip(scene)
    lo, hi = pooled_stretch([np.where(clear[None], clips[v], np.nan) for v in ('TOA', 'SR')])

    fig = plt.figure(figsize=(12.5, 4.6))
    gs = fig.add_gridspec(2, 5, width_ratios=[1.9, 0.85, 0.85, 0.12, 1.2],
                          wspace=0.32, hspace=0.32)
    for i, v in enumerate(('TOA', 'SR')):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(np.nan_to_num(stretch_rgb(clips[v], lo, hi), nan=1.0), interpolation='nearest')
        bare(ax)
        ax.text(0.015, 0.05, v, transform=ax.transAxes, fontsize=9, color=INK,
                bbox=dict(facecolor='white', alpha=0.85, lw=0, pad=1.5))
    for b in range(4):
        ax = fig.add_subplot(gs[b // 2, 1 + b % 2])
        x, y = clips['TOA'][b][clear], clips['SR'][b][clear]
        # Axes from 0, so the distance from the 1:1 line reads as the correction.
        lim = (0, 1.05 * np.percentile(np.concatenate([x, y]), 99.9))
        ax.hexbin(x, y, gridsize=60, extent=(*lim, *lim), bins='log', cmap='Greys', mincnt=1)
        ax.plot(lim, lim, color=C_SR, lw=1, ls='--')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.tick_params(colors=INK, labelsize=6.5, length=2)
        ax.locator_params(nbins=3)
        ax.set_title(BAND_LABEL[BANDS[b]], fontsize=8.5, color=INK, loc='left', pad=2)
        if b >= 2:
            ax.set_xlabel('TOA', fontsize=8, color=INK)
        if b % 2 == 0:
            ax.set_ylabel('SR', fontsize=8, color=INK)
        slope, icpt = np.polyfit(x, y, 1)
        click.echo(f'{BANDS[b]:5s} clip mean TOA {x.mean():.4f} SR {y.mean():.4f} '
                   f'(SR - TOA {y.mean() - x.mean():+.4f}); SR = {slope:.3f} TOA {icpt:+.4f}')

    w = means.pivot_table(index=['scene_id', 'instrument'], columns='version',
                          values=BANDS)
    ax = fig.add_subplot(gs[:, 4])
    diffs = [(w[(b, 'SR')] - w[(b, 'TOA')]).dropna().to_numpy() for b in BANDS]
    ax.boxplot(diffs, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.5),
                    boxprops=dict(facecolor='#d9d9d4', edgecolor=MUTED),
                    whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(1, 5), [BAND_LABEL[b] for b in BANDS])
    ax.set_ylabel('SR − TOA, plot mean', fontsize=8, color=INK)
    style(ax)
    n = len(diffs[0])
    for b, d in zip(BANDS, diffs):
        click.echo(f'{b:5s} SR - TOA over {len(d)} clips: median {np.median(d):+.4f}, '
                   f'p10/p90 {np.percentile(d, 10):+.4f}/{np.percentile(d, 90):+.4f}')
    by_inst = {(b, inst): np.median(grp) for b in BANDS
               for inst, grp in (w[(b, 'SR')] - w[(b, 'TOA')]).groupby(level='instrument')}
    click.echo('median SR - TOA by instrument: ' + ', '.join(
        f'{b}/{i} {v:+.4f}' for (b, i), v in by_inst.items()))
    s = next(x for x in scenes if x.scene_id == scene)
    click.echo(f'scene {scene} ({s.instrument}, {s.time}), clear {clear.mean():.3f}; '
               f'{n} clips in the distribution; stretch lo/hi '
               + ' '.join(f'{a:.3f}/{b:.3f}' for a, b in zip(lo, hi)))
    save(fig, outdir, 'sr_vs_toa')


@cli.command('before-after')
@out_option
def before_after(outdir):
    """Four clips from three Dove generations within 2 days as TOA, raw SR and
    calibrated SR, on one shared per-band stretch."""
    scenes, _ = example_scenes()
    data = {s.scene_id: plot_clip(s.scene_id) for s in scenes}
    rows = [('TOA', 'TOA'), ('SR', 'SR'), ('CAL', 'calibrated')]
    lo, hi = pooled_stretch([np.where(clear[None], c[v], np.nan)
                             for c, clear in data.values() for v, _ in rows])
    h, w = next(iter(data.values()))[1].shape
    fig, axes = plt.subplots(3, len(scenes), figsize=(3.1 * len(scenes), 3 * 3.1 * h / w + 0.7))
    for j, s in enumerate(scenes):
        clips, clear = data[s.scene_id]
        for i, (v, lab) in enumerate(rows):
            ax = axes[i, j]
            ax.imshow(np.nan_to_num(stretch_rgb(clips[v], lo, hi), nan=1.0),
                      interpolation='nearest')
            bare(ax)
            nir = np.nanmean(clips[v][3][clear])
            ax.text(0.02, 0.05, f'{nir:.3f}', transform=ax.transAxes, fontsize=7, color=INK,
                    bbox=dict(facecolor='white', alpha=0.85, lw=0, pad=1.2))
            if j == 0:
                ax.set_ylabel(lab, fontsize=10, color=INK)
            if i == 0:
                ax.set_title(f'{s.time:%Y-%m-%d}  {s.instrument}', fontsize=9, color=INK)
            click.echo(f'{s.scene_id} {s.instrument:6s} {v:3s} plot NIR (clear) {nir:.4f} '
                       f'clear {clear.mean():.3f}')
    fig.subplots_adjust(wspace=0.03, hspace=0.04)
    click.echo('stretch lo/hi (B,G,R,N) ' + ' '.join(f'{a:.3f}/{b:.3f}' for a, b in zip(lo, hi)))
    save(fig, outdir, 'calibration_before_after')


def _flat(color):
    from matplotlib.colors import ListedColormap
    return ListedColormap([color])


@cli.command('training-metrics')
@out_option
@click.option('--root', type=click.Path(path_type=Path), default=FLOWER, show_default=True,
              help='Folder holding the 202610_temporal* runs.')
@click.option('--sweep', type=click.Path(path_type=Path, exists=True),
              default=SPATIAL_SWEEP, show_default=True)
def training_metrics(outdir, root, sweep):
    """Left: temporal CV at epoch 12, two seeds per arm. Right: the spatial
    east/west benchmark against epoch (one seed)."""
    keys = ['test_iou', 'test_event/precision', 'test_event/recall', 'test_map/map_50']
    names = ['IoU', 'precision', 'recall', 'mAP@50']
    colors = {'TOA': C_TOA, 'calibrated': C_CAL}
    tcv = {arm: [json.load(open(root / r / 'temporal_cv.json'))['per_epoch'][str(EPOCH)]['pooled']
                 for r in runs] for arm, runs in TEMPORAL_RUNS.items()}
    sw = json.load(open(sweep))['results']

    fig = plt.figure(figsize=(12.5, 4.6))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.5, 0.02, 1, 1], wspace=0.32, hspace=0.45)
    ax = fig.add_subplot(gs[:, 0])
    x = np.arange(len(keys))
    w = 0.36
    for j, (arm, vals) in enumerate(tcv.items()):
        v = np.array([[p[k] for k in keys] for p in vals])
        xs = x + (j - 0.5) * w
        ax.bar(xs, v.mean(axis=0), w * 0.92, color=colors[arm], alpha=0.85, label=arm)
        for s in v:
            ax.scatter(xs, s, s=14, color='white', edgecolor=INK, lw=0.8, zorder=3)
        for k, name in enumerate(names):
            click.echo(f'temporal {arm:10s} {name:9s} seeds ' +
                       ' / '.join(f'{s[k]:.3f}' for s in v) + f'  mean {v[:, k].mean():.3f}')
    ax.set_xticks(x, names)
    ax.set_ylim(0, 0.6)
    style(ax)
    ax.set_title(f'temporal CV, epoch {EPOCH}', fontsize=10, color=INK, loc='left')
    ax.legend(fontsize=8, frameon=False, loc='upper left')

    for k, (key, name) in enumerate(zip(keys, names)):
        ax = fig.add_subplot(gs[k // 2, 2 + k % 2])
        for arm, run in SPATIAL_RUNS.items():
            eps = sorted(int(n.rsplit('_e', 1)[1]) for n in sw if n.rsplit('_e', 1)[0] == run)
            ys = [sw[f'{run}_e{e:03d}'][key] for e in eps]
            ax.plot(eps, ys, color=colors[arm], lw=2, marker='o', ms=3)
            i = int(np.argmax(ys))
            click.echo(f'spatial {arm:10s} {name:9s} peak {ys[i]:.3f} at e{eps[i]}; '
                       f'e{EPOCH} {ys[eps.index(EPOCH)]:.3f}')
        ax.set_xscale('log')
        ax.set_xticks([1, 3, 10, 30, 100], ['1', '3', '10', '30', '100'])
        ax.axvline(EPOCH, color=MUTED, lw=0.8, ls=':')
        style(ax)
        ax.set_title(name, fontsize=9, color=INK, loc='left')
        if k >= 2:
            ax.set_xlabel('epoch', fontsize=8, color=INK)
    fig.text(0.545, 0.985, 'spatial split (east/west), one seed', fontsize=10, color=INK, va='top')
    save(fig, outdir, 'calibration_training_metrics')


@cli.command()
@out_option
@click.option('--chip', default=None,
              help='Training chip basename (.tif); default: the chip with the median '
                   'mean model input among dry-season chips.')
@click.option('--n-draws', type=int, default=11, show_default=True)
@click.option('--seed', type=int, default=0, show_default=True)
@click.option('--size', type=int, default=512, show_default=True,
              help='Crop (chip pixels at 0.75 m), the training window size.')
def augmentation(outdir, chip, n_draws, seed, size):
    """Random draws of the training ColorJitter on one TOA chip's model input."""
    import torch
    import torchvision.transforms as T
    from types import SimpleNamespace
    from train_planet_image_maskrcnn import PlanetMaskRCNNDataset, build_input_channels

    def model_input(path):
        with rio.open(path) as src:
            arr = src.read().transpose(1, 2, 0).astype(np.float32)
        return build_input_channels(arr, ['red', 'green', 'blue'])

    if chip is None:
        tifs = sorted(p for p in TOA_CHIPS.glob('*_4band.tif') if p.name[4:6] in ('01', '02', '03'))
        means = {p: model_input(p).mean() for p in tifs}
        chip = sorted(means, key=means.get)[len(means) // 2].name
    img = model_input(TOA_CHIPS / chip)
    h, w = img.shape[:2]
    r0, c0 = (h - size) // 2, (w - size) // 2
    crop = img[r0:r0 + size, c0:c0 + size]

    # The dataset's own jitter path (RGB jittered jointly), as in training.
    jit = T.ColorJitter(brightness=0.2, contrast=0.2)
    ds = SimpleNamespace(jitter=jit, channel_kinds=['red', 'green', 'blue'])
    draws = []
    for i in range(n_draws):
        torch.manual_seed(seed + i)
        _, b, c, _, _ = T.ColorJitter.get_params(jit.brightness, jit.contrast, None, None)
        torch.manual_seed(seed + i)
        draws.append((b, c, PlanetMaskRCNNDataset._apply_jitter(ds, crop)))
    draws.sort(key=lambda d: (d[0], d[1]))
    q = np.clip(crop * 255, 0, 255).astype(np.uint8) / 255.0  # the uint8 round-trip
    click.echo(f'chip {chip}, crop rows {r0}:{r0 + size}, cols {c0}:{c0 + size}')
    click.echo('draws (brightness, contrast): ' +
               ', '.join(f'({b:.3f}, {c:.3f})' for b, c, _ in draws))

    ncol = 4
    nrow = int(np.ceil((n_draws + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.6 * nrow))
    panels = [q] + [d[2] for d in draws]
    for ax, p in zip(axes.flat, panels):
        ax.imshow(p, interpolation='nearest')
        bare(ax)
    for ax in axes.flat[len(panels):]:
        ax.axis('off')
    a0 = axes.flat[0]
    for sp in a0.spines.values():
        sp.set_visible(True)
        sp.set_color(INK)
        sp.set_linewidth(3)
    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    save(fig, outdir, 'augmentation_colorjitter')


if __name__ == '__main__':
    cli()
