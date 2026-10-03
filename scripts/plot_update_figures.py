#!/usr/bin/env python
"""Figures for the monthly progress updates (first written for September 2026).

Each subcommand writes <name>.png and <name>.pdf into OUTDIR (default
/Volumes/Earth03/flower/figs/202610_updates):

  coverage                   timeline of Planet scenes, drone mosaics, crown
                             maps and training chips, 2018 -> present
  ocm-example                50ha Planet scenes with their OmniCloudMask
                             classes overlaid
  classification-example     one drone mosaic with its HPC flowering and
                             deciduous probability maps
  classification-timeseries  per-crown flowering / deciduous fraction across
                             the phantom 2018-2024 and mavic 2024-2026 mosaics
  same-date                  per-crown classifier output on the dates flown by
                             both drones (2024-03-06, 2024-03-18)
  coreg-stats                drone -> Planet label transfer yield and AROSICS
                             shift magnitudes for each chip build
  maskrcnn-summary           single-panel mAP@50 vs epoch per training set,
                             from compare_maskrcnn_runs.py --json-out

The 50ha drone record is split by drone (see docs/data_sources.md, "Dataset
names"): "phantom" is the Phantom 4 Pro series, released as 24782016
(2018-04-04 .. 2023-10-24) and C3KW2X (2023-10-31 .. 2024-03-18), and
"mavic" is the Mavic 3M series from the STRI Globus share (2024-03-06 ..
2026-01-20). The September 2026 figures called these "STRI" and "globus".

Typical usage:
    python scripts/plot_update_figures.py coverage
    python scripts/plot_update_figures.py classification-timeseries --decimate 16
    python scripts/plot_update_figures.py --help
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import functools
import json
import re
from pathlib import Path

import click
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FLOWER = Path('/Volumes/Earth03/flower')
DEFAULT_OUT = FLOWER / 'figs' / '202610_updates'

# Training-chip builds (docs/planet_training_chips_50ha.md). The base set is
# the 56 vetted chips from the 2020-23 phantom (24782016) build; the other two
# are the raw builds, filtered to their Good chips via vetting.json.
PHANTOM_CHIPS = FLOWER / '20260706_full_label_application_x4_coreg_4band_stretch_stats_curated'
PHANTOM_EXT_CHIPS = FLOWER / '20261002_phantomext_label_application_x4_coreg_4band_stretch_stats'
MAVIC_CHIPS = FLOWER / '20260915_globus_label_application_x4_coreg_4band_stretch_stats_rerun'

PHANTOM_RELEASES = {'24782016': FLOWER / 'stri/24782016', 'C3KW2X': FLOWER / 'stri/C3KW2X'}
MAVIC_RGB = FLOWER / 'stri/globus/RGB'
CLASSIFICATIONS = FLOWER / 'results/classifications'
CROWNMAP = (FLOWER / 'stri/24784053/BCI_50ha_2022_09_29_crownmap_improved/'
            'BCI_50ha_2022_09_29_crownmap_improved.shp')
MAVIC_SCALING = Path(__file__).resolve().parent.parent / 'config/crown_classification_mavic.yml'

# Whole-island orthomosaics synced from Drive in 5dbbf8c (not on this volume).
WHOLE_ISLAND_DATES = [
    '2024-06-11', '2024-08-13', '2024-09-18', '2024-10-14', '2024-11-12',
    '2024-12-16', '2025-01-24', '2025-02-17', '2025-03-17', '2025-04-14',
    '2025-07-15', '2025-08-18', '2025-09-15',
]
# Dated crown outlines (crowns_20240803/BCI_50ha_crownmap_timeseries.shp) cover
# every phantom date, 2018-04-04 .. 2024-03-18. No classification product uses
# them: every phantom and mavic mosaic was classified with the static
# 2022-09-29 crown map (checked against the rasters' valid-pixel footprints).
DATED_CROWNS = ('2018-04-04', '2024-03-18')
CLASSIFIED_SPAN = ('2018-04-04', '2026-01-20')
# The 24782016 -> C3KW2X release boundary. The 2024 release reprocessed every
# date, so a step here is processing, not phenology (docs/data_sources.md).
RELEASE_BOUNDARY = ('2023-10-24', '2023-10-31')
SAME_DATES = ('2024_03_06', '2024_03_18')

DATE_RE = re.compile(r'(\d{4})_(\d{2})_(\d{2})')
C_PHANTOM, C_PHANTOM_EXT, C_MAVIC = '#1f77b4', '#2ca02c', '#d62728'
C_GREY = '#7f7f7f'
OCM_COLORS = {1: ('#ffffff', 'thick cloud'), 2: ('#ff9f1c', 'thin cloud'),
              3: ('#7b2cbf', 'cloud shadow')}

# key -> (label, colour, marker)
SERIES = {
    'phantom-24782016': ('phantom (24782016)', C_PHANTOM, 'o'),
    'phantom-C3KW2X': ('phantom (C3KW2X)', C_PHANTOM_EXT, '^'),
    'mavic': ('mavic (M3M)', C_MAVIC, 's'),
}


def save(fig, outdir, name):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'{name}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    click.echo(f'Wrote {outdir / name}.png/.pdf')


def path_date(p):
    m = DATE_RE.search(Path(p).name)
    return pd.Timestamp('-'.join(m.groups())) if m else None


def scene_date(scene_id):
    return pd.Timestamp(f'{scene_id[0:4]}-{scene_id[4:6]}-{scene_id[6:8]}')


def chip_stems(chipdir, vetting=None):
    stems = sorted(p.name[:-len('.png')] for p in Path(chipdir).glob('*.png')
                   if p.name.count('.') == 1)
    if vetting is not None:
        ratings = json.loads(Path(vetting).read_text())['ratings']
        stems = [s for s in stems
                 if ratings.get(f'{s}.png', {}).get('quality') == 'Good']
    return stems


def release_boundary():
    a, b = map(pd.Timestamp, RELEASE_BOUNDARY)
    return a + (b - a) / 2


# --------------------------------------------------------------------------
# dataset registry
# --------------------------------------------------------------------------
def phantom_dir(release, alignment):
    return PHANTOM_RELEASES[release] / f'BCI_50ha_timeseries_{alignment}_alignment'


@functools.cache
def c3kw2x_dates():
    return frozenset(path_date(p) for p in phantom_dir('C3KW2X', 'local').glob('*_local.tif'))


def classification_source(path):
    """Dataset, release, alignment and date of a drone mosaic or its classification."""
    name = Path(path).name
    d = path_date(name)
    if '_M3M_' in name:
        return dict(dataset='mavic', release='M3M', alignment='global', date=d,
                    series='mavic')
    m = re.search(r'_(global|local)(?:_classifications)?\.tif$', name)
    if m is None:
        raise ValueError(f'cannot tell the drone dataset of {name}')
    release = 'C3KW2X' if d in c3kw2x_dates() else '24782016'
    return dict(dataset='phantom', release=release, alignment=m.group(1), date=d,
                series=f'phantom-{release}')


def drone_ortho_for(cls_path):
    """The orthomosaic a *_classifications.tif was computed from."""
    name = Path(cls_path).name.replace('_classifications.tif', '.tif')
    src = classification_source(cls_path)
    if src['dataset'] == 'mavic':
        return MAVIC_RGB / name
    return phantom_dir(src['release'], src['alignment']) / name


def classification_files(series, alignment='local'):
    """Classification rasters of one SERIES key; phantom in one alignment."""
    if series == 'mavic':
        return sorted(CLASSIFICATIONS.glob('*_M3M_aligned_global_RGB_classifications.tif'))
    return [f for f in sorted(CLASSIFICATIONS.glob(f'*_{alignment}_classifications.tif'))
            if classification_source(f)['series'] == series]


def drone_dates(series):
    if series == 'mavic':
        return sorted({path_date(p) for p in MAVIC_RGB.glob('*.tif')})
    release = series.split('-', 1)[1]
    return sorted({path_date(p) for p in phantom_dir(release, 'local').glob('*_local.tif')})


@click.group()
def cli():
    pass


out_option = click.option('--outdir', type=click.Path(file_okay=False, path_type=Path),
                          default=DEFAULT_OUT, show_default=True)


# --------------------------------------------------------------------------
# coverage
# --------------------------------------------------------------------------
@cli.command()
@out_option
@click.option('--clear-threshold', default=0.5, show_default=True)
def coverage(outdir, clear_threshold):
    """Timeline of every data stream feeding the 50ha training set."""
    files = sorted((FLOWER / 'planet_clipped/4band').glob('*/*_4band.tif'))
    planet = pd.DataFrame({'id': [f.name[:-len('_4band.tif')] for f in files]})
    planet['date'] = planet['id'].map(scene_date)
    planet['month'] = planet['date'].dt.to_period('M').dt.to_timestamp()

    clear = pd.read_csv(FLOWER / 'whole_island/bci_clear_fraction.csv')
    clear['date'] = pd.to_datetime(clear['datetime_utc']).dt.tz_localize(None)
    clear['month'] = clear['date'].dt.to_period('M').dt.to_timestamp()
    clear_monthly = clear[clear['fraction_clear'] >= clear_threshold].groupby('month').size()

    drone = {k: drone_dates(k) for k in SERIES}
    island = [pd.Timestamp(d) for d in WHOLE_ISLAND_DATES]
    chips = {'phantom-24782016': [scene_date(s) for s in chip_stems(PHANTOM_CHIPS)]}
    for key, chipdir in (('phantom-C3KW2X', PHANTOM_EXT_CHIPS), ('mavic', MAVIC_CHIPS)):
        vet = chipdir / 'vetting.json'
        chips[key] = ([scene_date(s) for s in chip_stems(chipdir, vet)]
                      if vet.exists() else [])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.2), sharex=True,
                                   gridspec_kw={'height_ratios': [1.0, 1.6]})
    by = planet.groupby('month').size()
    ax1.bar(by.index, by.values, width=25, color=C_GREY,
            label=f'50ha Planet scenes ({len(planet):,})')
    ax1.plot(clear_monthly.index + pd.Timedelta(days=14), clear_monthly.values,
             color='k', lw=1.2,
             label=f'island scenes >= {clear_threshold:.0%} clear (OCM)')
    ax1.set_ylabel('scenes per month')
    ax1.set_ylim(0, by.max() * 1.2)
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    ax1.set_title('Planet imagery over BCI, through '
                  f'{planet["date"].max():%Y-%m-%d}')
    ax1.grid(axis='y', alpha=0.3)

    rows = [
        ('training chips\n(Planet dates)', [(chips[k], SERIES[k][1]) for k in SERIES]),
        ('whole-island\northomosaics', [(island, 'k')]),
        ('50ha mavic\n(global align)', [(drone['mavic'], C_MAVIC)]),
        ('50ha phantom\n(global + local align)',
         [(drone['phantom-24782016'], C_PHANTOM), (drone['phantom-C3KW2X'], C_PHANTOM_EXT)]),
    ]
    ys = {}
    for i, (label, series) in enumerate(rows):
        y = i + 1.5
        ys[label] = y
        for dates, color in series:
            ax2.vlines(dates, y - 0.3, y + 0.3, color=color, lw=1.1)

    d0, d1 = map(pd.Timestamp, DATED_CROWNS)
    s0, s1 = map(pd.Timestamp, CLASSIFIED_SPAN)
    ax2.barh(1.5 - 1.0, s1 - s0, left=s0, height=0.4, color='#bbbbbb')
    ax2.text(s0 + (s1 - s0) / 2, 0.5, 'static 2022-09-29 crown map used to classify every mosaic',
             ha='center', va='center', fontsize=8)
    ax2.barh(0, d1 - d0, left=d0, height=0.4, color='none', edgecolor=C_GREY,
             hatch='///', lw=1)
    ax2.text(d0 + (d1 - d0) / 2, 0, 'dated outlines available (not used)', ha='center',
             va='center', fontsize=8, color='#444', bbox=dict(fc='white', ec='none', pad=1))

    rb = release_boundary()
    top = len(rows) + 1.5
    for ax in (ax1, ax2):
        ax.axvline(rb, color='k', ls='--', lw=0.9)
    ax2.text(rb, top + 0.05, '24782016 | C3KW2X\nrelease boundary', ha='center',
             va='bottom', fontsize=7.5, color='#222')

    ax2.set_yticks([0, 0.5] + list(ys.values()),
                   ['dated outlines', 'crown map'] + list(ys))
    ax2.set_ylim(-0.5, top + 0.9)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.set_xlim(pd.Timestamp('2018-01-01'), pd.Timestamp('2026-10-31'))
    ax2.grid(axis='x', alpha=0.3)
    ax2.legend(handles=[Patch(color=c, label=l) for l, c, _ in SERIES.values()],
               loc='upper left', fontsize=8, ncol=3)
    n_phantom = len(drone['phantom-24782016']) + len(drone['phantom-C3KW2X'])
    ax2.set_title(f'Drone mosaics ({n_phantom} phantom = '
                  f'{len(drone["phantom-24782016"])} + {len(drone["phantom-C3KW2X"])}, '
                  f'{len(drone["mavic"])} mavic, {len(island)} whole-island) and '
                  'training chips (' + ' + '.join(str(len(chips[k])) for k in SERIES) + ')')
    fig.tight_layout()
    save(fig, outdir, 'coverage')
    for k in SERIES:
        click.echo(f'{k}: {len(drone[k])} mosaics, {len(chips[k])} training chips')


# --------------------------------------------------------------------------
# ocm-example
# --------------------------------------------------------------------------
def read_rgb_preview(path, stretch_mask=None):
    """2-98 % stretched RGB; percentiles over stretch_mask pixels if given."""
    import rasterio
    with rasterio.open(path) as src:
        a = src.read([1, 2, 3]).astype(np.float32)
    valid = a.sum(axis=0) > 0
    if stretch_mask is not None and (valid & stretch_mask).any():
        valid &= stretch_mask
    out = np.zeros(a.shape[1:] + (3,), np.float32)
    for i in range(3):
        b = a[i]
        lo, hi = np.percentile(b[valid], (2, 98)) if valid.any() else (0, 1)
        out[..., i] = np.clip((b - lo) / (hi - lo + 1e-6), 0, 1)
    return out


@cli.command('ocm-example')
@out_option
@click.option('--scene', 'scenes', multiple=True,
              help='Scene ids to show; default picks two candidate scenes.')
@click.option('--start', default=None,
              help='With --end, pick candidates acquired in [START, END] '
                   '(YYYY-MM-DD) instead of the scenes of the last Planet ingest.')
@click.option('--end', default=None)
def ocm_example(outdir, scenes, start, end):
    """50ha Planet scenes with their OmniCloudMask classes overlaid."""
    import rasterio
    ocm_dir = FLOWER / 'planet_clipped/ocm'
    rgb_dir = FLOWER / 'planet_clipped/rgb'

    def ocm_path(s):
        return ocm_dir / s[:4] / f'{s}_ocm.tif'

    def clear_frac(lab):
        valid = lab != 255
        return float((lab[valid] == 0).mean()) if valid.any() else np.nan

    if not scenes:
        if start and end:
            t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
            cands = sorted(p.name[:-len('_ocm.tif')] for p in ocm_dir.glob('*/*_ocm.tif')
                           if t0 <= scene_date(p.name) <= t1)
        else:
            inv = pd.read_csv(FLOWER / 'planet_clipped/inventory.csv')
            cands = sorted(inv.loc[inv['status'] == 'ok', 'scene_id'])
        stats = []
        for s in cands:
            p = ocm_path(s)
            if not p.exists():
                continue
            with rasterio.open(p) as src:
                lab = src.read(1)
            valid = lab != 255
            thin = float((lab[valid] == 2).mean()) if valid.any() else 0.0
            stats.append((s, clear_frac(lab), valid.mean(), thin))
        df = pd.DataFrame(stats, columns=['scene', 'clear', 'valid', 'thin'])
        df = df[df['valid'] > 0.95]
        # A mostly clear scene with some thin cloud/haze, and a half-clouded one.
        hazy = df[df['clear'].between(0.6, 0.9)].sort_values('thin').iloc[-1]['scene']
        mixed = df.iloc[(df['clear'] - 0.5).abs().argsort()].iloc[0]['scene']
        scenes = [hazy, mixed]
        click.echo(f'{len(df)} fully-covered candidate scenes with masks; '
                   f'showing {scenes}')

    fig, axes = plt.subplots(len(scenes), 2, figsize=(11, 3.3 * len(scenes)),
                             squeeze=False)
    for (a0, a1), s in zip(axes, scenes):
        with rasterio.open(ocm_path(s)) as src:
            lab = src.read(1)
        # Stretch on clear pixels so bright cloud does not black out the ground.
        rgb = read_rgb_preview(rgb_dir / s[:4] / f'{s}_rgb.tif', stretch_mask=lab == 0)
        a0.imshow(rgb)
        a1.imshow(rgb)
        over = np.zeros(lab.shape + (4,))
        for k, (c, _) in OCM_COLORS.items():
            rgba = plt.matplotlib.colors.to_rgba(c, 0.65)
            over[lab == k] = rgba
        a1.imshow(over)
        a0.set_title(f'{s}  ({scene_date(s):%Y-%m-%d})', fontsize=9)
        a1.set_title(f'OmniCloudMask: {clear_frac(lab):.0%} clear', fontsize=9)
        for a in (a0, a1):
            a.set_axis_off()
    fig.legend(handles=[Patch(fc=c, ec='k', lw=0.5, label=l)
                        for c, l in OCM_COLORS.values()],
               loc='lower center', ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, outdir, 'ocm_example')


# --------------------------------------------------------------------------
# classification-example
# --------------------------------------------------------------------------
def load_scaling():
    import yaml
    cfg = yaml.safe_load(MAVIC_SCALING.read_text())['uint16_to_uint8']
    return (np.array([b['gain'] for b in cfg]), np.array([b['offset'] for b in cfg]))


def read_decimated(path, factor, bands, window=None):
    """Read `bands` at 1/factor resolution; returns (array, transform)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.windows import Window
    with rasterio.open(path) as src:
        win = window or Window(0, 0, src.width, src.height)
        shape = (len(bands), max(1, int(win.height // factor)),
                 max(1, int(win.width // factor)))
        arr = src.read(bands, window=win, out_shape=shape,
                       resampling=Resampling.average)
        t = src.window_transform(win) * Affine.scale(win.width / shape[2],
                                                     win.height / shape[1])
    return arr, t


def mavic_rgb_display(arr):
    gains, offsets = load_scaling()
    rgb = np.clip(arr[:3].astype(np.float64) * gains[:, None, None]
                  + offsets[:, None, None], 0, 255) / 255
    rgb = np.moveaxis(rgb, 0, -1)
    rgb[arr[:3].sum(axis=0) == 0] = 1
    return rgb


def rgb_display(arr):
    """Display RGB as the classifier saw it: uint8 phantom as is, uint16 mavic scaled."""
    if arr.dtype == np.uint8:
        rgb = np.moveaxis(arr[:3], 0, -1) / 255.0
        rgb[arr[:3].sum(axis=0) == 0] = 1
        return rgb
    return mavic_rgb_display(arr)


def extent(arr, transform):
    """imshow extent (left, right, bottom, top) in map units."""
    h, w = arr.shape[-2:]
    left, top = transform * (0, 0)
    right, bottom = transform * (w, h)
    return (left, right, bottom, top)


def plot_crowns(ax, crowns, **kw):
    crowns.boundary.plot(ax=ax, **kw)


def crown_ids(crowns, shape, transform):
    from rasterio.features import rasterize
    return rasterize(((g, i + 1) for i, g in enumerate(crowns.geometry)),
                     out_shape=shape, transform=transform, fill=0).ravel()


def crown_band_means(crowns, cls_path, decimate, min_pixels=10):
    """Per-crown mean of each classification band; NaN under min_pixels."""
    n = len(crowns)
    arr, t = read_decimated(cls_path, decimate, [1, 2])
    ids = crown_ids(crowns, arr.shape[1:], t)
    counts = np.bincount(ids, minlength=n + 1)[1:]
    out = []
    for b in range(2):
        sums = np.bincount(ids, weights=np.nan_to_num(arr[b]).ravel(), minlength=n + 1)[1:]
        out.append(np.where(counts >= min_pixels, sums / np.maximum(counts, 1), np.nan))
    return out[0], out[1]


@cli.command('classification-example')
@out_option
@click.option('--series', type=click.Choice(list(SERIES)), default='phantom-C3KW2X',
              show_default=True)
@click.option('--date', 'date', default=None,
              help='Date YYYY_MM_DD; default = the SERIES date with the most '
                   'flowering crowns in classification_timeseries.csv.')
@click.option('--decimate', default=16, show_default=True)
@click.option('--zoom-m', default=120.0, show_default=True)
def classification_example(outdir, series, date, decimate, zoom_m):
    """One drone mosaic beside its flowering and deciduous probability maps."""
    import geopandas as gpd
    import rasterio
    from rasterio.windows import from_bounds

    if date is None:
        ts = pd.read_csv(outdir / 'classification_timeseries.csv')
        g = ts[ts['source'] == series]
        date = pd.Timestamp(g.loc[g['flowering_frac'].idxmax(), 'date']).strftime('%Y_%m_%d')
    cls_path = next(f for f in classification_files(series)
                    if path_date(f) == pd.Timestamp(date.replace('_', '-')))
    rgb_path = drone_ortho_for(cls_path)
    crowns = gpd.read_file(CROWNMAP)
    label = SERIES[series][0]

    click.echo(f'reading {rgb_path.name} at 1/{decimate} (no overviews; slow)')
    rgb, t_rgb = read_decimated(rgb_path, decimate, [1, 2, 3])
    cls, t_cls = read_decimated(cls_path, decimate, [1, 2])

    # Zoom on the most confidently flowering crown that is far enough inside
    # the plot for the zoom window to be fully labelled.
    half = zoom_m / 2
    ids = crown_ids(crowns, cls.shape[1:], t_cls)
    sums = np.bincount(ids, weights=np.nan_to_num(cls[0]).ravel(),
                       minlength=len(crowns) + 1)[1:]
    counts = np.bincount(ids, minlength=len(crowns) + 1)[1:]
    means = np.where(counts > 20, sums / np.maximum(counts, 1), 0)
    plot_area = crowns.union_all().convex_hull.buffer(-half)
    inside = crowns.centroid.within(plot_area).to_numpy()
    best = int(np.argmax(np.where(inside, means, -1)))
    cx, cy = crowns.geometry.iloc[best].centroid.coords[0]
    bounds = (cx - half, cy - half, cx + half, cy + half)

    def zoom_read(path, bands):
        with rasterio.open(path) as src:
            win = from_bounds(*bounds, transform=src.transform)
            win = win.round_offsets().round_lengths()
        return read_decimated(path, 2, bands, win)

    zrgb, zt_rgb = zoom_read(rgb_path, [1, 2, 3])
    zcls, zt_cls = zoom_read(cls_path, [1, 2])

    cb = crowns.total_bounds
    pad = 30
    full_lim = (cb[0] - pad, cb[2] + pad, cb[1] - pad, cb[3] + pad)
    zoom_lim = (bounds[0], bounds[2], bounds[1], bounds[3])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4),
                             gridspec_kw={'height_ratios': [0.62, 1]})
    rgb_title = (f'{label} RGB (uint16 -> uint8 as classified)' if rgb.dtype != np.uint8
                 else f'{label} RGB ({classification_source(cls_path)["alignment"]} align)')
    titles = [rgb_title, 'P(flowering)', 'P(deciduous)']
    for row, (img, t_img, c, t_c, lim, lw) in enumerate([
            (rgb, t_rgb, cls, t_cls, full_lim, 0.25),
            (zrgb, zt_rgb, zcls, zt_cls, zoom_lim, 1.2)]):
        axes[row, 0].imshow(rgb_display(img), extent=extent(img, t_img))
        plot_crowns(axes[row, 0], crowns, color='yellow', lw=lw)
        for k in (1, 2):
            axes[row, k].set_facecolor('#dddddd')
            im = axes[row, k].imshow(np.ma.masked_invalid(c[k - 1]), cmap='magma',
                                     vmin=0, vmax=1, extent=extent(c, t_c),
                                     interpolation='nearest')
            plot_crowns(axes[row, k], crowns, color='cyan', lw=lw)
        for k in range(3):
            ax = axes[row, k]
            ax.set_xlim(lim[0], lim[1])
            ax.set_ylim(lim[2], lim[3])
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[k])
                ax.add_patch(plt.Rectangle((bounds[0], bounds[1]), zoom_m, zoom_m,
                                           fill=False, ec='lime', lw=1.8))
            else:
                for side in ax.spines.values():
                    side.set_edgecolor('lime')
                    side.set_linewidth(1.8)
    fig.colorbar(im, ax=axes[:, 1:].ravel().tolist(), shrink=0.6,
                 label='classifier probability (grey = not classified)')
    fig.suptitle(f'HPC crown classification of the {date.replace("_", "-")} {label} '
                 f'mosaic; outlines = static 2022-09-29 crown map '
                 f'({len(crowns):,} crowns); bottom row = {zoom_m:.0f} m zoom',
                 fontsize=11)
    save(fig, outdir, 'classification_example')


# --------------------------------------------------------------------------
# classification-timeseries
# --------------------------------------------------------------------------
@cli.command('classification-timeseries')
@out_option
@click.option('--decimate', default=16, show_default=True,
              help='Read rasters at 1/N resolution (~0.75 m at 16).')
@click.option('--threshold', default=0.5, show_default=True,
              help='A crown counts as flowering/deciduous when its mean '
                   'probability exceeds this.')
@click.option('--min-pixels', default=10, show_default=True,
              help='Minimum decimated pixels inside a crown to score it.')
def classification_timeseries(outdir, decimate, threshold, min_pixels):
    """Fraction of crowns flowering / deciduous per mosaic date, 2018-2026.

    Phantom is scored on the locally aligned mosaics, mavic on its only
    (global) alignment. Results are cached per file in
    OUTDIR/classification_timeseries.csv.
    """
    import geopandas as gpd

    crowns = gpd.read_file(CROWNMAP)
    files = [f for k in SERIES for f in classification_files(k)]
    csv = outdir / 'classification_timeseries.csv'
    done = pd.read_csv(csv) if csv.exists() else pd.DataFrame(columns=['file'])
    rows = done.to_dict('records')
    seen = set(done['file'])
    with click.progressbar(files, label='scoring mosaics') as bar:
        for f in bar:
            if f.name in seen:
                continue
            src = classification_source(f)
            flower, decid = crown_band_means(crowns, f, decimate, min_pixels)
            scored = np.isfinite(flower)
            rows.append({'file': f.name, 'source': src['series'],
                         'alignment': src['alignment'], 'date': src['date'],
                         'n_scored': int(scored.sum()),
                         'flowering_frac': float((flower[scored] > threshold).mean()),
                         'deciduous_frac': float((decid[scored] > threshold).mean())})
            outdir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(csv, index=False)

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    fig, axes = plt.subplots(2, 1, figsize=(13, 6.0), sharex=True)
    rb = release_boundary()
    for ax, key in ((axes[0], 'flowering'), (axes[1], 'deciduous')):
        for series, (label, color, mk) in SERIES.items():
            g = df[df['source'] == series]
            vals = 100 * g[f'{key}_frac']
            ax.plot(g['date'], vals, marker=mk, ms=3.5, color=color, lw=1,
                    label=f'{label}, {len(g)} dates, median {vals.median():.1f}%')
            ax.hlines(vals.median(), g['date'].min(), g['date'].max(),
                      color=color, ls='--', lw=1.2)
        ax.axvline(rb, color='k', ls=':', lw=0.9)
        ax.set_ylabel(f'% crowns {key}')
        ax.set_ylim(0, None)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='upper left', title=key, title_fontsize=8)
    axes[0].text(rb, axes[0].get_ylim()[1], ' 24782016 | C3KW2X ', ha='center',
                 va='bottom', fontsize=7.5)
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[0].set_title(f'Crown-level phenology from drone classification: share of '
                      f'{int(df["n_scored"].median()):,} crowns (2022-09-29 map) '
                      f'with mean P > {threshold}', pad=14)
    fig.tight_layout()
    save(fig, outdir, 'classification_timeseries')
    for series in SERIES:
        g = df[df['source'] == series]
        click.echo(f'{series}: {len(g)} dates, median flowering '
                   f'{100 * g["flowering_frac"].median():.2f}%, deciduous '
                   f'{100 * g["deciduous_frac"].median():.2f}%')


# --------------------------------------------------------------------------
# same-date
# --------------------------------------------------------------------------
@cli.command('same-date')
@out_option
@click.option('--decimate', default=16, show_default=True,
              help='Read rasters at 1/N resolution (~0.75 m at 16).')
@click.option('--threshold', default=0.5, show_default=True)
def same_date(outdir, decimate, threshold):
    """Phantom vs mavic per-crown classifier output on the dates both flew.

    Writes same_date_comparison.png/.pdf and same_date_crowns.csv (one row per
    crown, date and mosaic).
    """
    import geopandas as gpd
    from scipy.stats import spearmanr

    crowns = gpd.read_file(CROWNMAP)
    mosaics = {'phantom local': '{d}_local', 'phantom global': '{d}_global',
               'mavic': '{d}_M3M_aligned_global_RGB'}
    recs, summary = [], []
    for d in SAME_DATES:
        for name, pat in mosaics.items():
            f = CLASSIFICATIONS / f'BCI_50ha_{pat.format(d=d)}_classifications.tif'
            flower, decid = crown_band_means(crowns, f, decimate)
            recs.append(pd.DataFrame({'date': d.replace('_', '-'), 'mosaic': name,
                                      'crown': np.arange(len(crowns)),
                                      'p_flower': flower, 'p_decid': decid}))
    df = pd.concat(recs)
    df.to_csv(outdir / 'same_date_crowns.csv', index=False)
    wide = df.pivot_table(index=['date', 'crown'], columns='mosaic',
                          values=['p_flower', 'p_decid']).dropna()

    fig, axes = plt.subplots(len(SAME_DATES), 2, figsize=(10, 4.8 * len(SAME_DATES)),
                             squeeze=False)
    for r, d in enumerate(SAME_DATES):
        w = wide.xs(d.replace('_', '-'), level='date')
        for c, (var, title) in enumerate((('p_flower', 'P(flowering)'),
                                          ('p_decid', 'P(deciduous)'))):
            ax = axes[r, c]
            x, y = w[(var, 'phantom local')], w[(var, 'mavic')]
            ax.scatter(x, y, s=5, alpha=0.4, color='k', lw=0)
            ax.plot([0, 1], [0, 1], color=C_GREY, lw=0.8)
            ax.axvline(threshold, color=C_PHANTOM, ls=':', lw=1)
            ax.axhline(threshold, color=C_MAVIC, ls=':', lw=1)
            frac = {m: 100 * (w[(var, m)] > threshold).mean() for m in mosaics}
            rho = spearmanr(x, y).statistic
            ax.text(0.02, 0.98,
                    f'crowns > {threshold}:\n'
                    f'  phantom local {frac["phantom local"]:.1f}%\n'
                    f'  phantom global {frac["phantom global"]:.1f}%\n'
                    f'  mavic {frac["mavic"]:.1f}%\n'
                    f'Spearman rho {rho:.2f} (n={len(w):,})',
                    transform=ax.transAxes, va='top', fontsize=8,
                    bbox=dict(fc='white', ec='#cccccc', alpha=0.9))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_xlabel(f'phantom (C3KW2X, local align) mean {title}')
            ax.set_ylabel(f'mavic (M3M) mean {title}')
            ax.set_title(f'{d.replace("_", "-")}: {title}', fontsize=10)
            summary.append(dict(date=d, var=var, rho=rho, n=len(w), **frac))
    fig.suptitle('Same-day flights, per-crown mean classifier probability '
                 '(2022-09-29 crown map)', fontsize=11)
    fig.tight_layout()
    save(fig, outdir, 'same_date_comparison')
    click.echo(pd.DataFrame(summary).round(3).to_string(index=False))


def same_date_paths(d):
    """(phantom local, mavic) classification rasters for a SAME_DATES entry."""
    return (CLASSIFICATIONS / f'BCI_50ha_{d}_local_classifications.tif',
            CLASSIFICATIONS / f'BCI_50ha_{d}_M3M_aligned_global_RGB_classifications.tif')


def read_bounds(path, bounds, bands, factor=1):
    """Read a map-bounds crop of `bands` at 1/factor resolution."""
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(path) as src:
        win = from_bounds(*bounds, transform=src.transform)
        win = win.round_offsets().round_lengths()
    return read_decimated(path, factor, bands, win)


@cli.command('same-date-maps')
@out_option
@click.option('--decimate', default=16, show_default=True)
def same_date_maps(outdir, decimate):
    """Whole-plot phantom vs mavic RGB and P(flowering) on the dates both flew."""
    import geopandas as gpd
    crowns = gpd.read_file(CROWNMAP)
    cb = crowns.total_bounds
    pad = 30
    bounds = (cb[0] - pad, cb[1] - pad, cb[2] + pad, cb[3] + pad)
    fig, axes = plt.subplots(len(SAME_DATES), 4, figsize=(20, 2.55 * len(SAME_DATES) + 0.5),
                             squeeze=False, layout='constrained')
    for r, d in enumerate(SAME_DATES):
        for c0, (label, cls) in enumerate(zip(('phantom (C3KW2X, local align)', 'mavic (M3M)'),
                                              same_date_paths(d))):
            rgb, t_rgb = read_bounds(drone_ortho_for(cls), bounds, [1, 2, 3], decimate)
            p, t_p = read_bounds(cls, bounds, [1], decimate)
            a_rgb, a_p = axes[r, 2 * c0], axes[r, 2 * c0 + 1]
            a_rgb.imshow(rgb_display(rgb), extent=extent(rgb, t_rgb))
            a_p.set_facecolor('#dddddd')
            im = a_p.imshow(np.ma.masked_invalid(p[0]), cmap='magma', vmin=0, vmax=1,
                            extent=extent(p, t_p), interpolation='nearest')
            a_rgb.set_title(f'{d.replace("_", "-")} {label} RGB', fontsize=9)
            a_p.set_title(f'{d.replace("_", "-")} {label} P(flowering)', fontsize=9)
            for a in (a_rgb, a_p):
                a.set_xlim(bounds[0], bounds[2])
                a.set_ylim(bounds[1], bounds[3])
                a.set_xticks([])
                a.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.01,
                 label='P(flowering) (grey = not classified)')
    fig.suptitle('Same-day flights by both drones, classified by the same flowering model '
                 '(windows placed by the static 2022-09-29 crown map)', fontsize=11)
    save(fig, outdir, 'same_date_maps')


@cli.command('same-date-examples')
@out_option
@click.option('--n-mavic-only', default=8, show_default=True,
              help='Crowns flowering on mavic only (P > 0.5 vs phantom < 0.2).')
@click.option('--pad-m', default=6.0, show_default=True)
def same_date_examples(outdir, n_mavic_only, pad_m):
    """Crowns where phantom and mavic classifications disagree on the same day.

    Reads OUTDIR/same_date_crowns.csv (from `same-date`). Writes
    same_date_mavic_only_flowering.png/.pdf (the dominant disagreement) and
    same_date_other_cases.png/.pdf (crowns flowering on both, flowering on
    phantom only, and deciduous on phantom only). Every crown is shown at
    full mosaic resolution with each sensor's RGB and probability map.
    """
    import geopandas as gpd
    crowns = gpd.read_file(CROWNMAP)
    df = pd.read_csv(outdir / 'same_date_crowns.csv')
    w = df.pivot_table(index=['date', 'crown'], columns='mosaic',
                       values=['p_flower', 'p_decid']).dropna()
    f, dc = w['p_flower'], w['p_decid']

    def pick(mask, key, n):
        """Top-n crown-dates by `key`, at most one row per crown."""
        s = key[mask].sort_values(ascending=False)
        s = s[~s.index.get_level_values('crown').duplicated()]
        return [(d, c) for d, c in s.index[:n]]

    mavic_only = pick((f['mavic'] > 0.5) & (f['phantom local'] < 0.2),
                      f['mavic'] - f['phantom local'], n_mavic_only)
    both = pick((f['mavic'] > 0.5) & (f['phantom local'] > 0.5),
                f[['mavic', 'phantom local']].min(axis=1), 2)
    phantom_only = pick((f['phantom local'] > 0.5) & (f['mavic'] < 0.3),
                        f['phantom local'] - f['mavic'], 3)
    decid_phantom = pick((dc['phantom local'] > 0.5) & (dc['mavic'] < 0.3),
                         dc['phantom local'] - dc['mavic'], 3)
    click.echo(f'{len(mavic_only)} mavic-only flowering, {len(both)} both, '
               f'{len(phantom_only)} phantom-only flowering, '
               f'{len(decid_phantom)} phantom-only deciduous')

    def draw(axes4, d, c, band, case):
        geom = crowns.geometry.iloc[c]
        x0, y0, x1, y1 = geom.bounds
        half = max(x1 - x0, y1 - y0) / 2 + pad_m
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        b = (cx - half, cy - half, cx + half, cy + half)
        var = 'p_flower' if band == 1 else 'p_decid'
        name = 'P(flowering)' if band == 1 else 'P(deciduous)'
        dd = d.replace('-', '_')
        for k, (label, cls, col) in enumerate(zip(('phantom', 'mavic'), same_date_paths(dd),
                                                  ('phantom local', 'mavic'))):
            rgb, t_rgb = read_bounds(drone_ortho_for(cls), b, [1, 2, 3])
            p, t_p = read_bounds(cls, b, [band], 4)
            a_rgb, a_p = axes4[2 * k], axes4[2 * k + 1]
            a_rgb.imshow(rgb_display(rgb), extent=extent(rgb, t_rgb))
            a_p.set_facecolor('#dddddd')
            a_p.imshow(np.ma.masked_invalid(p[0]), cmap='magma', vmin=0, vmax=1,
                       extent=extent(p, t_p), interpolation='nearest')
            for a in (a_rgb, a_p):
                gpd.GeoSeries([geom]).boundary.plot(ax=a, color='cyan', lw=0.9)
                a.set_xlim(b[0], b[2])
                a.set_ylim(b[1], b[3])
                a.set_xticks([])
                a.set_yticks([])
            a_rgb.set_title(f'{label} RGB', fontsize=8)
            a_p.set_title(f'{label} {name} = {w.loc[(d, c), (var, col)]:.2f}', fontsize=8)
        axes4[0].set_ylabel(f'{case}\n{d} crown {c}', fontsize=8)

    # Figure 1: the dominant disagreement, two crowns per row.
    rows = int(np.ceil(len(mavic_only) / 2))
    fig, axes = plt.subplots(rows, 8, figsize=(22, 2.9 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for i, (d, c) in enumerate(mavic_only):
        a4 = axes[i // 2, 4 * (i % 2):4 * (i % 2) + 4]
        for a in a4:
            a.set_axis_on()
        draw(a4, d, c, 1, 'mavic only')
    fig.suptitle('Crowns flowering on mavic but not phantom on the same day '
                 '(mean P(flowering) > 0.5 on mavic, < 0.2 on phantom)', fontsize=11)
    fig.tight_layout()
    save(fig, outdir, 'same_date_mavic_only_flowering')

    # Figure 2: the other cases, one crown per row.
    cases = ([(d, c, 1, 'both flowering') for d, c in both]
             + [(d, c, 1, 'phantom only') for d, c in phantom_only]
             + [(d, c, 2, 'deciduous,\nphantom only') for d, c in decid_phantom])
    fig, axes = plt.subplots(len(cases), 4, figsize=(11.5, 2.9 * len(cases)), squeeze=False)
    for row, (d, c, band, case) in zip(axes, cases):
        draw(row, d, c, band, case)
    fig.suptitle('Same-day crowns: agreement and the rarer disagreements', fontsize=11)
    fig.tight_layout()
    save(fig, outdir, 'same_date_other_cases')
    pd.DataFrame([dict(case=case.replace('\n', ' '), date=d, crown=c)
                  for d, c, _, case in [(d, c, 1, 'mavic only') for d, c in mavic_only]
                  + cases]).to_csv(outdir / 'same_date_examples.csv', index=False)


# --------------------------------------------------------------------------
# coreg-stats
# --------------------------------------------------------------------------
@cli.command('coreg-stats')
@out_option
def coreg_stats(outdir):
    """Label-transfer yield and AROSICS shift magnitudes for each chip build."""
    builds = []
    for name, chipdir, color, vet in (
            ('phantom 2020-23 build\n(24782016 local align)', PHANTOM_CHIPS, C_PHANTOM, None),
            ('phantom 2023-24 build\n(C3KW2X local align)', PHANTOM_EXT_CHIPS, C_PHANTOM_EXT,
             PHANTOM_EXT_CHIPS / 'vetting.json'),
            ('mavic 2024-26 build\n(M3M global align)', MAVIC_CHIPS, C_MAVIC,
             MAVIC_CHIPS / 'vetting.json')):
        log = pd.DataFrame(json.loads((chipdir / 'coreg_log.json').read_text()))
        ok = log[log['coreg_ok']]
        if vet is not None:
            q = pd.Series([v['quality'] for v in json.loads(vet.read_text())['ratings'].values()])
            counts = q.value_counts()
        else:
            # Labelbox vetting of the 2020-23 build, recorded in
            # docs/planet_training_chips_50ha.md.
            counts = pd.Series({'Good': 56, 'Fair': 25, 'Poor': 50})
        builds.append(dict(name=name, color=color, pairs=len(log), ok=len(ok),
                           good=int(counts.get('Good', 0)),
                           fair=int(counts.get('Fair', 0)),
                           poor=int(counts.get('Poor', 0)),
                           shifts=np.hypot(ok['x_shift_m'], ok['y_shift_m'])))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.6),
                                   gridspec_kw={'width_ratios': [1.25, 1]})
    stages = ['drone-Planet\npairs (±2 d)', 'AROSICS\ncoregistered', 'vetted\nGood']
    x = np.arange(len(stages))
    w = 0.8 / len(builds)
    for i, b in enumerate(builds):
        vals = [b['pairs'], b['ok'], b['good']]
        bars = ax1.bar(x + (i - (len(builds) - 1) / 2) * w, vals, w, color=b['color'],
                       label=b['name'].replace('\n', ' '))
        for bar, v in zip(bars, vals):
            pct = '' if v == b['pairs'] else f'\n{v / b["pairs"]:.0%}'
            ax1.text(bar.get_x() + bar.get_width() / 2, v + 4, f'{v}{pct}',
                     ha='center', va='bottom', fontsize=7.5)
    ax1.set_xticks(x, stages)
    ax1.set_ylabel('count')
    ax1.set_ylim(0, max(b['pairs'] for b in builds) * 1.25)
    ax1.set_title('Drone label -> Planet chip yield')
    ax1.legend(fontsize=8)

    bins = np.arange(0, 33, 1.5)
    for b in builds:
        ax2.hist(b['shifts'], bins=bins, histtype='step', lw=2, color=b['color'],
                 density=True,
                 label=f"{b['name'].splitlines()[0]}: median "
                       f"{np.median(b['shifts']):.1f} m (n={len(b['shifts'])})")
    ax2.axvline(3, color='k', ls=':', lw=1)
    ax2.text(3.3, ax2.get_ylim()[1] * 0.9, '1 Planet pixel', fontsize=8)
    ax2.set_xlabel('applied AROSICS shift magnitude (m)')
    ax2.set_ylabel('density')
    ax2.set_title('Global drone -> Planet shift per scene')
    ax2.legend(fontsize=8)
    fig.tight_layout()
    save(fig, outdir, 'coreg_stats')
    for b in builds:
        click.echo(f"{b['name'].splitlines()[0]}: {b['pairs']} pairs, {b['ok']} ok, "
                   f"{b['good']} Good / {b['fair']} Fair / {b['poor']} Poor, median shift "
                   f"{np.median(b['shifts']):.2f} m")


# --------------------------------------------------------------------------
# maskrcnn-summary
# --------------------------------------------------------------------------
RUN_COLORS = {'base': C_PHANTOM, 'ext': C_MAVIC, 'phantom': C_PHANTOM_EXT,
              'full': '#9467bd'}
# Peak-callout offsets (points) per run, so runs peaking at the same epoch
# do not print their labels on top of each other.
RUN_ANNOT = {'base': (8, 8), 'ext': (8, -18), 'phantom': (-88, 10), 'full': (-70, 14)}


@cli.command('maskrcnn-summary')
@out_option
@click.option('--sweep', type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None, help='compare_maskrcnn_runs.py --json-out file '
                                 '[default: OUTDIR/headtohead_sweep.json]')
@click.option('--label', 'labels', multiple=True, metavar='RUN=TEXT',
              help='Legend text per run (checkpoint labels are <run>_e<epoch>).')
def maskrcnn_summary(outdir, sweep, labels):
    """Single-panel mAP@50 vs epoch per training set, for a slide."""
    blob = json.loads((sweep or outdir / 'headtohead_sweep.json').read_text())
    names = {'base': 'phantom 2020-23 chips only (56)',
             'ext': '+ mavic 2024-26 chips (126)'}
    names.update(dict(lab.split('=', 1) for lab in labels))
    runs = {}
    for label, row in blob['results'].items():
        run, ep = label.rsplit('_e', 1)
        runs.setdefault(run, []).append((int(ep), row['test_map/map_50']))
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    curves = {}
    cycle = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'][4:])
    for run, pts in runs.items():
        pts.sort()
        e, m = map(np.array, zip(*pts))
        curves[run] = (e, m)
        color = RUN_COLORS.get(run) or next(cycle)
        ax.plot(e, m, marker='o', ms=4, color=color, lw=2, label=names.get(run, run))
        k = int(np.argmax(m))
        ax.plot(e[k], m[k], marker='*', ms=15, color=color)
        ax.annotate(f'{m[k]:.3f} @ ep {e[k]}', (e[k], m[k]),
                    textcoords='offset points', xytext=RUN_ANNOT.get(run, (6, 8)),
                    fontsize=9, color=color, fontweight='bold')
    if curves.keys() == {'base', 'ext'}:
        e, mb = curves['base']
        _, me = curves['ext']
        ax.fill_between(e, mb, me, where=me > mb, color=C_MAVIC, alpha=0.12,
                        interpolate=True, lw=0)
        ax.fill_between(e, mb, me, where=me <= mb, color=C_PHANTOM, alpha=0.12,
                        interpolate=True, lw=0)
    ax.set_xscale('log')
    ax.set_xlabel('training epoch')
    ax.set_ylabel('mask mAP@50')
    ax.set_ylim(0, max(m.max() for _, m in curves.values()) * 1.18)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_title(f"Mask R-CNN on the same {blob['n_chips']} phantom 2020-23 test chips")
    fig.tight_layout()
    save(fig, outdir, 'maskrcnn_summary')


if __name__ == '__main__':
    cli()
