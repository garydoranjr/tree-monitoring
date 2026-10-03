#!/usr/bin/env python
"""Figures for the September 2026 progress update.

Each subcommand writes <name>.png and <name>.pdf into OUTDIR (default
/Volumes/Earth03/flower/figs/202609_updates):

  coverage                   timeline of Planet scenes, drone mosaics, crown
                             outlines and training chips, 2018 -> present
  ocm-example                new 2026 50ha Planet scenes with their
                             OmniCloudMask classes overlaid
  classification-example     one globus mosaic with its HPC flowering and
                             deciduous probability maps
  classification-timeseries  per-crown flowering / deciduous fraction across
                             the STRI 2018-2023 and globus 2024-2026 mosaics
  coreg-stats                drone -> Planet label transfer yield and AROSICS
                             shift magnitudes, 2020-2023 vs globus builds
  maskrcnn-summary           single-panel mAP@50 vs epoch for the two training
                             sets, from compare_maskrcnn_runs.py --json-out

Typical usage:
    python scripts/plot_update_202609.py coverage
    python scripts/plot_update_202609.py classification-timeseries --decimate 16
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import json
import re
from pathlib import Path

import click
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

FLOWER = Path('/Volumes/Earth03/flower')
DEFAULT_OUT = FLOWER / 'figs' / '202609_updates'

OLD_CHIPS = FLOWER / '20260706_full_label_application_x4_coreg_4band_stretch_stats_curated'
GLOBUS_CHIPS = FLOWER / '20260915_globus_label_application_x4_coreg_4band_stretch_stats_rerun'
STRI_LOCAL = FLOWER / 'stri/24782016/BCI_50ha_timeseries_local_alignment'
GLOBUS_RGB = FLOWER / 'stri/globus/RGB'
CLASSIFICATIONS = FLOWER / 'results/classifications'
CROWNMAP = (FLOWER / 'stri/24784053/BCI_50ha_2022_09_29_crownmap_improved/'
            'BCI_50ha_2022_09_29_crownmap_improved.shp')
GLOBUS_SCALING = Path(__file__).resolve().parent.parent / 'config/crown_classification_globus.yml'

# Whole-island orthomosaics synced from Drive in 5dbbf8c (not on this volume).
WHOLE_ISLAND_DATES = [
    '2024-06-11', '2024-08-13', '2024-09-18', '2024-10-14', '2024-11-12',
    '2024-12-16', '2025-01-24', '2025-02-17', '2025-03-17', '2025-04-14',
    '2025-07-15', '2025-08-18', '2025-09-15',
]
# Dated crown outlines (STRI time-series shapefile, not on this volume): they
# cover every STRI mosaic through 2023-10-24, 14 further dates to 2024-02-28
# with no mosaic here, and the first two globus flights. Every later globus
# flight was classified with the static 2022-09-29 crown map.
DATED_CROWNS = ('2018-04-04', '2024-03-18')
STATIC_CROWNS = ('2024-03-18', '2026-01-20')
MOSAIC_GAP = ('2023-10-24', '2024-03-06')

DATE_RE = re.compile(r'(\d{4})_(\d{2})_(\d{2})')
C_OLD, C_NEW, C_GREY = '#1f77b4', '#d62728', '#7f7f7f'
OCM_COLORS = {1: ('#ffffff', 'thick cloud'), 2: ('#ff9f1c', 'thin cloud'),
              3: ('#7b2cbf', 'cloud shadow')}


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
    inv = pd.read_csv(FLOWER / 'planet_clipped/inventory.csv')
    new_ids = set(inv.loc[inv['status'] == 'ok', 'scene_id'])
    files = sorted((FLOWER / 'planet_clipped/4band').glob('*/*_4band.tif'))
    planet = pd.DataFrame({'id': [f.name[:-len('_4band.tif')] for f in files]})
    planet['date'] = planet['id'].map(scene_date)
    planet['new'] = planet['id'].isin(new_ids)
    planet['month'] = planet['date'].dt.to_period('M').dt.to_timestamp()

    clear = pd.read_csv(FLOWER / 'whole_island/bci_clear_fraction.csv')
    clear['date'] = pd.to_datetime(clear['datetime_utc']).dt.tz_localize(None)
    clear['month'] = clear['date'].dt.to_period('M').dt.to_timestamp()
    clear_monthly = clear[clear['fraction_clear'] >= clear_threshold].groupby('month').size()

    stri = sorted({path_date(p) for p in STRI_LOCAL.glob('*_local.tif')})
    globus = sorted({path_date(p) for p in GLOBUS_RGB.glob('*.tif')})
    island = [pd.Timestamp(d) for d in WHOLE_ISLAND_DATES]
    old_chips = [scene_date(s) for s in chip_stems(OLD_CHIPS)]
    new_chips = [scene_date(s) for s in chip_stems(GLOBUS_CHIPS, GLOBUS_CHIPS / 'vetting.json')]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6.8), sharex=True,
                                   gridspec_kw={'height_ratios': [1.1, 1.4]})
    width = 25
    by = planet.groupby(['month', 'new']).size().unstack(fill_value=0)
    ax1.bar(by.index, by.get(False, 0), width=width, color=C_OLD,
            label=f'50ha Planet scenes ({(~planet["new"]).sum():,})')
    ax1.bar(by.index, by.get(True, 0), width=width, bottom=by.get(False, 0),
            color=C_NEW, label=f'added Sep 2026 ({planet["new"].sum()})')
    ax1.plot(clear_monthly.index + pd.Timedelta(days=14), clear_monthly.values,
             color='k', lw=1.2,
             label=f'island scenes >= {clear_threshold:.0%} clear (OCM)')
    ax1.set_ylabel('scenes per month')
    ax1.set_ylim(0, by.sum(axis=1).max() * 1.15)
    ax1.legend(loc='upper left', fontsize=8, ncol=3)
    ax1.set_title('Planet imagery over BCI, through '
                  f'{planet["date"].max():%Y-%m-%d}')
    ax1.grid(axis='y', alpha=0.3)

    rows = [
        ('training chips\n(Planet dates)', [(old_chips, C_OLD), (new_chips, C_NEW)]),
        ('whole-island\northomosaics', [(island, C_NEW)]),
        ('50ha globus M3M\n(global align)', [(globus, C_NEW)]),
        ('50ha STRI series\n(local align)', [(stri, C_OLD)]),
    ]
    ys = {}
    for i, (label, series) in enumerate(rows):
        y = i + 1
        ys[label] = y
        for dates, color in series:
            ax2.vlines(dates, y - 0.3, y + 0.3, color=color, lw=1.1)

    y = 0
    d0, d1 = map(pd.Timestamp, DATED_CROWNS)
    s0, s1 = map(pd.Timestamp, STATIC_CROWNS)
    ax2.barh(y, d1 - d0, left=d0, height=0.45, color=C_OLD, alpha=0.6)
    ax2.barh(y, s1 - s0, left=s0, height=0.45, color='none', edgecolor=C_NEW,
             hatch='///', lw=1)
    ax2.text(s0 + (s1 - s0) / 2, y, 'static 2022-09-29 map', ha='center',
             va='center', fontsize=8, color=C_NEW,
             bbox=dict(fc='white', ec='none', pad=1))
    ax2.text(d0 + (d1 - d0) / 2, y, 'dated crown outlines', ha='center',
             va='center', fontsize=8, color='k')

    g0, g1 = map(pd.Timestamp, MOSAIC_GAP)
    for ax in (ax1, ax2):
        ax.axvspan(g0, g1, color=C_GREY, alpha=0.18, lw=0)
    ax2.text(g0 + (g1 - g0) / 2, len(rows) + 0.75, 'no 50ha\nmosaic', ha='center',
             va='center', fontsize=8, color='#444')

    ax2.set_yticks([0] + list(ys.values()), ['crown outlines'] + list(ys))
    ax2.set_ylim(-0.6, len(rows) + 1.2)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.set_xlim(pd.Timestamp('2018-01-01'), pd.Timestamp('2026-10-31'))
    ax2.grid(axis='x', alpha=0.3)
    ax2.legend(handles=[Patch(color=C_OLD, label='used before Aug 2026'),
                        Patch(color=C_NEW, label='new this month')],
               loc='upper left', fontsize=8)
    ax2.set_title(f'Drone mosaics ({len(stri)} STRI, {len(globus)} globus, '
                  f'{len(island)} whole-island) and training chips '
                  f'({len(old_chips)} + {len(new_chips)})')
    fig.tight_layout()
    save(fig, outdir, 'coverage')


# --------------------------------------------------------------------------
# ocm-example
# --------------------------------------------------------------------------
def read_rgb_preview(path):
    import rasterio
    with rasterio.open(path) as src:
        a = src.read([1, 2, 3]).astype(np.float32)
    valid = a.sum(axis=0) > 0
    out = np.zeros(a.shape[1:] + (3,), np.float32)
    for i in range(3):
        b = a[i]
        lo, hi = np.percentile(b[valid], (2, 98)) if valid.any() else (0, 1)
        out[..., i] = np.clip((b - lo) / (hi - lo + 1e-6), 0, 1)
    return out


@cli.command('ocm-example')
@out_option
@click.option('--scene', 'scenes', multiple=True,
              help='Scene ids to show; default picks two new 2026 scenes.')
def ocm_example(outdir, scenes):
    """New 2026 50ha scenes with their OmniCloudMask classes overlaid."""
    import rasterio
    ocm_dir = FLOWER / 'planet_clipped/ocm'
    rgb_dir = FLOWER / 'planet_clipped/rgb'
    inv = pd.read_csv(FLOWER / 'planet_clipped/inventory.csv')
    new = sorted(inv.loc[inv['status'] == 'ok', 'scene_id'])

    def ocm_path(s):
        return ocm_dir / s[:4] / f'{s}_ocm.tif'

    def clear_frac(lab):
        valid = lab != 255
        return float((lab[valid] == 0).mean()) if valid.any() else np.nan

    if not scenes:
        stats = []
        for s in new:
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
        click.echo(f'{len(df)} new fully-covered scenes with masks; '
                   f'showing {scenes}')

    fig, axes = plt.subplots(len(scenes), 2, figsize=(11, 3.3 * len(scenes)),
                             squeeze=False)
    for (a0, a1), s in zip(axes, scenes):
        rgb = read_rgb_preview(rgb_dir / s[:4] / f'{s}_rgb.tif')
        with rasterio.open(ocm_path(s)) as src:
            lab = src.read(1)
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
    cfg = yaml.safe_load(GLOBUS_SCALING.read_text())['uint16_to_uint8']
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


def globus_rgb_display(arr):
    gains, offsets = load_scaling()
    rgb = np.clip(arr[:3].astype(np.float64) * gains[:, None, None]
                  + offsets[:, None, None], 0, 255) / 255
    rgb = np.moveaxis(rgb, 0, -1)
    rgb[arr[:3].sum(axis=0) == 0] = 1
    return rgb


def extent(arr, transform):
    """imshow extent (left, right, bottom, top) in map units."""
    h, w = arr.shape[-2:]
    left, top = transform * (0, 0)
    right, bottom = transform * (w, h)
    return (left, right, bottom, top)


def plot_crowns(ax, crowns, **kw):
    crowns.boundary.plot(ax=ax, **kw)


@cli.command('classification-example')
@out_option
@click.option('--date', 'date', default=None,
              help='Globus date YYYY_MM_DD; default = date with the most '
                   'flowering crowns in classification_timeseries.csv.')
@click.option('--decimate', default=16, show_default=True)
@click.option('--zoom-m', default=120.0, show_default=True)
def classification_example(outdir, date, decimate, zoom_m):
    """One globus mosaic beside its flowering and deciduous probability maps."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from rasterio.windows import from_bounds

    if date is None:
        ts = pd.read_csv(outdir / 'classification_timeseries.csv')
        g = ts[ts['source'] == 'globus']
        date = pd.Timestamp(g.loc[g['flowering_frac'].idxmax(), 'date']).strftime('%Y_%m_%d')
    rgb_path = GLOBUS_RGB / f'BCI_50ha_{date}_M3M_aligned_global_RGB.tif'
    cls_path = CLASSIFICATIONS / f'BCI_50ha_{date}_M3M_aligned_global_RGB_classifications.tif'
    crowns = gpd.read_file(CROWNMAP)

    click.echo(f'reading {rgb_path.name} at 1/{decimate} (no overviews; slow)')
    rgb, t_rgb = read_decimated(rgb_path, decimate, [1, 2, 3])
    cls, t_cls = read_decimated(cls_path, decimate, [1, 2])

    # Zoom on the most confidently flowering crown that is far enough inside
    # the plot for the zoom window to be fully labelled.
    half = zoom_m / 2
    ids = rasterize(((g, i + 1) for i, g in enumerate(crowns.geometry)),
                    out_shape=cls.shape[1:], transform=t_cls, fill=0).ravel()
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
    titles = ['globus RGB (uint16 -> uint8 as classified)', 'P(flowering)',
              'P(deciduous)']
    for row, (img, t_img, c, t_c, lim, lw) in enumerate([
            (rgb, t_rgb, cls, t_cls, full_lim, 0.25),
            (zrgb, zt_rgb, zcls, zt_cls, zoom_lim, 1.2)]):
        axes[row, 0].imshow(globus_rgb_display(img), extent=extent(img, t_img))
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
    fig.suptitle(f'HPC crown classification of the {date.replace("_", "-")} globus '
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
    """Fraction of crowns flowering / deciduous per mosaic date, 2018-2026."""
    import geopandas as gpd
    from rasterio.features import rasterize

    crowns = gpd.read_file(CROWNMAP)
    n = len(crowns)
    sources = [('STRI local', sorted(CLASSIFICATIONS.glob('*_local_classifications.tif'))),
               ('globus', sorted(CLASSIFICATIONS.glob('*_M3M_aligned_global_RGB_classifications.tif')))]
    csv = outdir / 'classification_timeseries.csv'
    done = pd.read_csv(csv) if csv.exists() else pd.DataFrame(columns=['file'])
    rows = done.to_dict('records')
    seen = set(done['file'])
    with click.progressbar([(s, f) for s, fs in sources for f in fs],
                           label='scoring mosaics') as bar:
        for source, f in bar:
            if f.name in seen:
                continue
            arr, t = read_decimated(f, decimate, [1, 2])
            ids = rasterize(((g, i + 1) for i, g in enumerate(crowns.geometry)),
                            out_shape=arr.shape[1:], transform=t, fill=0).ravel()
            counts = np.bincount(ids, minlength=n + 1)[1:]
            rec = {'file': f.name, 'source': source, 'date': path_date(f)}
            scored = counts >= min_pixels
            rec['n_scored'] = int(scored.sum())
            for b, key in ((0, 'flowering'), (1, 'deciduous')):
                v = np.nan_to_num(arr[b].ravel())
                means = np.bincount(ids, weights=v, minlength=n + 1)[1:] / np.maximum(counts, 1)
                rec[f'{key}_frac'] = float((means[scored] > threshold).mean())
            rows.append(rec)
            pd.DataFrame(rows).to_csv(csv, index=False)

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    fig, axes = plt.subplots(2, 1, figsize=(13, 5.6), sharex=True)
    g0, g1 = map(pd.Timestamp, MOSAIC_GAP)
    for ax, key, color in ((axes[0], 'flowering', '#e377c2'),
                           (axes[1], 'deciduous', '#8c564b')):
        for source, mk, alpha in (('STRI local', 'o', 1.0), ('globus', 's', 1.0)):
            g = df[df['source'] == source]
            vals = 100 * g[f'{key}_frac']
            ax.plot(g['date'], vals, marker=mk, ms=3, color=color, lw=1,
                    alpha=alpha, label=f'{source} ({len(g)} dates)')
            ax.hlines(vals.median(), g['date'].min(), g['date'].max(),
                      color='k', ls='--', lw=1)
            ax.text(g['date'].max(), vals.median(), f' median {vals.median():.1f}%',
                    va='bottom', ha='right', fontsize=8)
        ax.axvspan(g0, g1, color=C_GREY, alpha=0.18, lw=0)
        ax.set_ylabel(f'% crowns {key}')
        ax.set_ylim(0, None)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='upper left', title=key, title_fontsize=8)
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[0].set_title(f'Crown-level phenology from drone classification: share of '
                      f'{int(df["n_scored"].median()):,} crowns (2022-09-29 map) '
                      f'with mean P > {threshold}')
    fig.tight_layout()
    save(fig, outdir, 'classification_timeseries')


# --------------------------------------------------------------------------
# coreg-stats
# --------------------------------------------------------------------------
@cli.command('coreg-stats')
@out_option
def coreg_stats(outdir):
    """Label-transfer yield and AROSICS shift magnitudes for both builds."""
    builds = []
    for name, chipdir, color, vet in (
            ('2020-23 build\n(STRI local align)', OLD_CHIPS, C_OLD, None),
            ('2024-26 build\n(globus global align)', GLOBUS_CHIPS, C_NEW,
             GLOBUS_CHIPS / 'vetting.json')):
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
                           shifts=np.hypot(ok['x_shift_m'], ok['y_shift_m'])))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.3),
                                   gridspec_kw={'width_ratios': [1.2, 1]})
    stages = ['drone-Planet\npairs (±2 d)', 'AROSICS\ncoregistered', 'vetted\nGood']
    x = np.arange(len(stages))
    w = 0.38
    for i, b in enumerate(builds):
        vals = [b['pairs'], b['ok'], b['good']]
        bars = ax1.bar(x + (i - 0.5) * w, vals, w, color=b['color'],
                       label=b['name'].replace('\n', ' '))
        for bar, v in zip(bars, vals):
            pct = '' if v == b['pairs'] else f'\n{v / b["pairs"]:.0%}'
            ax1.text(bar.get_x() + bar.get_width() / 2, v + 4, f'{v}{pct}',
                     ha='center', va='bottom', fontsize=8)
    ax1.set_xticks(x, stages)
    ax1.set_ylabel('count')
    ax1.set_ylim(0, max(b['pairs'] for b in builds) * 1.2)
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
                   f"{b['good']} Good / {b['fair']} Fair, median shift "
                   f"{np.median(b['shifts']):.2f} m")


# --------------------------------------------------------------------------
# maskrcnn-summary
# --------------------------------------------------------------------------
@cli.command('maskrcnn-summary')
@out_option
@click.option('--sweep', type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=DEFAULT_OUT / 'headtohead_sweep.json', show_default=True)
def maskrcnn_summary(outdir, sweep):
    """Single-panel mAP@50 vs epoch, extended vs base, for a slide."""
    blob = json.loads(sweep.read_text())
    runs = {}
    for label, row in blob['results'].items():
        run, ep = label.rsplit('_e', 1)
        runs.setdefault(run, []).append((int(ep), row['test_map/map_50']))
    names = {'base': ('2020-23 chips only (56)', C_OLD),
             'ext': ('+ 2024-26 globus chips (126)', C_NEW)}
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    curves = {}
    for run, pts in runs.items():
        pts.sort()
        e, m = map(np.array, zip(*pts))
        curves[run] = (e, m)
        label, color = names.get(run, (run, None))
        ax.plot(e, m, marker='o', ms=4, color=color, lw=2, label=label)
        k = int(np.argmax(m))
        ax.plot(e[k], m[k], marker='*', ms=15, color=color)
        ax.annotate(f'{m[k]:.3f} @ ep {e[k]}', (e[k], m[k]),
                    textcoords='offset points', xytext=(6, 8), fontsize=9,
                    color=color, fontweight='bold')
    if {'base', 'ext'} <= curves.keys():
        e, mb = curves['base']
        _, me = curves['ext']
        ax.fill_between(e, mb, me, where=me > mb, color=C_NEW, alpha=0.12,
                        interpolate=True, lw=0)
        ax.fill_between(e, mb, me, where=me <= mb, color=C_OLD, alpha=0.12,
                        interpolate=True, lw=0)
    ax.set_xscale('log')
    ax.set_xlabel('training epoch')
    ax.set_ylabel('mask mAP@50')
    ax.set_ylim(0, max(m.max() for _, m in curves.values()) * 1.18)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_title(f"Mask R-CNN on the same {blob['n_chips']} 2020-23 test chips")
    fig.tight_layout()
    save(fig, outdir, 'maskrcnn_summary')


if __name__ == '__main__':
    cli()
