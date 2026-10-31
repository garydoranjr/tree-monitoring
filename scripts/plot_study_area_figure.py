#!/usr/bin/env python
"""Study-area context figure for the November 2026 presentation.

Panama -> Barro Colorado Nature Monument / BCI -> the 50ha plot in drone and
Planet imagery -> one flowering crown seen by both sensors (before / during).
Outputs go to OUTDIR (default /Volumes/Earth03/flower/figs/20261103_presentation).

Subcommands:

  candidates  Rank large crowns that the drone classifier marks as flowering
              (results/crown_classifications{,_mavic}.nc4). For each, pick the
              peak-flowering flight, a non-flowering baseline flight 3-10 weeks
              earlier, and the nearest Planet scene to each flight that is clear
              over the crown. Score the Planet colour change of the crown
              against its surroundings. Writes materials/crown_candidates.csv
              and a contact sheet, materials/crown_candidates.png.
  figure      The 12.5 x 6 in study-area figure for one chosen crown
              (materials/selected_crown.json, written by `candidates --select`).

Planet scenes are moved onto the drone frame with the per-scene AROSICS shift
in crown_series_20261029/coreg_all_scenes.json (the shift takes drone-frame
geometry onto the Planet grid, as in build_crown_pixel_series.py, so the image
is displaced by minus the shift).

Typical usage:
    python scripts/plot_study_area_figure.py candidates
    python scripts/plot_study_area_figure.py candidates --select 1246:mavic
    python scripts/plot_study_area_figure.py figure
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import functools
import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_update_figures as pu  # noqa: E402

FLOWER = pu.FLOWER
DEFAULT_OUT = FLOWER / 'figs' / '20261103_presentation'
NC4 = {'phantom': FLOWER / 'results/crown_classifications.nc4',
       'mavic': FLOWER / 'results/crown_classifications_mavic.nc4'}
PLANET_SR = FLOWER / 'planet_clipped_sr/4band'
PLANET_OCM = FLOWER / 'planet_clipped/ocm'
COREG_JSON = FLOWER / 'crown_series_20261029/coreg_all_scenes.json'
BCNM_SR_INDEX = FLOWER / 'planet_bcnm_sr/index.csv'
BCNM_OCM = FLOWER / 'planet_bcnm_sr/ocm'
ISLAND = FLOWER / 'whole_island/Barro_Colorado_Island.geojson'
BCNM_LAND = FLOWER / 'whole_island/Barro_Colorado_Nature_Monument_Land_1118504159915377444.geojson'
PLOT_50HA = FLOWER / 'stri/24782016/aux_files/BCI_Plot_50ha.shp'
UTM = 'EPSG:32617'

MIN_AREA_M2 = 250
MIN_PEAK = 0.6
PLANET_START = pd.Timestamp('2020-01-01')


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
@functools.cache
def crownmap():
    import geopandas as gpd
    c = gpd.read_file(pu.CROWNMAP).to_crs(UTM)
    c = c[pd.to_numeric(c.tag, errors='coerce').notna()]
    c['tag'] = c.tag.astype(int)
    return c.drop_duplicates('tag').set_index('tag')


@functools.cache
def coreg_shifts():
    """{scene: (x_shift_m, y_shift_m)} for scenes with a successful shift."""
    out = {}
    for rec in json.loads(COREG_JSON.read_text()):
        if rec['coreg_ok']:
            out.setdefault(rec['scene'], (rec['x_shift_m'], rec['y_shift_m']))
    return out


@functools.cache
def planet_scenes():
    """50ha SR scenes with an OCM mask: DataFrame of scene, date, paths."""
    rows = []
    for p in sorted(PLANET_SR.glob('*/*_4band.tif')):
        scene = p.name[:-len('.tif')]
        ocm = PLANET_OCM / p.parent.name / p.name.replace('_4band.tif', '_ocm.tif')
        if ocm.exists():
            rows.append({'scene': scene, 'date': pd.Timestamp(p.name[:8]),
                         'sr': str(p), 'ocm': str(ocm)})
    return pd.DataFrame(rows)


def drone_file(series, date):
    d = pd.Timestamp(date).strftime('%Y_%m_%d')
    if series == 'mavic':
        return pu.MAVIC_RGB / f'BCI_50ha_{d}_M3M_aligned_global_RGB.tif'
    release = 'C3KW2X' if pd.Timestamp(date) in pu.c3kw2x_dates() else '24782016'
    return pu.phantom_dir(release, 'global') / f'BCI_50ha_{d}_global.tif'


def read_window(path, bounds, bands, factor=1, boundless=True):
    """Read map `bounds` from `path`; returns (array, imshow extent)."""
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(path) as src:
        win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        shape = (len(bands), max(1, int(win.height // factor)), max(1, int(win.width // factor)))
        arr = src.read(bands, window=win, out_shape=shape, boundless=boundless, fill_value=0)
        t = src.window_transform(win)
    left, top = t * (0, 0)
    right, bottom = t * (win.width, win.height)
    return arr, (left, right, bottom, top)


def shifted_bounds(bounds, shift):
    """Bounds in the Planet frame for drone-frame `bounds`."""
    dx, dy = shift
    return (bounds[0] + dx, bounds[1] + dy, bounds[2] + dx, bounds[3] + dy)


def unshift_extent(ext, shift):
    """imshow extent of a Planet read, moved back onto the drone frame."""
    dx, dy = shift
    return (ext[0] - dx, ext[1] - dx, ext[2] - dy, ext[3] - dy)


def planet_rgb(arr, lo, hi, gamma=0.8):
    """B,G,R(,NIR) SR -> display RGB with a fixed per-band (lo, hi) stretch."""
    rgb = np.stack([arr[2], arr[1], arr[0]], axis=-1).astype(np.float32)
    out = np.clip((rgb - lo) / (hi - lo), 0, 1) ** gamma
    out[rgb.sum(axis=-1) == 0] = 1
    return out


def stretch_limits(arrs, pct=(2, 98)):
    """Per-band (R, G, B) percentiles over the valid pixels of B,G,R arrays."""
    stack = np.concatenate([a[[2, 1, 0]].reshape(3, -1) for a in arrs], axis=1).astype(np.float32)
    stack = stack[:, stack.sum(axis=0) > 0]
    lo, hi = np.percentile(stack, pct, axis=1)
    return lo, hi


@functools.cache
def scene_stretch(path, pct=(0.5, 99.8)):
    """Per-band (R, G, B) stretch over the OCM-clear pixels of a 50ha scene."""
    import rasterio
    ocm = PLANET_OCM / Path(path).parent.name / Path(path).name.replace('_4band.tif', '_ocm.tif')
    with rasterio.open(path) as src:
        a = src.read([3, 2, 1]).astype(np.float32)
    with rasterio.open(ocm) as src:
        clear = src.read(1) == 0
    valid = a.sum(axis=0) > 0
    if clear.shape == valid.shape and (clear & valid).sum() > 1000:
        valid &= clear
    lo, hi = np.percentile(a[:, valid], pct, axis=1)
    return lo, hi


def crown_bounds(geom, pad_frac=0.3, min_half=0.0):
    x0, y0, x1, y1 = geom.bounds
    half = max(max(x1 - x0, y1 - y0) * (0.5 + pad_frac), min_half)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (cx - half, cy - half, cx + half, cy + half)


# --------------------------------------------------------------------------
# candidates
# --------------------------------------------------------------------------
def flowering_events(series):
    """Peak-flowering flight and a baseline flight per large flowering crown."""
    import xarray as xr
    ds = xr.open_dataset(NC4[series])
    ds = ds.sel(date=ds.date >= PLANET_START)
    cm = crownmap()
    fp = ds.flowering_probability.to_pandas()
    dp = ds.deciduous_probability.to_pandas()
    dates = pd.DatetimeIndex(fp.columns)
    rows = []
    for tag in fp.index:
        if tag not in cm.index or cm.geometry[tag].area < MIN_AREA_M2:
            continue
        f = fp.loc[tag]
        if not (f.max() >= MIN_PEAK):
            continue
        ev = f.idxmax()
        lag = (ev - dates).days
        ok = (lag >= 20) & (lag <= 70) & (f.values < 0.15) & (dp.loc[tag].values < 0.3)
        if not ok.any():
            continue
        cand = dates[ok]
        base = cand[np.argmin(np.abs((ev - cand).days - 35))]
        rows.append({'tag': int(tag), 'series': series, 'species': str(ds.species.sel(tag=tag).values),
                     'area_m2': cm.geometry[tag].area, 'event_date': ev, 'p_flower': f[ev],
                     'base_date': base, 'p_flower_base': f[base]})
    return pd.DataFrame(rows)


def crown_masks(geom, path, shift):
    """Crown and surrounding-ring pixel masks on a Planet raster's grid."""
    import rasterio
    from rasterio.features import geometry_mask
    from shapely.affinity import translate
    g = translate(geom, *shift)
    with rasterio.open(path) as src:
        crown = ~geometry_mask([g], src.shape, src.transform, all_touched=False)
        ring = ~geometry_mask([g.buffer(25).difference(g.buffer(4))], src.shape, src.transform)
    return crown, ring


def best_scene(geom, date, max_days=4):
    """Nearest Planet scene to `date` that is clear over the crown and its ring."""
    import rasterio
    sc = planet_scenes()
    sc = sc[(sc.date - date).abs() <= pd.Timedelta(days=max_days)].copy()
    if sc.empty:
        return None
    sc['days'] = (sc.date - date).dt.days
    sc['coreg'] = sc.scene.isin(coreg_shifts())
    sc = sc.sort_values(['days'], key=lambda s: s.abs())
    for r in sc.itertuples():
        shift = coreg_shifts().get(r.scene, (0.0, 0.0))
        crown, ring = crown_masks(geom, r.ocm, shift)
        with rasterio.open(r.ocm) as src:
            cls = src.read(1)
        if cls.shape != crown.shape or not crown.any():
            continue
        if (cls[crown] == 0).mean() == 1 and (cls[ring] == 0).mean() > 0.95:
            return {'scene': r.scene, 'sr': r.sr, 'days': int(r.days),
                    'coreg': bool(r.coreg), 'shift': shift}
    return None


def colour_change(geom, base, event):
    """Crown RGB change minus ring change, in units of ring change spread."""
    import rasterio
    vals = {}
    for k, s in (('base', base), ('event', event)):
        crown, ring = crown_masks(geom, s['sr'], s['shift'])
        with rasterio.open(s['sr']) as src:
            a = src.read([3, 2, 1]).astype(np.float32)
        vals[k] = (a[:, crown].mean(axis=1), a[:, ring].mean(axis=1), a[:, ring].std(axis=1))
    d = (vals['event'][0] - vals['base'][0]) - (vals['event'][1] - vals['base'][1])
    spread = (vals['event'][2] + vals['base'][2]) / 2
    return float(np.linalg.norm(d / spread)), d


def candidate_table():
    cm = crownmap()
    ev = pd.concat([flowering_events(s) for s in ('phantom', 'mavic')], ignore_index=True)
    click.echo(f'{len(ev)} crown events with a baseline flight')
    rows = []
    with click.progressbar(list(ev.itertuples()), label='matching Planet scenes') as bar:
        for r in bar:
            geom = cm.geometry[r.tag]
            b = best_scene(geom, r.base_date)
            e = best_scene(geom, r.event_date)
            if b is None or e is None:
                continue
            score, d = colour_change(geom, b, e)
            rows.append({**r._asdict(), 'base_scene': b['scene'], 'base_days': b['days'],
                         'base_coreg': b['coreg'], 'event_scene': e['scene'],
                         'event_days': e['days'], 'event_coreg': e['coreg'],
                         'planet_score': score, 'dR': d[0], 'dG': d[1], 'dB': d[2]})
    df = pd.DataFrame(rows).drop(columns='Index')
    return df.sort_values('planet_score', ascending=False)


def scene_info(scene):
    sc = planet_scenes().set_index('scene')
    return {'scene': scene, 'sr': sc.sr[scene], 'shift': coreg_shifts().get(scene, (0.0, 0.0))}


def crown_tiles(row, pad_frac=0.35):
    """Drone before/during and Planet before/during display tiles for one row."""
    geom = crownmap().geometry[row['tag']]
    bounds = crown_bounds(geom, pad_frac, min_half=22)
    tiles = []
    for key in ('base', 'event'):
        arr, ext = read_window(drone_file(row['series'], row[f'{key}_date']), bounds, [1, 2, 3, 4])
        tiles.append(('drone', key, pu.rgb_display(arr), ext))
    planet = []
    for key in ('base', 'event'):
        s = scene_info(row[f'{key}_scene'])
        arr, ext = read_window(s['sr'], shifted_bounds(bounds, s['shift']), [1, 2, 3])
        planet.append((key, arr, unshift_extent(ext, s['shift'])))
    for key, arr, ext in planet:
        lo, hi = scene_stretch(scene_info(row[f'{key}_scene'])['sr'])
        tiles.append(('planet', key, planet_rgb(arr, lo, hi), ext))
    return geom, bounds, tiles


def contact_sheet(df, path, n):
    import geopandas as gpd
    df = df.head(n)
    fig, axes = plt.subplots(len(df), 4, figsize=(8, 2.1 * len(df)), squeeze=False)
    for ax_row, (_, row) in zip(axes, df.iterrows()):
        geom, bounds, tiles = crown_tiles(row)
        for ax, (sensor, key, img, ext) in zip(ax_row, tiles):
            ax.imshow(img, extent=ext, interpolation='nearest')
            gpd.GeoSeries([geom]).boundary.plot(ax=ax, color='cyan', lw=0.8)
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
            ax.set_xticks([])
            ax.set_yticks([])
            if sensor == 'drone':
                d = row[f'{key}_date']
            else:
                d = pd.Timestamp(row[f'{key}_scene'][:8])
            ax.set_title(f'{sensor} {key} {pd.Timestamp(d):%Y-%m-%d}', fontsize=7)
        ax_row[0].set_ylabel(f"{row['tag']} {row['series']}\n{row['species']}\n"
                             f"{row['area_m2']:.0f} m², score {row['planet_score']:.1f}", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    click.echo(f'Wrote {path}')


@click.group()
def cli():
    pass


out_option = click.option('--outdir', type=click.Path(file_okay=False, path_type=Path),
                          default=DEFAULT_OUT, show_default=True)


@cli.command()
@out_option
@click.option('--n', default=20, show_default=True, help='Rows on the contact sheet.')
@click.option('--select', default=None,
              help='TAG:SERIES[:EVENT_DATE] to record in materials/selected_crown.json '
                   'from crown_candidates.csv instead of rebuilding the table.')
@click.option('--base', default=None,
              help='With --select, DRONE_DATE:PLANET_SCENE to use as the baseline instead '
                   'of the ranked pair, e.g. when the ranked Planet scene has no coreg shift.')
@click.option('--event', default=None,
              help='With --select, DRONE_DATE:PLANET_SCENE to use as the flowering pair '
                   'instead of the ranked one, e.g. for a scene clear over the whole plot.')
def candidates(outdir, n, select, base, event):
    """Rank flowering crowns by how visible the event is in Planet."""
    mat = outdir / 'materials'
    mat.mkdir(parents=True, exist_ok=True)
    csv = mat / 'crown_candidates.csv'
    if select:
        df = pd.read_csv(csv, parse_dates=['event_date', 'base_date'])
        tag, series, *rest = select.split(':')
        m = (df.tag == int(tag)) & (df.series == series)
        if rest:
            m &= df.event_date == pd.Timestamp(rest[0])
        row = {k: (v.item() if hasattr(v, 'item') else v)
               for k, v in df[m].iloc[0].to_dict().items()}
        for key, override in (('base', base), ('event', event)):
            if not override:
                continue
            d, scene = override.split(':')
            row.update({f'{key}_date': pd.Timestamp(d), f'{key}_scene': scene,
                        f'{key}_days': (pd.Timestamp(scene[:8]) - pd.Timestamp(d)).days,
                        f'{key}_coreg': scene in coreg_shifts()})
            import xarray as xr
            with xr.open_dataset(NC4[series]) as ds:
                row['p_flower' + ('_base' if key == 'base' else '')] = float(
                    ds.flowering_probability.sel(tag=int(tag), date=pd.Timestamp(d)))
        if base or event:
            # the ranking score was for the ranked pair, not the override
            for k in ('planet_score', 'dR', 'dG', 'dB'):
                row.pop(k, None)
        for k in ('event_date', 'base_date'):
            row[k] = f'{row[k]:%Y-%m-%d}'
        for k in ('base', 'event'):
            row[f'{k}_drone_file'] = str(drone_file(series, row[f'{k}_date']))
            s = scene_info(row[f'{k}_scene'])
            row[f'{k}_planet_file'] = s['sr']
            row[f'{k}_shift_m'] = list(s['shift'])
        (mat / 'selected_crown.json').write_text(json.dumps(row, indent=1, default=str))
        click.echo(f'Wrote {mat / "selected_crown.json"}')
        return
    df = candidate_table()
    df.to_csv(csv, index=False)
    click.echo(f'Wrote {csv} ({len(df)} rows)')
    contact_sheet(df, mat / 'crown_candidates.png', n)


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
NE_DIR = Path.home() / '.local/share/cartopy/shapefiles/natural_earth'
NE_ADMIN1 = NE_DIR / 'cultural/ne_10m_admin_1_states_provinces_lakes.shp'
DATED_CROWNS = FLOWER / 'crowns_20240803/BCI_50ha_crownmap_timeseries.shp'
BCNM_SCENE = '20210305_155108_65_227a'
PANAMA_LONLAT = (-83.2, 6.6, -77.0, 10.4)
PLOT_VIEW = (625745, 1011715, 626818, 1012304)
# Panel (b) view: the monument around BCI, trimmed east-west so the island
# stays large in a tall panel.
BCNM_VIEW = (623500, 1005600, 630500, 1018000)
CITIES = {'Panama City': (-79.52, 8.98), 'Colón': (-79.90, 9.36)}

FIG_W, FIG_H = 12.5, 6.0
C_BOX = '#ffd400'
C_CROWN = '#00e5ff'
WATER = '#cfe2f3'
FS_TITLE, FS_LABEL, FS_SMALL = 10.5, 8.5, 7.5


def halo(lw=2.0, color='black'):
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=lw, foreground=color)]


def add_scalebar(ax, length_m, label, loc=(0.05, 0.06), color='white', fontsize=FS_SMALL,
                 lw=3):
    """Horizontal bar of length_m map units, lower-left at axes fraction loc."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bx = x0 + loc[0] * (x1 - x0)
    by = y0 + loc[1] * (y1 - y0)
    edge = 'black' if color == 'white' else 'white'
    ax.plot([bx, bx + length_m], [by, by], color=color, lw=lw, solid_capstyle='butt',
            path_effects=halo(lw + 1.5, edge), zorder=20)
    ax.text(bx + length_m / 2, by + 0.025 * (y1 - y0), label, ha='center', va='bottom',
            color=color, fontsize=fontsize, path_effects=halo(1.8, edge), zorder=20)


def add_north(ax, loc=(0.92, 0.80), color='white', size=0.10):
    edge = 'black' if color == 'white' else 'white'
    ax.annotate('N', xy=(loc[0], loc[1] + size), xytext=(loc[0], loc[1]),
                xycoords='axes fraction', ha='center', va='top', fontsize=FS_LABEL,
                fontweight='bold', color=color, path_effects=halo(1.8, edge),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=1.5), zorder=20)


def panel_title(ax, letter, text, color='black'):
    ax.set_title(f'({letter}) {text}', fontsize=FS_TITLE, loc='left', pad=3, color=color)


def bare(ax):
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)


def box_patch(bounds, **kw):
    from matplotlib.patches import Rectangle
    return Rectangle((bounds[0], bounds[1]), bounds[2] - bounds[0], bounds[3] - bounds[1],
                     fill=False, **kw)


def connect(fig, ax_from, xy_from, ax_to, xy_to, coords_to='axes fraction', **kw):
    from matplotlib.patches import ConnectionPatch
    style = dict(color='0.35', lw=0.8, zorder=30)
    style.update(kw)
    fig.add_artist(ConnectionPatch(xyA=xy_from, coordsA=ax_from.transData, xyB=xy_to,
                                   coordsB=coords_to if coords_to != 'data' else ax_to.transData,
                                   axesB=ax_to, **style))


def dated_crown(tag, date):
    """The crown outline of `tag` drawn for drone `date` (crowns_20240803)."""
    import geopandas as gpd
    static = crownmap().geometry[tag]
    g = gpd.read_file(DATED_CROWNS, bbox=static.bounds)
    g = g[(pd.to_numeric(g.tag, errors='coerce') == tag)
          & (g.date == pd.Timestamp(date).strftime('%Y_%m_%d'))]
    return g.geometry.iloc[0] if len(g) else static


def utm(gdf):
    return gdf.to_crs(UTM)


def panama_view():
    import geopandas as gpd
    from shapely.geometry import box
    return utm(gpd.GeoSeries([box(*PANAMA_LONLAT)], crs=4326)).total_bounds


def draw_panama(ax, bcnm_bounds):
    import geopandas as gpd
    lon0, lat0, lon1, lat1 = PANAMA_LONLAT
    view = panama_view()
    adm = gpd.read_file(NE_ADMIN1, bbox=(lon0 - 2, lat0 - 2, lon1 + 2, lat1 + 2))
    countries = utm(adm.dissolve('admin').reset_index())
    ax.set_facecolor(WATER)
    for name, g in countries.groupby('admin'):
        fc = '#c7dcb0' if name == 'Panama' else '#ececec'
        g.plot(ax=ax, facecolor=fc, edgecolor='0.45', lw=0.4)
    ax.set_xlim(view[0], view[2])
    ax.set_ylim(view[1], view[3])
    ax.set_aspect('equal')
    for name, (lon, lat) in CITIES.items():
        x, y = utm(gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs=4326)).iloc[0].coords[0]
        ax.plot(x, y, 'o', ms=3, color='0.15', zorder=15)
        dx = 25e3 if name == 'Panama City' else -18e3
        ax.text(x + dx, y + (-14e3 if name == 'Panama City' else 14e3), name,
                fontsize=FS_SMALL, ha='left' if dx > 0 else 'right', color='0.15', zorder=15)
    labels = [('Caribbean Sea', -80.9, 9.75, 'italic', '#3a6d99'),
              ('Pacific Ocean', -80.6, 7.25, 'italic', '#3a6d99'),
              ('PANAMA', -81.0, 8.45, 'normal', '0.2'),
              ('COSTA\nRICA', -83.05, 9.35, 'normal', '0.4'),
              ('COLOMBIA', -77.1, 7.35, 'normal', '0.4')]
    for text, lon, lat, style, c in labels:
        x, y = utm(gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs=4326)).iloc[0].coords[0]
        ax.text(x, y, text, fontsize=FS_SMALL, style=style, color=c, va='center',
                ha='right' if text == 'COLOMBIA' else 'left',
                fontweight='bold' if text == 'PANAMA' else 'normal')
    ax.add_patch(box_patch(bcnm_bounds, edgecolor='#d62728', lw=1.4, zorder=16))
    bare(ax)
    add_scalebar(ax, 100e3, '100 km', loc=(0.05, 0.07), color='black')
    add_north(ax, loc=(0.93, 0.68), color='black', size=0.2)


def bcnm_image(view, factor=4):
    import rasterio
    from rasterio.enums import Resampling
    path = FLOWER / 'planet_bcnm_sr/4band' / BCNM_SCENE[:4] / f'{BCNM_SCENE}_4band.tif'
    with rasterio.open(path) as src:
        a = src.read([3, 2, 1], out_shape=(3, src.height // factor, src.width // factor),
                     resampling=Resampling.average).astype(np.float32)
        b = src.bounds
    valid = a.sum(axis=0) > 0
    lo, hi = np.percentile(a[:, valid], (1, 99.5), axis=1)
    rgb = np.clip((np.moveaxis(a, 0, -1) - lo) / (hi - lo), 0, 1) ** 0.8
    rgb[~valid] = 1
    return rgb, (b.left, b.right, b.bottom, b.top)


def plot_outline():
    import geopandas as gpd
    return utm(gpd.read_file(PLOT_50HA)).geometry.iloc[0]


def draw_bcnm(ax, view):
    import geopandas as gpd
    rgb, ext = bcnm_image(view)
    ax.imshow(rgb, extent=ext, interpolation='antialiased')
    utm(gpd.read_file(BCNM_LAND)).boundary.plot(ax=ax, color='white', lw=0.5, ls='--',
                                                alpha=0.8)
    island = utm(gpd.read_file(ISLAND))
    island.boundary.plot(ax=ax, color=C_BOX, lw=0.9)
    gpd.GeoSeries([plot_outline()]).boundary.plot(ax=ax, color='white', lw=0.8)
    ax.add_patch(box_patch(PLOT_VIEW, edgecolor='#d62728', lw=1.4, zorder=16))
    ax.set_xlim(view[0], view[2])
    ax.set_ylim(view[1], view[3])
    ax.set_aspect('equal')
    cx = island.geometry.iloc[0].centroid
    ax.text(cx.x - 1300, cx.y + 900, 'Barro Colorado\nIsland', color='white', fontsize=FS_LABEL,
            ha='center', va='center', fontweight='bold', path_effects=halo(2))
    ax.text(PLOT_VIEW[0] - 200, PLOT_VIEW[1] - 200, '50 ha plot', color='white',
            fontsize=FS_SMALL, ha='right', va='top', path_effects=halo(1.8))
    ax.text(629750, 1009900, 'Gatún\nLake', color='white', fontsize=FS_LABEL, style='italic',
            ha='center', va='center', path_effects=halo(2))
    bare(ax)
    add_scalebar(ax, 2000, '2 km', loc=(0.06, 0.05))
    add_north(ax, loc=(0.08, 0.84), size=0.09)


def draw_plot_drone(ax, sel, crown_box):
    import geopandas as gpd
    import rasterio
    path = sel['event_drone_file']
    with rasterio.open(path) as src:
        factor = max(1, int(round(1.0 / src.res[0])))
    arr, ext = read_window(path, PLOT_VIEW, [1, 2, 3, 4], factor=factor)
    ax.imshow(pu.rgb_display(arr), extent=ext, interpolation='antialiased')
    gpd.GeoSeries([plot_outline()]).boundary.plot(ax=ax, color='white', lw=0.8)
    ax.add_patch(box_patch(crown_box, edgecolor=C_BOX, lw=1.3, zorder=16))
    ax.set_xlim(PLOT_VIEW[0], PLOT_VIEW[2])
    ax.set_ylim(PLOT_VIEW[1], PLOT_VIEW[3])
    ax.set_aspect('equal')
    bare(ax)
    add_scalebar(ax, 200, '200 m', loc=(0.05, 0.07))


def draw_plot_planet(ax, sel, crown_box):
    import geopandas as gpd
    shift = sel['event_shift_m']
    arr, ext = read_window(sel['event_planet_file'], shifted_bounds(PLOT_VIEW, shift), [1, 2, 3])
    lo, hi = scene_stretch(sel['event_planet_file'])
    ax.imshow(planet_rgb(arr, lo, hi), extent=unshift_extent(ext, shift),
              interpolation='nearest')
    gpd.GeoSeries([plot_outline()]).boundary.plot(ax=ax, color='white', lw=0.8)
    ax.add_patch(box_patch(crown_box, edgecolor=C_BOX, lw=1.3, zorder=16))
    ax.set_xlim(PLOT_VIEW[0], PLOT_VIEW[2])
    ax.set_ylim(PLOT_VIEW[1], PLOT_VIEW[3])
    ax.set_aspect('equal')
    bare(ax)
    add_scalebar(ax, 200, '200 m', loc=(0.05, 0.07))


def draw_tile(ax, img, ext, outline, bounds, label, nearest, scalebar):
    import geopandas as gpd
    ax.imshow(img, extent=ext, interpolation='nearest' if nearest else 'antialiased')
    gpd.GeoSeries([outline]).boundary.plot(ax=ax, color=C_CROWN, lw=0.9)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect('equal')
    bare(ax)
    for s in ax.spines.values():
        s.set_edgecolor(C_BOX)
        s.set_linewidth(1.3)
    ax.text(0.5, -0.04, label, transform=ax.transAxes, ha='center', va='top',
            fontsize=FS_SMALL)
    if scalebar:
        add_scalebar(ax, 20, '20 m', loc=(0.07, 0.07), lw=2.5)


@cli.command()
@out_option
@click.option('--name', default='study_area_context', show_default=True)
@click.option('--dpi', default=300, show_default=True)
def figure(outdir, name, dpi):
    """The 12.5 x 6 in study-area context figure."""
    import geopandas as gpd
    mat = outdir / 'materials'
    sel = json.loads((mat / 'selected_crown.json').read_text())
    tag = int(sel['tag'])

    view = BCNM_VIEW
    outlines = {k: dated_crown(tag, sel[f'{k}_date']) for k in ('base', 'event')}
    crown_box = crown_bounds(outlines['event'], 0.35, min_half=24)

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    def axes(x, y, w, h):
        return fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H])

    # column 1: Panama, centred vertically
    pan_w = 2.8
    pv = panama_view()
    pan_h = pan_w * (pv[3] - pv[1]) / (pv[2] - pv[0])
    ax_pan = axes(0.06, (FIG_H - pan_h) / 2, pan_w, pan_h)
    # column 2: BCNM
    bw = 2.75
    bh = bw * (view[3] - view[1]) / (view[2] - view[0])
    ax_b = axes(3.12, (FIG_H - bh) / 2 - 0.1, bw, bh)
    # columns 3-4: 50 ha plot and crown tiles
    x0 = 6.2
    gap = 0.18
    cw = (FIG_W - 0.06 - x0 - gap) / 2
    ch = cw * (PLOT_VIEW[3] - PLOT_VIEW[1]) / (PLOT_VIEW[2] - PLOT_VIEW[0])
    tgap = 0.1
    tile = (cw - tgap) / 2
    # centre the block (titles above c/d; date labels and the crown caption
    # below the tiles)
    block = 0.3 + ch + 0.9 + tile + 0.55
    ytop = FIG_H - (FIG_H - block) / 2 - 0.3 - ch
    ax_c = axes(x0, ytop, cw, ch)
    ax_d = axes(x0 + cw + gap, ytop, cw, ch)
    ytile = ytop - 0.9 - tile
    tiles = {('drone', 'base'): axes(x0, ytile, tile, tile),
             ('drone', 'event'): axes(x0 + tile + tgap, ytile, tile, tile),
             ('planet', 'base'): axes(x0 + cw + gap, ytile, tile, tile),
             ('planet', 'event'): axes(x0 + cw + gap + tile + tgap, ytile, tile, tile)}

    click.echo('drawing Panama')
    draw_panama(ax_pan, view)
    panel_title(ax_pan, 'a', 'Panama')
    click.echo('drawing BCNM')
    draw_bcnm(ax_b, view)
    panel_title(ax_b, 'b', 'Barro Colorado Nature Monument')
    click.echo('drawing 50 ha plot')
    draw_plot_drone(ax_c, sel, crown_box)
    panel_title(ax_c, 'c', f"50 ha plot, drone {sel['event_date']}")
    draw_plot_planet(ax_d, sel, crown_box)
    panel_title(ax_d, 'd', f"PlanetScope {pd.Timestamp(sel['event_scene'][:8]):%Y-%m-%d}")

    click.echo('drawing crown tiles')
    for key in ('base', 'event'):
        arr, ext = read_window(sel[f'{key}_drone_file'], crown_box, [1, 2, 3, 4])
        word = 'before' if key == 'base' else 'flowering'
        draw_tile(tiles[('drone', key)], pu.rgb_display(arr), ext, outlines[key], crown_box,
                  f"{sel[f'{key}_date']} ({word})", False, key == 'base')
        shift = sel[f'{key}_shift_m']
        arr, ext = read_window(sel[f'{key}_planet_file'], shifted_bounds(crown_box, shift),
                               [1, 2, 3])
        lo, hi = scene_stretch(sel[f'{key}_planet_file'])
        d = pd.Timestamp(sel[f'{key}_scene'][:8])
        draw_tile(tiles[('planet', key)], planet_rgb(arr, lo, hi), unshift_extent(ext, shift),
                  outlines[key], crown_box, f'{d:%Y-%m-%d} ({word})', True, key == 'base')
    species = sel['species']
    # (e)/(f) centred over each tile pair, between the zoom connectors that
    # meet the pair's outer corners
    for xpair, letter, sensor_name in ((x0, 'e', 'drone'), (x0 + cw + gap, 'f', 'PlanetScope')):
        fig.text((xpair + cw / 2) / FIG_W, (ytile + tile + 0.05) / FIG_H,
                 f'({letter}) {sensor_name}', ha='center', va='bottom', fontsize=FS_TITLE)
    # crown caption below the tiles' date labels, clear of the connectors
    fig.text((x0 + cw + gap / 2) / FIG_W, (ytile - 0.3) / FIG_H,
             f'One $\\it{{{species.split()[0]}}}$ $\\it{{{species.split()[1]}}}$ crown '
             f'({sel["area_m2"]:.0f} m²), before and during flowering',
             ha='center', va='top', fontsize=FS_LABEL)

    # zoom connectors
    connect(fig, ax_pan, (view[2], view[3]), ax_b, (0, 1))
    connect(fig, ax_pan, (view[2], view[1]), ax_b, (0, 0))
    connect(fig, ax_b, (PLOT_VIEW[2], PLOT_VIEW[3]), ax_c, (0, 1), color='#d62728')
    connect(fig, ax_b, (PLOT_VIEW[2], PLOT_VIEW[1]), ax_c, (0, 0), color='#d62728')
    for sensor, ax_src in (('drone', ax_c), ('planet', ax_d)):
        connect(fig, ax_src, (crown_box[0], crown_box[1]), tiles[(sensor, 'base')], (0, 1),
                color=C_BOX, lw=0.9)
        connect(fig, ax_src, (crown_box[2], crown_box[1]), tiles[(sensor, 'event')], (1, 1),
                color=C_BOX, lw=0.9)

    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'{name}.{ext}', dpi=dpi)
    plt.close(fig)
    click.echo(f'Wrote {outdir / name}.png/.pdf')


# --------------------------------------------------------------------------
# drone-cards
# --------------------------------------------------------------------------
# Wikimedia Commons photos (CC BY-SA 4.0), downloaded to materials/drone_photos.
# crop = (x0, x1, y_centre) as fractions of the photo; the height follows from
# the 3:2 card aspect.
DRONE_CARDS = {
    'phantom': dict(
        name='DJI Phantom 4 Pro', file='phantom4pro_03.jpg', crop=(0.05, 0.97, 0.47),
        credit='Photo: A.BourgeoisP, CC BY-SA 4.0, via Wikimedia Commons',
        source='https://commons.wikimedia.org/wiki/File:DJI_Phantom_4_Pro_-_03.jpg'),
    'mavic': dict(
        name='DJI Mavic 3 Multispectral', file='m3m_FA373WYPN9_cropped.jpg',
        crop=(0.06, 1.0, 0.50),
        credit='Photo: ZLEA, CC BY-SA 4.0, via Wikimedia Commons',
        source='https://commons.wikimedia.org/wiki/'
               'File:DJI_Mavic_3M_Multispectral_FA373WYPN9_(4-9-2024)_(cropped).jpg'),
}
CARD_ASPECT = 1.5


def crop_photo(img, crop):
    h, w = img.shape[:2]
    x0, x1 = int(crop[0] * w), int(crop[1] * w)
    ch = min(h, int(round((x1 - x0) / CARD_ASPECT)))
    y0 = int(np.clip(crop[2] * h - ch / 2, 0, h - ch))
    return img[y0:y0 + ch, x0:x1]


def make_cards(outdir, cards, photos, prefix, width, dpi):
    """One small figure per entry of `cards`: the photo, the name underneath."""
    img_h = width / CARD_ASPECT
    text_h = 0.62
    for key, card in cards.items():
        fig = plt.figure(figsize=(width, img_h + text_h))
        ax = fig.add_axes([0, text_h / (img_h + text_h), 1, img_h / (img_h + text_h)])
        ax.imshow(crop_photo(plt.imread(photos / card['file']), card['crop']))
        bare(ax)
        fig.text(0.5, (text_h - 0.27) / (img_h + text_h), card['name'], ha='center',
                 va='center', fontsize=15)
        fig.text(0.995, 0.012, card['credit'], ha='right', va='bottom', fontsize=5.5,
                 color='0.45')
        name = f'{prefix}_{key}'
        for ext in ('png', 'pdf'):
            fig.savefig(outdir / f'{name}.{ext}', dpi=dpi)
        plt.close(fig)
        click.echo(f'Wrote {outdir / name}.png/.pdf')


@cli.command('drone-cards')
@out_option
@click.option('--width', default=4.0, show_default=True, help='Card width in inches.')
@click.option('--dpi', default=300, show_default=True)
def drone_cards(outdir, width, dpi):
    """One small figure per drone model: its photo with the name underneath."""
    make_cards(outdir, DRONE_CARDS, outdir / 'materials/drone_photos', 'drone', width, dpi)


# Dove Classic and Dove-R share the 3U Dove bus and the PS2 telescope (Planet,
# docs.planet.com/data/imagery/planetscope), and no freely licensed photo of a
# Dove-R exists, so one public-domain NASA photo stands for both.
DOVE_CARDS = {
    'classic': dict(
        name='Dove Classic / Dove-R', file='nasa_iss38_planet_launch.jpg',
        crop=(0.376, 0.705, 0.412),
        credit='Photo: NASA (ISS Expedition 38, 2014), public domain',
        source='https://commons.wikimedia.org/wiki/File:Planet_Labs_satellite_launch_from_ISS.jpg'),
    'superdove': dict(
        name='SuperDove', file='planet_superdove_render.jpg', crop=(0.04, 0.80, 0.55),
        credit='Image: © Planet Labs PBC',
        source='https://www.planet.com/our-constellations/'),
}


@cli.command('dove-cards')
@out_option
@click.option('--width', default=4.0, show_default=True, help='Card width in inches.')
@click.option('--dpi', default=300, show_default=True)
def dove_cards(outdir, width, dpi):
    """One small figure per Dove generation: its photo with the name underneath."""
    make_cards(outdir, DOVE_CARDS, outdir / 'materials/dove_photos', 'dove', width, dpi)


# --------------------------------------------------------------------------
# drone-timeline
# --------------------------------------------------------------------------
@cli.command('drone-timeline')
@out_option
@click.option('--size', nargs=2, type=float, default=(12.5, 2.6), show_default=True,
              help='Figure width and height in inches.')
@click.option('--dpi', default=300, show_default=True)
def drone_timeline(outdir, size, dpi):
    """Dates of the 50ha drone flights: phantom (both releases), then mavic."""
    import matplotlib.dates as mdates
    phantom = sorted(set(pu.drone_dates('phantom-24782016')) | set(pu.drone_dates('phantom-C3KW2X')))
    mavic = pu.drone_dates('mavic')
    rows = [(DRONE_CARDS['phantom']['name'], phantom, pu.C_PHANTOM),
            (DRONE_CARDS['mavic']['name'], mavic, pu.C_MAVIC)]
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0.04, 0.2, 0.94, 0.72])
    for i, (label, dates, color) in enumerate(rows):
        y = len(rows) - i
        ax.vlines(dates, y - 0.3, y + 0.3, color=color, lw=1.2)
        # the first series' label starts at its first flight, later ones end
        # at their last flight so they stay inside the axes
        left = i == 0
        ax.text(dates[0] - pd.Timedelta(days=20) if left else dates[-1] + pd.Timedelta(days=20),
                y + 0.42,
                f'{label}: {len(dates)} flights, {dates[0]:%b %Y} – {dates[-1]:%b %Y}',
                ha='left' if left else 'right', va='bottom', fontsize=11, color=color)
    ax.set_ylim(0.45, len(rows) + 0.95)
    ax.set_yticks([])
    for side in ('left', 'right', 'top'):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', labelsize=11)
    ax.set_xlim(pd.Timestamp('2018-01-01'), pd.Timestamp('2026-04-01'))
    ax.grid(axis='x', alpha=0.3)
    name = 'drone_timeline'
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'{name}.{ext}', dpi=dpi)
    plt.close(fig)
    click.echo(f'Wrote {outdir / name}.png/.pdf ({len(phantom)} phantom, {len(mavic)} mavic)')


# --------------------------------------------------------------------------
# planet-timeline
# --------------------------------------------------------------------------
PLANET_CLIPPED = FLOWER / 'planet_clipped/4band'
# instrument -> (label, colour), oldest model first
DOVE_MODELS = {
    'PS2': ('Dove Classic', '#7570b3'),
    'PS2.SD': ('Dove-R', '#1b9e77'),
    'PSB.SD': ('SuperDove', '#d95f02'),
}
# Satellites with no SR order, so absent from BCNM_SR_INDEX; their
# instrument is read from the Planet order metadata
# (planet/BCI_50ha_Dove-Classic_psscene_analytic_udm2/PSScene/*_1105_metadata.json).
EXTRA_INSTRUMENTS = {'1105': 'PS2'}


def planet_scene_table():
    """Every 50ha Planet scene: id, date, satellite and instrument."""
    files = sorted(PLANET_CLIPPED.glob('*/*_4band.tif'))
    df = pd.DataFrame({'scene': [f.name[:-len('_4band.tif')] for f in files]})
    df['date'] = pd.to_datetime(df.scene.str[:8])
    df['sat'] = df.scene.str.split('_').str[-1]
    idx = pd.read_csv(BCNM_SR_INDEX, usecols=['satellite_id', 'instrument']).drop_duplicates()
    inst = dict(zip(idx.satellite_id, idx.instrument)) | EXTRA_INSTRUMENTS
    df['instrument'] = df.sat.map(inst)
    if df.instrument.isna().any():
        raise click.ClickException(f'no instrument for satellites {sorted(set(df.sat[df.instrument.isna()]))}')
    return df


@cli.command('planet-timeline')
@out_option
@click.option('--size', nargs=2, type=float, default=(12.5, 5.5), show_default=True,
              help='Figure width and height in inches.')
@click.option('--dpi', default=300, show_default=True)
def planet_timeline(outdir, size, dpi):
    """Dates of the 50ha Planet scenes, one row per satellite, by Dove model."""
    import matplotlib.dates as mdates
    df = planet_scene_table()
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0.2, 0.1, 0.78, 0.86])
    y, gap = 0, 12
    rows = {}
    for inst, (label, color) in reversed(list(DOVE_MODELS.items())):
        g = df[df.instrument == inst]
        sats = g.groupby('sat').date.min().sort_values(ascending=False).index
        y0 = y
        for s in sats:
            rows[s] = y
            y += 1
        ax.scatter(g.date, g.sat.map(rows), marker='|', s=6, lw=0.6, color=color)
        ax.text(-0.01, (y0 + y - 1) / 2, f'{label} ({inst})\n{len(sats)} satellites, '
                f'{len(g):,} scenes', transform=ax.get_yaxis_transform(), ha='right',
                va='center', fontsize=10, color=color)
        y += gap
    # all satellites together: one tick per day with at least one scene
    days = df.date.drop_duplicates()
    span = (df.date.max() - df.date.min()).days + 1
    ax.vlines(days, -gap - 14, -gap, color='0.2', lw=0.4)
    ax.text(-0.01, -gap - 7, f'any satellite\n{len(days):,} of {span:,} days '
            f'({len(days) / span:.0%})', transform=ax.get_yaxis_transform(), ha='right',
            va='center', fontsize=10, color='0.2')
    ax.axhline(-gap / 2, color='0.75', lw=0.6)
    ax.set_ylim(-gap - 16, y - gap + 2)
    ax.set_yticks([])
    for side in ('left', 'right', 'top'):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', labelsize=11)
    ax.set_xlim(pd.Timestamp('2019-12-01'), df.date.max() + pd.Timedelta(days=30))
    ax.grid(axis='x', alpha=0.3)
    name = 'planet_timeline'
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'{name}.{ext}', dpi=dpi)
    plt.close(fig)
    by = df.groupby('instrument').agg(sats=('sat', 'nunique'), scenes=('scene', 'size'))
    click.echo(f'Wrote {outdir / name}.png/.pdf\n{by}')


# --------------------------------------------------------------------------
# cloud-seasonality
# --------------------------------------------------------------------------
FIXED_STRETCH = Path(__file__).resolve().parent.parent / 'config/planet_fixed_stretch.yml'
MONTH_LETTERS = 'JFMAMJJASOND'


def ocm_clear_table(cache):
    """Per 50ha scene: share of the plot's valid pixels that OCM calls clear."""
    import rasterio
    from rasterio.features import geometry_mask
    if cache.exists():
        return pd.read_csv(cache, parse_dates=['date'])
    df = planet_scene_table()
    plot = plot_outline()
    rows = []
    with click.progressbar(list(df.itertuples()), label='reading OCM masks') as bar:
        for r in bar:
            ocm = PLANET_OCM / f'{r.date:%Y}' / f'{r.scene}_ocm.tif'
            if not ocm.exists():
                continue
            with rasterio.open(ocm) as src:
                cls = src.read(1)
                inside = ~geometry_mask([plot], src.shape, src.transform)
            valid = inside & (cls != 255)
            n = valid.sum()
            rows.append({'scene': r.scene, 'date': r.date, 'valid_px': int(n),
                         'clear': (cls[valid] == 0).mean() if n else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False)
    return out


def fixed_rgb(arr):
    """B,G,R SR (x1e4) -> display RGB with config/planet_fixed_stretch.yml."""
    import yaml
    cfg = yaml.safe_load(FIXED_STRETCH.read_text())
    lo = np.array(cfg['lo'][:3]) / cfg['scale']
    hi = np.array(cfg['hi'][:3]) / cfg['scale']
    return planet_rgb(arr, lo[[2, 1, 0]], hi[[2, 1, 0]], gamma=0.8)


PLANET_CALFILL = FLOWER / 'planet_clipped_calfill/4band'


@cli.command('cloud-seasonality')
@out_option
@click.option('--montage-year', default=2023, show_default=True,
              help='Year of the montage scenes (2023 is the only year with 15+ scenes '
                   'covering the plot in every month).')
@click.option('--per-month', default=15, show_default=True, help='Tiles per montage row.')
@click.option('--chunk-m', default=240.0, show_default=True,
              help='Side of each montage tile, metres, centred on the plot.')
@click.option('--min-valid', default=0.9, show_default=True,
              help='Skip scenes covering less than this share of the plot.')
@click.option('--seed', default=0, show_default=True)
@click.option('--dpi', default=300, show_default=True)
def cloud_seasonality(outdir, montage_year, per_month, chunk_m, min_valid, seed, dpi):
    """2-week OCM-clear fraction averaged over years, and a one-year montage."""
    import matplotlib.dates as mdates
    clear = ocm_clear_table(outdir / 'materials/planet_ocm_clear.csv')
    full = clear.valid_px.max()
    clear = clear[clear.valid_px >= min_valid * full].dropna(subset=['clear'])
    years = sorted(clear.date.dt.year.unique())
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # 2-week mean clear fraction: each year's mean per bin, then the mean over years
    days = 14
    nbins = 365 // days
    period = np.minimum((clear.date.dt.dayofyear - 1) // days, nbins - 1)  # fold the year's tail in
    per_year = clear.assign(year=clear.date.dt.year, period=period).groupby(['year', 'period']).clear.mean()
    mean = per_year.groupby('period').mean()
    base = pd.Timestamp('2001-01-01')  # x axis is day of year
    wx = lambda k: base + pd.Timedelta(days=days * k + days / 2)  # noqa: E731
    ax = fig.add_axes([0.7 / FIG_W, 0.75 / FIG_H, 4.15 / FIG_W, 4.55 / FIG_H])
    # BCI dry season, roughly mid-December to the end of April
    for a, b in (('2001-01-01', '2001-04-30'), ('2001-12-15', '2002-01-01')):
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='#f3e2c0', lw=0, zorder=0)
    ax.text(pd.Timestamp('2001-02-28'), 97, 'dry season', ha='center', va='top', fontsize=9,
            color='#9a6b1f')
    ax.bar([wx(k) for k in mean.index], 100 * mean.values, width=days - 1.5,
           color='#4c72b0', zorder=2)
    ax.set_xlim(base, base + pd.Timedelta(days=365))
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.tick_params(labelsize=9)
    ax.set_ylabel('plot canopy clear of cloud (%)', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)

    # (b) montage: random scenes per month of one year
    import yaml
    cfg = yaml.safe_load(FIXED_STRETCH.read_text())
    forest_lo = np.array(cfg['lo'][:3]) / cfg['scale']  # B, G, R in SR x1e4
    forest_hi = np.array(cfg['hi'][:3]) / cfg['scale']
    rng = np.random.default_rng(seed)
    pool = clear[clear.date.dt.year == montage_year].assign(month=clear.date.dt.month)
    pool = pool[[(PLANET_CALFILL / f'{d:%Y}' / f'{s}_4band.tif').exists()
                 for s, d in zip(pool.scene, pool.date)]]
    cx, cy = plot_outline().centroid.coords[0]
    half = chunk_m / 2
    bounds = (cx - half, cy - half, cx + half, cy + half)
    g, top, bottom = 0.03, FIG_H - 0.45, 0.3
    t = (top - bottom - 11 * g) / 12
    x0 = FIG_W - 0.45 - per_month * t - (per_month - 1) * g
    for m in range(1, 13):
        rows = pool[pool.month == m]
        pick = rows.iloc[rng.choice(len(rows), min(per_month, len(rows)), replace=False)]
        # clearest tile on the left. A pixel counts as clear only if OCM calls it
        # clear and every band lies within the fixed stretch's clear-forest range
        # (0.5th-99.9th percentile): OCM passes some haze, bright cloud and shadow
        # as clear. Ties are broken by mean brightness, low first.
        tiles = []
        for r in pick.itertuples():
            arr, ext = read_window(PLANET_CALFILL / f'{r.date:%Y}' / f'{r.scene}_4band.tif',
                                   bounds, [1, 2, 3])
            cls, _ = read_window(PLANET_OCM / f'{r.date:%Y}' / f'{r.scene}_ocm.tif', bounds, [1])
            valid = cls[0] != 255
            forest = ((arr >= forest_lo[:, None, None])
                      & (arr <= forest_hi[:, None, None])).all(axis=0)
            share = ((cls[0] == 0) & forest)[valid].mean() if valid.any() else 0.0
            ok = arr.sum(axis=0) > 0
            bright = arr[:, ok].mean() if ok.any() else np.inf
            tiles.append((-share, bright, arr, ext))
        tiles.sort(key=lambda x: (x[0], x[1]))
        y = top - m * t - (m - 1) * g
        for j, (_, _, arr, ext) in enumerate(tiles):
            ax = fig.add_axes([(x0 + j * (t + g)) / FIG_W, y / FIG_H, t / FIG_W, t / FIG_H])
            ax.imshow(fixed_rgb(arr), extent=ext, interpolation='nearest')
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():  # frame, so white cloud reads as a tile
                sp.set_linewidth(0.4)
                sp.set_edgecolor('0.6')
        fig.text((x0 - 0.1) / FIG_W, (y + t / 2) / FIG_H, MONTH_LETTERS[m - 1],
                 ha='right', va='center', fontsize=11, fontweight='bold')
        fig.text((FIG_W - 0.4) / FIG_W, (y + t / 2) / FIG_H, f'of {len(rows)}',
                 ha='left', va='center', fontsize=7, color='0.45')
    # panel descriptions live in planet_intro_captions.md, not the image
    name = 'planet_cloud_seasonality'
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'{name}.{ext}', dpi=dpi)
    plt.close(fig)
    by_month = clear.groupby(clear.date.dt.month).clear.mean().round(2)
    click.echo(f'Wrote {outdir / name}.png/.pdf\nmean clear fraction by month: '
               f'{by_month.to_dict()}')


# --------------------------------------------------------------------------
# planet-calibration
# --------------------------------------------------------------------------
PLANET_CAL = FLOWER / 'planet_clipped_cal/4band'
# Four clear clips (>= 97% OCM-clear), all with their own MODIS fit, from all
# three Dove generations within 2 days: the >= 3-satellite group <= 4 days
# long with the largest spread in raw-SR plot-mean visible reflectance
# (raw NIR spread 0.13, calibrated 0.007).
CAL_EXAMPLE = ('20210130_132220_104a', '20210130_152842_1038',
               '20210201_155304_09_2307', '20210201_160348_72_1066')


@cli.command('planet-calibration')
@out_option
@click.option('--chunk-m', default=360.0, show_default=True,
              help='Side of each image tile, metres, centred on the plot.')
@click.option('--min-clear', default=0.8, show_default=True,
              help='Plot clear fraction for a clip to enter the time series.')
@click.option('--dpi', default=300, show_default=True)
def planet_calibration(outdir, chunk_m, min_clear, dpi):
    """Raw vs MODIS-calibrated Planet SR: example clips and the plot-mean NIR record."""
    from shapely.geometry import shape
    import evaluate_planet_calibration as ev
    from util import load_config
    means = pd.read_csv(ev.CAL / 'planet_calibration_plot_means.csv', parse_dates=['time'])
    means = means[means.clear_frac >= min_clear]
    have_cal = set(means.scene_id[means.version == 'CAL'])
    means = means[means.scene_id.isin(have_cal)]
    passes = ev.per_acquisition(means)
    region = shape(load_config(ev.REPO / 'config/clip_50ha_plot.yml')['region'])
    modis = ev.modis_plot_series(FLOWER / 'modis/mcd43a1_bcnm/reference_nadir_sza45.nc', region)
    modis = modis[(modis.index >= passes.time.min()) & (modis.index <= passes.time.max())]
    ex = means[means.scene_id.isin(CAL_EXAMPLE)].set_index(['scene_id', 'version'])

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    # left: the example clips, raw SR above calibrated. One stretch shared by
    # all eight tiles (per band, 1st-99th percentile of them pooled), so
    # differences between tiles are real reflectance differences.
    cx, cy = plot_outline().centroid.coords[0]
    half = chunk_m / 2
    bounds = (cx - half, cy - half, cx + half, cy + half)
    t, g, x0 = 1.62, 0.08, 0.7
    y_rows = {'SR': FIG_H - 0.75 - t, 'CAL': FIG_H - 0.75 - 2 * t - 0.4}
    roots = {'SR': PLANET_SR, 'CAL': PLANET_CAL}
    tiles = {(sid, v): read_window(roots[v] / sid[:4] / f'{sid}_4band.tif', bounds, [1, 2, 3])
             for sid in CAL_EXAMPLE for v in y_rows}
    lo, hi = stretch_limits([a for a, _ in tiles.values()], pct=(1, 99))
    for j, sid in enumerate(CAL_EXAMPLE):
        for version, y in y_rows.items():
            ax = fig.add_axes([(x0 + j * (t + g)) / FIG_W, y / FIG_H, t / FIG_W, t / FIG_H])
            arr, ext = tiles[(sid, version)]
            ax.imshow(planet_rgb(arr, lo, hi), extent=ext, interpolation='nearest')
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
            bare(ax)
            nir = ex.loc[(sid, version), 'nir']
            ax.text(0.04, 0.04, f'NIR {nir:.3f}', transform=ax.transAxes, fontsize=7.5,
                    va='bottom', bbox=dict(fc='white', ec='none', alpha=0.8, pad=1.2))
            if version == 'SR':
                inst = ex.loc[(sid, version), 'instrument']
                ax.set_title(f'{pd.Timestamp(sid[:8]):%d %b %Y}\n{DOVE_MODELS[inst][0]}',
                             fontsize=9, color=ev.INSTRUMENT_COLOR[inst], pad=3)
            if j == 0 and version == 'SR':
                add_scalebar(ax, 100, '100 m', loc=(0.08, 0.2), lw=2.5)
    for version, y in y_rows.items():
        fig.text((x0 - 0.15) / FIG_W, (y + t / 2) / FIG_H,
                 {'SR': 'raw surface\nreflectance', 'CAL': 'calibrated\nto MODIS'}[version],
                 rotation=90, ha='center', va='center', fontsize=10)

    # right: plot-mean NIR per pass, raw and calibrated, with the MODIS reference
    left = x0 + 4 * t + 3 * g + 0.95
    w = FIG_W - left - 0.2
    h = (FIG_H - 1.25) / 2
    axes = {}
    for k, version in enumerate(('SR', 'CAL')):
        ax = fig.add_axes([left / FIG_W, (0.65 + (1 - k) * (h + 0.15)) / FIG_H, w / FIG_W,
                           h / FIG_H], sharex=axes.get('SR'), sharey=axes.get('SR'))
        axes[version] = ax
        s = passes[passes.version == version]
        for inst, gi in s.groupby('instrument'):
            ax.scatter(gi.time, gi.nir, s=7, lw=0, color=ev.INSTRUMENT_COLOR[inst],
                       label=DOVE_MODELS[inst][0], zorder=2)
        ax.plot(modis.index, modis.nir, color='k', lw=1.2, label='MODIS reference', zorder=3)
        e = ex.xs(version, level='version')
        ax.scatter(e.time, e.nir, s=40, facecolor='none', edgecolor='k', lw=1.1, zorder=4)
        m, _ = ev.series_metrics(s, 'nir')
        ax.text(0.01, 0.03, f'median change between passes ≤ 3 days apart: {m["jitter"]:.3f}',
                transform=ax.transAxes, fontsize=8, va='bottom')
        ax.set_ylabel(f'plot-mean NIR\n({"raw" if version == "SR" else "calibrated"})',
                      fontsize=9.5)
        ax.tick_params(labelsize=8.5, labelbottom=version == 'CAL')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=0.3)
    axes['SR'].legend(loc='lower left', fontsize=7.5, ncol=2, frameon=False, markerscale=2,
                      bbox_to_anchor=(0, 1.0))
    name = 'planet_calibration'
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'{name}.{ext}', dpi=dpi)
    plt.close(fig)
    click.echo(f'Wrote {outdir / name}.png/.pdf ({passes.version.value_counts().to_dict()} passes)')


# --------------------------------------------------------------------------
# crown-resolution
# --------------------------------------------------------------------------
# One example crown per size class (about 3, 5, 11, 35 and 60 Planet pixels),
# each from a species whose median crown area (2022-09-29 map) is close to
# that size and near its species' median. Each has a drone-classified event
# (P > 0.5 on a phantom flight) with a >= 95%-clear, co-registered Planet
# scene within 1 day:
#   (tag, species, event, drone flight, Planet scene)
SIZE_EXAMPLES = (
    (2421, 'Gustavia superba', 'flowering', '2023-02-14', '20230215_152354_97_2479_4band'),
    (289442, 'Cordia bicolor', 'flowering', '2024-01-25', '20240126_155354_34_24d5_4band'),
    (617175, 'Jacaranda copaia', 'flowering', '2023-02-22', '20230221_143625_17_2445_4band'),
    (3661, 'Cavanillesia platanifolia', 'leafless', '2023-04-25',
     '20230426_152625_06_2478_4band'),
    (4146, 'Ceiba pentandra', 'leafless', '2024-01-25', '20240126_155354_34_24d5_4band'),
)
RES_MAP_DATE = '2022-09-29'


@cli.command('crown-resolution')
@out_option
@click.option('--tile-m', default=46.0, show_default=True,
              help='Ground width of every example tile, metres.')
@click.option('--dpi', default=300, show_default=True)
def crown_resolution(outdir, tile_m, dpi):
    """Crown-size histogram and example crown events in drone vs Planet."""
    import geopandas as gpd
    import rasterio
    from matplotlib.ticker import FuncFormatter, LogLocator
    crowns = crownmap().copy()
    crowns['area'] = crowns.geometry.area
    px_area = 9.0
    ex = [dict(tag=t, latin=sp, kind=k, flight=pd.Timestamp(f), scene=sc,
               area=crowns.area[t], geom=dated_crown(t, f))
          for t, sp, k, f, sc in SIZE_EXAMPLES]
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # histogram of crown area, log scale, with Planet pixels on a second axis
    ax = fig.add_axes([0.07, 4.85 / FIG_H, 0.88, 0.75 / FIG_H])
    bins = np.logspace(np.log10(2), np.log10(2000), 50)
    ax.hist(crowns.area, bins=bins, color='0.6', ec='white', lw=0.3)
    ax.set_xscale('log')
    ax.set_xlim(bins[0], bins[-1])
    plain = FuncFormatter(lambda v, _: f'{v:g}')
    ax.xaxis.set_major_formatter(plain)
    ax.set_xlabel('crown area (m²)', fontsize=9.5, labelpad=1, loc='right')
    ax.set_ylabel('crowns', fontsize=9.5)
    ax.tick_params(labelsize=8.5)
    ax.spines[['right']].set_visible(False)
    med = crowns.area.median()
    ax.axvline(med, color='k', lw=1, ls=':')
    ax.text(med * 1.04, ax.get_ylim()[1] * 0.95, f'median {med:.0f} m²', fontsize=8,
            va='top')
    for i, e in enumerate(ex, start=1):
        ax.axvline(e['area'], color=C_BOX, lw=1.4)
        # numbers below the tick labels, clear of both axes' ticks
        ax.annotate(str(i), (e['area'], 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -24), textcoords='offset points', ha='center', va='top',
                    fontsize=9, fontweight='bold', color='#9a7a00')
    top = ax.secondary_xaxis('top', functions=(lambda a: a / px_area, lambda p: p * px_area))
    top.xaxis.set_major_formatter(plain)
    top.set_xlabel('Planet pixels (3 × 3 m)', fontsize=9.5, labelpad=2)
    top.tick_params(labelsize=8.5)
    ax.text(0.995, 0.88, f'{len(crowns):,} crowns',
            transform=ax.transAxes, ha='right', va='top', fontsize=8, color='0.35')

    # Planet tiles: MODIS-calibrated SR (calfill), placed with each scene's
    # AROSICS shift. Each band's dark point is its own 1st percentile over the
    # five tiles, but all bands share one range (the widest 1st-99th percentile
    # span): per-band ranges stretch the narrow visible bands of forest into
    # colour noise.
    reads = []
    for e in ex:
        cx, cy = e['geom'].centroid.coords[0]
        e['box'] = (cx - tile_m / 2, cy - tile_m / 2, cx + tile_m / 2, cy + tile_m / 2)
        # only scenes with a successful AROSICS match: scene_info falls back to a
        # zero shift otherwise, which can leave a scene tens of metres off
        if e['scene'] not in coreg_shifts() or not any(coreg_shifts()[e['scene']]):
            raise click.ClickException(f"{e['scene']} has no successful co-registration")
        e['shift'] = coreg_shifts()[e['scene']]
        e['cal'] = PLANET_CALFILL / e['scene'][:4] / f"{e['scene']}.tif"
        b = e['box']
        pb = (b[0] - 6, b[1] - 6, b[2] + 6, b[3] + 6)
        reads.append(read_window(e['cal'], shifted_bounds(pb, e['shift']), [1, 2, 3]))
    plo, phi = stretch_limits([a for a, _ in reads], pct=(1, 99))
    phi = plo + (phi - plo).max()

    n = len(ex)
    t, gap = 1.72, 0.5
    x0 = (FIG_W - n * t - (n - 1) * gap) / 2 + 0.25
    y_planet = 0.1
    y_drone = y_planet + t + 0.1
    for i, e in enumerate(ex):
        b, (dx, dy) = e['box'], e['shift']
        x = x0 + i * (t + gap)
        # drone on the event flight, with the Planet pixel grid in the drone frame
        a1 = fig.add_axes([x / FIG_W, y_drone / FIG_H, t / FIG_W, t / FIG_H])
        arr, ext = read_window(drone_file('phantom', e['flight']), b, [1, 2, 3, 4], factor=2)
        a1.imshow(pu.rgb_display(arr), extent=ext)
        with rasterio.open(e['cal']) as src:
            gx0, gy0 = src.transform.c, src.transform.f
        kx = np.arange(np.floor((b[0] + dx - gx0) / 3), np.ceil((b[2] + dx - gx0) / 3) + 1)
        ky = np.arange(np.floor((b[1] + dy - gy0) / 3), np.ceil((b[3] + dy - gy0) / 3) + 1)
        for k in kx:
            a1.axvline(gx0 + 3 * k - dx, color='#ff00ff', lw=0.52, alpha=0.85)
        for k in ky:
            a1.axhline(gy0 + 3 * k - dy, color='#ff00ff', lw=0.52, alpha=0.85)
        # Planet within a day of the flight
        a2 = fig.add_axes([x / FIG_W, y_planet / FIG_H, t / FIG_W, t / FIG_H])
        parr, pext = reads[i]
        a2.imshow(planet_rgb(parr, plo, phi, gamma=0.9), extent=unshift_extent(pext, (dx, dy)),
                  interpolation='nearest')
        for a in (a1, a2):
            gpd.GeoSeries([e['geom']]).boundary.plot(ax=a, color=C_CROWN, lw=1.1)
            a.set_xlim(b[0], b[2])
            a.set_ylim(b[1], b[3])
            bare(a)
        pday = pd.Timestamp(e['scene'][:8])
        a1.set_title(f"{i + 1}. {italic_name(e['latin'])}\n{e['area']:.0f} m² ≈ "
                     f"{e['area'] / px_area:.0f} pixels\n{e['kind']}, {e['flight']:%d %b %Y}",
                     fontsize=8.5, pad=3)
        a2.text(0.03, 0.03, f'{pday:%d %b}', transform=a2.transAxes, fontsize=7.5,
                va='bottom', color='white', path_effects=halo(1.5))
        if i == 0:
            add_scalebar(a1, 10, '10 m', loc=(0.07, 0.07), lw=2.5)
            add_scalebar(a2, 10, '10 m', loc=(0.07, 0.2), lw=2.5)
    xl = (x0 - 0.22) / FIG_W
    fig.text(xl, (y_drone + t / 2) / FIG_H, 'drone (4.5 cm)\n+ Planet pixel grid',
             rotation=90, ha='center', va='center', fontsize=9.5)
    fig.text(xl, (y_planet + t / 2) / FIG_H, 'PlanetScope (3 m)', rotation=90,
             ha='center', va='center', fontsize=9.5)
    name = 'crown_resolution'
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'{name}.{ext}', dpi=dpi)
    plt.close(fig)
    click.echo(f'Wrote {outdir / name}.png/.pdf')
    for e in ex:
        click.echo(f"{e['tag']} {e['latin']} {e['area']:.0f} m2 {e['kind']} {e['flight']:%Y-%m-%d} "
                   f"{e['scene']} shift {e['shift'][0]:.1f},{e['shift'][1]:.1f}")


def italic_name(latin):
    return ' '.join(f'$\\it{{{w}}}$' for w in latin.split())


if __name__ == '__main__':
    cli()
