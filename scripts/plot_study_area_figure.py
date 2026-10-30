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


if __name__ == '__main__':
    cli()
