#!/usr/bin/env python
"""Build a self-contained interactive HTML viewer of per-crown drone and
Planet classification confidence over time.

The page shows the crown map (clickable; colored by species, or by drone or
Planet confidence on a chosen date) above two linked time series:

- the selected crown: drone p_flowering, p_deciduous and the training-mask
  event probability 1 - (1 - p_flowering)(1 - p_deciduous) at each flight,
  and the temporal Planet Mask R-CNN's max_score and pos_frac at each scene;
- the manual labels of the same crown (Vicente's, and Paulino's unverified
  ones) at the flights they were made on: flowering yes / maybe / no and
  leafless (leafing < 50%);
- its species: the share of crowns that are events per flight and per
  scene, with the rules of plot_crown_species_comparison.py.

Data are embedded as JSON and plotly.js is inlined, so the file opens offline.
serve_crown_timeseries.py serves the same page with drone and Planet image
layers under the map.
"""
import json
import os

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from plotly.offline import get_plotlyjs

from plot_crown_species_comparison import APPLY, NC4S, PHANTOM_END

CROWNMAP = ('/Volumes/Earth03/flower/stri/24784053/'
            'BCI_50ha_2022_09_29_crownmap_improved/'
            'BCI_50ha_2022_09_29_crownmap_improved.shp')
DATED_CROWNS = ('/Volumes/Earth03/flower/crowns_20240803/'
                'BCI_50ha_crownmap_timeseries.shp')
LABELS = '/Volumes/Earth03/flower/vicente_20260618/vicente_20260618_labels.csv'
PAULINO = '/Volumes/Earth03/flower/vicente_202610/latest_paulino_label.gpkg'
TRAIN_GPKG = '/Volumes/Earth03/flower/vicente_202509/flowering_dataset.gpkg'
FLOWER_CODE = {'no': 0, 'maybe': 1, 'yes': 2}
TEMPLATE = os.path.join(os.path.dirname(__file__),
                        'crown_timeseries_template.html')


def q(x, scale=100):
    """Quantize probabilities to ints (NaN -> -1) to keep the JSON small."""
    a = np.asarray(x, dtype=float)
    return np.where(np.isnan(a), -1, np.round(a * scale)).astype(int).tolist()


def load_drone(nc4s):
    """(dates, {tag: {'f': [...], 'd': [...]}}, species by tag)."""
    frames, species = [], {}
    for path in nc4s:
        ds = xr.open_dataset(path)
        df = ds[['flowering_probability', 'deciduous_probability']] \
            .to_dataframe().reset_index()
        if 'mavic' in os.path.basename(path):
            df = df[df['date'] > PHANTOM_END]
        frames.append(df)
        species.update(zip(ds.tag.values.tolist(), ds.species.values.tolist()))
    df = pd.concat(frames, ignore_index=True)
    dates = np.sort(df['date'].unique())
    f = df.pivot(index='tag', columns='date', values='flowering_probability')
    d = df.pivot(index='tag', columns='date', values='deciduous_probability')
    f, d = f.reindex(columns=dates), d.reindex(columns=dates)
    return dates, f, d, species


def load_manual_labels(labels=LABELS, paulino=PAULINO, dated=DATED_CROWNS,
                       train_gpkg=TRAIN_GPKG):
    """Manual crown-date labels keyed by crown tag.

    Vicente's labels (one row per annotator, keyed by crown uuid) are reduced
    to one per crown-date with aggregate_gt_labels and bridged to tag through
    the dated crown map, dropping crowns that share a tag. Paulino's
    unverified labels (already keyed by tag) fill crown-dates Vicente did not
    label. Returns a frame with tag, date (YYYY-MM-DD), fl (0 no, 1 maybe,
    2 yes), leaf (leafing %, 0-100), src (0 Vicente, 1 Paulino) and train
    (1 when that crown-date is one of the drone classifier's training labels).
    """
    from compare_classifications_to_gt import (
        aggregate_gt_labels, load_gt, load_labels_gpkg)
    parts = []
    if labels:
        gt, _ = aggregate_gt_labels(load_gt(labels))
        pairs = gpd.read_file(dated, ignore_geometry=True,
                              columns=['GlobalID', 'tag'])
        pairs = pairs.rename(columns={'GlobalID': 'uuid'}).drop_duplicates()
        pairs['tag'] = pairs['tag'].astype(int)
        pairs = pairs[~pairs['tag'].duplicated(keep=False)
                      & ~pairs['uuid'].duplicated(keep=False)]
        gt = gt.merge(pairs, on='uuid')
        gt['src'] = 0
        parts.append(gt)
    if paulino:
        gt, _, _, _ = load_labels_gpkg(paulino)
        gt, _ = aggregate_gt_labels(gt)
        gt['tag'] = gt['uuid'].astype(int)
        gt['src'] = 1
        parts.append(gt)
    if not parts:
        return None
    lab = pd.concat(parts, ignore_index=True)
    lab['date'] = lab['dt'].dt.strftime('%Y-%m-%d')
    lab = lab.sort_values('src').drop_duplicates(['tag', 'date'])
    lab['fl'] = lab['isFlowering'].map(FLOWER_CODE)
    lab['leaf'] = lab['leafing'].round()
    lab['train'] = 0
    if train_gpkg and os.path.exists(train_gpkg):
        tr = gpd.read_file(train_gpkg, layer='flowering_dataset',
                           ignore_geometry=True, columns=['polygon_id'])
        lab['train'] = (lab['polygon_id'].str.replace('_', '-').isin(
            set(tr['polygon_id'].str.replace('_', '-')))
            & (lab['src'] == 0)).astype(int)
    return lab[['tag', 'date', 'fl', 'leaf', 'src', 'train']]


def build_data(apply_dir=APPLY, crownmap=CROWNMAP, nc4s=NC4S,
               store_min_clear=0.1, simplify_m=0.3, manual=None):
    """The page's data: drone and Planet per-crown series, scenes and crown
    outlines in map coordinates (x - x0, y1 - y)."""
    dates, pf, pd_, species = load_drone(nc4s)

    scenes = pd.read_csv(f'{apply_dir}/scene_summary.csv',
                         parse_dates=['date', 'drone_date'])
    scenes = scenes.sort_values(['date', 'scene']).reset_index(drop=True)
    scene_idx = {s: i for i, s in enumerate(scenes['scene'])}
    crowns = pd.read_csv(
        f'{apply_dir}/crown_timeseries.csv',
        usecols=['scene', 'tag', 'mnemonic', 'crown_px', 'crown_clear_frac',
                 'pos_frac', 'max_score'])
    info = crowns.drop_duplicates('tag').set_index('tag')[['mnemonic', 'crown_px']]
    obs = crowns[crowns['crown_clear_frac'] >= store_min_clear].copy()
    obs['si'] = obs['scene'].map(scene_idx)
    obs = obs.sort_values(['tag', 'si'])

    gdf = gpd.read_file(crownmap)
    gdf['tag'] = pd.to_numeric(gdf['tag'], errors='coerce')
    # A few tags are shared by several polygons; all are drawn for the tag.
    gdf = gdf[gdf['tag'].isin(info.index)]
    x0, y0, x1, y1 = gdf.total_bounds
    gdf['geometry'] = gdf.geometry.simplify(simplify_m)

    def path(geoms):
        polys = [p for g in geoms
                 for p in (g.geoms if g.geom_type == 'MultiPolygon' else [g])]
        out = []
        for p in polys:
            pts = ' '.join(f'{x - x0:.1f},{y1 - y:.1f}'
                           for x, y in p.exterior.coords[:-1])
            out.append(f'M{pts}Z')
        return ''.join(out)

    tags = sorted(info.index)
    date_idx = {str(d)[:10]: i for i, d in enumerate(dates)}
    by_tag_labels = {}
    if manual is not None:
        manual = manual.assign(li=manual['date'].map(date_idx))
        off = manual['li'].isna()
        manual = manual[~off].astype({'li': int}).sort_values(['tag', 'li'])
        by_tag_labels = {t: g for t, g in manual.groupby('tag')}
        shown = manual[manual['tag'].isin(info.index)]
        print(f'manual labels: {len(shown):,} crown-dates on '
              f'{shown["tag"].nunique()} viewer crowns '
              f'({int((shown["src"] == 1).sum()):,} from Paulino); '
              f'{int(off.sum())} not on a plotted flight, '
              f'{manual["tag"].nunique() - shown["tag"].nunique()} labeled '
              f'crowns not in the viewer')
    planet = {t: g for t, g in obs.groupby('tag')}
    crown_rows = []
    for t in tags:
        geom = gdf.loc[gdf['tag'] == t, 'geometry']
        g = planet.get(t)
        row = {
            'tag': int(t),
            'sp': species.get(t) or 'unknown',
            'mn': info.loc[t, 'mnemonic'],
            'px': int(info.loc[t, 'crown_px']),
            'path': path(geom) if len(geom) else '',
        }
        if t in pf.index:
            row['f'] = q(pf.loc[t].values)
            row['d'] = q(pd_.loc[t].values)
        if g is not None:
            row['ps'] = g['si'].tolist()
            row['pc'] = q(g['crown_clear_frac'])
            row['pm'] = q(g['max_score'])
            row['pp'] = q(g['pos_frac'])
        m = by_tag_labels.get(t)
        if m is not None:
            row['li'] = m['li'].tolist()
            row['lf'] = m['fl'].fillna(-1).astype(int).tolist()
            row['ll'] = m['leaf'].fillna(-1).astype(int).tolist()
            row['lsrc'] = m['src'].tolist()
            row['ltr'] = m['train'].tolist()
        crown_rows.append(row)

    print(f'{len(crown_rows)} crowns, {len(dates)} flights, '
          f'{len(scenes)} scenes, {len(obs):,} Planet crown-scenes')
    return {
        'width': round(x1 - x0, 1), 'height': round(y1 - y0, 1),
        'origin': [float(x0), float(y1)],
        'droneDates': [str(d)[:10] for d in dates],
        'scenes': {
            'date': scenes['date'].dt.strftime('%Y-%m-%d').tolist(),
            'name': scenes['scene'].tolist(),
            'env': scenes['in_envelope'].astype(int).tolist(),
            'coreg': scenes['coreg_ok'].astype(int).tolist(),
            'dtrain': scenes['days_to_nearest_train'].astype(int).tolist(),
            'ddrone': scenes['days_to_drone'].astype(int).tolist(),
            'droneDate': scenes['drone_date'].dt.strftime('%Y-%m-%d').tolist(),
            'ndet': scenes['n_detections'].astype(int).tolist(),
        },
        'crowns': crown_rows,
        'phantomEnd': str(PHANTOM_END.date()),
    }


def render_page(data, plotlyjs=None):
    """The page with `data` embedded; plotly.js is inlined unless `plotlyjs`
    gives a script URL to load it from instead."""
    with open(TEMPLATE) as fh:
        html = fh.read()
    if plotlyjs is None:
        html = html.replace('/*PLOTLYJS*/', get_plotlyjs())
    else:
        html = html.replace('<script>/*PLOTLYJS*/</script>',
                            f'<script src="{plotlyjs}"></script>')
    return html.replace('/*DATA*/', json.dumps(data, separators=(',', ':')))


@click.command()
@click.argument('output', type=click.Path(dir_okay=False))
@click.option('--apply-dir', default=APPLY, show_default=True,
              type=click.Path(exists=True, file_okay=False))
@click.option('--crownmap', default=CROWNMAP, show_default=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option('--nc4', 'nc4s', multiple=True, default=NC4S, show_default=True)
@click.option('--store-min-clear', default=0.1, show_default=True,
              help='Planet crown-scenes below this clear fraction are not '
                   'embedded; the page filters further.')
@click.option('--simplify-m', default=0.3, show_default=True,
              help='Crown outline simplification tolerance (m).')
@click.option('--labels', default=LABELS, show_default=True,
              help="Vicente's manual labels CSV ('' to leave out).")
@click.option('--paulino-labels', default=PAULINO, show_default=True,
              help="Paulino's unverified labels gpkg ('' to leave out).")
def main(output, apply_dir, crownmap, nc4s, store_min_clear, simplify_m,
         labels, paulino_labels):
    manual = load_manual_labels(labels, paulino_labels)
    data = build_data(apply_dir, crownmap, nc4s, store_min_clear, simplify_m,
                      manual)
    with open(output, 'w') as fh:
        fh.write(render_page(data))
    print(f'Wrote {output} ({os.path.getsize(output) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
