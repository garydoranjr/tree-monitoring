#!/usr/bin/env python
"""Build a self-contained interactive HTML viewer of per-crown drone and
Planet classification confidence over time.

The page shows the crown map (clickable; coloured by species, or by drone or
Planet confidence on a chosen date) above two linked time series:

- the selected crown: drone p_flowering, p_deciduous and the training-mask
  event probability 1 - (1 - p_flowering)(1 - p_deciduous) at each flight,
  and the temporal Planet Mask R-CNN's max_score and pos_frac at each scene;
- its species: the share of crowns that are events per flight and per
  scene, with the rules of plot_crown_species_comparison.py.

Data are embedded as JSON and plotly.js is inlined, so the file opens offline.
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
def main(output, apply_dir, crownmap, nc4s, store_min_clear, simplify_m):
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
        crown_rows.append(row)

    data = {
        'width': round(x1 - x0, 1), 'height': round(y1 - y0, 1),
        'droneDates': [str(d)[:10] for d in dates],
        'scenes': {
            'date': scenes['date'].dt.strftime('%Y-%m-%d').tolist(),
            'name': scenes['scene'].tolist(),
            'env': scenes['in_envelope'].astype(int).tolist(),
            'coreg': scenes['coreg_ok'].astype(int).tolist(),
            'dtrain': scenes['days_to_nearest_train'].astype(int).tolist(),
            'ddrone': scenes['days_to_drone'].astype(int).tolist(),
            'droneDate': scenes['drone_date'].dt.strftime('%Y-%m-%d').tolist(),
        },
        'crowns': crown_rows,
        'phantomEnd': str(PHANTOM_END.date()),
    }
    with open(TEMPLATE) as fh:
        html = fh.read()
    html = html.replace('/*PLOTLYJS*/', get_plotlyjs())
    html = html.replace('/*DATA*/', json.dumps(data, separators=(',', ':')))
    with open(output, 'w') as fh:
        fh.write(html)
    print(f'Wrote {output} ({os.path.getsize(output) / 1e6:.1f} MB, '
          f'{len(crown_rows)} crowns, {len(dates)} flights, '
          f'{len(scenes)} scenes, {len(obs):,} Planet crown-scenes)')


if __name__ == '__main__':
    main()
