#!/usr/bin/env python
"""Build an exhaustive set of 50ha-plot Planet 3B clips from local copies.

Walks `csdap/planet/PSScene-*/assets/{ortho_analytic_4b,ortho_udm2}` and the
`planet/{planet_bci_orders*,BCI_*}/<order>/PSScene/` order roots, picks the
best (4-band, UDM2) source per scene, clips both to the 50ha polygon, and
renders a per-scene 2nd/98th percentile RGB from the clipped 4-band.

Outputs under `<output-root>/`:
    4band/<YYYY>/<scene_id>_4band.tif   uint16 4-band, original CRS / nodata
    udm2/<YYYY>/<scene_id>_udm2.tif     uint8 8-band UDM2 mask
    rgb/<YYYY>/<scene_id>_rgb.tif       uint8 3-band RGB
    inventory.csv                       per-scene source + status
    missing_at_planet.csv               scenes Planet has but we don't (API)
"""
import asyncio
import csv
import os
import re
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import click
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_geom
from shapely.geometry import box, mapping, shape
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from util import load_config  # noqa: E402

SCENE_RE = re.compile(r'(\d{8}_\d{6}(?:_\d{1,3})?_[0-9a-f]{4})')

# CSDAP roots (full unclipped scenes)
CSDAP_FOURBAND_SUBDIR = 'ortho_analytic_4b'
CSDAP_UDM2_SUBDIR = 'ortho_udm2'

# Planet-direct order roots — discovered by listing top-level dirs whose
# names match these patterns.
PLANET_ORDER_DIR_PATTERNS = (
    re.compile(r'^planet_bci_orders(_\d{4})?$'),  # planet_bci_orders, _2025
    re.compile(r'^BCI_.*_psscene_.*_udm2$'),       # BCI_50ha_*, BCNM_*
)


def _has_4band(p: Path) -> bool:
    return p.suffix.lower() == '.tif' and 'rgb' not in p.name.lower() \
        and 'ndvi' not in p.name.lower() and '_sr' not in p.name.lower()


def discover_csdap(csdap_root: Path):
    """Yield (scene_id, four_band_path|None, udm2_path|None) per CSDAP scene."""
    planet_dir = csdap_root / 'planet'
    if not planet_dir.is_dir():
        return
    for scene_dir in planet_dir.iterdir():
        if not scene_dir.is_dir() or not scene_dir.name.startswith('PSScene-'):
            continue
        scene_id = scene_dir.name[len('PSScene-'):]

        four_band = None
        fb_dir = scene_dir / 'assets' / CSDAP_FOURBAND_SUBDIR
        if fb_dir.is_dir():
            for p in fb_dir.iterdir():
                if p.name.endswith('_3B_AnalyticMS.tif') and _has_4band(p):
                    four_band = p
                    break

        udm2 = None
        u_dir = scene_dir / 'assets' / CSDAP_UDM2_SUBDIR
        if u_dir.is_dir():
            for p in u_dir.iterdir():
                if p.name.endswith('_3B_udm2.tif'):
                    udm2 = p
                    break

        if four_band or udm2:
            yield scene_id, four_band, udm2


def discover_planet_direct(planet_root: Path):
    """Yield (scene_id, four_band_path|None, udm2_path|None) per Planet-direct
    order root. Walks every PSScene/ subdir under each order dir."""
    if not planet_root.is_dir():
        return
    order_roots = [
        d for d in planet_root.iterdir()
        if d.is_dir() and any(
            pat.match(d.name) for pat in PLANET_ORDER_DIR_PATTERNS)
    ]

    # scene_id -> (4band, udm2)
    by_scene: dict[str, list[Path | None]] = defaultdict(lambda: [None, None])

    for root in order_roots:
        for ps_dir in root.rglob('PSScene'):
            if not ps_dir.is_dir():
                continue
            for f in ps_dir.iterdir():
                name = f.name
                m = SCENE_RE.match(name)
                if not m:
                    continue
                sid = m.group(1)
                low = name.lower()
                if 'rgb' in low or 'ndvi' in low or '_sr' in low:
                    continue
                if name.endswith('_3B_AnalyticMS_clip.tif'):
                    if by_scene[sid][0] is None:
                        by_scene[sid][0] = f
                elif name.endswith('_3B_udm2_clip.tif'):
                    if by_scene[sid][1] is None:
                        by_scene[sid][1] = f

    for sid, (fb, ud) in by_scene.items():
        if fb or ud:
            yield sid, fb, ud


def choose_sources(csdap_root: Path, planet_root: Path):
    """Combine CSDAP and Planet-direct discoveries into one dict keyed by
    scene_id. CSDAP wins for both 4-band and UDM2; Planet-direct fills gaps."""
    chosen: dict[str, dict] = {}

    for sid, fb, ud in discover_csdap(csdap_root):
        chosen[sid] = {
            'scene_id': sid,
            'four_band_src': str(fb) if fb else '',
            'udm2_src': str(ud) if ud else '',
            'four_band_origin': 'csdap' if fb else '',
            'udm2_origin': 'csdap' if ud else '',
        }

    for sid, fb, ud in discover_planet_direct(planet_root):
        rec = chosen.setdefault(sid, {
            'scene_id': sid, 'four_band_src': '', 'udm2_src': '',
            'four_band_origin': '', 'udm2_origin': '',
        })
        if not rec['four_band_src'] and fb:
            rec['four_band_src'] = str(fb)
            rec['four_band_origin'] = 'planet_direct'
        if not rec['udm2_src'] and ud:
            rec['udm2_src'] = str(ud)
            rec['udm2_origin'] = 'planet_direct'

    return chosen


def _filter_by_date(records: dict, start: date | None, end: date | None):
    if start is None and end is None:
        return records
    out = {}
    for sid, rec in records.items():
        d = datetime.strptime(sid[:8], '%Y%m%d').date()
        if start and d < start:
            continue
        if end and d > end:
            continue
        out[sid] = rec
    return out


def _clip_to(region, src_path: Path, out_path: Path) -> str:
    """Clip a single raster to `region`. Returns a status string."""
    with rasterio.open(src_path) as data:
        if not box(*data.bounds).intersects(region):
            return 'no_intersection'
        try:
            out_img, out_trans = rio_mask(
                data, shapes=[region], crop=True)
        except ValueError:
            return 'no_intersection'
        if data.nodata is not None and np.all(out_img == data.nodata):
            return 'nodata'
        meta = data.meta.copy()
        meta.update({
            'transform': out_trans,
            'height': out_img.shape[1],
            'width': out_img.shape[2],
            'driver': 'GTiff',
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, 'w', **meta) as out:
        out.write(out_img)
    return 'ok'


def _render_rgb(four_band_path: Path, out_path: Path) -> str:
    """Per-band 2/98 percentile stretch on the clipped 4-band. Planet
    AnalyticMS band order is [Blue, Green, Red, NIR] (1,2,3,4). Output as
    [Red, Green, Blue]."""
    with rasterio.open(four_band_path) as src:
        nodata = src.nodata
        if src.count < 4:
            return 'error'
        red = src.read(3).astype(np.float32)
        green = src.read(2).astype(np.float32)
        blue = src.read(1).astype(np.float32)
        validity = src.dataset_mask() > 0
        crs = src.crs
        transform = src.transform
        height, width = red.shape

    bands = (red, green, blue)
    out = np.zeros((3, height, width), dtype=np.uint8)
    for i, b in enumerate(bands):
        valid = validity.copy()
        if nodata is not None:
            valid &= b != nodata
        if not valid.any():
            continue
        lo, hi = np.percentile(b[valid], (2.0, 98.0))
        if hi <= lo:
            continue
        scaled = np.clip((b - lo) * (255.0 / (hi - lo)), 0, 255)
        scaled = np.where(valid, scaled, 0)
        out[i] = scaled.astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, 'w', driver='GTiff', dtype='uint8', count=3,
        height=height, width=width, crs=crs, transform=transform,
        photometric='RGB',
    ) as dst:
        dst.write(out)
    return 'ok'


def process_scene(rec: dict, region, output_root: Path, force: bool) -> str:
    sid = rec['scene_id']
    year = sid[:4]
    out_4 = output_root / '4band' / year / f'{sid}_4band.tif'
    out_u = output_root / 'udm2' / year / f'{sid}_udm2.tif'
    out_r = output_root / 'rgb' / year / f'{sid}_rgb.tif'

    have_all = all(p.exists() and p.stat().st_size > 0
                   for p in (out_4, out_u, out_r))
    if have_all and not force:
        return 'skipped_existing'

    # 4-band first; RGB depends on it.
    status_4 = 'missing_src'
    if rec['four_band_src']:
        if force or not (out_4.exists() and out_4.stat().st_size > 0):
            status_4 = _clip_to(region, Path(rec['four_band_src']), out_4)
        else:
            status_4 = 'ok'
    if status_4 != 'ok':
        # Don't make RGB without a clipped 4-band
        out_r_status = 'skipped'
    else:
        if force or not (out_r.exists() and out_r.stat().st_size > 0):
            out_r_status = _render_rgb(out_4, out_r)
        else:
            out_r_status = 'ok'

    status_u = 'missing_src'
    if rec['udm2_src']:
        if force or not (out_u.exists() and out_u.stat().st_size > 0):
            status_u = _clip_to(region, Path(rec['udm2_src']), out_u)
        else:
            status_u = 'ok'

    # Aggregate: any non-ok status surfaces; else 'ok'.
    statuses = [status_4, status_u, out_r_status]
    if all(s == 'ok' for s in statuses):
        return 'ok'
    if any(s == 'no_intersection' for s in statuses):
        return 'no_intersection'
    if any(s == 'nodata' for s in statuses):
        return 'nodata'
    return ';'.join(f'{k}={v}' for k, v in
                    zip(('four_band', 'udm2', 'rgb'), statuses))


# ---------- Planet API missing-files report ----------

def _utm_polygon_to_lonlat(region, src_crs='EPSG:32617'):
    return transform_geom(src_crs, 'EPSG:4326', mapping(region))


async def _query_planet(geom_lonlat, search_config, start: date, end: date):
    from planet import Auth, Session, data_filter
    auth = Auth.from_env()
    instruments = []
    publishing = []
    for sf in search_config.get('string_filters', []):
        if sf['field_name'] == 'instrument':
            instruments = sf['values']
        elif sf['field_name'] == 'publishing_stage':
            publishing = sf['values']

    conds = [
        data_filter.permission_filter(),
        data_filter.geometry_filter(geom_lonlat),
        data_filter.string_in_filter(
            field_name='item_type', values=['PSScene']),
        data_filter.date_range_filter(
            'acquired',
            gte=datetime.combine(start, datetime.min.time()),
            lt=datetime.combine(end, datetime.min.time())),
        data_filter.asset_filter(['ortho_analytic_4b']),
    ]
    if instruments:
        conds.append(data_filter.string_in_filter(
            field_name='instrument', values=instruments))
    if publishing:
        conds.append(data_filter.string_in_filter(
            field_name='publishing_stage', values=publishing))

    sfilter = data_filter.and_filter(conds)
    async with Session(auth=auth) as sess:
        client = sess.client('data')
        # limit=0 disables the SDK's default 100-item paging cap
        return [i async for i in client.search(
            ['PSScene'], sfilter, limit=0)]


def planet_api_missing_report(region, search_config, start: date, end: date,
                              local_scene_ids: set[str], output_root: Path):
    geom_lonlat = _utm_polygon_to_lonlat(region)
    items = asyncio.run(_query_planet(geom_lonlat, search_config, start,
                                      datetime(end.year, end.month, end.day)
                                      .date()))
    api_ids = {it['id'] for it in items}
    missing = api_ids - local_scene_ids
    by_id = {it['id']: it for it in items}

    out_csv = output_root / 'missing_at_planet.csv'
    rows = []
    for sid in sorted(missing):
        it = by_id[sid]
        props = it.get('properties', {})
        rows.append({
            'scene_id': sid,
            'acquired': props.get('acquired', ''),
            'instrument': props.get('instrument', ''),
            'cloud_cover': props.get('cloud_cover', ''),
        })
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'scene_id', 'acquired', 'instrument', 'cloud_cover'])
        w.writeheader()
        w.writerows(rows)

    # Per-month summary
    from collections import Counter
    by_month = Counter(sid[:6] for sid in missing)
    print(f'\nPlanet API: {len(api_ids)} scenes match the 50ha AOI from '
          f'{start} to {end}.')
    print(f'Local clips cover: {len(api_ids & local_scene_ids)}.')
    print(f'Missing locally:   {len(missing)} '
          f'(written to {out_csv})')
    if by_month:
        print('Per month:')
        for m in sorted(by_month):
            print(f'  {m}: {by_month[m]}')


# ---------- CLI ----------

# Statuses that mean "the source raster genuinely has no usable pixels in the
# 50ha polygon." Re-running just re-confirms this, so on subsequent runs we
# skip these scenes if their source paths haven't changed.
TERMINAL_NEGATIVE_STATUSES = {'no_intersection', 'nodata'}


def _load_prior_inventory(output_root: Path) -> dict[str, dict]:
    """Return prior inventory.csv keyed by scene_id, or {} if absent."""
    inv_csv = output_root / 'inventory.csv'
    if not inv_csv.is_file():
        return {}
    with open(inv_csv, newline='') as f:
        return {row['scene_id']: row for row in csv.DictReader(f)}


def _parse_date(s):
    return datetime.strptime(s, '%Y-%m-%d').date()


@click.command()
@click.option('--clip-config', type=click.Path(
    path_type=Path, exists=True, dir_okay=False),
    default=Path('config/clip_50ha_plot.yml'), show_default=True)
@click.option('--search-config', type=click.Path(
    path_type=Path, exists=True, dir_okay=False),
    default=Path('config/planet_search.yml'), show_default=True)
@click.option('--output-root', type=click.Path(path_type=Path),
              default=Path('/Volumes/Earth03/flower/planet_clipped'),
              show_default=True)
@click.option('--csdap-root', type=click.Path(path_type=Path),
              default=Path('/Volumes/Earth03/flower/csdap'),
              show_default=True)
@click.option('--planet-root', type=click.Path(path_type=Path),
              default=Path('/Volumes/Earth03/flower/planet'),
              show_default=True)
@click.option('--start-date', type=str, default='2020-01-01',
              show_default=True)
@click.option('--end-date', type=str, default=None,
              help='Exclusive end date (default: today + 1 day)')
@click.option('--no-api-check', is_flag=True,
              help='Skip the Planet API missing-files report.')
@click.option('--force', is_flag=True,
              help='Re-clip and re-render even if outputs already exist.')
@click.option('--limit', type=int, default=0,
              help='Process only the first N scenes (for smoke tests).')
def main(clip_config, search_config, output_root, csdap_root, planet_root,
         start_date, end_date, no_api_check, force, limit):
    clip_cfg = load_config(clip_config)
    search_cfg = load_config(search_config)
    region = shape(clip_cfg['region'])

    start = _parse_date(start_date)
    end = (_parse_date(end_date) if end_date
           else date.today())

    # Source inventory
    print('Discovering local sources...')
    records = choose_sources(csdap_root, planet_root)
    records = _filter_by_date(records, start, end)
    print(f'  {len(records)} unique scene IDs in [{start}, {end}].')

    items = sorted(records.values(), key=lambda r: r['scene_id'])
    if limit:
        items = items[:limit]

    output_root.mkdir(parents=True, exist_ok=True)

    prior = _load_prior_inventory(output_root) if not force else {}
    skipped_negatives = 0

    inv_rows = []
    for rec in tqdm(items, desc='Clipping'):
        sid = rec['scene_id']
        prev = prior.get(sid)
        if (prev
                and prev.get('status') in TERMINAL_NEGATIVE_STATUSES
                and prev.get('four_band_src', '') == rec['four_band_src']
                and prev.get('udm2_src', '') == rec['udm2_src']):
            # Same source, previously confirmed empty inside 50ha polygon.
            inv_rows.append({
                'scene_id': sid,
                'four_band_origin': rec['four_band_origin'],
                'udm2_origin': rec['udm2_origin'],
                'four_band_src': rec['four_band_src'],
                'udm2_src': rec['udm2_src'],
                'status': prev['status'],
            })
            skipped_negatives += 1
            continue
        try:
            status = process_scene(rec, region, output_root, force)
        except Exception as e:  # noqa: BLE001
            status = f'error:{type(e).__name__}:{e}'
        inv_rows.append({
            'scene_id': sid,
            'four_band_origin': rec['four_band_origin'],
            'udm2_origin': rec['udm2_origin'],
            'four_band_src': rec['four_band_src'],
            'udm2_src': rec['udm2_src'],
            'status': status,
        })

    if skipped_negatives:
        print(f'Skipped {skipped_negatives} scenes with prior terminal '
              'negative status (no_intersection / nodata, unchanged source).')

    inv_csv = output_root / 'inventory.csv'
    with open(inv_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'scene_id', 'four_band_origin', 'udm2_origin',
            'four_band_src', 'udm2_src', 'status'])
        w.writeheader()
        w.writerows(inv_rows)
    print(f'Wrote {inv_csv}')

    # Locally accounted-for scene IDs: those with an OK 4-band clip on
    # disk, plus those whose source has been confirmed empty over the
    # 50ha polygon (TERMINAL_NEGATIVE_STATUSES). The latter would never
    # produce a 4band file but should not be reported as "missing at
    # Planet" — re-fetching won't help.
    fourband_dir = output_root / '4band'
    local_ids = {p.name.split('_4band.tif')[0]
                 for p in fourband_dir.rglob('*_4band.tif')
                 if p.stat().st_size > 0}
    local_ids |= {row['scene_id'] for row in inv_rows
                  if row['status'] in TERMINAL_NEGATIVE_STATUSES}

    if not no_api_check:
        try:
            planet_api_missing_report(
                region, search_cfg, start, end, local_ids, output_root)
        except Exception as e:  # noqa: BLE001
            print(f'\nPlanet API check failed: {e}', file=sys.stderr)


if __name__ == '__main__':
    main()
