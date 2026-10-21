#!/usr/bin/env python
"""Download MODIS MCD43A1 BRDF parameters + mandatory QA over BCNM via AppEEARS.

Submits an AppEEARS area task for the BRDF/Albedo model parameters (iso, vol,
geo) and the per-band mandatory quality of the four MODIS bands that match the
PlanetScope 4-band sensor (MODIS B3 blue, B4 green, B1 red, B2 NIR), in the
native MODIS sinusoidal projection, as netCDF. The AOI is the Planet order box
of `config/planet_search.yml` buffered by ~5 km so that QA30 and the 5 km
neighbourhood climatology (see scripts/build_modis_reference.py) are defined at
the box edges.

Credentials are read from the `urs.earthdata.nasa.gov` entry of ~/.netrc and
are never printed.

Typical usage:
    # one-month test task to check layer/file naming
    python scripts/fetch_modis_appeears.py --start 2024-02-01 --end 2024-02-29 \\
        --out-dir /Volumes/Earth03/flower/modis/mcd43a1_bcnm/test
    # full record
    python scripts/fetch_modis_appeears.py --start 2005-01-01 --end 2026-10-15
    # resume polling/downloading tasks submitted earlier
    python scripts/fetch_modis_appeears.py --task-id <id> [--task-id <id> ...]
"""
from __future__ import annotations

import json
import netrc
import time
from datetime import date
from pathlib import Path

import click
import requests

API = 'https://appeears.earthdatacloud.nasa.gov/api'
PRODUCT = 'MCD43A1.061'
MODIS_BANDS = [1, 2, 3, 4]
LAYERS = (
    [f'BRDF_Albedo_Parameters_Band{b}' for b in MODIS_BANDS]
    + [f'BRDF_Albedo_Band_Mandatory_Quality_Band{b}' for b in MODIS_BANDS]
)
# config/planet_search.yml box (lon -79.899..-79.791, lat 9.089..9.215) + ~5 km
AOI = (-79.945, 9.043, -79.745, 9.261)
DEFAULT_OUT = Path('/Volumes/Earth03/flower/modis/mcd43a1_bcnm')


def login() -> str:
    auth = netrc.netrc().authenticators('urs.earthdata.nasa.gov')
    if auth is None:
        raise click.ClickException('no urs.earthdata.nasa.gov entry in ~/.netrc')
    user, _, password = auth
    r = check(requests.post(f'{API}/login', auth=(user, password), timeout=60))
    return r.json()['token']


def task_request(name: str, start: date, end: date) -> dict:
    x0, y0, x1, y1 = AOI
    geo = {'type': 'FeatureCollection', 'features': [{
        'type': 'Feature', 'properties': {},
        'geometry': {'type': 'Polygon', 'coordinates': [[
            [x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]},
    }]}
    return {
        'task_type': 'area',
        'task_name': name,
        'params': {
            'dates': [{'startDate': start.strftime('%m-%d-%Y'),
                       'endDate': end.strftime('%m-%d-%Y')}],
            'layers': [{'product': PRODUCT, 'layer': l} for l in LAYERS],
            'output': {'format': {'type': 'netcdf4'}, 'projection': 'native'},
            'geo': geo,
        },
    }


def check(r: requests.Response) -> requests.Response:
    if not r.ok:
        raise click.ClickException(f'{r.status_code} {r.url}: {r.text[:2000]}')
    return r


def get(url: str, headers: dict, **kw) -> requests.Response:
    """GET, logging in again once if the token was invalidated (403)."""
    r = requests.get(url, headers=headers, **kw)
    if r.status_code == 403:
        headers['Authorization'] = f'Bearer {login()}'
        r = requests.get(url, headers=headers, **kw)
    return check(r)


def wait_for(task_id: str, headers: dict, poll: int) -> None:
    while True:
        r = get(f'{API}/task/{task_id}', headers, timeout=60)
        status = r.json()['status']
        click.echo(f'{time.strftime("%H:%M:%S")} task {task_id}: {status}')
        if status == 'done':
            return
        if status in ('error', 'expired'):
            raise click.ClickException(json.dumps(r.json()))
        time.sleep(poll)


def download(task_id: str, headers: dict, out_dir: Path) -> None:
    r = get(f'{API}/bundle/{task_id}', headers, timeout=60)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in r.json()['files']:
        dest = out_dir / f['file_name']
        if dest.exists() and dest.stat().st_size == f['file_size']:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with get(f'{API}/bundle/{task_id}/{f["file_id"]}', headers,
                 stream=True, timeout=600) as s:
            with open(dest, 'wb') as fh:
                for chunk in s.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        click.echo(f'downloaded {dest}')


@click.command()
@click.option('--start', type=click.DateTime(['%Y-%m-%d']), default='2005-01-01',
              show_default=True)
@click.option('--end', type=click.DateTime(['%Y-%m-%d']),
              default=date.today().isoformat(), show_default=True)
@click.option('--out-dir', type=click.Path(path_type=Path), default=DEFAULT_OUT,
              show_default=True)
@click.option('--task-name', default='bcnm_mcd43a1', show_default=True)
@click.option('--task-id', 'task_ids', multiple=True,
              help='Resume existing task(s) instead of submitting new ones.')
@click.option('--chunk-years', type=int, default=3, show_default=True,
              help='Split the date range into tasks of this many years '
                   '(AppEEARS caps the number of files per task).')
@click.option('--poll', type=int, default=120, show_default=True,
              help='Seconds between status checks.')
def main(start, end, out_dir, task_name, task_ids, chunk_years, poll):
    # No logout: it invalidates the token of any other running instance.
    headers = {'Authorization': f'Bearer {login()}'}
    out_dir.mkdir(parents=True, exist_ok=True)
    if not task_ids:
        task_ids = []
        for y0 in range(start.year, end.year + 1, chunk_years):
            c0 = max(start.date(), date(y0, 1, 1))
            c1 = min(end.date(), date(y0 + chunk_years - 1, 12, 31))
            name = f'{task_name}_{c0:%Y%m%d}_{c1:%Y%m%d}'
            req = task_request(name, c0, c1)
            r = check(requests.post(f'{API}/task', json=req, headers=headers, timeout=60))
            task_ids.append(r.json()['task_id'])
            (out_dir / f'{name}.json').write_text(
                json.dumps(dict(req, task_id=task_ids[-1]), indent=2))
            click.echo(f'submitted task {task_ids[-1]} ({name})')
    for tid in task_ids:
        wait_for(tid, headers, poll)
        download(tid, headers, out_dir / tid)

if __name__ == '__main__':
    main()
