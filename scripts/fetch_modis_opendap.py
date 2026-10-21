#!/usr/bin/env python
"""Download MODIS MCD43A1 BRDF parameters + mandatory QA over BCNM via OPeNDAP.

Faster alternative to scripts/fetch_modis_appeears.py (whose queue can sit in
"pending" for many hours). Each daily MCD43A1.061 granule of tile h10v08 is
listed through CMR and only the pixel window covering the AOI is read from
NASA's cloud OPeNDAP (Hyrax) server with a DAP4 index constraint: ~160 KB and
a few seconds per day instead of the 138 MB HDF4 granule. Granules that
OPeNDAP cannot serve (missing .dmrpp sidecar, HTTP 404; ~250 days, mostly
2006-2012) are downloaded whole and cut locally. The layers are the
same as the AppEEARS request: BRDF_Albedo_Parameters_Band{1-4} (iso, vol, geo)
and BRDF_Albedo_Band_Mandatory_Quality_Band{1-4}.

Each day is written as `<out>/daily/<YYYY>/MCD43A1.A<YYYYDDD>.nc` with raw int16
parameters (scale_factor/_FillValue kept), dims (time, y, x, param) and
sinusoidal pixel-centre coordinates. `--combine` concatenates the days into
`<out>/yearly/mcd43a1_h10v08_<YYYY>.nc`, the input of
scripts/build_modis_reference.py.

Credentials come from the `urs.earthdata.nasa.gov` entry of ~/.netrc (requests
applies it on the Earthdata login redirect) and are never printed.

Typical usage:
    python scripts/fetch_modis_opendap.py --start 2024-02-01 --end 2024-02-29 --combine
    python scripts/fetch_modis_opendap.py --start 2005-01-01 --end 2026-10-15 --combine
"""
from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

import click
import numpy as np
import requests
import xarray as xr
from tqdm import tqdm

from fetch_modis_appeears import AOI, MODIS_BANDS

CMR = 'https://cmr.earthdata.nasa.gov/search/granules.json'
COLLECTION = 'C2343116130-LPCLOUD'  # MCD43A1.061, LP DAAC cloud
OPENDAP = f'https://opendap.earthdata.nasa.gov/collections/{COLLECTION}/granules'
DATA_URL = 'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MCD43A1.061'
TILE = 'h10v08'
# MODIS sinusoidal tile grid (sphere R = 6371007.181 m)
R_EARTH = 6371007.181
TILE_SIZE = 1111950.5197665
X_ORIGIN, Y_ORIGIN = -20015109.355798, 10007554.677899
NPIX = 2400
PIXEL = TILE_SIZE / NPIX
DEFAULT_OUT = Path('/Volumes/Earth03/flower/modis/mcd43a1_bcnm/opendap')

_local = threading.local()


def session() -> requests.Session:
    if not hasattr(_local, 's'):
        _local.s = requests.Session()
    return _local.s


def tile_window(aoi=AOI, tile=TILE) -> tuple[int, int, int, int, float, float]:
    """(row0, row1, col0, col1) inclusive pixel window of `aoi` in `tile`, plus
    the tile's upper-left corner (x, y) in sinusoidal metres."""
    h, v = int(tile[1:3]), int(tile[4:6])
    x0t = X_ORIGIN + h * TILE_SIZE
    y0t = Y_ORIGIN - v * TILE_SIZE
    lon0, lat0, lon1, lat1 = aoi
    xs = [R_EARTH * np.radians(lo) * np.cos(np.radians(la))
          for lo in (lon0, lon1) for la in (lat0, lat1)]
    ys = [R_EARTH * np.radians(la) for la in (lat0, lat1)]
    c0, c1 = int(np.floor((min(xs) - x0t) / PIXEL)), int(np.ceil((max(xs) - x0t) / PIXEL))
    r0, r1 = int(np.floor((y0t - max(ys)) / PIXEL)), int(np.ceil((y0t - min(ys)) / PIXEL))
    if min(r0, c0) < 0 or max(r1, c1) >= NPIX:
        raise click.ClickException(f'AOI {aoi} is not inside tile {tile}')
    return r0, r1, c0, c1, x0t, y0t


def list_granules(start: date, end: date, tile=TILE) -> dict[date, str]:
    """Granule names by acquisition date (latest production if duplicated)."""
    params = {'collection_concept_id': COLLECTION, 'page_size': 2000,
              'temporal': f'{start:%Y-%m-%d}T00:00:00Z,{end:%Y-%m-%d}T23:59:59Z',
              'readable_granule_name': f'MCD43A1.A*.{tile}.*',
              'options[readable_granule_name][pattern]': 'true'}
    headers, names = {}, []
    while True:
        r = requests.get(CMR, params=params, headers=headers, timeout=120)
        r.raise_for_status()
        entries = r.json()['feed']['entry']
        names += [e['title'] for e in entries]
        after = r.headers.get('CMR-Search-After')
        if not entries or not after:
            break
        headers['CMR-Search-After'] = after
    by_day: dict[date, str] = {}
    for n in sorted(names):  # production timestamp sorts last
        m = re.match(r'MCD43A1\.A(\d{7})\.', n)
        d = datetime.strptime(m.group(1), '%Y%j').date()
        if start <= d <= end:
            by_day[d] = n
    return by_day


def constraint(r0, r1, c0, c1) -> str:
    s = f'[{r0}:{r1}][{c0}:{c1}]'
    grp = '/MOD_Grid_BRDF/Data_Fields'
    return ';'.join(f'{grp}/BRDF_Albedo_Parameters_Band{b}{s}[0:2];'
                    f'{grp}/BRDF_Albedo_Band_Mandatory_Quality_Band{b}{s}'
                    for b in MODIS_BANDS)


def get_with_retry(url: str, retries: int = 5, **kw) -> requests.Response | None:
    """GET with exponential backoff; None if the server answers 404."""
    for attempt in range(retries):
        try:
            r = session().get(url, timeout=600, **kw)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(10 * 2 ** attempt)


def read_opendap(granule: str, window, tmp: Path) -> xr.Dataset | None:
    r0, r1, c0, c1 = window[:4]
    r = get_with_retry(f'{OPENDAP}/{granule}.dap.nc4',
                       params={'dap4.ce': constraint(r0, r1, c0, c1)})
    if r is None:
        return None
    tmp.write_bytes(r.content)
    with xr.open_dataset(tmp, group='/MOD_Grid_BRDF/Data_Fields',
                         mask_and_scale=False) as raw:
        ds = raw.load()
    tmp.unlink()
    return ds.rename({'YDim_MOD_Grid_BRDF': 'y', 'XDim_MOD_Grid_BRDF': 'x',
                      'Num_Parameters_MOD_Grid_BRDF': 'param'})


def read_hdf(granule: str, window, tmp: Path) -> xr.Dataset:
    """Fallback for granules OPeNDAP cannot serve (no .dmrpp sidecar, 404):
    download the full HDF4 granule (~140 MB) and cut the window locally."""
    import netCDF4  # libnetcdf built with HDF4 reads MODIS HDF-EOS2 directly
    r0, r1, c0, c1 = window[:4]
    r = get_with_retry(f'{DATA_URL}/{granule}/{granule}.hdf')
    if r is None:
        raise FileNotFoundError(f'{granule}.hdf not found')
    tmp.write_bytes(r.content)
    data = {}
    with netCDF4.Dataset(tmp) as nc:
        for b in MODIS_BANDS:
            for name, dims in ((f'BRDF_Albedo_Parameters_Band{b}', ('y', 'x', 'param')),
                               (f'BRDF_Albedo_Band_Mandatory_Quality_Band{b}', ('y', 'x'))):
                v = nc[name]
                v.set_auto_maskandscale(False)
                attrs = {k: v.getncattr(k) for k in v.ncattrs()}
                fill = attrs.pop('_FillValue')
                arr = v[r0:r1 + 1, c0:c1 + 1].astype(v.dtype.newbyteorder('='))
                data[name] = xr.Variable(dims, np.ma.filled(arr, fill), attrs)
                data[name].encoding['_FillValue'] = fill
    tmp.unlink()
    return xr.Dataset(data)


def fetch_day(day: date, granule: str, window, out_dir: Path) -> Path:
    dest = out_dir / 'daily' / f'{day:%Y}' / f'MCD43A1.A{day:%Y%j}.nc'
    if dest.exists():
        return dest
    r0, r1, c0, c1, x0t, y0t = window
    dest.parent.mkdir(parents=True, exist_ok=True)
    ds = read_opendap(granule, window, dest.with_suffix('.dap.nc4'))
    if ds is None:
        ds = read_hdf(granule, window, dest.with_suffix('.hdf'))
    ds = ds.assign_coords(
        y=y0t - (np.arange(r0, r1 + 1) + 0.5) * PIXEL,
        x=x0t + (np.arange(c0, c1 + 1) + 0.5) * PIXEL,
        param=['iso', 'vol', 'geo'],
    ).expand_dims(time=[np.datetime64(day, 'ns')])
    ds.attrs = {'granule': granule, 'tile': TILE,
                'crs': f'+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R={R_EARTH} +units=m +no_defs'}
    part = dest.with_suffix('.part')
    ds.to_netcdf(part)
    part.rename(dest)
    return dest


def combine(out_dir: Path) -> None:
    for ydir in sorted((out_dir / 'daily').iterdir()):
        files = sorted(ydir.glob('MCD43A1.A*.nc'))
        if not files:
            continue
        ds = xr.concat([xr.open_dataset(f, mask_and_scale=False) for f in files], 'time')
        ds.attrs.pop('granule', None)
        dest = out_dir / 'yearly' / f'mcd43a1_{TILE}_{ydir.name}.nc'
        dest.parent.mkdir(parents=True, exist_ok=True)
        enc = {v: {'zlib': True, 'complevel': 4} for v in ds.data_vars}
        ds.to_netcdf(dest, encoding=enc)
        click.echo(f'wrote {dest} ({ds.sizes["time"]} days)')


@click.command()
@click.option('--start', type=click.DateTime(['%Y-%m-%d']), default='2005-01-01',
              show_default=True)
@click.option('--end', type=click.DateTime(['%Y-%m-%d']),
              default=date.today().isoformat(), show_default=True)
@click.option('--out-dir', type=click.Path(path_type=Path), default=DEFAULT_OUT,
              show_default=True)
@click.option('--workers', type=int, default=4, show_default=True,
              help='Parallel OPeNDAP requests.')
@click.option('--combine/--no-combine', 'do_combine', default=False,
              help='Concatenate the daily files into yearly files afterwards.')
def main(start, end, out_dir, workers, do_combine):
    window = tile_window()
    r0, r1, c0, c1 = window[:4]
    click.echo(f'{TILE} window rows {r0}-{r1}, cols {c0}-{c1} '
               f'({r1 - r0 + 1} x {c1 - c0 + 1} pixels)')
    granules = list_granules(start.date(), end.date())
    click.echo(f'{len(granules)} granules from CMR')
    failed = []
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(fetch_day, d, g, window, out_dir): d for d, g in granules.items()}
        for f in tqdm(as_completed(futs), total=len(futs), unit='day'):
            try:
                f.result()
            except Exception as e:  # keep going; re-running retries missing days
                failed.append(futs[f])
                tqdm.write(f'{futs[f]}: {e}')
    if failed:
        click.echo(f'{len(failed)} days failed (re-run to retry): '
                   f'{", ".join(map(str, sorted(failed)[:20]))}')
    if do_combine:
        combine(out_dir)


if __name__ == '__main__':
    main()
