#!/usr/bin/env python
"""Test whether failed Planet-to-drone registrations fail because the shift
exceeds AROSICS's search limit.

compute_coreg_shift (apply_drone_labels_coreg.py) searches at most
max_shift = 10 Planet pixels (30 m) per axis. For every failed scene in a
coreg JSON (as written by coreg_planet_to_nearest_drone.py), this reruns
AROSICS at each --max-shift, other settings unchanged, against the same
ortho. max_shift is in Planet pixels (3 m), so the default 20 and 30 search
60 m and 90 m. It also runs an independent brute-force match:
1. The Planet red band is resampled to a 1 m grid on the ortho.
2. Pixels that OmniCloudMask does not call clear, and nodata, are set to the
   clear mean.
3. The part of the scene that stays inside the ortho under any shift up to
   --ncc-search m is correlated (normalized cross-correlation) against the
   ortho's red band, block-averaged to 1 m, over every shift in that range.

Controls are scenes that already registered, so the NCC sign convention and
the stability of a wider search can be checked:
- --controls-per-group successes from each ortho group that has a failure;
- every success with a shift above --near-cap-m on either axis.

Writes one CSV row per scene, appended after every ortho group. A rerun skips
scenes already in it. Columns:
- the JSON record's scene, date, clear_fraction, days_to_drone, coreg_ok,
  x_shift_m, y_shift_m, and role (failed / control);
- ok_<k>, x_<k>, y_<k> for AROSICS at max_shift k;
- ncc_x, ncc_y (m, the shift to apply to the Planet scene, as AROSICS's
  corrected_shifts_map), ncc_peak, ncc_second (best NCC more than 15 m from
  the peak) and ncc_clear (clear share of the template).

See docs/handoff_rerun_coregistration.md.
"""
import csv
import json
import multiprocessing as mp
import os

import click
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from skimage.feature import match_template

from apply_drone_labels_coreg import (compute_coreg_shift, find_ocm_mask,
                                      load_as_geoarray)
from build_crown_pixel_series import OCM_CLEAR, read_ocm

NCC_RES = 1.0
SECOND_EXCLUDE_M = 15.0


def drone_red_1m(dronefile):
    """The ortho's red band block-averaged to NCC_RES, its transform, and its
    valid mask."""
    with rasterio.open(dronefile) as src:
        b = src.bounds
        w = int((b.right - b.left) // NCC_RES)
        h = int((b.top - b.bottom) // NCC_RES)
        red = src.read(1, out_shape=(h, w), resampling=Resampling.average,
                       masked=True)
        transform = from_origin(b.left, b.top, NCC_RES, NCC_RES)
        crs = src.crs
    valid = ~np.ma.getmaskarray(red) & (red.filled(0) > 0)
    return red.filled(0).astype(np.float32), valid, transform, crs


def ncc_shift(planetfile, ocm_path, drone, search_m):
    """Brute-force NCC shift of a Planet scene against the 1 m ortho."""
    red_d, valid_d, transform, crs = drone
    with rasterio.open(planetfile) as src:
        red_p = src.read(3).astype(np.float32)
        valid_p = src.read_masks(3) > 0
        clear = valid_p
        if ocm_path is not None:
            cls, _ = read_ocm(ocm_path, src)
            clear = valid_p & (cls == OCM_CLEAR)
        on_d = np.zeros(red_d.shape, np.float32)
        clr_d = np.zeros(red_d.shape, np.uint8)
        reproject(red_p, on_d, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs=crs,
                  resampling=Resampling.bilinear)
        reproject(clear.astype(np.uint8), clr_d, src_transform=src.transform,
                  src_crs=src.crs, dst_transform=transform, dst_crs=crs,
                  resampling=Resampling.nearest)
    m = int(search_m / NCC_RES)
    tpl = on_d[m:-m, m:-m].copy()
    tclr = clr_d[m:-m, m:-m].astype(bool)
    if tclr.mean() < 0.05:
        return np.nan, np.nan, np.nan, np.nan, float(tclr.mean())
    tpl[~tclr] = tpl[tclr].mean()
    img = red_d.copy()
    img[~valid_d] = img[valid_d].mean()
    r = match_template(img, tpl)
    i, j = np.unravel_index(np.argmax(r), r.shape)
    peak = float(r[i, j])
    yy, xx = np.ogrid[:r.shape[0], :r.shape[1]]
    far = (yy - i) ** 2 + (xx - j) ** 2 > (SECOND_EXCLUDE_M / NCC_RES) ** 2
    second = float(r[far].max()) if far.any() else np.nan
    # The best match is (j - m, i - m) pixels right and down of the scene's
    # claimed position. The correction to apply is its negative, in
    # AROSICS's corrected_shifts_map convention (checked on scenes AROSICS
    # registers: NCC (-11, -33) m against AROSICS (-10.4, -31.8) m).
    return (-(j - m) * NCC_RES, (i - m) * NCC_RES, peak, second,
            float(tclr.mean()))


def group_rows(args):
    """Every scene of one ortho group; runs in a worker."""
    dronefile, recs, planetdir, ocmdir, max_shifts, search_m = args
    drone_ga = load_as_geoarray(dronefile)
    drone = drone_red_1m(dronefile)
    rows = []
    for rec in recs:
        planetfile = os.path.join(planetdir, rec['planet_rel'])
        row = {k: rec[k] for k in ('scene', 'date', 'clear_fraction',
                                   'days_to_drone', 'coreg_ok', 'x_shift_m',
                                   'y_shift_m', 'role')}
        row['drone_file'] = dronefile
        for k in max_shifts:
            x, y, ok = compute_coreg_shift(dronefile, planetfile,
                                           planet_match_band=3,
                                           drone_ga=drone_ga, max_shift=k)
            row.update({f'ok_{k}': ok, f'x_{k}': x, f'y_{k}': y})
        ocm = find_ocm_mask(planetfile, planetdir, ocmdir)
        (row['ncc_x'], row['ncc_y'], row['ncc_peak'], row['ncc_second'],
         row['ncc_clear']) = ncc_shift(planetfile, ocm, drone, search_m)
        rows.append(row)
    return rows


@click.command()
@click.argument('coreg_json', type=click.Path(exists=True, dir_okay=False))
@click.argument('planetdir', type=click.Path(exists=True, file_okay=False))
@click.argument('ocmdir', type=click.Path(exists=True, file_okay=False))
@click.argument('out_csv', type=click.Path(dir_okay=False))
@click.option('--max-shift', 'max_shifts', multiple=True, type=int,
              default=(20, 30), show_default=True,
              help='AROSICS max_shift values to try (3 m Planet pixels: '
                   '20 is 60 m).')
@click.option('--ncc-search', 'search_m', default=90.0, show_default=True,
              help='NCC search half-width (m).')
@click.option('--controls-per-group', default=1, show_default=True)
@click.option('--near-cap-m', default=20.0, show_default=True)
@click.option('--path-map', default='/scratch/tree-monitoring=/Volumes/Earth03/flower',
              show_default=True, help='OLD=NEW prefix rewrite for drone_file.')
@click.option('--workers', default=2, show_default=True,
              help='Each holds one ortho (up to ~4 GB).')
@click.option('--seed', default=0, show_default=True)
def main(coreg_json, planetdir, ocmdir, out_csv, max_shifts, search_m,
         controls_per_group, near_cap_m, path_map, workers, seed):
    """Rerun failed registrations in COREG_JSON at wider search limits."""
    old, new = path_map.split('=')
    recs = pd.DataFrame(json.load(open(coreg_json)))
    recs['drone_file'] = recs['drone_file'].str.replace(old, new, regex=False)
    failed = recs[~recs['coreg_ok']].assign(role='failed')
    ok = recs[recs['coreg_ok']]
    rng = np.random.default_rng(seed)
    ctrl = [g.iloc[rng.permutation(len(g))[:controls_per_group]]
            for f, g in ok.groupby('drone_file')
            if f in set(failed['drone_file'])]
    near = ok[np.maximum(ok['x_shift_m'].abs(), ok['y_shift_m'].abs()) > near_cap_m]
    ctrl = pd.concat(ctrl + [near]).drop_duplicates('scene').assign(role='control')
    todo = pd.concat([failed, ctrl], ignore_index=True)
    done = set(pd.read_csv(out_csv)['scene']) if os.path.exists(out_csv) else set()
    todo = todo[~todo['scene'].isin(done)]
    missing = sorted(set(todo['drone_file']) - {f for f in todo['drone_file']
                                                if os.path.exists(f)})
    if missing:
        raise click.ClickException(f'orthos not found, e.g. {missing[0]}')
    # clearest groups first, so the most informative rows arrive early
    order = todo.groupby('drone_file')['clear_fraction'].max().sort_values(
        ascending=False).index
    jobs = [(f, todo[todo['drone_file'] == f].to_dict('records'), planetdir,
             ocmdir, max_shifts, search_m) for f in order]
    click.echo(f'{len(failed)} failed + {len(ctrl)} controls; {len(todo)} to do '
               f'in {len(jobs)} ortho groups')
    ctx = mp.get_context('spawn')
    with ctx.Pool(workers, maxtasksperchild=1) as pool:
        for n, rows in enumerate(pool.imap_unordered(group_rows, jobs), 1):
            new_file = not os.path.exists(out_csv)
            with open(out_csv, 'a', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                if new_file:
                    w.writeheader()
                w.writerows(rows)
            click.echo(f'{n}/{len(jobs)} groups', err=True)


if __name__ == '__main__':
    main()
