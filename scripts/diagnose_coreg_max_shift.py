#!/usr/bin/env python
"""Test whether failed Planet-to-drone registrations fail because the shift
exceeds AROSICS's search limit.

Until November 2026, compute_coreg_shift (apply_drone_labels_coreg.py)
searched at most max_shift = 10 Planet pixels (30 m) per axis. For every
failed scene in a coreg JSON (as written by coreg_planet_to_nearest_drone.py),
this reruns AROSICS alone (arosics_shift) at each --max-shift, other settings
unchanged, against the same ortho. max_shift is in Planet pixels (3 m), so the
default 20 and 30 search 60 m and 90 m. It also runs the independent
brute-force NCC match of coreg_ncc.py, searching --ncc-search m.

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

from apply_drone_labels_coreg import (arosics_shift, find_ocm_mask,
                                      load_as_geoarray)
from coreg_ncc import ncc_shift


def group_rows(args):
    """Every scene of one ortho group; runs in a worker."""
    dronefile, recs, planetdir, ocmdir, max_shifts, search_m = args
    drone_ga = load_as_geoarray(dronefile)
    rows = []
    for rec in recs:
        planetfile = os.path.join(planetdir, rec['planet_rel'])
        row = {k: rec[k] for k in ('scene', 'date', 'clear_fraction',
                                   'days_to_drone', 'coreg_ok', 'x_shift_m',
                                   'y_shift_m', 'role')}
        row['drone_file'] = dronefile
        for k in max_shifts:
            x, y, ok = arosics_shift(dronefile, planetfile,
                                     planet_match_band=3, drone_ga=drone_ga,
                                     max_shift=k)
            row.update({f'ok_{k}': ok, f'x_{k}': x, f'y_{k}': y})
        ocm = find_ocm_mask(planetfile, planetdir, ocmdir)
        ncc = ncc_shift(planetfile, dronefile, ocm, search_m=search_m)
        row.update(ncc_x=ncc['ncc_x_m'], ncc_y=ncc['ncc_y_m'],
                   **{k: ncc[k] for k in ('ncc_peak', 'ncc_second',
                                          'ncc_clear')})
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
