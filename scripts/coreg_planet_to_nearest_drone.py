#!/usr/bin/env python
"""Coregister Planet scenes away from drone dates to the nearest drone ortho.

The temporal-interpolation model is applied to Planet scenes that have no
drone flight within a few days, so there is no same-date ortho to align
them to. Each such scene is matched instead to the drone ortho nearest in
time, with the same AROSICS settings as the training build
(`compute_coreg_shift` in apply_drone_labels_coreg.py: Red band in both
images, ws 200, max_shift 10). The canopy geometry changes little over a
few weeks, and the shift being estimated is Planet's own geolocation error.

Scenes within --exclude-within-days of any drone flight (the training-time
scenes) and scenes whose OmniCloudMask clear fraction is below
--min-clear-fraction are skipped. Scenes are processed grouped by ortho, each
group in a fresh worker process so a ~4 GB mavic ortho is read once and its
memory is returned afterwards. The output is rewritten after every group,
and a rerun skips scenes already in it.

Output records follow coreg_log.json, plus the scene's path relative to
PLANETDIR (so the file can be used on another machine), its date, the
ortho's date, and the distance in days to the nearest drone flight.
See docs/planet_temporal_generalization.md.
"""
import json
import logging
import multiprocessing as mp
import os
import re
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

from apply_drone_labels_coreg import (
    DATE_PATTERN,
    compute_clear_fraction,
    compute_coreg_shift,
    find_ocm_mask,
    get_planet_date,
    load_as_geoarray,
    write_coreg_log,
)

log = logging.getLogger(__name__)


def index_orthos(drone_dirs):
    """{date: ortho path} over every *.tif in the drone dirs. Where two
    dirs hold the same date, the first dir given wins."""
    orthos = {}
    for d in drone_dirs:
        for f in sorted(Path(d).glob('*.tif')):
            m = re.search(DATE_PATTERN, f.name)
            if m is None:
                continue
            date = pd.to_datetime(m.group(), format='%Y_%m_%d')
            orthos.setdefault(date, str(f))
    if not orthos:
        raise click.ClickException(f'no dated *.tif orthos in {drone_dirs}')
    return pd.Series(orthos).sort_index()


def coreg_group(args):
    """Coregister every scene in one ortho group; runs in a worker."""
    dronefile, scenes = args
    drone_ga = load_as_geoarray(dronefile)
    records = []
    for rec in scenes:
        x_shift, y_shift, ok = compute_coreg_shift(
            dronefile, rec['planet_file'], planet_match_band=3,
            drone_ga=drone_ga,
        )
        records.append({**rec, 'drone_file': dronefile, 'coreg_ok': ok,
                        'x_shift_m': x_shift, 'y_shift_m': y_shift})
    return records


@click.command()
@click.argument('planetdir', type=click.Path(exists=True, file_okay=False))
@click.argument('ocmdir', type=click.Path(exists=True, file_okay=False))
@click.argument('out_json', type=click.Path(dir_okay=False))
@click.option('--drone-dir', 'drone_dirs', multiple=True, required=True,
              type=click.Path(exists=True, file_okay=False),
              help='Directory of dated drone orthos; repeatable.')
@click.option('--exclude-within-days', default=2, show_default=True,
              help='Skip scenes this close to any drone flight (the scenes '
                   'the training chips were built from).')
@click.option('--min-clear-fraction', default=0.1, show_default=True,
              help='Skip scenes with less of the 50ha cutout clear in OCM.')
@click.option('--start', default=None, help='Earliest scene date, YYYY-MM-DD.')
@click.option('--end', default=None, help='Latest scene date, YYYY-MM-DD.')
@click.option('--workers', default=1, show_default=True,
              help='Ortho groups processed in parallel. Each worker holds '
                   'one ortho in memory (up to ~4 GB for mavic).')
@click.option('--dry-run', is_flag=True,
              help='Report the scene selection by year and stop.')
def main(planetdir, ocmdir, out_json, drone_dirs, exclude_within_days,
         min_clear_fraction, start, end, workers, dry_run):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    orthos = index_orthos(drone_dirs)
    flight_dates = orthos.index.values

    done = []
    if os.path.exists(out_json):
        with open(out_json) as fh:
            done = json.load(fh)
    done_scenes = {r['scene'] for r in done}

    planetfiles = sorted(Path(planetdir).rglob('*4band.tif'))
    pending = []
    n_near = n_cloudy = n_no_ocm = 0
    for pf in planetfiles:
        date = pd.to_datetime(get_planet_date(str(pf)), format='%Y%m%d')
        if (start and date < pd.Timestamp(start)) or \
                (end and date > pd.Timestamp(end)):
            continue
        if pf.stem in done_scenes:
            continue
        days = abs((flight_dates - date.to_datetime64())
                   / pd.Timedelta(days=1))
        if days.min() <= exclude_within_days:
            n_near += 1
            continue
        ocm_path = find_ocm_mask(str(pf), planetdir, ocmdir)
        if ocm_path is None:
            n_no_ocm += 1
            continue
        clear = compute_clear_fraction(ocm_path)
        if clear is None or clear < min_clear_fraction:
            n_cloudy += 1
            continue
        nearest = int(days.argmin())
        pending.append({
            'scene': pf.stem,
            'planet_file': str(pf),
            'planet_rel': str(pf.relative_to(planetdir)),
            'date': date.strftime('%Y-%m-%d'),
            'clear_fraction': clear,
            'drone_date': pd.Timestamp(flight_dates[nearest]).strftime('%Y-%m-%d'),
            'days_to_drone': int(days[nearest]),
            '_ortho': orthos.iloc[nearest],
        })
    log.info('%d scenes to coregister (%d already done; skipped %d within '
             '%d days of a flight, %d below clear fraction %.2f, %d without '
             'an OCM mask)', len(pending), len(done_scenes), n_near,
             exclude_within_days, n_cloudy, min_clear_fraction, n_no_ocm)

    if dry_run:
        df = pd.DataFrame(pending)
        if not df.empty:
            df['year'] = df['date'].str[:4]
            print(f"{len(df)} scenes in {df['_ortho'].nunique()} ortho groups")
            print(df.groupby('year').agg(
                scenes=('scene', 'size'),
                median_days_to_drone=('days_to_drone', 'median'),
                max_days_to_drone=('days_to_drone', 'max'),
            ).to_string())
        return

    groups = {}
    for rec in pending:
        groups.setdefault(rec.pop('_ortho'), []).append(rec)

    records = list(done)
    # maxtasksperchild=1: a fresh process per ortho, so memory does not creep
    # across orthos (it did in reapply_drone_labels.py's single-process run).
    ctx = mp.get_context('spawn')
    with ctx.Pool(workers, maxtasksperchild=1) as pool:
        for recs in tqdm(pool.imap_unordered(coreg_group, groups.items()),
                         total=len(groups)):
            records.extend(recs)
            records.sort(key=lambda r: r['scene'])
            write_coreg_log(records, out_json)

    n_ok = sum(r['coreg_ok'] for r in records)
    log.info('%d records (%d coregistered) in %s', len(records), n_ok, out_json)


if __name__ == '__main__':
    main()
