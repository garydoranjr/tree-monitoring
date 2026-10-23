#!/usr/bin/env python
"""Give every Planet SR 50ha clip a MODIS calibration offset, borrowing one
for the date-satellite groups that calibrate_planet_modis.py left uncalibrated.

Most groups (61%, mostly wet season) have too few clear land pixels for their
own fit. Their offset per band is borrowed, in order of preference, from:

  satellite   median offset of calibrated groups of the same satellite
              within --sat-days days
  instrument  median offset of calibrated groups of the same instrument
              (PS2 / PS2.SD / PSB.SD) within --inst-days days
  month       median offset of the instrument's calibrated groups in the
              same calendar month (any year)
  instrument_all  median offset of all the instrument's calibrated groups

The windows are short because the offset has a seasonal (BRDF) cycle, e.g.
PSB.SD NIR from -0.04 in December to -0.09 in May. Calibrated groups keep
their own fit (source "fit").

The same rules are scored leave-one-group-out on the calibrated groups: each
one's offset is predicted from the others and compared with its fit
(--validation CSV, summarised on stdout).

Outputs:
  --output       one row per scene and band: offset, source, n_groups (the
                 groups the median is over) and days_to_nearest
  --fill-root    <fill-root>/4band/<YYYY>/<scene>_4band.tif for every SR clip:
                 a symlink to the calibrated clip for calibrated scenes, SR +
                 borrowed offset (uint16 reflectance x 10000, nodata 0) for
                 the rest

Typical usage:
    python scripts/fill_planet_calibration.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_planet_modis import write_calibrated  # noqa: E402

FLOWER = Path('/Volumes/Earth03/flower')
REPO = Path(__file__).resolve().parent.parent
BANDS = ['blue', 'green', 'red', 'nir']


def load_groups(calibration: Path) -> pd.DataFrame:
    """One row per date-satellite group: instrument, calibrated, scene list and
    one offset column per band."""
    cal = pd.read_csv(calibration, parse_dates=['date'], dtype={'satellite_id': str})
    key = ['date', 'satellite_id', 'instrument', 'calibrated', 'scene_ids']
    g = cal.set_index(key + ['band'])['offset'].unstack('band').reset_index()
    g.columns.name = None
    return g[key + BANDS]


def borrow(target: pd.Series, donors: pd.DataFrame, sat_days: int, inst_days: int):
    """(offsets per band, source, n_groups, days_to_nearest) for `target` from
    the calibrated `donors` (which must not include `target` itself)."""
    dt = (donors.date - target.date).dt.days.abs()
    rules = [('satellite', (donors.satellite_id == target.satellite_id) & (dt <= sat_days)),
             ('instrument', (donors.instrument == target.instrument) & (dt <= inst_days)),
             ('month', (donors.instrument == target.instrument)
              & (donors.date.dt.month == target.date.month)),
             ('instrument_all', donors.instrument == target.instrument)]
    for source, sel in rules:
        if sel.any():
            return (donors.loc[sel, BANDS].median().values, source, int(sel.sum()),
                    int(dt[sel].min()))
    return np.full(4, np.nan), 'none', 0, -1


@click.command()
@click.option('--calibration', type=click.Path(path_type=Path, exists=True),
              default=REPO / 'results/planet_modis_calibration.csv', show_default=True)
@click.option('--sr-root', type=click.Path(path_type=Path, exists=True),
              default=FLOWER / 'planet_clipped_sr', show_default=True)
@click.option('--cal-root', type=click.Path(path_type=Path, exists=True),
              default=FLOWER / 'planet_clipped_cal', show_default=True)
@click.option('--fill-root', type=click.Path(path_type=Path),
              default=FLOWER / 'planet_clipped_calfill', show_default=True)
@click.option('--sat-days', type=int, default=30, show_default=True)
@click.option('--inst-days', type=int, default=15, show_default=True)
@click.option('--output', type=click.Path(path_type=Path),
              default=REPO / 'results/planet_scene_offsets.csv', show_default=True)
@click.option('--validation', type=click.Path(path_type=Path),
              default=REPO / 'results/planet_offset_borrow_validation.csv', show_default=True)
@click.option('--no-clips', is_flag=True, help='Write the tables only.')
def main(calibration, sr_root, cal_root, fill_root, sat_days, inst_days, output,
         validation, no_clips):
    groups = load_groups(calibration)
    ok = groups[groups.calibrated].reset_index(drop=True)

    # leave-one-group-out score of the borrowing rules on calibrated groups
    rows = []
    for i, g in tqdm(ok.iterrows(), total=len(ok), desc='validation'):
        pred, source, n, days = borrow(g, ok.drop(index=i), sat_days, inst_days)
        rows += [dict(date=g.date.date(), satellite_id=g.satellite_id,
                      instrument=g.instrument, band=b, source=source, n_groups=n,
                      days_to_nearest=days, fit=g[b], borrowed=pred[k],
                      error=pred[k] - g[b]) for k, b in enumerate(BANDS)]
    val = pd.DataFrame(rows)
    val.to_csv(validation, index=False)
    stats = dict(n='size', bias='mean', rmse=lambda e: np.sqrt(np.mean(e ** 2)),
                 median_abs=lambda e: np.median(np.abs(e)))
    base = val.groupby('band').fit.agg(rms_fit=lambda f: np.sqrt(np.mean(f ** 2)))
    with pd.option_context('display.float_format', '{:.4f}'.format, 'display.width', 160):
        click.echo('leave-one-group-out error of borrowed offsets (borrowed - fit):')
        click.echo(val.groupby(['source', 'band']).error.agg(**stats).to_string())
        click.echo(val.groupby('band').error.agg(**stats).join(base).reindex(BANDS)
                   .to_string())
        click.echo('(rms_fit: error of applying no offset, i.e. raw SR)')

    # offsets for every scene
    rows = []
    for _, g in tqdm(groups.iterrows(), total=len(groups), desc='groups'):
        if g.calibrated:
            off, source, n, days = g[BANDS].values.astype(float), 'fit', 1, 0
        else:
            off, source, n, days = borrow(g, ok, sat_days, inst_days)
        for sid in g.scene_ids.split(';'):
            rows += [dict(scene_id=sid, date=g.date.date(), satellite_id=g.satellite_id,
                          instrument=g.instrument, band=b, offset=off[k], source=source,
                          n_groups=n, days_to_nearest=days) for k, b in enumerate(BANDS)]
    sc = pd.DataFrame(rows)
    sc.to_csv(output, index=False)
    per = sc[sc.band == 'nir']
    click.echo(f'{per.scene_id.nunique()} scenes by offset source: '
               f'{per.source.value_counts().to_dict()}; wrote {output}')

    if no_clips:
        return
    off = sc.pivot_table(index='scene_id', columns='band', values='offset')[BANDS]
    src = per.set_index('scene_id').source
    n_link = n_write = n_missing = 0
    for clip in tqdm(sorted((sr_root / '4band').glob('*/*_4band.tif')), desc='clips'):
        sid = clip.name[:-len('_4band.tif')]
        out = fill_root / '4band' / clip.parent.name / clip.name
        if sid not in off.index or src[sid] == 'none':
            n_missing += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.is_symlink() or out.exists():
            out.unlink()
        if src[sid] == 'fit':
            cal = cal_root / '4band' / clip.parent.name / clip.name
            os.symlink(cal, out)
            n_link += 1
        else:
            write_calibrated(clip, out, np.ones(4), off.loc[sid].values)
            n_write += 1
    click.echo(f'{fill_root}: {n_link} calibrated (symlinks), {n_write} borrowed, '
               f'{n_missing} without an offset')


if __name__ == '__main__':
    main()
