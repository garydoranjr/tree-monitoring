#!/usr/bin/env python
"""What drives per-chip skill in the blocked temporal CV, and where does the
calibrated run differ from the TOA one?

Joins the per-chip held-out IoU of two runs (eval_temporal_cv.py
temporal_cv_windows.csv, both windows summed, at --epoch) to chip and scene
properties, then prints:

1. Spearman correlation of each numeric property with the IoU of each run
   and with their difference (second minus first), raw and partial on the
   chip's days to the nearest training chip (rank residuals). The partial
   column removes the shared decline with distance, which the persistence
   baseline shows is largely temporal autocorrelation of the labels.
2. IoU pooled from summed counts by each categorical property (season,
   instrument, drone, offset source), for both runs.

Properties: day of year and season (dry = January to April), instrument, sun
elevation and view angle (SR index.csv); clear fraction and saturated clear
share under the fixed stretch (the calibrated set's chips.csv); the TOA
model-input red median and crown contrast-to-noise (planet_fixed_stretch_metrics.csv);
the scene's calibration offset (mean absolute RGB offset,
planet_scene_offsets.csv); labelled crowns per chip; the drone and its
coregistration shift (coreg_log.json); and, with --persistence, the
persistence baseline's area Jaccard for the chip (persistence_baseline.py
--out-prefix, the _cv.csv).

Typical usage:
    python scripts/temporal_cv_chip_drivers.py \\
        --windows toa=/Volumes/Earth03/flower/202610_temporal/temporal_cv_windows.csv \\
        --windows cal=/Volumes/Earth03/flower/202610_temporal_cal/temporal_cv_windows.csv \\
        --persistence persistence_cv.csv
"""
import json

import click
import numpy as np
import pandas as pd
from scipy import stats

from plot_temporal_cv_detail import chip_table
from util import calibration_results

FLOWER = '/Volumes/Earth03/flower'
CAL = calibration_results()
CAL_CHIPS = f'{FLOWER}/20261012_full_label_extended2_mkl_x4_coreg_4band_cal_curated'
TOA_CHIPS = f'{FLOWER}/20261012_full_label_extended2_mkl_x4_coreg_4band_stretch_stats_curated'
NUMERIC = ['days_to_nearest_train', 'doy', 'sun_elevation', 'view_angle', 'clear_frac',
           'saturated_clear', 'toa_red_median', 'toa_red_cnr', 'cal_offset_rgb', 'n_gt',
           'coreg_shift_m', 'persistence_jaccard']
CATEGORICAL = ['season', 'instrument', 'drone', 'offset_source']


def chip_properties(chips_csv, index_csv, metrics_csv, offsets_csv, coreg_json):
    p = pd.read_csv(chips_csv)[['chip', 'scene_id', 'offset_source', 'clear_frac',
                                'saturated_clear']]
    idx = pd.read_csv(index_csv, usecols=['scene_id', 'instrument', 'sun_elevation',
                                          'view_angle'])
    m = pd.read_csv(metrics_csv, usecols=['chip', 'current_red_median', 'current_red_crown_cnr'])
    m.columns = ['chip', 'toa_red_median', 'toa_red_cnr']
    off = pd.read_csv(offsets_csv)
    off = (off[off.band.isin(['blue', 'green', 'red'])].assign(a=lambda d: d.offset.abs())
           .groupby('scene_id', as_index=False)['a'].mean()
           .rename(columns={'a': 'cal_offset_rgb'}))
    with open(coreg_json) as fh:
        log = pd.DataFrame(json.load(fh))
    log = pd.DataFrame({
        'scene_id': log['scene'].str.replace('_4band', '', regex=False),
        'drone': np.where(log['drone_file'].str.contains('/globus/'), 'mavic', 'phantom'),
        'coreg_shift_m': np.hypot(log['x_shift_m'], log['y_shift_m'])})
    return (p.merge(idx, on='scene_id', how='left').merge(m, on='chip', how='left')
             .merge(off, on='scene_id', how='left')
             .merge(log.drop_duplicates('scene_id'), on='scene_id', how='left'))


def rank_resid(y, x):
    ry, rx = stats.rankdata(y), stats.rankdata(x)
    return ry - np.polyval(np.polyfit(rx, ry, 1), rx)


@click.command()
@click.option('--windows', 'window_csvs', multiple=True, required=True, metavar='LABEL=CSV',
              help='Two runs, as label=path; differences are second minus first.')
@click.option('--epoch', default=12, show_default=True)
@click.option('--persistence', default=None, type=click.Path(exists=True, dir_okay=False),
              help='persistence_baseline.py <prefix>_cv.csv.')
@click.option('--chips-csv', default=f'{CAL_CHIPS}/chips.csv', show_default=True)
@click.option('--index-csv', default=f'{FLOWER}/planet_bcnm_sr/index.csv', show_default=True)
@click.option('--metrics-csv', default=str(CAL / 'planet_fixed_stretch_metrics.csv'),
              show_default=True)
@click.option('--offsets-csv', default=str(CAL / 'planet_scene_offsets.csv'),
              show_default=True)
@click.option('--coreg-json', default=f'{TOA_CHIPS}/coreg_log.json', show_default=True)
@click.option('--out-csv', default=None, help='Write the joined per-chip table.')
def main(window_csvs, epoch, persistence, chips_csv, index_csv, metrics_csv, offsets_csv,
         coreg_json, out_csv):
    if len(window_csvs) != 2:
        raise click.UsageError('give exactly two --windows')
    runs = []
    tabs = []
    for spec in window_csvs:
        label, path = spec.split('=', 1)
        runs.append(label)
        t = chip_table(pd.read_csv(path), epoch)
        keep = ['chip', 'days_to_nearest_train', 'planet_date', 'n_gt', 'inter_px', 'union_px', 'iou']
        tabs.append(t[keep].set_index(['chip', 'days_to_nearest_train', 'planet_date', 'n_gt'])
                    .add_suffix(f'_{label}'))
    a, b = runs
    df = pd.concat(tabs, axis=1, join='inner').reset_index()
    df['d_iou'] = df[f'iou_{b}'] - df[f'iou_{a}']
    df['doy'] = df['planet_date'].dt.dayofyear
    df['season'] = np.where(df['planet_date'].dt.month <= 4, 'dry (Jan-Apr)', 'wet (May-Dec)')
    df = df.merge(chip_properties(chips_csv, index_csv, metrics_csv, offsets_csv, coreg_json),
                  on='chip', how='left')
    if persistence:
        pc = pd.read_csv(persistence)
        pc['persistence_jaccard'] = pc['inter_px'] / pc['union_px']
        df = df.merge(pc[['chip', 'persistence_jaccard']], on='chip', how='left')
    else:
        df['persistence_jaccard'] = np.nan

    rows = []
    for col in NUMERIC:
        ok = df[col].notna() & df[f'iou_{a}'].notna() & df[f'iou_{b}'].notna()
        d = df[ok]
        if len(d) < 10:
            continue
        r = {'property': col, 'chips': len(d)}
        for target in (f'iou_{a}', f'iou_{b}', 'd_iou'):
            r[f'rho_{target}'] = stats.spearmanr(d[col], d[target])[0]
            if col != 'days_to_nearest_train':
                r[f'partial_{target}'] = stats.pearsonr(
                    rank_resid(d[target], d['days_to_nearest_train']),
                    rank_resid(d[col], d['days_to_nearest_train']))[0]
        rows.append(r)
    corr = pd.DataFrame(rows).set_index('property')

    cats = []
    for col in CATEGORICAL:
        for value, g in df.groupby(col):
            r = {'property': col, 'value': value, 'chips': len(g)}
            for lab in runs:
                r[f'iou_{lab}'] = g[f'inter_px_{lab}'].sum() / g[f'union_px_{lab}'].sum()
            r['d_iou'] = r[f'iou_{b}'] - r[f'iou_{a}']
            cats.append(r)
    cats = pd.DataFrame(cats)

    with pd.option_context('display.float_format', '{:+.2f}'.format, 'display.width', 200):
        click.echo(f'Spearman correlation with per-chip IoU at epoch {epoch} '
                   f'(d_iou = {b} - {a}); partial = controlling for days to nearest training chip:')
        click.echo(corr.to_string())
    with pd.option_context('display.float_format', '{:.3f}'.format, 'display.width', 200):
        click.echo('\nIoU pooled by group:')
        click.echo(cats.to_string(index=False))
    if out_csv:
        df.to_csv(out_csv, index=False)
        click.echo(f'wrote {out_csv}')


if __name__ == '__main__':
    main()
