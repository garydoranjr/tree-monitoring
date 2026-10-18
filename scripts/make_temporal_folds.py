#!/usr/bin/env python
"""Assign Planet training chips to blocked temporal cross-validation folds.

The timeline of chips is cut into contiguous blocks spanning at most
--block-days, and the blocks are dealt in time order to --n-folds folds.
Each block goes to the fold with the fewest chips so far, excluding the
fold of the block just before it, so neighbouring blocks are never held
out together and every held-out gap has training data on both sides. Chips that share a drone flight (and so share a label)
are kept in the same block. See docs/planet_temporal_generalization.md.
"""
import os
import json
from glob import glob

import click
import numpy as np
import pandas as pd
from PIL import Image

from apply_drone_labels_coreg import get_cls_date, get_planet_date
from train_planet_image_maskrcnn import binary_mask_to_instances, get_split


def chip_dates(imagedir):
    """One row per chip: its Planet date and the drone flight date(s) whose
    labels were applied to it (from the coreg_ok records of coreg_log.json)."""
    chips = sorted(
        os.path.basename(f)[:-len('.mask.png')]
        for f in glob(os.path.join(imagedir, '*.mask.png'))
    )
    with open(os.path.join(imagedir, 'coreg_log.json')) as fh:
        log = json.load(fh)
    drone = {}
    for rec in log:
        if rec.get('coreg_ok'):
            drone.setdefault(rec['scene'], set()).add(
                pd.to_datetime(get_cls_date(rec['label']), format='%Y_%m_%d')
            )
    missing = [c for c in chips if c not in drone]
    if missing:
        raise click.ClickException(
            f'{len(missing)} chip(s) have no coreg_ok record, e.g. {missing[:3]}'
        )
    return pd.DataFrame({
        'chip': chips,
        'planet_date': [
            pd.to_datetime(get_planet_date(c), format='%Y%m%d') for c in chips
        ],
        'drone_dates': [sorted(drone[c]) for c in chips],
    })


def flight_groups(df):
    """Union-find over chips that share a drone flight date."""
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first_chip = {}
    for i, dates in enumerate(df['drone_dates']):
        for d in dates:
            if d in first_chip:
                parent[find(i)] = find(first_chip[d])
            else:
                first_chip[d] = i
    return np.array([find(i) for i in range(len(df))])


def assign_blocks(df, block_days):
    """Cut flight groups, in time order, into blocks spanning at most
    `block_days` of Planet dates. A group is never split across blocks."""
    groups = (
        df.groupby('group')['planet_date'].agg(['min', 'max'])
        .sort_values(['min', 'max'])
    )
    block_of = {}
    block, start = -1, None
    for g, row in groups.iterrows():
        if start is None or (row['max'] - start).days >= block_days:
            block += 1
            start = row['min']
        block_of[g] = block
    return df['group'].map(block_of)


def assign_folds(df, n_folds):
    """Deal blocks in time order to the smallest fold (by chips) that the
    previous block did not go to."""
    sizes = df.groupby('block').size()
    load = np.zeros(n_folds, dtype=int)
    fold_of = {}
    prev = None
    for b in sorted(sizes.index):
        load_masked = load.copy()
        if prev is not None:
            load_masked[prev] = np.iinfo(int).max
        k = int(np.argmin(load_masked))
        fold_of[b] = k
        load[k] += sizes[b]
        prev = k
    return df['block'].map(fold_of)


def days_to_nearest_train(df):
    """For each chip, days to the nearest Planet date of a chip in another
    fold, i.e. to the nearest training chip when its own fold is held out."""
    out = np.empty(len(df), dtype=int)
    dates = df['planet_date'].values
    folds = df['fold'].values
    for i in range(len(df)):
        other = dates[folds != folds[i]]
        out[i] = np.abs((other - dates[i]) / np.timedelta64(1, 'D')).min()
    return out


def count_instances(imagedir, chips, size, min_instance_size):
    counts = []
    for c in chips:
        mask = np.array(Image.open(os.path.join(imagedir, f'{c}.mask.png')))
        mask = (mask == 255).astype(np.uint8)
        n = 0
        for split in ('left', 'right'):
            _, crop = get_split(mask, mask, split, size)
            n += binary_mask_to_instances(crop, min_instance_size).shape[0]
        counts.append(n)
    return counts


@click.command()
@click.argument('imagedir', type=click.Path(exists=True, file_okay=False))
@click.argument('outcsv', type=click.Path(dir_okay=False))
@click.option('--block-days', default=14, show_default=True,
              help='Maximum span of Planet dates within one held-out block.')
@click.option('--n-folds', default=6, show_default=True)
@click.option('--size', default=512, show_default=True,
              help='Window size, only for the instance counts in the summary.')
@click.option('--min-instance-size', default=64, show_default=True,
              help='Only for the instance counts in the summary.')
def main(imagedir, outcsv, block_days, n_folds, size, min_instance_size):
    df = chip_dates(imagedir)
    df['group'] = flight_groups(df)
    df['block'] = assign_blocks(df, block_days)
    df['fold'] = assign_folds(df, n_folds)
    span = df.groupby('block')['planet_date'].agg(['min', 'max'])
    df['block_start'] = df['block'].map(span['min'])
    df['block_end'] = df['block'].map(span['max'])
    df['days_to_nearest_train'] = days_to_nearest_train(df)
    df['n_instances'] = count_instances(
        imagedir, df['chip'], size, min_instance_size,
    )

    out = df.sort_values('planet_date')[[
        'chip', 'planet_date', 'drone_dates', 'block', 'block_start',
        'block_end', 'fold', 'days_to_nearest_train', 'n_instances',
    ]].copy()
    out['chip'] = out['chip'] + '.tif'
    for col in ('planet_date', 'block_start', 'block_end'):
        out[col] = out[col].dt.strftime('%Y-%m-%d')
    out['drone_dates'] = out['drone_dates'].apply(
        lambda ds: ';'.join(d.strftime('%Y-%m-%d') for d in ds)
    )
    out.to_csv(outcsv, index=False)

    n_flights = len({d for ds in df['drone_dates'] for d in ds})
    print(f'{len(df)} chips, {n_flights} drone flights, '
          f'{df["block"].nunique()} blocks of <= {block_days} days, '
          f'{n_folds} folds -> {outcsv}\n')
    rows = []
    for k, g in df.groupby('fold'):
        d = g['days_to_nearest_train']
        rows.append({
            'fold': k, 'blocks': g['block'].nunique(), 'chips': len(g),
            'instances': g['n_instances'].sum(),
            'days_median': int(d.median()), 'days_max': d.max(),
            'chips_>30d': int((d > 30).sum()),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    d = df['days_to_nearest_train']
    print(f'\nAll held-out chips: days to nearest training chip '
          f'median {d.median():.0f}, 90th pct {d.quantile(0.9):.0f}, '
          f'max {d.max()}; {(d > 30).sum()} chip(s) beyond 30 days')


if __name__ == '__main__':
    main()
