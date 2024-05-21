#!/usr/bin/env python
import os
import json
import click
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

from planet_coreg import stem


def load_offsets(coregfile):

    key = stem(coregfile)

    with open(coregfile, 'r') as f:
        results = json.load(f)

    return {
        (key, tgt): (
            r['corrected_shifts_px']['x'],
            r['corrected_shifts_px']['y'],
        )
        for tgt, r in results.items()
        if r['success']
    }


def offsets_to_matrix(offsets):
    keys = set([])
    for src, tgt in offsets.keys():
        keys |= set([src, tgt])
    keys = sorted(keys)
    n = len(keys)

    xy = np.full((n, n, 2), np.nan)
    for i, k1 in tqdm(list(enumerate(keys)), 'Converting'):
        for j, k2 in enumerate(keys):
            key = (k1, k2)
            if k1 == k2 or key not in offsets:
                continue
            else:
                xy[i, j, :] = offsets[key]

    return keys, xy


def load_offset_matrix(coregfiles):

    offsets = {}
    for f in tqdm(coregfiles, 'Loading'):
        offsets.update(load_offsets(f))

    return offsets_to_matrix(offsets)


def iterate(offset, xy):
    for i, dxy in enumerate(offset):
        diff = xy[i, :] + offset
        offset[i, :] = np.nanmedian(diff, axis=0)
    offset -= np.nanmean(offset, axis=0)
    return offset


@click.command()
@click.argument('coregdir')
@click.argument('outputfile')
@click.option('-m', '--maxiter', type=int, default=100)
@click.option('-t', '--tolerance', type=float, default=1e-6)
def main(coregdir, outputfile, maxiter, tolerance):

    files = sorted(glob(os.path.join(coregdir, '*.json')))

    keys, xy = load_offset_matrix(files)

    offset = np.zeros((len(xy), 2))

    it = tqdm(list(range(maxiter)), 'Optimizing')
    for i in it:
        prev = np.array(offset)
        offset = iterate(offset, xy)
        delta = np.sqrt(np.nanmean(np.square(prev - offset)))
        if delta < tolerance:
            it.close()
            break

    output = [
        {
            'key': key,
            'x_offset': -float(ox),
            'y_offset': -float(oy),
        }
        for key, (ox, oy) in zip(keys, offset)
    ]

    with open(outputfile, 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == '__main__':
    main()