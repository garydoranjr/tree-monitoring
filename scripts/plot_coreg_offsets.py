#!/usr/bin/env python
import os
import json
import click
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from werkzeug.security import safe_join

from util import load_config


def load_offsets(coregfile):
    with open(coregfile, 'r') as f:
        results = json.load(f)

    return np.array([
        (
            r['corrected_shifts_px']['x'],
            r['corrected_shifts_px']['y'],
        )
        for r in results.values()
        if r['success']
    ])


@click.command()
@click.argument('coregdir')
@click.argument('configfile')
@click.argument('outputfile')
@click.option('-b', '--nbins', type=int, default=100)
def main(coregdir, configfile, outputfile, nbins):

    config = load_config(configfile)
    coreg_args = config.get('coreg_args', {})
    vmax = coreg_args.get('max_shift', 5)
    vmax = 10
    vmax *= 3
    vmax = 20

    files = sorted(glob(safe_join(coregdir, '*.json')))

    offsets = [load_offsets(f) for f in tqdm(files, 'Loading')]

    xy = np.vstack([o for o in offsets if o.size > 0])

    fig, ax = plt.subplots(figsize=(8, 6))

    bins = np.linspace(-vmax, vmax, nbins + 1)

    ax.set_facecolor('k')
    h = ax.hist2d(
        3*xy[:, 0], 3*xy[:, 1],
        bins=(bins, bins), norm=LogNorm(vmin=1e0, vmax=1e2),
        cmap='magma',
    )
    cbar = fig.colorbar(h[3])
    cbar.set_label('Count', fontsize=14)
    ax.set_xlabel('x Shift (m)', fontsize=14)
    ax.set_ylabel('y Shift (m)', fontsize=14)
    ax.set_aspect('equal')

    fig.savefig(outputfile)



if __name__ == '__main__':
    main()
