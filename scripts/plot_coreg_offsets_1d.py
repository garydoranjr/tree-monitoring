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

    files = sorted(glob(safe_join(coregdir, '*.json')))

    offsets = [load_offsets(f) for f in tqdm(files[::5], 'Loading')]

    xy = np.vstack([o for o in offsets if o.size > 0])
    delta = 3*np.sqrt(np.sum(xy**2, axis=1))
    delta = delta[delta > 2.0]
    print(np.min(delta))

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(delta, bins=np.linspace(0, 20, nbins),
        density=True, cumulative=True,
        histtype='step', lw=2, edgecolor='k')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Offset (m)', fontsize=14)
    ax.set_ylabel('Cumulative Fraction', fontsize=14)

    plt.show()
    exit()

    ax.set_facecolor('k')
    h = ax.hist2d(
        xy[:, 0], xy[:, 1],
        bins=(bins, bins), norm=LogNorm(), cmap='magma'
    )
    cbar = fig.colorbar(h[3])
    cbar.set_label('Count', fontsize=14)
    ax.set_xlabel('x Shift', fontsize=14)
    ax.set_ylabel('y Shift', fontsize=14)
    ax.set_aspect('equal')

    fig.savefig(outputfile)



if __name__ == '__main__':
    main()
