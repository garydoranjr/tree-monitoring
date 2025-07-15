#!/usr/bin/env python
import os
import csv
import json
import click
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt


M_PER_PIX = 3.0


def load_offsets(coregfile):
    with open(coregfile, 'r') as f:
        data = json.load(f)

    return {
        e['key']: np.array([e['x_offset'], e['y_offset']])
        for e in data
    }


def offset_getter(offsets):

    def f(row):
        src = offsets[row['source_id']]
        tgt = offsets[row['target_id']]
        return src - tgt

    return f


@click.command()
@click.argument('labelfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('coregfile', type=click.Path(
    path_type=Path, exists=False
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
def main(labelfile, coregfile, outputfile):

    label_df = pd.read_csv(labelfile)
    offsets = load_offsets(coregfile)

    calc_offset = offset_getter(offsets)

    errors = []

    for i, row in label_df.iterrows():
        try:
            est = calc_offset(row)
        except KeyError: continue
        act = np.array([row['dx'], row['dy']])
        error = (est - act)
        errors.append(error)

    errors = np.array(errors)

    error_mag = M_PER_PIX * np.sqrt(np.sum(errors**2, axis=1))

    print(np.average(error_mag))
    print(f'n = {len(error_mag)}')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(
        error_mag,
        bins=np.linspace(0, 50, 101),
        cumulative=True, density=True,
        histtype='step', linewidth=2,
    )
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Error (m)', fontsize=16)
    ax.set_ylabel('Cumulative Fraction', fontsize=16)

    fig.savefig(outputfile)


if __name__ == '__main__':
    main()
