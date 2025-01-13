#!/usr/bin/env python
import os
import csv
import json
import click
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


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

    for i, row in label_df.iterrows():
        est = calc_offset(row)
        act = np.array([row['dx'], row['dy']])
        error = (est - act)
        print(3 * np.sqrt(np.sum(error**2)))


if __name__ == '__main__':
    main()
