#!/usr/bin/env python
import os
import csv
import json
import click
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


PROJECT_ID = 'cm4t6hgnp09i507wceocs1sq1'
SCALE_FACTOR = 2.0


def parse_id(external_id):
    return tuple(os.path.splitext(external_id)[0].split('-'))


def parse_labels(annotations):
    frames = [v for k, v in sorted(annotations['frames'].items())]
    assert len(frames) == 2
    f1 = frames[0]['objects']
    f2 = frames[1]['objects']

    points = defaultdict(list)
    for f in (f1, f2):
        for k, v in f.items():
            points[k].append(v['point'])

    pairs = [v for v in points.values() if len(v) == 2]
    return pairs


def pairs_to_diffs(pairs):
    return np.array([
        [p2['x'] - p1['x'], p2['y'] - p1['y']]
        for p1, p2 in pairs
    ])


@click.command()
@click.argument('labelfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
def main(labelfile, outputfile):

    data = []
    with open(labelfile, 'r') as f:
        for line in f:
            data.append(json.loads(line))


    all_annotations = []
    for row in data:
        project = row['projects'][PROJECT_ID]
        src, tgt = parse_id(row['data_row']['external_id'])
        labels = project['labels'][0]
        pairs = parse_labels(labels['annotations'])
        diffs = pairs_to_diffs(pairs) / SCALE_FACTOR
        n = len(diffs)
        dx, dy = np.average(diffs, axis=0)
        sdx, sdy = np.std(diffs, axis=0)
        all_annotations.append({
            'source_id': src,
            'target_id': tgt,
            'n': n,
            'dx': dx,
            'dy': dy,
            'stdev_x': sdx,
            'stdev_y': sdy,
        })

    df = pd.DataFrame(all_annotations)
    df.to_csv(outputfile, index=False)


if __name__ == '__main__':
    main()
