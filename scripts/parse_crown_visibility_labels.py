#!/usr/bin/env python
import os
import csv
import json
import click
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


PROJECT_ID = 'cm9j8546c17vg072p6m3wgnty'
PROJECT_ID = 'cmc5e6wce09g007zean63g0jh'


def parse_labels(annotations):
    objects = annotations['objects']

    return [
        o['bounding_box']
        for o in objects
    ]


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
        filename = row['data_row']['external_id']
        labels = project['labels']
        if len(labels) == 0: continue

        boxes = parse_labels(labels[0]['annotations'])
        for b in boxes:
            b['filename'] = filename

        all_annotations += boxes

    df = pd.DataFrame.from_dict(all_annotations)
    df = df[['filename', 'top', 'left', 'width', 'height']]
    df.to_csv(outputfile, index=False)


if __name__ == '__main__':
    main()
