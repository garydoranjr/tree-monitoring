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
PROJECT_ID = 'cmcgtwkuf0vjn07185e728a0z'
PROJECT_ID = 'cme1pxuyc0pjk07494fzmff4d'


def parse_object(o):
    cls = o['classifications']
    if len(cls) == 0:
        is_event = False
    else:
        assert len(cls) == 1
        cls = cls[0]

        ans = cls['checklist_answers']
        assert len(ans) == 1
        ans = ans[0]
        is_event = ans['value']
        assert is_event

    return o['bounding_box'] | { 'is_event': is_event }


def parse_labels(annotations):
    objects = annotations['objects']
    return [
        parse_object(o)
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

        for label in labels:
            labeler = label['label_details']['created_by']
            boxes = parse_labels(label['annotations'])
            for b in boxes:
                b['filename'] = filename
                b['labeler'] = labeler

            all_annotations += boxes

    df = pd.DataFrame.from_dict(all_annotations)
    df = df[['filename', 'labeler', 'top', 'left', 'width', 'height', 'is_event']]
    df.to_csv(outputfile, index=False)


if __name__ == '__main__':
    main()
