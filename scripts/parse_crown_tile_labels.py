#!/usr/bin/env python
import os
import csv
import json
import click
from pathlib import Path


PROJECT_ID = 'clyrjcv2d02wu07y1fqse2npf'


def parse_id(row):
    external_id = row['data_row']['external_id']
    parts = external_id.split('.')[0].split('_')
    tag = int(parts[0])
    image_id = '_'.join(parts[1:])
    return tag, image_id


def get_annotation(row, project_id=PROJECT_ID):
    project = row['projects'][project_id]
    labels = project['labels']
    annotations = set([])
    for label in labels:
        cls = label['annotations']['classifications']
        assert len(cls) == 1
        annotations.add(cls[0]['radio_answer']['value'])
    assert len(annotations) == 1
    return annotations.pop()


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

    labels = []
    for row in data:
        tag, image_id = parse_id(row)
        label = get_annotation(row)
        labels.append({
            'tag': tag,
            'image_id': image_id,
            'label': label,
        })

    with open(outputfile, 'w') as f:
        fieldnames = ['tag', 'image_id', 'label']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labels)


if __name__ == '__main__':
    main()
