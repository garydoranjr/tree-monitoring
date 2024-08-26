#!/usr/bin/env python
import os
import csv
import json
import click
from pathlib import Path


PROJECT_ID = 'clvd3ln9m0o7j071u73o6hrgw'
MISSING_STR = ''
HEADERS = (
    'tag',
    'species',
    'author',
    'frame',
    'date',
    'leafing',
    'fruting_flowering_event',
    'event_color',
    'data_quality_issues',
)


def parse_id(external_id):
    parts = external_id.split('.')[0].split('_')
    species = ' '.join(parts[:2])
    tag = int(parts[-1])
    return tag, species


def get_classifications(cls):
    a = {
        'leafing': MISSING_STR,
        'fruting_flowering_event': MISSING_STR,
        'event_color': MISSING_STR,
        'data_quality_issues': MISSING_STR,
    }

    for c in cls:
        value = c['value']

        match value:
            case 'leafing' | 'fruting_flowering_event' | 'event_color':
                a[value] = c['radio_answer']['value']
                continue

            case 'data_quality_issues':
                a[value] = '|'.join([
                    i['value'] for i in c['checklist_answers']
                ])
                continue

            case _:
                raise ValueError(f'Unknown annotation "{value}"')

    return a


def parse_labels(label, frames):
    author = label['label_details']['created_by']
    annotations = []
    for f, frame in sorted(label['annotations']['frames'].items(), key=lambda i: int(i[0])):
        cls = get_classifications(frame['classifications'])
        annotations.append(cls | {
            'author': author,
            'frame': int(f),
            'date': frames[int(f) - 1],
        })
    return annotations


@click.command()
@click.argument('labelfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('frameinfo', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
def main(labelfile, frameinfo, outputfile):

    data = []
    with open(labelfile, 'r') as f:
        for line in f:
            data.append(json.loads(line))


    with open(frameinfo, 'r') as f:
        frames = json.load(f)


    all_annotations = []
    for row in data:
        tag, species = parse_id(row['data_row']['external_id'])
        project = row['projects'][PROJECT_ID]
        parse = lambda l: parse_labels(l, frames[str(tag)])
        annotations = sum(map(parse, project['labels']), [])
        all_annotations += [
            (a | { 'tag': tag, 'species': species })
            for a in annotations
        ]


    with open(outputfile, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(all_annotations)

if __name__ == '__main__':
    main()
