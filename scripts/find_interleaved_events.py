#!/usr/bin/env python
import os
import csv
import json
import click
import pandas as pd
from pathlib import Path


def is_event_pair(last, current):
    if last is None: return False
    if last['tag'] != current['tag']: return False
    diff = current['frame'] - last['frame']
    return (diff == 1)


@click.command()
@click.argument('labelfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
def main(labelfile, outputfile):

    df = pd.read_csv(labelfile)

    # Filter rows
    df = df[df['data_quality_issues'].isnull()]
    df = df[df['fruting_flowering_event'] == 'full']
    df = df[df['event_color'].notnull()]

    candidates = {}

    for i, row in df.iterrows():
        tag = row['tag']
        species = row['species']
        frame = row['frame']
        date = row['date']
        color = row['event_color']

        if (tag, frame) not in candidates:
            candidates[tag, frame] = {
                'tag': tag,
                'species': species,
                'frame': frame,
                'date': date,
                'colors': set([color]),
            }
        else:
            candidates[tag, frame]['colors'].add(color)

    for v in candidates.values():
        v['colors'] = sorted(v['colors'])

    events = []
    last = None
    for _, c in sorted(candidates.items()):
        if is_event_pair(last, c):
            events.append((last, c))
        last = c

    print(candidates)
    print(len(candidates))

    print(events)
    print(len(events))

    for e1, e2 in events:
        tag = e1['tag']
        species = e1['species']
        d1 = e1['date']
        d2 = e2['date']
        c = ', '.join(sorted(set(e1['colors']) | set(e2['colors'])))
        print(f"{tag}: {species} [{d1} - {d2}] ({c})")


if __name__ == '__main__':
    main()
