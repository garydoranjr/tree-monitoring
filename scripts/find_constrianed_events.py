#!/usr/bin/env python
import os
import csv
import json
import click
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


def is_event_pair(last, current):
    if last is None: return False
    if last['tag'] != current['tag']: return False
    diff = current['frame'] - last['frame']
    return (diff == 1)


def is_event_center(current):
    if 'full' not in current['events']: return False
    if len(current['colors']) < 1: return False
    if len(current['quality']) > 0: return False
    return True


def parse_date(f):
    parts = f.split('_')
    dt = '_'.join(parts[:2])
    return np.datetime64(datetime.strptime(dt, '%Y%m%d_%H%M%S'))


@click.command()
@click.argument('labelfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('assessmentfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
def main(labelfile, assessmentfile, outputfile):

    df = pd.read_csv(labelfile)

    asmt = np.load(assessmentfile)
    tags = asmt['tags'].astype(int)
    files = asmt['files']
    values = asmt['values']
    dates = np.array([
        parse_date(f) for f in files
    ])

    def select_relevant_files(tag, start, end, threshold=0.5):
        vt = np.squeeze(values[tags == tag])
        good = (
            (start <= dates) &
            (dates <= end) &
            (vt >= threshold)
        )
        fi = files[good]
        vi = vt[good]

        return fi.tolist(), vi.tolist()

    candidates = {}

    for i, row in df.iterrows():
        tag = row['tag']
        species = row['species']
        event = row['fruting_flowering_event']
        quality = row['data_quality_issues']
        frame = row['frame']
        date = row['date']
        color = row['event_color']

        if (tag, frame) not in candidates:
            candidates[tag, frame] = {
                'tag': tag,
                'species': species,
                'events': set([event]),
                'quality': set([quality]),
                'frame': frame,
                'date': date,
                'colors': set([color]),
            }
        else:
            candidates[tag, frame]['colors'].add(color)
            candidates[tag, frame]['events'].add(event)
            candidates[tag, frame]['quality'].add(quality)

    for v in candidates.values():
        for k in ('colors', 'events', 'quality'):
            if np.nan in v[k]: v[k].remove(np.nan)
            v[k] = sorted(v[k])

    events = []
    for _, c in sorted(candidates.items()):
        if is_event_center(c):
            t = c['tag']
            f = c['frame']
            start = (t, f - 1)
            end = (t, f + 1)
            if start in candidates and end in candidates:
                events.append((candidates[start], c, candidates[end]))

    results = []

    for s, c, e in events:
        t = c['tag']
        e = dict(c)
        fls, confs = select_relevant_files(
            t, np.datetime64(s['date']), np.datetime64(e['date'])
        )
        if len(fls) == 0: continue
        e['files'] = fls
        e['confidences'] = confs
        results.append(e)

    with open(outputfile, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
