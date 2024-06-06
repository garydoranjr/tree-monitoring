#!/usr/bin/env python
import os
import json
import click
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from werkzeug.security import safe_join


from validate_planet import MANIFEST_FILE

MONTHS = 'JFMAMJJASOND'


def get_date_from_path(path):
    base = os.path.basename(path)
    first = base.split('_')[0]
    try:
        return datetime.strptime(first, '%Y%m%d')
    except ValueError:
        return None


def get_date(orderdir, subdir):
    order_path = safe_join(orderdir, subdir)

    manifest_file = safe_join(order_path, MANIFEST_FILE)
    if not os.path.exists(manifest_file):
        return None

    with open(manifest_file, 'r') as f:
        manifest = json.load(f)

    for f in manifest['files']:
        month = get_date_from_path(f['path'])
        if month is not None:
            return month


def months_str(year, months):
    mstr = ''
    for i, m in enumerate(MONTHS, 1):
        mstr += m if i in months else ' '
    return f'{year} [{mstr}]'


@click.command()
@click.argument('orderdir', type=click.Path(
    path_type=Path, exists=True
))
def main(orderdir):

    subdirs = [
        d for d in os.listdir(orderdir)
        if os.path.isdir(safe_join(orderdir, d))
    ]

    results = defaultdict(set)
    for subdir in tqdm(subdirs, 'Scanning Orders'):
        date = get_date(orderdir, subdir)
        if date is not None:
            ym = (date.year, date.month)
            results[date.year].add(date.month)

    for y, months in sorted(results.items()):
        print(months_str(y, months))


if __name__ == '__main__':
    main()
