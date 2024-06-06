#!/usr/bin/env python
import os
import json
import click
import hashlib
from tqdm import tqdm
from pathlib import Path
from typing import NamedTuple
from werkzeug.security import safe_join


MANIFEST_FILE = 'manifest.json'
MARKER = '.validated'


class Validation(NamedTuple):
    valid: bool = True
    reason: str = ""


def file_digest(filename):
    with open(filename, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def validate(orderdir, subdir):
    order_path = safe_join(orderdir, subdir)

    marker_file = safe_join(order_path, MARKER)
    if os.path.exists(marker_file):
        return Validation(True, 'Order marked valid')

    manifest_file = safe_join(order_path, MANIFEST_FILE)
    if not os.path.exists(manifest_file):
        return Validation(False, 'Missing Manifest')

    with open(manifest_file, 'r') as f:
        manifest = json.load(f)

    file_list = manifest['files']
    for f in tqdm(file_list, f'Checking {subdir}', leave=False):
        path = safe_join(order_path, f['path'])
        if not os.path.exists(path):
            return Validation(False, f"Missing file: {f['path']}")

    for f in tqdm(file_list, f'Validating {subdir}', leave=False):
        path = safe_join(order_path, f['path'])
        expected_digest = f['digests']['md5']
        actual_digest = file_digest(path)
        if expected_digest != actual_digest:
            return Validation(False, f"File md5sum mismatch: {f['path']}")

    # Mark valid
    Path(marker_file).touch()

    return Validation()


@click.command()
@click.argument('orderdir', type=click.Path(
    path_type=Path, exists=True
))
def main(orderdir):

    subdirs = [
        d for d in os.listdir(orderdir)
        if os.path.isdir(safe_join(orderdir, d))
    ]

    results = {}
    for subdir in tqdm(subdirs, 'Validating Orders'):
        results[subdir] = validate(orderdir, subdir)

    if all(v.valid for v in results.values()):
        print('\nValidation complete: all orders are valid!')

    else:
        print('\nValidation report:')
        for s, v in results.items():
            if not v.valid:
                print(f'{s} is invalid: {v.reason}')


if __name__ == '__main__':
    main()
