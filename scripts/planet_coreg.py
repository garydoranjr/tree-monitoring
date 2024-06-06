#!/usr/bin/env python
import os
import json
import click
import warnings
import numpy as np
from glob import glob
from tqdm import tqdm
from arosics import COREG
from geoarray import GeoArray
from werkzeug.security import safe_join

from util import load_config


def stem(path, suffix=None):
    return os.path.splitext(
        os.path.basename(path)
    )[0] + ('' if suffix is None else suffix)


def coreg(ref, tgt, coreg_args):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, message=".*footprint.*"
        )

        try:
            coreg = COREG(
                ref, tgt, ignore_errors=True, q=True, **coreg_args
            )
            coreg.calculate_spatial_shifts()
        except (AssertionError, AttributeError):
            return { 'success': False }

        return coreg.coreg_info


@click.command()
@click.argument('imagedir')
@click.argument('outputdir')
@click.argument('configfile')
@click.argument('referenceindex', type=int)
def main(imagedir, outputdir, configfile, referenceindex):

    config = load_config(configfile)
    coreg_args = config.get('coreg_args', {})

    files = sorted(glob(safe_join(imagedir, config['glob_pattern'])))
    if referenceindex >= len(files):
        raise ValueError(f'Index {referenceindex} out of range for {len(files)} files')

    reference_file = files.pop(referenceindex)
    ref = GeoArray(reference_file)

    outputfile = safe_join(
        outputdir,
        stem(reference_file, suffix='.json')
    )

    if os.path.exists(outputfile):
        print('Already completed')
        return

    results = {}

    for target_file in tqdm(files):
        tgt = GeoArray(target_file)
        key = stem(target_file)
        results[key] = coreg(ref, tgt, coreg_args)

    with open(outputfile, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
