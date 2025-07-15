#!/usr/bin/env python
import os
import click
from tqdm import tqdm
from glob import glob
from pathlib import Path
from werkzeug.security import safe_join

from calculate_ndvi import ndvi


def find_metadata(image):
    base, ext = os.path.splitext(image)
    parts = base.split('_')
    assert parts[-1] == 'clip'
    new = '_'.join(parts[:-1]) + '_metadata_clip.xml'
    assert os.path.exists(new)
    return new


def get_outputfile(image):
    base, ext = os.path.splitext(image)
    return base + '_ndvi' + ext


@click.command()
@click.argument('inputdir', type=click.Path(
    path_type=Path, exists=True
))
def main(inputdir):

    inputfiles = glob(safe_join(inputdir, '*MS_clip.tif'))

    jobs = [
        (f, find_metadata(f), get_outputfile(f))
        for f in inputfiles
    ]

    for job in tqdm(jobs, 'Calculating NDVI'):
        ndvi(*job)


if __name__ == '__main__':
    main()
