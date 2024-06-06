#!/usr/bin/env python
import os
import click
from tqdm import tqdm
from pathlib import Path
from werkzeug.security import safe_join

from clip_planet_image import clip

@click.command()
@click.argument('inputfiles', nargs=-1,
    type=click.Path(
    path_type=Path, exists=True
))
@click.argument('configfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputdir', type=click.Path(
    path_type=Path, exists=True
))
def main(inputfiles, configfile, outputdir):

    jobs = [
        (f, configfile, safe_join(outputdir, os.path.basename(f)))
        for f in inputfiles
    ]

    for job in tqdm(jobs, 'Clipping images'):
        clip(*job)


if __name__ == '__main__':
    main()
