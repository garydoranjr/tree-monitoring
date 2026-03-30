#!/usr/bin/env python
import os
import json
import click
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
from shutil import copy
from pathlib import Path


def concat_candidates(tag, files, sequencedir, outputfile):
    base = os.path.join(sequencedir, str(tag))
    montage = np.vstack([
        np.hstack([
            np.array(Image.open(os.path.join(base, p, f + '.png')))
            for p in ('poly', 'nopoly')
        ])
        for f in files
    ])
    img = Image.fromarray(montage)
    img.save(outputfile)


def make_candidate(c, sequencedir, crownroot, outputdir):
    spc = c['species'].replace(' ', '_')
    datestr = c['date'].replace('-', '_')
    base = f"{spc}_{c['tag']}_{datestr}"

    imgs = glob(os.path.join(crownroot, '*', base + '.png'))
    img = sorted(imgs)[-1]

    odir = os.path.join(outputdir, base)
    if not os.path.exists(odir):
        os.makedirs(odir)

    montagefile = os.path.join(
        odir, base + '_candidates.png'
    )
    concat_candidates(c['tag'], c['files'], sequencedir, montagefile)

    copy(img, odir)


@click.command()
@click.argument('candidatefile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('sequencedir', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('crownroot', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputdir', type=click.Path(
    path_type=Path, exists=True
))
def main(candidatefile, sequencedir, crownroot, outputdir):

    with open(candidatefile, 'r') as f:
        candidates = json.load(f)

    for c in tqdm(candidates, 'Collecting'):
        make_candidate(c, sequencedir, crownroot, outputdir)


if __name__ == '__main__':
    main()
