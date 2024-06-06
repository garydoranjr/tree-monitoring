#!/usr/bin/env python
import os
import click
import numpy as np
from glob import glob
from tqdm import tqdm
import rasterio as rio
from pathlib import Path
from werkzeug.security import safe_join


def find_matching_mask(path):
    head, tail = os.path.split(path)
    prefix = '_'.join(tail.split('_')[:2])
    candidates = glob(safe_join(head, f'{prefix}*udm2*.tif'))
    if len(candidates) != 1:
        return None
    return candidates[0]


def get_clear_pc(path):
    with rio.open(path) as data:
        clear = data.read(1)
        return np.average(clear)


@click.command()
@click.argument('inputdir', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
def main(inputdir, outputfile):

    rgb_files = sorted(glob(safe_join(inputdir, '*rgb.tif')))
    mask_files = list(map(find_matching_mask, rgb_files))

    data = []
    pairs = list(zip(rgb_files, mask_files))
    for rgbf, maskf in tqdm(pairs, 'Scanning'):
        pc = get_clear_pc(maskf)
        base = os.path.splitext(os.path.basename(rgbf))[0]
        data.append((base, pc))

    with open(outputfile, 'w') as f:
        f.write('File,PercentClear\n')
        for base, pc in data:
            f.write(f'{base},{pc}\n')


if __name__ == '__main__':
    main()
