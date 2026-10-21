#!/usr/bin/env python
"""Static water mask over the BCNM Planet order box from PlanetScope SR.

For every staged SR scene (see scripts/stage_planet_sr.py) with an OCM mask,
counts per 3 m pixel how often it is OCM-clear and how often its clear NIR
surface reflectance is below --nir-threshold. A pixel is water when it was
clear at least --min-clear times and dark in NIR in more than half of those.
Gatun Lake and the Canal dominate the box, so the MODIS calibration must use
only 500 m pixels well inside land (scripts/calibrate_planet_modis.py).

Output: uint8 GeoTIFF on the shared box grid (scripts/planet_box_grid.py):
1 = water, 0 = land, 255 = never observed clear often enough.

Typical usage:
    python scripts/build_bcnm_water_mask.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import click
import numpy as np
import rasterio as rio
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from planet_box_grid import UTM, box_grid, warp_to_box  # noqa: E402

DEFAULT_STAGE = Path('/Volumes/Earth03/flower/planet_bcnm_sr')
NIR_BAND = 4


@click.command()
@click.option('--stage-dir', type=click.Path(path_type=Path, exists=True),
              default=DEFAULT_STAGE, show_default=True,
              help='Output of stage_planet_sr.py (index.csv, ocm/).')
@click.option('--output', type=click.Path(path_type=Path), default=None,
              help='Default: <stage-dir>/aux/bcnm_water_mask.tif')
@click.option('--nir-threshold', type=float, default=0.08, show_default=True,
              help='NIR SR below this (reflectance) counts as water.')
@click.option('--min-clear', type=int, default=3, show_default=True)
def main(stage_dir, output, nir_threshold, min_clear):
    output = output or stage_dir / 'aux' / 'bcnm_water_mask.tif'
    transform, width, height = box_grid()
    n_clear = np.zeros((height, width), np.uint16)
    n_dark = np.zeros((height, width), np.uint16)
    rows = list(csv.DictReader(open(stage_dir / 'index.csv')))
    for r in tqdm(rows, desc='scenes'):
        sid = r['scene_id']
        ocm_path = stage_dir / 'ocm' / sid[:4] / f'{sid}_ocm.tif'
        if not ocm_path.exists():
            continue
        nir = warp_to_box(Path(r['sr_path']), [NIR_BAND], transform, width,
                          height, 0, np.uint16)[0]
        cls = warp_to_box(ocm_path, [1], transform, width, height, 255,
                          np.uint8)[0]
        clear = (cls == 0) & (nir > 0)
        n_clear += clear
        n_dark += clear & (nir < nir_threshold * 10000)
    mask = np.full((height, width), 255, np.uint8)
    seen = n_clear >= min_clear
    mask[seen] = (2 * n_dark[seen] > n_clear[seen]).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    with rio.open(output, 'w', driver='GTiff', width=width, height=height,
                  count=1, dtype='uint8', crs=UTM, transform=transform,
                  nodata=255, compress='lzw') as dst:
        dst.write(mask, 1)
    click.echo(f'{output}: water {np.mean(mask == 1):.1%}, land '
               f'{np.mean(mask == 0):.1%}, unseen {np.mean(mask == 255):.1%}')


if __name__ == '__main__':
    main()
