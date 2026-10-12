#!/usr/bin/env python
"""Rebuild a label-application set's crown masks from new drone classifications.

apply_drone_labels_coreg.py coregisters each drone *ortho* to its Planet
scene, so the AROSICS shifts, Planet chips, cloud masks and QA PNGs depend
only on the orthos and the Planet scenes. Re-classifying the same orthos
(e.g. the mavic 3x3 colour-transfer run) changes only the `*.mask.png`
crown masks. This copies SRC_SET to DST_SET and regenerates each mask from
the classification of the same name in LABEL_DIR, with the shift recorded
in SRC_SET/coreg_log.json and the same masking code
(load_label -> apply_shift_to_label -> reproject_match -> OCM crown filter).
The vetting ratings (vetting.json), which judge Planet image quality, carry
over unchanged.

Records are replayed in log order, so where one Planet scene paired with two
label dates the later date's mask wins, as in the original build.

Typical usage:
    F=/Volumes/Earth03/flower
    python scripts/reapply_drone_labels.py \
        $F/20260915_globus_label_application_x4_coreg_4band_stretch_stats_rerun \
        $F/results/classifications \
        $F/20261012_globus_mkl_label_application_x4_coreg_4band_stretch_stats
"""
import json
import multiprocessing
import shutil
from functools import partial
from pathlib import Path

import click
import imageio.v3 as iio
import numpy as np
from rasterio.enums import Resampling
from tqdm import tqdm

from apply_drone_labels_coreg import (
    apply_shift_to_label,
    crown_filter_by_ocm,
    find_ocm_mask,
    load_label,
    load_planet,
    reproject_ocm_to_grid,
)

FLOWER = Path('/Volumes/Earth03/flower')


def write_label_masks(item, label_dir, dst_set, planets, planetdir, maskdir, resize, mode):
    """Write the masks of every coregistered scene paired with one label date.

    Runs in a worker that is replaced after each label: a full-resolution
    mavic classification is several GB, and memory crept up across label
    dates when they all ran in one process, until it was killed on a 16 GB
    machine.
    """
    label, records = item
    conf = load_label(label_dir / label, mode)
    scenes = []
    for r in records:
        planetfile = planets[r['scene']]
        pimg = load_planet(planetfile, resize)
        shifted = apply_shift_to_label(conf, r['x_shift_m'], r['y_shift_m'])
        conf_resampled = shifted.rio.reproject_match(pimg, resampling=Resampling.average)
        ocm_path = find_ocm_mask(planetfile, planetdir, maskdir)
        if ocm_path is not None:
            mask = crown_filter_by_ocm(conf_resampled, reproject_ocm_to_grid(ocm_path, pimg))
        else:
            mask = (conf_resampled.values > 0.5).astype(np.uint8) * 255
        # 4-band chips are named after the scene stem (<scene>_4band).
        iio.imwrite(dst_set / f'{r["scene"]}.mask.png', mask)
        scenes.append(r['scene'])
    return scenes


@click.command()
@click.argument('src_set', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument('label_dir', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument('dst_set', type=click.Path(file_okay=False, path_type=Path))
@click.option('--planetdir', type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=FLOWER / 'planet_clipped/4band', show_default=True)
@click.option('-k', '--maskdir', type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=FLOWER / 'planet_clipped/ocm', show_default=True,
              help='OmniCloudMask directory, as -k in the original build.')
@click.option('-r', '--resize', default=4.0, show_default=True,
              help='Planet upsampling factor, as -r in the original build.')
@click.option('-m', '--mode', default='both', show_default=True)
def main(src_set, label_dir, dst_set, planetdir, maskdir, resize, mode):
    if dst_set.exists() and any(dst_set.iterdir()):
        raise click.UsageError(f'{dst_set} exists and is not empty')
    log = json.loads((src_set / 'coreg_log.json').read_text())
    ok = [r for r in log if r['coreg_ok']]
    planets = {p.stem: p for p in planetdir.rglob('*4band.tif')}

    shutil.copytree(src_set, dst_set, ignore=shutil.ignore_patterns('*.mask.png'))
    click.echo(f'Copied {src_set.name} -> {dst_set} (without masks)')

    by_label = {}
    for r in ok:
        by_label.setdefault(r['label'], []).append(r)
    work = partial(write_label_masks, label_dir=label_dir, dst_set=dst_set, planets=planets,
                   planetdir=planetdir, maskdir=maskdir, resize=resize, mode=mode)
    written = set()
    # One label date at a time, in log order, so later dates overwrite
    # earlier ones on a shared scene exactly as in the original build.
    with multiprocessing.get_context('spawn').Pool(1, maxtasksperchild=1) as pool:
        for scenes in tqdm(pool.imap(work, by_label.items()), total=len(by_label)):
            written.update(scenes)
    missing = [p.name for p in src_set.glob('*.mask.png')
               if p.name[:-len('.mask.png')] not in written]
    click.echo(f'Wrote {len(written)} masks from {len(ok)} coregistered records '
               f'({len(by_label)} label dates)')
    if missing:
        raise click.ClickException(f'{len(missing)} source masks were not regenerated: {missing[:5]}')


if __name__ == '__main__':
    main()
