#!/usr/bin/env python
"""Build a MODIS-calibrated copy of a Planet Mask R-CNN chip set.

Each 4-band training chip written by apply_drone_labels_coreg.py is a 50ha
TOA clip resampled (bilinear, x4) onto a finer grid with the same origin; the
drone coregistration shift is applied to the labels, not to the image. So the
calibrated chip is the scene's calibrated 50ha clip (planet_clipped_calfill:
own MODIS fit, or a borrowed offset; see fill_planet_calibration.py)
reprojected onto the source chip's grid, and the labels, folds and curation
carry over unchanged.

For every <name>_4band.tif in SRC_DIR this writes to OUT_DIR:
  <name>_4band.tif       calibrated reflectance x 10^4 (uint16, nodata 0) on
                         the source chip's grid
  <name>_4band.png       RGB preview with the fixed --stretch, i.e. exactly
                         the RGB the model sees with --fixed-stretch
  .mask.png .drone.png .ocm.png   copied unchanged, as is coreg_log.json
and chips.csv: scene, offset source, and the share of OCM-clear pixels (all,
and crown-mask pixels) saturated at the stretch's upper limit in any of R, G,
B.

Typical usage:
    python scripts/build_calibrated_chips.py \\
        /Volumes/Earth03/flower/20261012_full_label_extended2_mkl_x4_coreg_4band_stretch_stats_curated \\
        /Volumes/Earth03/flower/20261012_full_label_extended2_mkl_x4_coreg_4band_cal_curated
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import click
import imageio.v3 as iio
import numpy as np
import pandas as pd
import rasterio as rio
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apply_planet_maskrcnn import load_chip  # noqa: E402
from train_planet_image_maskrcnn import (  # noqa: E402
    _load_clear_mask, build_input_channels, load_stretch,
)
from util import calibration_results  # noqa: E402

FLOWER = Path('/Volumes/Earth03/flower')
REPO = Path(__file__).resolve().parent.parent
CAL = calibration_results()
SIDECARS = ('.mask.png', '.drone.png', '.ocm.png')
DATASET_FILES = ('coreg_log.json',)  # copied as-is: same scenes, same shifts


@click.command()
@click.argument('src_dir', type=click.Path(path_type=Path, exists=True, file_okay=False))
@click.argument('out_dir', type=click.Path(path_type=Path))
@click.option('--fill-root', type=click.Path(path_type=Path, exists=True),
              default=FLOWER / 'planet_clipped_calfill', show_default=True)
@click.option('--offsets', type=click.Path(path_type=Path, exists=True),
              default=CAL / 'planet_scene_offsets.csv', show_default=True)
@click.option('--stretch', 'stretch_path', type=click.Path(path_type=Path, exists=True),
              default=REPO / 'config/planet_fixed_stretch.yml', show_default=True)
def main(src_dir, out_dir, fill_root, offsets, stretch_path):
    stretch = load_stretch(stretch_path)
    source = (pd.read_csv(offsets).query("band == 'nir'")
                .set_index('scene_id')['source'])
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in DATASET_FILES:
        if (src_dir / f).exists():
            shutil.copy2(src_dir / f, out_dir / f)
    rows = []
    for chip in tqdm(sorted(src_dir.glob('*_4band.tif')), desc='chips'):
        name = chip.name[:-len('.tif')]
        sid = name[:-len('_4band')]
        clip = fill_root / '4band' / sid[:4] / chip.name
        if not clip.exists():
            raise click.ClickException(f'no calibrated clip for {sid}: {clip}')
        with rio.open(chip) as src:
            grid = {'crs': src.crs, 'transform': src.transform,
                    'height': src.height, 'width': src.width}
            profile = src.profile
        img = load_chip(clip, grid)
        assert img.shape[1:] == (grid['height'], grid['width'])
        assert img.rio.transform() == grid['transform'] and img.rio.crs == grid['crs']
        dest = out_dir / chip.name
        profile.update(dtype='uint16', nodata=0, compress='lzw')
        with rio.open(dest, 'w', **profile) as dst:
            dst.write(img.values.astype(np.uint16))
        for suffix in SIDECARS:
            side = src_dir / f'{name}{suffix}'
            if side.exists():
                shutil.copy2(side, out_dir / side.name)

        arr = img.values.transpose(1, 2, 0).astype(np.float32)
        rgb = build_input_channels(arr, ['red', 'green', 'blue'], stretch)
        iio.imwrite(out_dir / f'{name}.png', np.rint(rgb * 255).astype(np.uint8))
        valid = (arr > 0).all(axis=2)
        clear = valid & _load_clear_mask(str(dest), valid.shape)
        crown = clear & (np.array(Image.open(src_dir / f'{name}.mask.png')) == 255)
        sat = (rgb >= 1).any(axis=2)
        rows.append(dict(chip=chip.name, scene_id=sid, offset_source=source.get(sid, ''),
                         clear_frac=clear.sum() / max(valid.sum(), 1),
                         saturated_clear=sat[clear].mean() if clear.any() else np.nan,
                         saturated_crown=sat[crown].mean() if crown.any() else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'chips.csv', index=False)
    click.echo(f'{len(df)} chips -> {out_dir}; offset sources '
               f'{df.offset_source.value_counts().to_dict()}')
    click.echo('saturated share of clear pixels (median / p90 / max over chips): '
               f'{df.saturated_clear.median():.4f} / {df.saturated_clear.quantile(.9):.4f} / '
               f'{df.saturated_clear.max():.4f}; crown pixels '
               f'{df.saturated_crown.median():.4f} / {df.saturated_crown.quantile(.9):.4f} / '
               f'{df.saturated_crown.max():.4f}')


if __name__ == '__main__':
    main()
