#!/usr/bin/env python
"""Derive one fixed per-band input stretch for the Planet crown classifier from
MODIS-calibrated 50ha reflectance.

The classifier's default input stretch (train_planet_image_maskrcnn.py,
build_input_channels) is per image: each band from its 0th to 99.9th
percentile. Chips contain nodata pixels, so in practice each band is divided
by its own 99.9th percentile, and the same forest gets a different input
value on every date. With calibrated reflectance a single stretch can be
shared by all images instead:

    input = clip((reflectance - lo) / (hi - lo), 0, 1)   per band

lo and hi are percentiles (--lo-pct, --hi-pct) of OCM-clear reflectance pooled
over all calibrated clips (planet_clipped_cal), with the same number of
randomly sampled pixels from every clip (--per-clip) so that frequently imaged
seasons do not dominate. Clips under --min-clear clear are skipped. The high
upper percentile keeps bright (flowering) crowns from saturating.

Writes a YAML file read by train_planet_image_maskrcnn.py --fixed-stretch:
    scale: 1.0e-4, lo: [B, G, R, N], hi: [B, G, R, N], plus provenance

Typical usage:
    python scripts/derive_planet_fixed_stretch.py
"""
from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import rasterio as rio
import yaml
from tqdm import tqdm

FLOWER = Path('/Volumes/Earth03/flower')
REPO = Path(__file__).resolve().parent.parent
BANDS = ['blue', 'green', 'red', 'nir']


@click.command()
@click.option('--cal-root', type=click.Path(path_type=Path, exists=True),
              default=FLOWER / 'planet_clipped_cal', show_default=True)
@click.option('--ocm-root', type=click.Path(path_type=Path, exists=True),
              default=FLOWER / 'planet_clipped/ocm', show_default=True)
@click.option('--per-clip', type=int, default=20000, show_default=True)
@click.option('--min-clear', type=float, default=0.5, show_default=True)
@click.option('--lo-pct', type=float, default=0.5, show_default=True)
@click.option('--hi-pct', type=float, default=99.9, show_default=True)
@click.option('--seed', type=int, default=0, show_default=True)
@click.option('--output', type=click.Path(path_type=Path),
              default=REPO / 'config/planet_fixed_stretch.yml', show_default=True)
def main(cal_root, ocm_root, per_clip, min_clear, lo_pct, hi_pct, seed, output):
    rng = np.random.default_rng(seed)
    samples, n_used, n_skip = [], 0, 0
    for clip in tqdm(sorted((cal_root / '4band').glob('*/*_4band.tif')), desc='clips'):
        ocm = ocm_root / clip.parent.name / clip.name.replace('_4band.tif', '_ocm.tif')
        if not ocm.exists():
            n_skip += 1
            continue
        with rio.open(clip) as src, rio.open(ocm) as m:
            if (src.height, src.width) != (m.height, m.width):
                n_skip += 1  # the ~3% of masks on the 7 px inset grid
                continue
            a = src.read().reshape(4, -1)
            clear = (m.read(1).ravel() == 0) & (a > 0).all(axis=0)
        if clear.mean() < min_clear:
            n_skip += 1
            continue
        idx = np.flatnonzero(clear)
        samples.append(a[:, rng.choice(idx, min(per_clip, idx.size), replace=False)])
        n_used += 1
    refl = np.concatenate(samples, axis=1) * 1e-4
    lo, hi = np.percentile(refl, [lo_pct, hi_pct], axis=1)
    med = np.median(refl, axis=1)
    out = dict(scale=1e-4, lo=[round(float(v), 4) for v in lo],
               hi=[round(float(v), 4) for v in hi], bands=BANDS,
               percentiles=[lo_pct, hi_pct], n_clips=n_used, per_clip=per_clip,
               min_clear=min_clear,
               source=f'OCM-clear pixels of {cal_root}/4band (MODIS-calibrated SR), '
                      'scripts/derive_planet_fixed_stretch.py')
    output.write_text(yaml.safe_dump(out, sort_keys=False, default_flow_style=None))
    click.echo(f'{n_used} clips used, {n_skip} skipped; wrote {output}')
    for b, l, h, m in zip(BANDS, lo, hi, med):
        click.echo(f'  {b:5s} lo {l:.4f}  hi {h:.4f}  median {m:.4f} -> input {(m - l) / (h - l):.2f}')


if __name__ == '__main__':
    main()
