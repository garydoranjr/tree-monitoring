#!/usr/bin/env python
"""Render example 50ha clips before (raw SR) and after MODIS cross-calibration.

Picks pairs of clear clips from different instruments acquired a few days apart
(from results/planet_calibration_plot_means.csv, written by
evaluate_planet_calibration.py), ranked by the raw-SR plot-mean NIR difference
between the two. Two renderings:

  --stretch model  (default) the RGB the Mask R-CNN classifier sees:
                   `build_input_channels` from train_planet_image_maskrcnn.py,
                   a per-image, per-band 0-99.9 percentile stretch to [0, 1],
                   for the TOA clip the current models use (planet_clipped),
                   raw SR and calibrated SR.
  --stretch fixed  true colour (R, G, B) and false colour (NIR, R, G) of raw
                   and calibrated SR with one linear stretch per band shared
                   by all panels (1st-99th percentile of all shown pixels), so
                   brightness differences between panels are real.

The model stretch starts at each band's minimum over the whole clip, nodata
included. Nearly every clip (and every training chip) has some nodata (0)
pixels, so the stretch is value / 99.9th percentile per band, which a
calibration offset does change; on the few clips without nodata (43 TOA clips,
2020-2022) it starts at the true minimum and would cancel the offset. The
script prints the largest raw-SR vs calibrated model-input difference.

Typical usage:
    python scripts/plot_planet_calibration_examples.py
    python scripts/plot_planet_calibration_examples.py --stretch fixed
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio as rio  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_planet_calibration import FLOWER, INK, REPO  # noqa: E402
from train_planet_image_maskrcnn import build_input_channels  # noqa: E402

SCHEMES = {'true colour': (2, 1, 0), 'false colour': (3, 2, 1)}  # band indices B,G,R,N


def pick_pairs(means: pd.DataFrame, n_pairs: int, max_days: int, min_clear: float):
    w = (means.pivot_table(index=['scene_id', 'instrument', 'satellite_id', 'time',
                                  'clear_frac'], columns='version', values='nir')
              .reset_index())
    w = w[w.clear_frac >= min_clear].sort_values('time').reset_index(drop=True)
    cands = []
    for i, a in w.iterrows():
        for _, b in w.iloc[i + 1:].iterrows():
            if (b.time - a.time).days > max_days:
                break
            if a.instrument != b.instrument:
                cands.append((abs(a.SR - b.SR), a, b))
    pairs, used = [], set()
    for _, a, b in sorted(cands, key=lambda c: -c[0]):
        # one pair per acquisition week, no scene or satellite reused
        if used & {a.scene_id, b.scene_id, a.satellite_id, b.satellite_id,
                   a.time.strftime('%G%V'), b.time.strftime('%G%V')}:
            continue
        pairs.append((a, b))
        used |= {a.scene_id, b.scene_id, a.satellite_id, b.satellite_id,
                 a.time.strftime('%G%V'), b.time.strftime('%G%V')}
        if len(pairs) == n_pairs:
            break
    return pairs


def read(root: Path, sid: str) -> np.ndarray:
    """Raw (4, H, W) values: SR x 10^4, or DN for the TOA clips."""
    with rio.open(root / '4band' / sid[:4] / f'{sid}_4band.tif') as src:
        return src.read().astype(np.float32)


def fixed_panels(imgs: dict) -> dict:
    """{(row label, scene): RGB} with one stretch per band over all images."""
    refl = {k: np.where(a == 0, np.nan, a * 1e-4) for k, a in imgs.items()}
    stack = np.concatenate([a.reshape(4, -1) for a in refl.values()], axis=1)
    lo, hi = np.nanpercentile(stack, [1, 99], axis=1)
    click.echo('stretch lo/hi (B,G,R,N): ' + ' '.join(f'{a:.3f}/{b:.3f}' for a, b in zip(lo, hi)))
    out = {}
    for (v, sid), a in refl.items():
        for sch, idx in SCHEMES.items():
            idx = list(idx)
            rgb = (a[idx] - lo[idx, None, None]) / (hi[idx] - lo[idx])[:, None, None]
            out[(f'{v}\n{sch}', sid)] = np.clip(np.nan_to_num(rgb, nan=1.0), 0, 1).transpose(1, 2, 0)
    return out


def model_panels(imgs: dict) -> dict:
    """{(row label, scene): RGB} exactly as `build_input_channels` feeds the model."""
    return {(f'{v}\nmodel input', sid): build_input_channels(a.transpose(1, 2, 0),
                                                             ['red', 'green', 'blue'])
            for (v, sid), a in imgs.items()}


@click.command()
@click.option('--plot-means', type=click.Path(path_type=Path, exists=True),
              default=REPO / 'results/planet_calibration_plot_means.csv', show_default=True)
@click.option('--toa-root', type=click.Path(path_type=Path),
              default=FLOWER / 'planet_clipped', show_default=True)
@click.option('--sr-root', type=click.Path(path_type=Path),
              default=FLOWER / 'planet_clipped_sr', show_default=True)
@click.option('--cal-root', type=click.Path(path_type=Path),
              default=FLOWER / 'planet_clipped_cal', show_default=True)
@click.option('--stretch', type=click.Choice(['model', 'fixed']), default='model',
              show_default=True)
@click.option('--n-pairs', type=int, default=2, show_default=True)
@click.option('--max-days', type=int, default=3, show_default=True)
@click.option('--min-clear', type=float, default=0.97, show_default=True)
@click.option('--output', type=click.Path(path_type=Path), default=None,
              help='Default figs/planet_calibration/example_clips_<stretch>.png')
def main(plot_means, toa_root, sr_root, cal_root, stretch, n_pairs, max_days, min_clear,
         output):
    output = output or REPO / f'figs/planet_calibration/example_clips_{stretch}.png'
    means = pd.read_csv(plot_means, parse_dates=['time'], dtype={'satellite_id': str})
    scenes = [s for pair in pick_pairs(means, n_pairs, max_days, min_clear) for s in pair]
    versions = {'raw SR': sr_root, 'calibrated': cal_root}
    if stretch == 'model':
        versions = {'TOA (current)': toa_root, **versions}
    imgs = {(v, s.scene_id): read(root, s.scene_id)
            for s in scenes for v, root in versions.items()}
    panels = model_panels(imgs) if stretch == 'model' else fixed_panels(imgs)
    if stretch == 'model':
        d = max(np.abs(panels[('raw SR\nmodel input', s.scene_id)]
                       - panels[('calibrated\nmodel input', s.scene_id)]).max() for s in scenes)
        click.echo(f'largest |raw SR - calibrated| model-input difference: {d:.4f} '
                   f'({d * 255:.1f} of 255)')

    rows = list(dict.fromkeys(r for r, _ in panels))
    h, w = next(iter(panels.values())).shape[:2]
    fig, axes = plt.subplots(len(rows), len(scenes), figsize=(3.2 * len(scenes),
                             3.2 * len(rows) * h / w + 0.8), squeeze=False)
    for j, s in enumerate(scenes):
        for i, row in enumerate(rows):
            ax = axes[i, j]
            ax.imshow(panels[(row, s.scene_id)], interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            col = next(c for k, c in (('TOA', 'TOA'), ('raw', 'SR'), ('cal', 'CAL'))
                       if row.startswith(k))
            ax.text(0.02, 0.04, f'plot NIR {s[col]:.3f}', transform=ax.transAxes,
                    fontsize=7, color=INK,
                    bbox=dict(facecolor='white', alpha=0.8, lw=0, pad=1.5))
            if i == 0:
                ax.set_title(f'{s.time:%Y-%m-%d %H:%M}\n{s.instrument}, sat. {s.satellite_id}',
                             fontsize=8, color=INK)
            if j == 0:
                ax.set_ylabel(row, fontsize=8, color=INK)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    click.echo(f'wrote {output}: ' + ', '.join(s.scene_id for s in scenes))


if __name__ == '__main__':
    main()
