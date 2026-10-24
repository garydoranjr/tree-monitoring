#!/usr/bin/env python
"""Show the Planet training chips as the Mask R-CNN sees them, with the current
input (TOA chip, per-image percentile stretch) and with the calibrated chip and
the fixed stretch (build_calibrated_chips.py, config/planet_fixed_stretch.yml).

Both renderings go through train_planet_image_maskrcnn.build_input_channels, so
the RGB shown is exactly the [0, 1] model input (x 255), before the fixed
ImageNet mean/std normalisation.

Writes to figs/planet_calibration/:
  training_inputs_current_vs_fixed.png  ~8 chips (darkest and brightest current
      input, the most saturated fixed input, a borrowed-offset chip and dates
      spread over the record), a --crop window of each, current above new
  training_inputs_contact_{current,fixed}.png  every chip, whole, by date
and prints per-channel consistency metrics across chips (SD of the clear-pixel
median input) and crown contrast (crown minus background mean input, and the
same divided by the background SD), also saved to
results/planet_fixed_stretch_metrics.csv.

Typical usage:
    python scripts/plot_planet_fixed_stretch_examples.py
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
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_planet_calibration import INK, MUTED  # noqa: E402
from train_planet_image_maskrcnn import (  # noqa: E402
    _load_clear_mask, build_input_channels, load_stretch,
)

FLOWER = Path('/Volumes/Earth03/flower')
REPO = Path(__file__).resolve().parent.parent
RGB = ['red', 'green', 'blue']
BORROWED, HAZY = '#eb6834', '#d03b3b'


def model_rgb(path: Path, stretch=None) -> np.ndarray:
    with rio.open(path) as src:
        arr = src.read().transpose(1, 2, 0).astype(np.float32)
    return build_input_channels(arr, RGB, stretch)


def chip_metrics(cur, new, clear, crown):
    bg = clear & ~crown
    out = {}
    for v, x in (('current', cur), ('fixed', new)):
        for i, c in enumerate(RGB):
            out[f'{v}_{c}_median'] = np.median(x[..., i][clear]) if clear.any() else np.nan
            if crown.any() and bg.any():
                d = x[..., i][crown].mean() - x[..., i][bg].mean()
                out[f'{v}_{c}_crown_diff'] = d
                out[f'{v}_{c}_crown_cnr'] = d / (x[..., i][bg].std() + 1e-6)
    return out


def contact_sheet(chips, imgs, out, title_fn, flags, ncol=10):
    nrow = int(np.ceil(len(chips) / ncol))
    h, w = next(iter(imgs.values())).shape[:2]
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.9, nrow * 1.9 * h / w + 0.3),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, c in zip(axes.flat, chips):
        ax.axis('on')
        ax.imshow(imgs[c], interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        color = flags.get(c)
        for sp in ax.spines.values():
            sp.set_visible(color is not None)
            if color:
                sp.set_color(color)
                sp.set_linewidth(2.5)
        ax.set_title(title_fn(c), fontsize=6, color=INK, pad=2)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=150)
    plt.close(fig)


@click.command()
@click.option('--current-dir', type=click.Path(path_type=Path, exists=True),
              default=FLOWER / '20261012_full_label_extended2_mkl_x4_coreg_4band_stretch_stats_curated',
              show_default=True)
@click.option('--fixed-dir', type=click.Path(path_type=Path, exists=True),
              default=FLOWER / '20261012_full_label_extended2_mkl_x4_coreg_4band_cal_curated',
              show_default=True)
@click.option('--stretch', 'stretch_path', type=click.Path(path_type=Path, exists=True),
              default=REPO / 'config/planet_fixed_stretch.yml', show_default=True)
@click.option('--index', type=click.Path(path_type=Path, exists=True),
              default=FLOWER / 'planet_bcnm_sr/index.csv', show_default=True)
@click.option('--crop', type=int, default=400, show_default=True,
              help='Side of the square window (chip pixels, 0.75 m) in the comparison.')
@click.option('--hazy', type=float, default=0.1, show_default=True,
              help='Flag chips with more than this share of clear pixels saturated.')
@click.option('--fig-dir', type=click.Path(path_type=Path),
              default=REPO / 'figs/planet_calibration', show_default=True)
@click.option('--metrics', type=click.Path(path_type=Path),
              default=REPO / 'results/planet_fixed_stretch_metrics.csv', show_default=True)
def main(current_dir, fixed_dir, stretch_path, index, crop, hazy, fig_dir, metrics):
    stretch = load_stretch(stretch_path)
    info = pd.read_csv(fixed_dir / 'chips.csv').set_index('chip')
    inst = pd.read_csv(index).set_index('scene_id')['instrument']
    chips = sorted(info.index)
    cur, new, rows = {}, {}, []
    for c in tqdm(chips, desc='chips'):
        cur[c] = model_rgb(current_dir / c)
        new[c] = model_rgb(fixed_dir / c, stretch)
        valid = (new[c] > 0).any(axis=2)
        clear = valid & _load_clear_mask(str(fixed_dir / c), valid.shape)
        crown = clear & (np.array(Image.open(fixed_dir / c.replace('.tif', '.mask.png'))) == 255)
        rows.append(dict(chip=c, **chip_metrics(cur[c], new[c], clear, crown)))
    met = pd.DataFrame(rows).set_index('chip').join(info)
    met.to_csv(metrics)

    summary = []
    for v in ('current', 'fixed'):
        for ch in RGB:
            summary.append(dict(version=v, channel=ch,
                                sd_of_median=met[f'{v}_{ch}_median'].std(),
                                p10_median=met[f'{v}_{ch}_median'].quantile(.1),
                                p90_median=met[f'{v}_{ch}_median'].quantile(.9),
                                crown_diff=met[f'{v}_{ch}_crown_diff'].median(),
                                crown_cnr=met[f'{v}_{ch}_crown_cnr'].median()))
    summary = pd.DataFrame(summary)
    with pd.option_context('display.float_format', '{:.3f}'.format, 'display.width', 160):
        click.echo('across chips: SD and p10/p90 of the clear-pixel median input; '
                   'median over chips of crown - background mean input and of that / background SD')
        click.echo(summary.to_string(index=False))
    summary.to_csv(metrics.with_name(metrics.stem + '_summary.csv'), index=False)

    def label(c, short=False):
        sid = c[:-len('_4band.tif')]
        s = f'{sid[:4]}-{sid[4:6]}-{sid[6:8]} {inst.get(sid, "?")}'
        return s if short else f'{s}, sat. {sid.rsplit("_", 1)[-1]}'

    flags = {c: BORROWED for c in chips if info.loc[c, 'offset_source'] != 'fit'}
    flags.update({c: HAZY for c in chips if info.loc[c, 'saturated_clear'] > hazy})
    fig_dir.mkdir(parents=True, exist_ok=True)

    # comparison: extremes of the current input, the worst fixed one, a borrowed
    # chip, then dates spread over the record
    cur_mean = met[[f'current_{c}_median' for c in RGB]].mean(axis=1)
    pick = [cur_mean.idxmin(), cur_mean.idxmax(), met.saturated_clear.idxmax()]
    borrowed = [c for c in chips if flags.get(c) == BORROWED and c not in pick]
    pick += borrowed[:1]
    rest = [c for c in chips if c not in pick]
    pick += [rest[int(i)] for i in np.linspace(0, len(rest) - 1, 4)]
    pick = sorted(dict.fromkeys(pick))
    h, w = next(iter(new.values())).shape[:2]
    r0, c0 = (h - crop) // 2, (w - crop) // 2
    fig, axes = plt.subplots(2, len(pick), figsize=(2.3 * len(pick), 5.3), squeeze=False)
    for j, c in enumerate(pick):
        for i, (lab, imgs) in enumerate((('current input\n(TOA, per-image stretch)', cur),
                                          ('new input\n(calibrated, fixed stretch)', new))):
            ax = axes[i, j]
            ax.imshow(imgs[c][r0:r0 + crop, c0:c0 + crop], interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                src = info.loc[c, 'offset_source']
                ax.set_title(f'{label(c)}\noffset: {src}', fontsize=7, color=INK)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8, color=INK)
        if c in flags:
            axes[1, j].text(0.03, 0.04, 'hazy' if flags[c] == HAZY else 'borrowed offset',
                            transform=axes[1, j].transAxes, fontsize=7, color=INK,
                            bbox=dict(facecolor='white', alpha=0.85, lw=0, pad=1.5))
    fig.tight_layout()
    fig.savefig(fig_dir / 'training_inputs_current_vs_fixed.png', dpi=200)
    plt.close(fig)

    for v, imgs in (('current', cur), ('fixed', new)):
        contact_sheet(chips, {c: imgs[c][::4, ::4] for c in chips},
                      fig_dir / f'training_inputs_contact_{v}.png',
                      lambda c: label(c, short=True), flags if v == 'fixed' else {})
    click.echo(f'figures in {fig_dir}; comparison chips: {", ".join(pick)}')


if __name__ == '__main__':
    main()
