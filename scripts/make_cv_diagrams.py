#!/usr/bin/env python
"""Diagrams of the two cross-validation designs for the Planet Mask R-CNN.

Subcommands (outputs in OUTDIR, default the November 2026 presentation; each
is half a 12.5 x 6 in slide):

  spatial   the east/west split (_split_window in train_planet_image_maskrcnn.py):
            every chip is cut at x = 626281.5 m; the model trains on the
            left (west) 512 x 512 window of every chip and is tested on the
            right (east) one, so every date is in both train and test
  temporal  the first temporal CV (config/temporal_folds_20261012.csv,
            make_temporal_folds.py): chips grouped by flight, cut into blocks
            of at most 14 days, dealt to 6 folds; a fold is held out whole
            and both windows of every chip are used

Typical usage:
    python scripts/make_cv_diagrams.py spatial
    python scripts/make_cv_diagrams.py temporal --fold 2
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_study_area_figure as sa  # noqa: E402
from train_planet_image_maskrcnn import _split_window  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CHIPS = sa.FLOWER / '20261012_full_label_extended2_mkl_x4_coreg_4band_cal_curated'
FOLDS = REPO / 'config/temporal_folds_20261012.csv'
EXAMPLE_CHIP = '20230314_152522_22_247f_4band'
WINDOW = 512  # px, train_planet_image_maskrcnn default size
CHIP_RES = 0.75
SIZE = (6.1, 6.0)  # half the slide
C_TRAIN, C_TEST = '#4c72b0', '#dd8452'


@click.group()
def cli():
    pass


out_option = click.option('--outdir', type=click.Path(file_okay=False, path_type=Path),
                          default=sa.DEFAULT_OUT, show_default=True)


def save(fig, outdir, name, dpi=300):
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'{name}.{ext}', dpi=dpi)
    plt.close(fig)
    click.echo(f'Wrote {outdir / name}.png/.pdf')


@cli.command()
@out_option
def spatial(outdir):
    """East/west split of every chip."""
    import rasterio
    import imageio.v3 as iio
    from matplotlib.patches import Rectangle
    from skimage.measure import find_contours
    img = iio.imread(CHIPS / f'{EXAMPLE_CHIP}.png')
    mask = iio.imread(CHIPS / f'{EXAMPLE_CHIP}.mask.png') > 0
    with rasterio.open(CHIPS / f'{EXAMPLE_CHIP}.tif') as src:
        t = src.transform
    h, w = img.shape[:2]
    ext = (t.c, t.c + w * t.a, t.f + h * t.e, t.f)
    wins = {k: _split_window(h, w, k, WINDOW) for k in ('left', 'right')}
    split_x = t.c + wins['left'][3] * t.a

    W = SIZE[0]
    H = W * h / w
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, extent=ext)
    for c in find_contours(mask.astype(float), 0.5):
        ax.plot(t.c + c[:, 1] * t.a, t.f + c[:, 0] * t.e, color='#ffe11a', lw=0.35)
    for key, color, label in (('left', C_TRAIN, 'TRAIN\nwest window'),
                              ('right', C_TEST, 'TEST\neast window')):
        r0, r1, c0, c1 = wins[key]
        x0, x1 = t.c + c0 * t.a, t.c + c1 * t.a
        y1, y0 = t.f + r0 * t.e, t.f + r1 * t.e
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=3))
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=color, alpha=0.12, lw=0))
        ax.text((x0 + x1) / 2, y1 + 8, label.split('\n')[0], color='white', ha='center',
                va='bottom', fontsize=12, fontweight='bold',
                path_effects=sa.halo(3.5, color))
    ax.axvline(split_x, color='white', lw=1.5, ls='--')
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    sa.bare(ax)
    for s in ax.spines.values():
        s.set_visible(False)
    sa.add_scalebar(ax, 100, '100 m', loc=(0.02, 0.05), lw=2.5)
    save(fig, outdir, 'cv_spatial')


@cli.command()
@out_option
@click.option('--fold', default=2, show_default=True,
              help='Fold drawn as held out in the static image.')
@click.option('--seconds', default=1.5, show_default=True, help='Seconds per fold in the GIF.')
@click.option('--dpi', default=150, show_default=True, help='GIF resolution.')
def temporal(outdir, fold, seconds, dpi):
    """Six folds of contiguous date blocks; the GIF cycles the held-out fold."""
    import io
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch
    from PIL import Image
    df = pd.read_csv(FOLDS, parse_dates=['planet_date', 'block_start', 'block_end'])
    nf = df.fold.nunique()
    W, H = SIZE[0], 3.6
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.15, 0.17, 0.82, 0.8])
    rows = []
    for f in range(nf):
        y = nf - 1 - f
        blocks = df[df.fold == f].drop_duplicates('block')
        bars = [ax.barh(y, (b.block_end - b.block_start).days + 6,
                        left=b.block_start - pd.Timedelta(days=3), height=0.62,
                        alpha=0.25, lw=0)[0] for b in blocks.itertuples()]
        ticks = ax.vlines(df.planet_date[df.fold == f], y - 0.31, y + 0.31, lw=1.0)
        label = ax.text(-0.01, y, '', transform=ax.get_yaxis_transform(), ha='right',
                        va='center', fontsize=9)
        rows.append((f, bars, ticks, label))
    ax.set_yticks([])
    ax.set_ylim(-0.6, nf - 0.4)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', labelsize=9)
    ax.set_xlim(pd.Timestamp('2020-12-01'), pd.Timestamp('2026-02-15'))
    ax.grid(axis='x', alpha=0.3)
    ax.spines[['left', 'right', 'top']].set_visible(False)
    fig.legend(handles=[Patch(color=C_TRAIN, alpha=0.6, label='train'),
                        Patch(color=C_TEST, alpha=0.6, label='test (held out)')],
               loc='lower right', ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.99, -0.01))

    def show(k):
        for f, bars, ticks, label in rows:
            held = f == k
            color = C_TEST if held else C_TRAIN
            for b in bars:
                b.set_color(color)
                b.set_alpha(0.25)
            ticks.set_color(color)
            label.set_text(f'fold {f}' + ('\nheld out' if held else ''))
            label.set_color(color)
            label.set_fontweight('bold' if held else 'normal')

    show(fold)
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'cv_temporal.{ext}', dpi=300)
    frames = []
    for k in range(nf):
        show(k)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi)
        frames.append(Image.open(buf).convert('RGB'))
    plt.close(fig)
    frames = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    gif = outdir / 'cv_temporal.gif'
    frames[0].save(gif, save_all=True, append_images=frames[1:],
                   duration=int(seconds * 1000), loop=0)
    click.echo(f'Wrote {gif} ({nf} frames, {seconds:g} s each)')


if __name__ == '__main__':
    cli()
