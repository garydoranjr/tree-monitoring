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

    W, H = SIZE
    fig = plt.figure(figsize=SIZE)
    ah = 0.92 * W / H * h / w * (h * CHIP_RES + 40) / (h * CHIP_RES)
    ax = fig.add_axes([0.04, 0.96 - ah, 0.92, ah])
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
    ax.set_ylim(ext[2], ext[3] + 40)
    sa.bare(ax)
    for s in ax.spines.values():
        s.set_visible(False)
    sa.add_scalebar(ax, 100, '100 m', loc=(0.02, 0.05), lw=2.5)
    side = WINDOW * CHIP_RES
    notes = [
        'All 142 training chips (PlanetScope, 2021–2026) are cut\n'
        '   at the same line (dashed).',
        f'Train on the west {side:.0f} × {side:.0f} m window of every chip;\n'
        '   test on the east window.',
        'Every date is in both train and test: new trees, seen dates.',
    ]
    fig.text(0.04, 0.96 - ah - 0.05, '\n'.join('• ' + n for n in notes), fontsize=10,
             va='top', linespacing=1.5)
    fig.text(0.04, 0.96 - ah - 0.33, 'Yellow: drone-derived crown labels on one chip '
             f'(PlanetScope {pd.Timestamp(EXAMPLE_CHIP[:8]):%d %b %Y}).', fontsize=8,
             color='0.35')
    save(fig, outdir, 'cv_spatial')


@cli.command()
@out_option
@click.option('--fold', default=2, show_default=True, help='Fold drawn as held out.')
def temporal(outdir, fold):
    """Six folds of contiguous date blocks; one held out at a time."""
    import matplotlib.dates as mdates
    df = pd.read_csv(FOLDS, parse_dates=['planet_date', 'block_start', 'block_end'])
    nf = df.fold.nunique()
    W, H = SIZE
    fig = plt.figure(figsize=SIZE)
    ax = fig.add_axes([0.15, 0.38, 0.82, 0.54])
    for f in range(nf):
        y = nf - 1 - f
        held = f == fold
        color = C_TEST if held else C_TRAIN
        blocks = df[df.fold == f].drop_duplicates('block')
        for b in blocks.itertuples():
            ax.barh(y, (b.block_end - b.block_start).days + 6, left=b.block_start
                    - pd.Timedelta(days=3), height=0.62, color=color, alpha=0.25, lw=0)
        d = df[df.fold == f]
        ax.vlines(d.planet_date, y - 0.31, y + 0.31, color=color, lw=1.0)
        ax.text(-0.01, y, f'fold {f}' + ('\nheld out' if held else ''),
                transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=9,
                color=color, fontweight='bold' if held else 'normal')
    ax.set_yticks([])
    ax.set_ylim(-0.6, nf - 0.4)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', labelsize=9)
    ax.set_xlim(pd.Timestamp('2020-12-01'), pd.Timestamp('2026-02-15'))
    ax.grid(axis='x', alpha=0.3)
    ax.spines[['left', 'right', 'top']].set_visible(False)
    ax.set_title(f'{len(df)} chips in {df.block.nunique()} blocks of ≤ 14 days, dealt to '
                 f'{nf} folds', fontsize=10.5, pad=6)
    held = df[df.fold == fold]
    notes = [
        'Chips from one drone flight stay in one block;\n'
        '   neighboring blocks never share a fold.',
        f'Hold out one fold (here fold {fold}, {len(held)} chips), train on the\n'
        '   other five; repeat for every fold.',
        'Both windows of every chip are used: seen trees, new dates.',
        f'Held-out chips are a median {df.days_to_nearest_train.median():.0f} days '
        f'(range {df.days_to_nearest_train.min()}–{df.days_to_nearest_train.max()})\n'
        '   from the nearest training chip.',
    ]
    fig.text(0.04, 0.27, '\n'.join('• ' + n for n in notes), fontsize=10, va='top',
             linespacing=1.45)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=C_TRAIN, alpha=0.6, label='train'),
                        Patch(color=C_TEST, alpha=0.6, label='test (held-out dates)')],
               loc='upper right', ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.98, 0.335))
    save(fig, outdir, 'cv_temporal')


if __name__ == '__main__':
    cli()
