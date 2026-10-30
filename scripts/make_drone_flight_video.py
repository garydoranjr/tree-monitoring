#!/usr/bin/env python
"""Time-lapse video of the weekly mavic (M3M) flights over the 50ha plot.

Each orthomosaic in stri/globus/RGB is cropped to the 50ha plot view used by
plot_study_area_figure.py, block-averaged to --width pixels across (about
0.6 m at the default 1800 px) and shown with the fixed uint16 -> uint8 mapping
in config/crown_classification_mavic.yml, so brightness differences between
frames are the flights' own. Each frame carries the flight date (upper right)
and a scale bar (lower right).

Reading a 4 GB mosaic takes about two minutes, so the downsampled crops are
cached as PNGs in OUTDIR/<name>_frames/cache and reused on later runs.

Typical usage:
    python scripts/make_drone_flight_video.py
    python scripts/make_drone_flight_video.py --fps 4 --start 2025-01-01 --end 2025-12-31
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_update_figures as pu  # noqa: E402
from plot_study_area_figure import DEFAULT_OUT, PLOT_VIEW, add_scalebar, halo  # noqa: E402


def read_frame(path, view, width):
    """Block-averaged display RGB (uint8) of `view` from a mavic mosaic."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
    height = int(round(width * (view[3] - view[1]) / (view[2] - view[0])))
    with rasterio.open(path) as src:
        win = from_bounds(*view, transform=src.transform)
        arr = src.read([1, 2, 3], window=win, out_shape=(3, height, width),
                       resampling=Resampling.average, boundless=True, fill_value=0)
    return (pu.rgb_display(arr) * 255).round().astype(np.uint8)


def cache_frame(path, cache, view, width):
    out = cache / f'{pu.path_date(path):%Y_%m_%d}.png'
    if not out.exists():
        plt.imsave(out, read_frame(path, view, width))
    return out


def render_frame(img, date, out, view, dpi=150, bar_m=200):
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, extent=(view[0], view[2], view[1], view[3]), interpolation='none')
    ax.set_xlim(view[0], view[2])
    ax.set_ylim(view[1], view[3])
    ax.set_axis_off()
    fs = 0.024 * h
    x_frac = 1 - 0.04 - bar_m / (view[2] - view[0])
    add_scalebar(ax, bar_m, f'{bar_m} m', loc=(x_frac, 0.05), fontsize=fs, lw=0.008 * h)
    ax.text(0.975, 0.955, f'{date:%Y-%m-%d}', transform=ax.transAxes, ha='right', va='top',
            color='white', fontsize=fs * 1.15, fontweight='bold', path_effects=halo(3))
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


@click.command()
@click.option('--outdir', type=click.Path(file_okay=False, path_type=Path),
              default=DEFAULT_OUT, show_default=True)
@click.option('--name', default='mavic_2025_weekly', show_default=True)
@click.option('--start', default='2025-01-01', show_default=True)
@click.option('--end', default='2025-12-31', show_default=True)
@click.option('--width', default=1800, show_default=True,
              help='Frame width in pixels (the height follows the 50ha view).')
@click.option('--fps', default=3.0, show_default=True)
@click.option('--jobs', '-j', default=3, show_default=True,
              help='Mosaics read at once; the reads are disk-bound.')
def main(outdir, name, start, end, width, fps, jobs):
    files = [p for p in sorted(pu.MAVIC_RGB.glob('*_M3M_aligned_global_RGB.tif'))
             if pd.Timestamp(start) <= pu.path_date(p) <= pd.Timestamp(end)]
    click.echo(f'{len(files)} mavic flights, {start} .. {end}')
    frames = outdir / f'{name}_frames'
    cache = frames / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(jobs) as ex:
        futs = [ex.submit(cache_frame, p, cache, PLOT_VIEW, width) for p in files]
        with click.progressbar(as_completed(futs), length=len(futs),
                               label='reading mosaics') as bar:
            for f in bar:
                f.result()
    for old in frames.glob('frame_*.png'):
        old.unlink()
    for i, p in enumerate(files):
        img = plt.imread(cache / f'{pu.path_date(p):%Y_%m_%d}.png')[..., :3]
        render_frame(img, pu.path_date(p), frames / f'frame_{i:03d}.png', PLOT_VIEW)
    out = outdir / f'{name}.mp4'
    # H.264 / yuv420p plays in PowerPoint and Keynote; even dimensions for yuv420p
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps),
                    '-i', str(frames / 'frame_%03d.png'),
                    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', '-crf', '18', '-movflags', '+faststart',
                    str(out)], check=True)
    click.echo(f'Wrote {out} ({len(files)} frames at {fps} fps)')


if __name__ == '__main__':
    main()
