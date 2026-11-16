#!/usr/bin/env python
"""Time-lapse video of every coregistered Planet scene over the 50ha plot.

Each scene with coreg_ok in the coregistration log is read at the 50ha plot
view used by plot_study_area_figure.py, offset by its (x_shift_m, y_shift_m),
so every frame sits on the drone frame and the plot stays still on screen.
All frames use the fixed stretch in config/planet_fixed_stretch.yml, so
brightness differences between frames are the scenes' own. Each frame carries
the acquisition date (lower right) and a scale bar (lower left). Pixels the
scene does not cover are black.

Typical usage:
    python scripts/make_planet_coreg_video.py
    python scripts/make_planet_coreg_video.py --fps 12 --min-clear 0.5
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_study_area_figure as sa  # noqa: E402

COREG_JSON = sa.FLOWER / 'crown_series_20261101/coreg_all_scenes.json'


def render_frame(path, shift, date, out, view, width, dpi=150, bar_m=200):
    arr, ext = sa.read_window(path, sa.shifted_bounds(view, shift), [1, 2, 3])
    img = sa.fixed_rgb(arr)
    img[arr.sum(axis=0) == 0] = 0
    height = int(round(width * (view[3] - view[1]) / (view[2] - view[0])))
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor('black')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, extent=sa.unshift_extent(ext, shift), interpolation='nearest')
    ax.set_xlim(view[0], view[2])
    ax.set_ylim(view[1], view[3])
    ax.set_axis_off()
    fs = 0.024 * height
    sa.add_scalebar(ax, bar_m, f'{bar_m} m', loc=(0.04, 0.05), fontsize=fs, lw=0.008 * height)
    ax.text(0.975, 0.04, f'{date:%Y-%m-%d}', transform=ax.transAxes, ha='right', va='bottom',
            color='white', fontsize=fs * 1.15, fontweight='bold', path_effects=sa.halo(3))
    fig.savefig(out, dpi=dpi, facecolor='black')
    plt.close(fig)


@click.command()
@click.option('--coreg-json', type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=COREG_JSON, show_default=True)
@click.option('--planet-dir', type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=sa.PLANET_CALFILL, show_default=True,
              help='4band clips, as YEAR/<scene>.tif.')
@click.option('--outdir', type=click.Path(file_okay=False, path_type=Path),
              default=sa.DEFAULT_OUT, show_default=True)
@click.option('--name', default='planet_coreg_sequence', show_default=True)
@click.option('--start', default='2000-01-01', show_default=True)
@click.option('--end', default='2100-01-01', show_default=True)
@click.option('--min-clear', default=0.0, show_default=True,
              help="Skip scenes whose log clear_fraction is below this.")
@click.option('--width', default=1800, show_default=True,
              help='Frame width in pixels (the height follows the 50ha view).')
@click.option('--fps', default=8.0, show_default=True)
@click.option('--jobs', '-j', default=6, show_default=True)
@click.option('--ffmpeg', default='ffmpeg', show_default=True, help='ffmpeg executable.')
def main(coreg_json, planet_dir, outdir, name, start, end, min_clear, width, fps, jobs, ffmpeg):
    recs = [r for r in json.loads(coreg_json.read_text()) if r['coreg_ok']
            and pd.Timestamp(start) <= pd.Timestamp(r['date']) <= pd.Timestamp(end)
            and r['clear_fraction'] >= min_clear]
    recs = sorted({r['scene']: r for r in recs}.values(), key=lambda r: r['scene'])
    have = [r for r in recs if (planet_dir / r['planet_rel']).exists()]
    click.echo(f'{len(recs)} coregistered scenes, {len(recs) - len(have)} missing from {planet_dir}')
    frames = outdir / f'{name}_frames'
    frames.mkdir(parents=True, exist_ok=True)
    for old in frames.glob('frame_*.png'):
        old.unlink()
    with ProcessPoolExecutor(jobs) as ex:
        futs = [ex.submit(render_frame, planet_dir / r['planet_rel'],
                          (r['x_shift_m'], r['y_shift_m']), pd.Timestamp(r['date']),
                          frames / f'frame_{i:04d}.png', sa.PLOT_VIEW, width)
                for i, r in enumerate(have)]
        with click.progressbar(futs, label='rendering frames') as bar:
            for f in bar:
                f.result()
    out = outdir / f'{name}.mp4'
    # H.264 / yuv420p plays in PowerPoint and Keynote; even dimensions for yuv420p
    subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-framerate', str(fps),
                    '-i', str(frames / 'frame_%04d.png'),
                    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', '-crf', '18', '-movflags', '+faststart',
                    str(out)], check=True)
    click.echo(f'Wrote {out} ({len(have)} frames at {fps} fps, {len(have) / fps:.0f} s)')


if __name__ == '__main__':
    main()
