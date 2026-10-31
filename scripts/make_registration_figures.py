#!/usr/bin/env python
"""Planet -> drone image registration figures for the November 2026 presentation.

Subcommands (outputs in OUTDIR, default the presentation folder):

  blink   GIF/MP4 in which a Planet scene blinks on and off over the same-week
          drone orthomosaic: top, Planet at its own georeferencing; bottom,
          Planet moved by its AROSICS shift (crown_series_20261029/
          coreg_all_scenes.json). Outlines of crowns the drone classifier calls
          leafless that day are drawn fixed in every frame as reference marks.
  shifts  the initial misregistration of every co-registered Planet scene: the
          AROSICS shift vectors (by Dove model) and their magnitudes.

Both are sized for one third of a 12.5 x 6 in slide.

Typical usage:
    python scripts/make_registration_figures.py blink
    python scripts/make_registration_figures.py shifts
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import json
import subprocess
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_update_figures as pu  # noqa: E402
import plot_study_area_figure as sa  # noqa: E402

SIZE = (4.15, 6.0)  # one third of the 12.5 x 6 in slide
# 100%-clear scene one day after a phantom flight, with a large AROSICS shift
# that an independent brute-force brightness match confirms: (-11, +17) m
BLINK_SCENE = '20240209_155640_21_24ad_4band'
BLINK_DRONE = '2024-02-08'
C_OUTLINE = '#ffe11a'


@click.group()
def cli():
    pass


out_option = click.option('--outdir', type=click.Path(file_okay=False, path_type=Path),
                          default=sa.DEFAULT_OUT, show_default=True)


def leafless_crowns(day, threshold=0.5):
    """Dated crown outlines the phantom classifier calls leafless on `day`."""
    import geopandas as gpd
    import xarray as xr
    day = pd.Timestamp(day)
    c = gpd.read_file(sa.DATED_CROWNS)
    c = c[c.date == day.strftime('%Y_%m_%d')].copy()
    c['tag'] = pd.to_numeric(c.tag, errors='coerce')
    with xr.open_dataset(sa.NC4['phantom']) as ds:
        dp = ds.deciduous_probability.sel(date=day).to_series()
    c['dp'] = c.tag.map(dp)
    return c[c.dp >= threshold]


def busiest_window(crowns, side, view=sa.PLOT_VIEW, step=10):
    """side x side window inside `view` holding the most crown centroids."""
    x = crowns.geometry.centroid.x.values
    y = crowns.geometry.centroid.y.values
    best = None
    for x0 in np.arange(view[0], view[2] - side + 1, step):
        for y0 in np.arange(view[1], view[3] - side + 1, step):
            n = ((x >= x0 + 10) & (x < x0 + side - 10) & (y >= y0 + 10) & (y < y0 + side - 10)).sum()
            if best is None or n > best[0]:
                best = (n, (x0, y0, x0 + side, y0 + side))
    return best[1]


@cli.command()
@out_option
@click.option('--side', default=150.0, show_default=True, help='Window side, metres.')
@click.option('--period', default=0.7, show_default=True, help='Seconds per blink state.')
@click.option('--dpi', default=150, show_default=True)
def blink(outdir, side, period, dpi):
    """Planet blinking over the drone mosaic, before and after registration."""
    import geopandas as gpd
    crowns = leafless_crowns(BLINK_DRONE)
    b = busiest_window(crowns, side)
    shift = sa.coreg_shifts()[BLINK_SCENE]
    sr = sa.scene_info(BLINK_SCENE)['sr']
    drone, dext = sa.read_window(sa.drone_file('phantom', BLINK_DRONE), b, [1, 2, 3, 4],
                                 factor=4)
    drone = pu.rgb_display(drone)
    lo, hi = sa.scene_stretch(sr)
    pad = (b[0] - 30, b[1] - 30, b[2] + 30, b[3] + 30)
    raw, rext = sa.read_window(sr, pad, [1, 2, 3])
    reg, gext = sa.read_window(sr, sa.shifted_bounds(pad, shift), [1, 2, 3])
    planet = {'raw': (sa.planet_rgb(raw, lo, hi), rext),
              'reg': (sa.planet_rgb(reg, lo, hi), sa.unshift_extent(gext, shift))}
    mag = np.hypot(*shift)

    fig = plt.figure(figsize=SIZE, dpi=dpi)
    W, H = SIZE
    s = 2.4
    x0 = (W - s) / 2
    ys = {'raw': H - 0.42 - s, 'reg': 0.38}
    over = {}
    for key, y in ys.items():
        ax = fig.add_axes([x0 / W, y / H, s / W, s / H])
        ax.imshow(drone, extent=dext)
        img, ext = planet[key]
        over[key] = ax.imshow(img, extent=ext, interpolation='nearest')
        crowns.boundary.plot(ax=ax, color=C_OUTLINE, lw=1.0)
        ax.set_xlim(b[0], b[2])
        ax.set_ylim(b[1], b[3])
        sa.bare(ax)
        sa.add_scalebar(ax, 50, '50 m', loc=(0.06, 0.06), lw=2.5)
    fig.text(0.5, (ys['raw'] + s + 0.08) / H, 'Before registration', ha='center',
             va='bottom', fontsize=11)
    fig.text(0.5, (ys['reg'] + s + 0.08) / H,
             f'After registration (shift {mag:.0f} m)', ha='center', va='bottom', fontsize=11)
    label = fig.text(0.5, 0.06 / H, '', ha='center', va='bottom', fontsize=9.5)
    texts = {True: f'PlanetScope {pd.Timestamp(BLINK_SCENE[:8]):%d %b %Y}',
             False: f'drone {pd.Timestamp(BLINK_DRONE):%d %b %Y}'}

    frames = []
    for on in (True, False):
        for im in over.values():
            im.set_visible(on)
        label.set_text(texts[on])
        fig.canvas.draw()
        frames.append(bytes(fig.canvas.buffer_rgba()))
    w, h = fig.canvas.get_width_height()
    for on, f in zip((True, False), frames):
        if on:
            plt.imsave(outdir / 'registration_blink_planet.png',
                       np.frombuffer(f, np.uint8).reshape(h, w, 4))
    plt.close(fig)

    mp4 = outdir / 'registration_blink.mp4'
    n = max(1, int(round(period * 20)))
    ff = subprocess.Popen(
        ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba',
         '-s', f'{w}x{h}', '-r', '20', '-i', '-', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-movflags', '+faststart',
         str(mp4)], stdin=subprocess.PIPE)
    for _ in range(4):  # 4 blinks; loop the slide
        for f in frames:
            for _ in range(n):
                ff.stdin.write(f)
    ff.stdin.close()
    if ff.wait():
        raise click.ClickException('ffmpeg failed')
    gif = outdir / 'registration_blink.gif'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(mp4), '-vf',
                    'fps=10,split[a][b];[a]palettegen[p];[b][p]paletteuse',
                    '-loop', '0', str(gif)], check=True)
    click.echo(f'Wrote {mp4}, {gif} and registration_blink_planet.png\n'
               f'window {b}, shift {shift[0]:.1f}, {shift[1]:.1f} m, {len(crowns)} leafless '
               'crowns on the drone date')


@cli.command()
@out_option
@click.option('--dpi', default=300, show_default=True)
def shifts(outdir, dpi):
    """AROSICS shift of every co-registered Planet scene, by Dove model."""
    recs = pd.DataFrame(json.loads(sa.COREG_JSON.read_text()))
    ok = recs[recs.coreg_ok].copy()
    ok['sat'] = ok.scene.str.replace('_4band', '').str.split('_').str[-1]
    idx = pd.read_csv(sa.BCNM_SR_INDEX, usecols=['satellite_id', 'instrument']).drop_duplicates()
    inst = dict(zip(idx.satellite_id, idx.instrument)) | sa.EXTRA_INSTRUMENTS
    ok['instrument'] = ok.sat.map(inst)
    ok['mag'] = np.hypot(ok.x_shift_m, ok.y_shift_m)

    W, H = SIZE
    fig = plt.figure(figsize=SIZE)
    ax = fig.add_axes([0.62 / W, 2.42 / H, 3.25 / W, 3.25 / H])
    lim = 35
    for r, lab in ((3, '1 px'), (10, '10 m')):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color='0.45', lw=0.8, ls=':'))
        ax.text(-r * 0.71 - 0.3, r * 0.71 + 0.3, lab, fontsize=7.5, color='0.35', ha='right',
                va='bottom')
    # compute_coreg_shift's AROSICS search window: max_shift 10 Planet pixels per axis
    ax.add_patch(plt.Rectangle((-30, -30), 60, 60, fill=False, color='0.45', lw=0.8, ls='--'))
    ax.text(29, -31, 'search limit', fontsize=7.5, color='0.35', ha='right', va='top')
    # largest group first, so the rarer models sit on top
    for key, (label, color) in reversed(list(sa.DOVE_MODELS.items())):
        g = ok[ok.instrument == key]
        ax.scatter(g.x_shift_m, g.y_shift_m, s=7, lw=0, alpha=0.7, color=color,
                   label=f'{label} ({len(g)})')
    ax.axhline(0, color='0.75', lw=0.6)
    ax.axvline(0, color='0.75', lw=0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('east shift (m)', fontsize=9.5, labelpad=1)
    ax.set_ylabel('north shift (m)', fontsize=9.5, labelpad=1)
    ax.tick_params(labelsize=8)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc='upper left', fontsize=7.5, frameon=True, framealpha=0.9, markerscale=2,
              handletextpad=0.2, borderpad=0.3)
    ax.set_title('Planet-to-drone shift per scene', fontsize=10.5, pad=4)

    ax2 = fig.add_axes([0.62 / W, 0.5 / H, 3.25 / W, 1.3 / H])
    bins = np.arange(0, 37, 1.5)
    ax2.hist(ok.mag, bins=bins, color='0.55', ec='white', lw=0.3)
    med = ok.mag.median()
    ax2.axvline(med, color='k', lw=1.2)
    ax2.axvline(3, color='0.4', lw=0.8, ls=':')
    ax2.text(3.3, ax2.get_ylim()[1] * 0.98, '1 px', fontsize=7.5, color='0.35', va='top')
    ax2.text(med + 0.5, ax2.get_ylim()[1] * 0.92, f'median {med:.1f} m', fontsize=8,
             va='top')
    ax2.set_xlabel('shift magnitude (m)', fontsize=9.5, labelpad=1)
    ax2.set_ylabel('scenes', fontsize=9.5)
    ax2.tick_params(labelsize=8)
    ax2.spines[['top', 'right']].set_visible(False)
    name = 'registration_shifts'
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'{name}.{ext}', dpi=dpi)
    plt.close(fig)
    q = ok.mag.quantile([0.25, 0.5, 0.75, 0.9]).round(1).to_dict()
    click.echo(f'Wrote {outdir / name}.png/.pdf\n{len(ok)} of {len(recs)} scenes co-registered; '
               f'magnitude quartiles {q}; > 3 m: {(ok.mag > 3).mean():.0%}, '
               f'> 10 m: {(ok.mag > 10).mean():.0%}\n'
               f"by model: {ok.groupby('instrument').mag.median().round(1).to_dict()}")


if __name__ == '__main__':
    cli()
