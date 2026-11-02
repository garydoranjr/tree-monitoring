#!/usr/bin/env python
"""How drone crown labels become PlanetScope training labels (presentation slide).

Timeline (label_transfer_timeline.{png,pdf}): drone flights (Phantom, Mavic)
and the PlanetScope scenes within +-2 days of a labeled flight
(apply_drone_labels_coreg.py --timewindow 2), those AROSICS co-registered,
and those kept as training chips after vetting.

Animation, two examples from the training set
(20261012_full_label_extended2_mkl_x4_coreg_4band_cal_curated), each a drone
tile (the chip's shift-corrected `.drone.png`) and its Planet chip (`.png`):
  A  a clear scene: the label outlines slide from the drone to Planet
  B  a half-cloudy scene: the outlines slide over, the OmniCloudMask cloud and
     shadow mask fades in, and crowns touching it are dropped, as
     crown_filter_by_ocm() does when the chip is built
Labels are the drone classifier's flowering-or-deciduous mask (`--mode both`,
P > 0.5) moved by the scene's AROSICS shift; example B's unfiltered mask is
rebuilt with apply_drone_labels_coreg's own functions and checked against the
chip's `.mask.png`.

The animation is written as label_transfer.mp4 / .gif (looping), with its
first and last frames as label_transfer_start.png and label_transfer.png.

Typical usage:
    python scripts/make_label_transfer_figure.py
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
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_update_figures as pu  # noqa: E402
import plot_study_area_figure as sa  # noqa: E402

FLOWER = sa.FLOWER
CHIPS = FLOWER / '20261012_full_label_extended2_mkl_x4_coreg_4band_cal_curated'
TIMEWINDOW = 2  # days, apply_drone_labels_coreg.py --timewindow
CHIP_RES = 0.75  # m, --resize 4
EXAMPLES = (
    dict(key='A', stem='20230314_152522_22_247f_4band', cloudy=False),
    dict(key='B', stem='20230327_152535_70_247f_4band', cloudy=True),
)
C_LABEL = '#ffe11a'
C_DROP = '#ff2d2d'
C_CLOUD = (1.0, 0.45, 0.1)
W = sa.FIG_W
TIMELINE_H, EX_H = 2.9, 3.45  # inches; the two halves of the slide


def chip_records():
    log = pd.DataFrame(json.loads((CHIPS / 'coreg_log.json').read_text()))
    log['ldate'] = pd.to_datetime(log.label.str.extract(r'(\d{4}_\d{2}_\d{2})')[0],
                                  format='%Y_%m_%d')
    log['pdate'] = pd.to_datetime(log.scene.str[:8])
    log['chip'] = [(CHIPS / f'{s}.mask.png').exists() for s in log.scene]
    return log


def unfiltered_mask(rec):
    """The chip's label mask before crown_filter_by_ocm, and the OCM classes, on the chip grid."""
    import rioxarray  # noqa: F401
    import xarray as xr
    from rasterio.enums import Resampling
    import apply_drone_labels_coreg as adl
    chip = xr.open_dataarray(CHIPS / f"{rec['scene']}.tif", engine='rasterio')
    label = pu.CLASSIFICATIONS / rec['label']
    conf = adl.apply_shift_to_label(adl.load_label(label, 'both'), rec['x_shift_m'],
                                    rec['y_shift_m'])
    conf = conf.rio.reproject_match(chip, resampling=Resampling.average)
    ocm = sa.PLANET_OCM / rec['scene'][:4] / rec['scene'].replace('_4band', '_ocm.tif')
    ocm = adl.reproject_ocm_to_grid(ocm, chip)
    return conf.values > 0.5, ocm.values


def components(mask, ocm):
    """Connected label components and whether each touches non-clear OCM pixels."""
    from scipy.ndimage import label
    lab, n = label(mask)
    drop = np.zeros(n + 1, bool)
    if ocm is not None:
        bad = np.bincount(lab[ocm != 0].ravel(), minlength=n + 1) > 0
        drop[1:] = bad[1:]
    return lab, drop


def pick_window(lab, drop, ocm, side_px, cloudy):
    """side_px square of the chip: most label components (and, for a cloudy
    chip, a cloud/shadow share near one half)."""
    from scipy.ndimage import find_objects
    h, w = lab.shape
    cen = np.array([((s[0].start + s[0].stop) / 2, (s[1].start + s[1].stop) / 2)
                    for s in find_objects(lab)])
    best = None
    for r0 in range(0, h - side_px + 1, 20):
        for c0 in range(0, w - side_px + 1, 20):
            inside = ((cen[:, 0] >= r0) & (cen[:, 0] < r0 + side_px)
                      & (cen[:, 1] >= c0) & (cen[:, 1] < c0 + side_px))
            score = inside.sum()
            if cloudy:
                cf = (ocm[r0:r0 + side_px, c0:c0 + side_px] != 0).mean()
                kept, gone = (inside & ~drop[1:]).sum(), (inside & drop[1:]).sum()
                score = min(kept, gone) * (1 - 2 * abs(cf - 0.5))
            if best is None or score > best[0]:
                best = (score, r0, c0)
    return best[1], best[2]


def contours(lab, ids, r0, c0, side_px):
    """Outline polylines of components `ids` in window-fraction coordinates (x, y up)."""
    from skimage.measure import find_contours
    out = {}
    sub = lab[r0:r0 + side_px, c0:c0 + side_px]
    for i in ids:
        m = np.pad(sub == i, 1)
        if not m.any():
            continue
        out[i] = [np.column_stack([(c[:, 1] - 0.5) / side_px, 1 - (c[:, 0] - 0.5) / side_px])
                  for c in find_contours(m.astype(float), 0.5)]
    return out


@click.command()
@click.option('--outdir', type=click.Path(file_okay=False, path_type=Path),
              default=sa.DEFAULT_OUT, show_default=True)
@click.option('--side-m', default=200.0, show_default=True, help='Example tile side, meters.')
@click.option('--fps', default=20, show_default=True)
@click.option('--dpi', default=150, show_default=True)
def main(outdir, side_m, fps, dpi):
    import imageio.v3 as iio
    log = chip_records()
    side = int(round(side_m / CHIP_RES))

    # ---- timeline (its own image) ----
    import matplotlib.dates as mdates
    tfig = plt.figure(figsize=(W, TIMELINE_H))
    ax = tfig.add_axes([0.13, 0.13, 0.84, 0.8])
    phantom = sorted(set(pu.drone_dates('phantom-24782016')) | set(pu.drone_dates('phantom-C3KW2X')))
    mavic = pu.drone_dates('mavic')
    labeled = set(log.ldate)
    scenes = log.groupby('scene').agg(pdate=('pdate', 'first'), ok=('coreg_ok', 'any'),
                                      chip=('chip', 'any'))
    rows = [('Phantom flights', phantom, pu.C_PHANTOM, 3),
            ('Mavic flights', mavic, pu.C_MAVIC, 2),
            ('paired', list(scenes.pdate), '0.82', 1),
            ('co-registered', list(scenes.pdate[scenes.ok]), '0.55', 1),
            ('chips', list(scenes.pdate[scenes.chip]), '0.1', 1)]
    for label, dates, color, y in rows:
        ax.vlines(dates, y - 0.3, y + 0.3, color=color, lw=1.0 if y > 1 else 1.3)
    ax.set_yticks([3, 2, 1], ['Phantom flights', 'Mavic flights',
                              f'PlanetScope\nwithin ±{TIMEWINDOW} days'], fontsize=10)
    ax.set_ylim(0.4, 3.6)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', labelsize=9.5)
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(pd.Timestamp('2019-11-01'), pd.Timestamp('2026-03-01'))
    ax.grid(axis='x', alpha=0.3)
    ax.spines[['left', 'right', 'top']].set_visible(False)
    ax.legend(handles=[Line2D([], [], color='0.82', lw=1.5, label=f'{len(scenes)} paired with a '
                              'labeled flight'),
                       Line2D([], [], color='0.55', lw=1.5, label=f'{scenes.ok.sum()} '
                              'co-registered to the drone mosaic'),
                       Line2D([], [], color='0.1', lw=1.5, label=f'{scenes.chip.sum()} '
                              'kept as training chips (vetted Good)')],
              loc='upper right', bbox_to_anchor=(1.0, 1.04), ncol=1, fontsize=8.5,
              frameon=False)
    n_lab = sum(1 for d in phantom + mavic if d in labeled)
    for ext in ('png', 'pdf'):
        tfig.savefig(outdir / f'label_transfer_timeline.{ext}', dpi=300)
    plt.close(tfig)

    # ---- examples (the animation) ----
    fig = plt.figure(figsize=(W, EX_H), dpi=dpi)

    t, g, pg = 2.62, 0.08, 0.5
    x0 = (W - 4 * t - 2 * g - pg) / 2
    y0 = 0.42
    H = EX_H
    xs = [x0, x0 + t + g, x0 + 2 * t + g + pg, x0 + 3 * t + 2 * g + pg]
    lines = []  # (Line2D, xy in window fraction, drone-tile x, planet-tile x, comp id, example)
    cloud_imgs = []
    for k, ex in enumerate(EXAMPLES):
        rec = log[log.scene == ex['stem']].iloc[0].to_dict()
        final = iio.imread(CHIPS / f"{ex['stem']}.mask.png") > 0
        if ex['cloudy']:
            raw, ocm = unfiltered_mask(rec)
            lab, drop = components(raw, ocm)
            kept = (lab > 0) & ~drop[lab]
            agree = (kept == final).mean()
            click.echo(f"{ex['key']}: rebuilt mask matches .mask.png on {agree:.4%} of pixels")
        else:
            ocm = None
            lab, drop = components(final, None)
        r0, c0 = pick_window(lab, drop, ocm, side, ex['cloudy'])
        win = (slice(r0, r0 + side), slice(c0, c0 + side))
        drone = iio.imread(CHIPS / f"{ex['stem']}.drone.png")[2 * r0:2 * (r0 + side),
                                                               2 * c0:2 * (c0 + side)]
        planet = iio.imread(CHIPS / f"{ex['stem']}.png")[win]
        xd, xp = xs[2 * k], xs[2 * k + 1]
        for x, img, title in ((xd, drone, f"drone {rec['ldate']:%d %b %Y}"),
                              (xp, planet, f"PlanetScope {rec['pdate']:%d %b %Y}")):
            a = fig.add_axes([x / W, y0 / H, t / W, t / H])
            a.imshow(img, extent=(0, 1, 0, 1), interpolation='nearest' if img is planet
                     else 'antialiased')
            if img is planet and ex['cloudy']:
                over = np.zeros(planet.shape[:2] + (4,))
                over[..., :3] = C_CLOUD
                over[..., 3] = (ocm[win] != 0) * 0.55
                cloud_imgs.append(a.imshow(over, extent=(0, 1, 0, 1), interpolation='nearest',
                                           alpha=0.0))
            a.set_xlim(0, 1)
            a.set_ylim(0, 1)
            sa.bare(a)
            a.set_title(title, fontsize=9.5, pad=3)
            if x == xd:
                a.plot([0.06, 0.06 + 50 / side_m], [0.06, 0.06], color='white', lw=2.5,
                       solid_capstyle='butt',
                       path_effects=sa.halo(4))
                a.text(0.06 + 25 / side_m, 0.085, '50 m', ha='center', va='bottom',
                       fontsize=8, color='white', path_effects=sa.halo(1.8))
        fig.text((xd + t + g / 2) / W, (y0 + t + 0.27) / H,
                 ('Clear scene' if not ex['cloudy']
                                     else 'Half-cloudy scene: crowns touching cloud are dropped'),
                 ha='center', fontsize=10.5, fontweight='bold' if False else 'normal')
        ids = np.unique(lab[win])
        ids = ids[ids > 0]
        for i, polys in contours(lab, ids, r0, c0, side).items():
            for p in polys:
                ln = Line2D([], [], color=C_LABEL, lw=0.9, transform=fig.transFigure)
                fig.add_artist(ln)
                lines.append((ln, p, xd, xp, bool(drop[i]), ex['cloudy']))

    fig.legend(handles=[Line2D([], [], color=C_LABEL, lw=1.5, label='drone label '
                               '(flowering or leafless crown)'),
                        plt.Rectangle((0, 0), 1, 1, fc=C_CLOUD, alpha=0.55,
                                      label='cloud or shadow (OmniCloudMask)'),
                        Line2D([], [], color=C_DROP, lw=1.5, label='dropped: touches cloud')],
               loc='lower center', ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.005))

    # ---- animation ----
    def ease(u):
        u = np.clip(u, 0, 1)
        return u * u * (3 - 2 * u)

    def draw(tsec):
        slide = ease((tsec - 1.0) / 1.5)
        cloud = ease((tsec - 3.3) / 0.8)
        red = ease((tsec - 4.4) / 0.4)
        gone = ease((tsec - 5.2) / 0.8)
        for ln, p, xd, xp, dropped, cloudy in lines:
            x = xd + slide * (xp - xd)
            ln.set_data((x + p[:, 0] * t) / W, (y0 + p[:, 1] * t) / H)
            if dropped:
                ln.set_color(C_DROP if red > 0.5 else C_LABEL)
                ln.set_alpha(1 - gone)
                ln.set_linewidth(0.9 + 0.6 * red * (1 - gone))
        for im in cloud_imgs:
            im.set_alpha(cloud)

    total = 8.0
    mp4 = outdir / 'label_transfer.mp4'
    draw(0)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    ff = subprocess.Popen(
        ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba',
         '-s', f'{w}x{h}', '-r', str(fps), '-i', '-', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-movflags', '+faststart',
         str(mp4)], stdin=subprocess.PIPE)
    with click.progressbar(range(int(total * fps)), label='rendering') as bar:
        for f in bar:
            draw(f / fps)
            fig.canvas.draw()
            ff.stdin.write(bytes(fig.canvas.buffer_rgba()))
    ff.stdin.close()
    if ff.wait():
        raise click.ClickException('ffmpeg failed')
    draw(total)
    fig.savefig(outdir / 'label_transfer.png', dpi=300)
    draw(0)
    fig.savefig(outdir / 'label_transfer_start.png', dpi=300)
    plt.close(fig)
    gif = outdir / 'label_transfer.gif'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(mp4), '-vf',
                    'fps=12,scale=1280:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff'
                    '[p];[b][p]paletteuse=dither=sierra2_4a', '-loop', '0', str(gif)],
                   check=True)
    click.echo(f'Wrote label_transfer_timeline.png/.pdf, {mp4}, {gif}, label_transfer.png '
               f'(final frame) and label_transfer_start.png\nlabeled flights {n_lab}; '
               f'Planet scenes paired {len(scenes)}, co-registered {scenes.ok.sum()}, '
               f'training chips {scenes.chip.sum()}')


if __name__ == '__main__':
    main()
