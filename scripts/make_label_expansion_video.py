#!/usr/bin/env python
"""Animation of hand labels expanding to classifier labels over the whole 50ha plot.

One Phantom flight (plot_study_area_figure.LABEL_DATE, 2023-03-07). The loop:
  1. hold: Vicente's training labels of events (thick outlines: flowering,
     deciduous); every unlabelled crown under a grey mask
  2. cross-fade: the mask fades out while the classifier's flowering and
     deciduous calls fade in (thin outlines); crowns it calls no event get none
  3. hold: hand labels plus classifier labels
  4. cross-fade back to 1
Labels and classes come from plot_study_area_figure.label_tables(). Writes
<name>.mp4 (H.264; set the slide to loop) and a looping <name>.gif.

Typical usage:
    python scripts/make_label_expansion_video.py
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import subprocess
import sys
from pathlib import Path

import click
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_update_figures as pu  # noqa: E402
import plot_study_area_figure as sa  # noqa: E402

MASK = '#7f7f7f'
EVENTS = ('flowering', 'deciduous')


def ease(t):
    """Smoothstep 0 -> 1."""
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


@click.command()
@click.option('--outdir', type=click.Path(file_okay=False, path_type=Path),
              default=sa.DEFAULT_OUT, show_default=True)
@click.option('--name', default='label_expansion', show_default=True)
@click.option('--fps', default=20, show_default=True)
@click.option('--hold', nargs=2, type=float, default=(1.5, 2.0), show_default=True,
              help='Seconds held on the hand labels and on the classifier labels.')
@click.option('--fade', default=0.5, show_default=True, help='Seconds per cross-fade.')
@click.option('--threshold', default=0.5, show_default=True)
@click.option('--gif-width', default=1280, show_default=True)
def main(outdir, name, fps, hold, fade, threshold, gif_width):
    view = sa.PLOT_VIEW
    lab, new = sa.label_tables(view, threshold)
    events = new[new.cls != 'none']
    n_new = new.cls.value_counts()
    click.echo(f'hand {lab.cls.value_counts().to_dict()}  classifier {n_new.to_dict()}')

    path = pu.phantom_dir('24782016', 'local') / f'BCI_50ha_{sa.LABEL_DATE.replace("-", "_")}_local.tif'
    click.echo('reading the mosaic')
    arr, ext = sa.read_window(path, view, [1, 2, 3, 4], factor=8)
    img = pu.rgb_display(arr)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=150)  # 1920 x 1080
    ax = fig.add_axes([0.02, 0.105, 0.96, 0.88])
    ax.imshow(img, extent=ext)
    ax.set_xlim(view[0], view[2])
    ax.set_ylim(view[1], view[3])
    ax.set_axis_off()
    # grey mask over crowns without a hand label
    new.plot(ax=ax, facecolor=MASK, edgecolor='none', alpha=0.7)
    mask = ax.collections[-1]
    # classifier calls: flowering and deciduous only
    fades = []
    for c in ('flowering', 'deciduous'):
        g = events[events.cls == c]
        if len(g):
            g.boundary.plot(ax=ax, color=sa.LABEL_COLORS[c], lw=1.1)
            fades.append(ax.collections[-1])
    # hand labels of events, always on; hand-labelled no-event crowns, like the
    # classifier's, get no outline (they are unmasked, being labelled)
    for c in EVENTS:
        col = sa.LABEL_COLORS[c]
        g = lab[lab.cls == c]
        if len(g):
            g.boundary.plot(ax=ax, color=col, lw=2.4)
    sa.add_scalebar(ax, 100, '100 m', loc=(0.015, 0.03), fontsize=10)

    n_lab = lab.cls.value_counts()
    handles = [Line2D([], [], color=sa.LABEL_COLORS[c], lw=2.4,
                      label=f'{sa.LABEL_NAMES[c]}: hand label ({n_lab.get(c, 0)})')
               for c in EVENTS]
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.02, 0.0), ncol=2,
               fontsize=11, frameon=False, handlelength=1.8)
    leg_cls = fig.legend(handles=[Line2D([], [], color='0.3', lw=1.1, label='classifier')],
                         loc='lower left', bbox_to_anchor=(0.02, 0.04), fontsize=11,
                         frameon=False, handlelength=1.8)
    leg_mask = fig.legend(handles=[Patch(facecolor=MASK, alpha=0.7, label='unlabelled')],
                          loc='lower left', bbox_to_anchor=(0.02, 0.04), fontsize=11,
                          frameon=False, handlelength=1.8)

    def state(s):
        """s = 0: hand labels + mask; s = 1: classifier labels."""
        mask.set_alpha(0.7 * (1 - s))
        for c in fades:
            c.set_alpha(s)
        for t in leg_mask.get_texts() + leg_mask.legend_handles:
            t.set_alpha((1 - s) * (0.7 if isinstance(t, Patch) else 1))
        for t in leg_cls.get_texts() + leg_cls.legend_handles:
            t.set_alpha(s)

    nh = [int(round(h * fps)) for h in hold]
    nf = int(round(fade * fps))
    seq = ([0.0] * nh[0] + [ease((k + 1) / nf) for k in range(nf)] + [1.0] * nh[1]
           + [ease(1 - (k + 1) / nf) for k in range(nf)])

    mp4 = outdir / f'{name}.mp4'
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    ff = subprocess.Popen(
        ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba',
         '-s', f'{w}x{h}', '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-pix_fmt',
         'yuv420p', '-crf', '18', '-movflags', '+faststart', str(mp4)],
        stdin=subprocess.PIPE)
    cache = {}
    with click.progressbar(seq, label='rendering frames') as bar:
        for s in bar:
            key = round(s, 4)
            if key not in cache:
                state(s)
                fig.canvas.draw()
                cache[key] = bytes(fig.canvas.buffer_rgba())
            ff.stdin.write(cache[key])
    ff.stdin.close()
    if ff.wait():
        raise click.ClickException('ffmpeg failed')
    plt.close(fig)
    gif = outdir / f'{name}.gif'
    subprocess.run(
        ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(mp4), '-vf',
         f'fps={min(fps, 15)},scale={gif_width}:-1:flags=lanczos,split[a][b];'
         '[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a',
         '-loop', '0', str(gif)], check=True)
    click.echo(f'Wrote {mp4} ({len(seq) / fps:.1f} s loop) and {gif}')


if __name__ == '__main__':
    main()
