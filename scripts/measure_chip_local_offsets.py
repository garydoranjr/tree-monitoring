#!/usr/bin/env python
"""Measure the residual, spatially varying drone-to-Planet offset in training chips.

`apply_drone_labels_coreg.py` registers each drone ortho to its Planet scene
with a single AROSICS shift, and writes the shifted ortho as the `.drone.png`
sidecar on an exact 2x subdivision of the chip grid. Whatever misalignment is
left after that shift is what the crown labels inherit. A single rigid shift
cannot remove a non-linear warp, so the leftover shows up as offsets that vary
across the chip.

This script tiles each chip into windows and phase-correlates the Planet
preview against the drone sidecar (downsampled onto the chip grid) inside each
window. Windows that are cloudy (per the `.ocm.png` overlay), that fall outside
the drone footprint, or whose post-shift normalized cross-correlation is weak
are dropped. Two per-chip summaries are reported:

* ``median_offset_m`` -- magnitude of the median window vector, i.e. the rigid
  residual the global shift left behind;
* ``spread_m`` -- median distance of the window vectors from that median
  vector, i.e. the non-rigid component a single shift cannot fix.

Cross-sensor phase correlation against 3 m Planet data is noisy, so these
numbers are best read as a comparison between chip sets processed the same way,
not as absolute registration accuracy.

Outputs (in OUTDIR):
  <prefix>_windows.csv   one row per accepted window
  <prefix>_chips.csv     one row per chip
  <prefix>.png / .pdf    distributions per chip set plus a quiver map for the
                         chip with the largest spread

Typical usage:
    python scripts/measure_chip_local_offsets.py \\
        --set "2020-23 local-align=/Volumes/Earth03/flower/20260706_full_label_application_x4_coreg_4band_stretch_stats_curated" \\
        --set "2024-26 mavic=/Volumes/Earth03/flower/20260915_globus_label_application_x4_coreg_4band_stretch_stats_rerun" \\
        --vetting "2024-26 mavic=/Volumes/Earth03/flower/20260915_globus_label_application_x4_coreg_4band_stretch_stats_rerun/vetting.json" \\
        /Volumes/Earth03/flower/figs/202609_updates

A set can span several chip directories (repeat ``--set`` with the same NAME),
and ``--only-stems NAME=FILE`` restricts it to the chip stems listed in FILE,
one per line, e.g. to compare two alignments of the same Planet scenes.
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation

Image.MAX_IMAGE_PIXELS = None

# Chips are 3 m Planet cutouts upsampled 4x (`--resize 4`).
CHIP_RES_M = 0.75
QUALITY_RANK = {'Poor': 0, 'Fair': 1, 'Good': 2}
SET_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']


def load_chip(chip_png):
    """Return (planet_gray, drone_gray, valid) on the chip grid for one chip."""
    stem = chip_png.name[:-len('.png')]
    planet = Image.open(chip_png).convert('RGB')
    drone = Image.open(chip_png.with_name(f'{stem}.drone.png')).convert('RGB')
    drone = drone.resize(planet.size, Image.BILINEAR)
    ocm_path = chip_png.with_name(f'{stem}.ocm.png')

    p = np.asarray(planet, dtype=np.float32)
    d = np.asarray(drone, dtype=np.float32)
    valid = (p.sum(axis=2) > 0) & (d.sum(axis=2) > 0)
    if ocm_path.exists():
        ocm = np.asarray(Image.open(ocm_path).convert('RGBA'))
        valid &= ocm[..., 3] == 0
    return p.mean(axis=2), d.mean(axis=2), valid


def standardize(a, sigma):
    """Blur to roughly the Planet resolution and z-score, so the drone's fine
    texture does not dominate the correlation."""
    a = gaussian_filter(a, sigma)
    return (a - a.mean()) / (a.std() + 1e-6)


def window_offsets(planet, drone, valid, win, min_valid, min_ncc, max_shift_px,
                   sigma):
    """Phase-correlate each window; yield accepted window records."""
    h, w = planet.shape
    for r0 in range(0, h - win + 1, win):
        for c0 in range(0, w - win + 1, win):
            sl = (slice(r0, r0 + win), slice(c0, c0 + win))
            if valid[sl].mean() < min_valid:
                continue
            p = standardize(planet[sl], sigma)
            d = standardize(drone[sl], sigma)
            shift, _, _ = phase_cross_correlation(p, d, upsample_factor=10,
                                                  normalization=None)
            dy, dx = shift
            if max(abs(dy), abs(dx)) > max_shift_px:
                continue
            # Post-shift NCC on the overlap, as a match-quality gate.
            iy, ix = int(round(dy)), int(round(dx))
            ps = p[max(iy, 0):win + min(iy, 0), max(ix, 0):win + min(ix, 0)]
            ds = d[max(-iy, 0):win + min(-iy, 0), max(-ix, 0):win + min(-ix, 0)]
            ncc = float(np.corrcoef(ps.ravel(), ds.ravel())[0, 1])
            if not np.isfinite(ncc) or ncc < min_ncc:
                continue
            yield {
                'row': r0 + win // 2, 'col': c0 + win // 2,
                # Positive dx: the drone content must move east to match Planet;
                # positive dy: it must move south (image rows).
                'dx_m': dx * CHIP_RES_M, 'dy_m': dy * CHIP_RES_M,
                'offset_m': float(np.hypot(dx, dy) * CHIP_RES_M),
                'ncc': ncc,
            }


def chip_summary(wdf):
    mx, my = wdf['dx_m'].median(), wdf['dy_m'].median()
    spread = np.hypot(wdf['dx_m'] - mx, wdf['dy_m'] - my)
    return {
        'n_windows': len(wdf),
        'median_dx_m': mx, 'median_dy_m': my,
        'median_offset_m': float(np.hypot(mx, my)),
        'spread_m': float(spread.median()),
        'max_offset_m': float(wdf['offset_m'].max()),
    }


def parse_pairs(values, what):
    out = {}
    for k, p in parse_multi(values, what).items():
        out[k] = p[-1]
    return out


def parse_multi(values, what):
    """NAME=PATH pairs; a NAME given more than once keeps every PATH, in order."""
    out = {}
    for v in values:
        if '=' not in v:
            raise click.BadParameter(f'{what} must be NAME=PATH, got {v!r}')
        k, p = v.split('=', 1)
        out.setdefault(k, []).append(Path(p))
    return out


@click.command()
@click.argument('outdir', type=click.Path(file_okay=False, path_type=Path))
@click.option('--set', 'sets', multiple=True, required=True,
              help='NAME=CHIPDIR; repeatable, and a NAME may be repeated to '
                   'pool several directories. Order sets the plot colours.')
@click.option('--vetting', 'vettings', multiple=True,
              help='NAME=vetting.json; restrict that set to chips rated at '
                   'least --min-quality. Repeat to pool several rating files.')
@click.option('--only-stems', 'only_stems', multiple=True,
              help='NAME=FILE; restrict that set to the chip stems listed in '
                   'FILE (one per line, with or without .png).')
@click.option('--min-quality', type=click.Choice(list(QUALITY_RANK)),
              default='Good', show_default=True)
@click.option('--window', default=128, show_default=True,
              help='Window size in chip pixels (0.75 m).')
@click.option('--min-valid', default=0.9, show_default=True,
              help='Minimum clear, in-footprint fraction of a window.')
@click.option('--min-ncc', default=0.3, show_default=True,
              help='Minimum post-shift normalized cross-correlation.')
@click.option('--max-shift-m', default=20.0, show_default=True,
              help='Discard window shifts larger than this as spurious.')
@click.option('--sigma', default=1.5, show_default=True,
              help='Gaussian blur (chip pixels) applied before correlating.')
@click.option('--min-windows', default=5, show_default=True,
              help='Chips with fewer accepted windows are not summarized.')
@click.option('--prefix', default='residual_offsets', show_default=True)
@click.option('--color', 'set_colors', multiple=True,
              help='NAME=COLOR; overrides the order-based colour of a set.')
def main(outdir, sets, vettings, only_stems, min_quality, window, min_valid,
         min_ncc, max_shift_m, sigma, min_windows, prefix, set_colors):
    outdir.mkdir(parents=True, exist_ok=True)
    sets = parse_multi(sets, '--set')
    vettings = parse_multi(vettings, '--vetting')
    only_stems = parse_pairs(only_stems, '--only-stems')
    colors = dict(zip(sets, SET_COLORS))
    colors.update({k: str(v) for k, v in parse_pairs(set_colors, '--color').items()})

    wrows, crows = [], []
    for name, chipdirs in sets.items():
        chips = sorted((p for d in chipdirs for p in d.glob('*.png')
                        if p.name.count('.') == 1), key=lambda p: p.name)
        if name in vettings:
            ratings = {}
            for v in vettings[name]:
                ratings.update(json.loads(v.read_text())['ratings'])
            keep = {k for k, v in ratings.items()
                    if QUALITY_RANK[v['quality']] >= QUALITY_RANK[min_quality]}
            chips = [p for p in chips if p.name in keep]
        if name in only_stems:
            listed = {ln.strip().removesuffix('.png')
                      for ln in only_stems[name].read_text().splitlines() if ln.strip()}
            chips = [p for p in chips if p.name[:-len('.png')] in listed]
        click.echo(f'{name}: {len(chips)} chips from '
                   + ', '.join(str(d) for d in chipdirs))

        for chip in chips:
            chipdir = chip.parent
            stem = chip.name[:-len('.png')]
            planet, drone, valid = load_chip(chip)
            recs = list(window_offsets(planet, drone, valid, window, min_valid,
                                       min_ncc, max_shift_m / CHIP_RES_M,
                                       sigma))
            for r in recs:
                wrows.append({'set': name, 'stem': stem, **r})
            if len(recs) >= min_windows:
                crows.append({'set': name, 'stem': stem, 'chipdir': str(chipdir),
                              'width': planet.shape[1], 'height': planet.shape[0],
                              **chip_summary(pd.DataFrame(recs))})

    wdf, cdf = pd.DataFrame(wrows), pd.DataFrame(crows)
    wdf.to_csv(outdir / f'{prefix}_windows.csv', index=False)
    cdf.to_csv(outdir / f'{prefix}_chips.csv', index=False)

    click.echo('\nper-set summary (chips with >= %d windows):' % min_windows)
    for name, g in cdf.groupby('set', sort=False):
        w = wdf[wdf['set'] == name]
        click.echo(f'  {name}: {len(g)} chips, {len(w)} windows | window '
                   f'|d| median {w["offset_m"].median():.2f} m, p90 '
                   f'{w["offset_m"].quantile(0.9):.2f} m, > 3 m '
                   f'{(w["offset_m"] > 3).mean():.1%} | chip spread median '
                   f'{g["spread_m"].median():.2f} m | chip rigid residual '
                   f'median {g["median_offset_m"].median():.2f} m')

    plot(wdf, cdf, list(sets), colors, outdir / prefix, window, max_shift_m)


def plot(wdf, cdf, names, colors, outbase, window, max_shift_m):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6),
                             gridspec_kw={'width_ratios': [1, 1, 1.25]})

    bins = np.linspace(0, max_shift_m, 41)
    ax = axes[0]
    for name in names:
        w = wdf.loc[wdf['set'] == name, 'offset_m']
        ax.hist(w, bins=bins, density=True, histtype='step', lw=2,
                color=colors[name],
                label=f'{name} (median {w.median():.1f} m, n={len(w)})')
    ax.set_xlabel('window offset magnitude after global shift (m)')
    ax.set_ylabel('density')
    ax.set_title(f'Per-window residual offset ({window * CHIP_RES_M:.0f} m windows)')
    ax.legend(fontsize=8)

    ax = axes[1]
    data = [cdf.loc[cdf['set'] == n, 'spread_m'] for n in names]
    parts = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=True)
    for patch, n in zip(parts['boxes'], names):
        patch.set_facecolor(colors[n])
        patch.set_alpha(0.5)
    for i, d in enumerate(data, 1):
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(d))
        ax.scatter(np.full(len(d), i) + jitter, d, s=10, color='k', alpha=0.5,
                   zorder=3)
    ax.set_xticks(range(1, len(names) + 1),
                  [f'{n}\n({len(d)} chips)' for n, d in zip(names, data)])
    ax.set_ylabel('within-chip offset spread (m)')
    ax.set_title('Non-rigid component per chip')
    ax.grid(axis='y', alpha=0.3)

    ax = axes[2]
    worst = cdf.sort_values('spread_m', ascending=False).iloc[0]
    w = wdf[wdf['stem'] == worst['stem']]
    img = Image.open(Path(worst['chipdir']) / f"{worst['stem']}.png")
    ax.imshow(img)
    q = ax.quiver(w['col'], w['row'], w['dx_m'], w['dy_m'], w['offset_m'],
                  angles='xy', scale_units='xy', scale=1 / 8, cmap='autumn',
                  width=0.005)
    ax.quiverkey(q, 0.84, 1.04, 5, '5 m', labelpos='E', coordinates='axes')
    ax.set_title(f"{worst['set']}: {worst['stem'].replace('_4band', '')}\n"
                 f"largest spread ({worst['spread_m']:.1f} m); arrows x8",
                 fontsize=9)
    ax.set_axis_off()

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{outbase}.{ext}', dpi=150, bbox_inches='tight')
    click.echo(f'Wrote {outbase}.png/.pdf')


if __name__ == '__main__':
    main()
