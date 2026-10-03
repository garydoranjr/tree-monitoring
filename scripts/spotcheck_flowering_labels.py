#!/usr/bin/env python
"""Spot-check crowns the drone classifier labels as flowering.

The HPC SegFormer classification of the mavic mosaics marks far more crowns
as flowering than the phantom record does (median 1.8 % vs 0.5 % of crowns in
the September 2026 update; see `plot_update_figures.py
classification-timeseries`). This script puts the positives in front of a
person so they can be judged against the imagery.

Subcommands:

  sample  Score every crown's mean P(flowering) on N evenly spaced dates per
          series, sample crown-dates above --threshold, and render contact
          sheets (drone RGB with the crown outline | P(flowering) map).
          Writes spotcheck_sample.csv plus mavic_flowering_<k>.png,
          phantom_flowering_reference.png (24782016 release) and
          phantom_c3kw2x_flowering.png (C3KW2X release).
  zoom    Render chosen crown-dates from spotcheck_sample.csv at native mosaic
          resolution, for a close look or a slide.

Mavic RGB is shown after the same fixed uint16 -> uint8 mapping the
classifier saw (config/crown_classification_mavic.yml); phantom RGB is uint8
and shown as is.

Typical usage:
    python scripts/spotcheck_flowering_labels.py sample
    python scripts/spotcheck_flowering_labels.py zoom zoom_mavic_likely_tp \\
        2025-06-17:854 2024-06-04:1523 2026-01-20:1881
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

from pathlib import Path

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plot_update_figures as pu

DEFAULT_OUT = pu.DEFAULT_OUT / 'spotcheck_flowering'


def load_crowns():
    import geopandas as gpd
    return gpd.read_file(pu.CROWNMAP)


def crown_means(crowns, cls_path, decimate):
    """Mean P(flowering) per crown, NaN for crowns under 10 decimated pixels."""
    from rasterio.features import rasterize
    n = len(crowns)
    arr, t = pu.read_decimated(cls_path, decimate, [1])
    ids = rasterize(((g, i + 1) for i, g in enumerate(crowns.geometry)),
                    out_shape=arr.shape[1:], transform=t, fill=0).ravel()
    counts = np.bincount(ids, minlength=n + 1)[1:]
    sums = np.bincount(ids, weights=np.nan_to_num(arr[0]).ravel(),
                       minlength=n + 1)[1:]
    return np.where(counts >= 10, sums / np.maximum(counts, 1), np.nan)


def read_crop(path, bounds, bands, target_px=None):
    """Read a map-bounds crop; decimate so the width is about target_px."""
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(path) as src:
        win = from_bounds(*bounds, transform=src.transform)
        win = win.round_offsets().round_lengths()
    factor = 1 if target_px is None else max(1, int(win.width // target_px))
    return pu.read_decimated(path, factor, bands, win)


def to_display(arr):
    return pu.rgb_display(arr)


def crown_bounds(geom, pad_frac, pad_m):
    x0, y0, x1, y1 = geom.bounds
    half = max(x1 - x0, y1 - y0) * (0.5 + pad_frac) + pad_m
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (cx - half, cy - half, cx + half, cy + half)


def outline(ax, geom, lw):
    import geopandas as gpd
    gpd.GeoSeries([geom]).boundary.plot(ax=ax, color='cyan', lw=lw)


def sample_source(crowns, source, files, n_dates, n_crowns, threshold,
                  decimate, seed):
    idx = np.unique(np.linspace(0, len(files) - 1, n_dates).astype(int))
    rows = []
    with click.progressbar([files[i] for i in idx],
                           label=f'scoring {source}') as bar:
        for f in bar:
            m = crown_means(crowns, f, decimate)
            for i in np.flatnonzero(m > threshold):
                rows.append({'source': source, 'cls': str(f),
                             'date': pu.path_date(f), 'crown': int(i),
                             'p_flower': float(m[i]),
                             'area_m2': float(crowns.geometry.iloc[i].area)})
    df = pd.DataFrame(rows)
    click.echo(f'{source}: {len(df)} crown-dates above {threshold} on '
               f'{len(idx)} dates; sampling {min(n_crowns, len(df))}')
    return df.sample(min(n_crowns, len(df)), random_state=seed).sort_values('date')


def contact_sheet(crowns, df, path, per_row=4):
    rows = int(np.ceil(len(df) / per_row))
    fig, axes = plt.subplots(rows, per_row * 2,
                             figsize=(per_row * 2 * 2.3, rows * 2.55))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.set_axis_off()
    for j, (_, r) in enumerate(df.iterrows()):
        geom = crowns.geometry.iloc[r['crown']]
        b = crown_bounds(geom, 0.35, 5)
        img, ti = read_crop(pu.drone_ortho_for(r['cls']), b, [1, 2, 3], target_px=220)
        cls, tc = read_crop(r['cls'], b, [1], target_px=220)
        a0 = axes[j // per_row, 2 * (j % per_row)]
        a1 = axes[j // per_row, 2 * (j % per_row) + 1]
        a0.imshow(to_display(img), extent=pu.extent(img, ti))
        a1.set_facecolor('#dddddd')
        a1.imshow(np.ma.masked_invalid(cls[0]), cmap='magma', vmin=0, vmax=1,
                  extent=pu.extent(cls, tc), interpolation='nearest')
        for a in (a0, a1):
            outline(a, geom, 1)
            a.set_xlim(b[0], b[2])
            a.set_ylim(b[1], b[3])
        a0.set_title(f"#{j + 1} {r['date']:%Y-%m-%d} crown {r['crown']}",
                     fontsize=7)
        a1.set_title(f"mean P={r['p_flower']:.2f}", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    click.echo(f'Wrote {path}')


@click.group()
def cli():
    pass


@cli.command()
@click.option('--outdir', type=click.Path(file_okay=False, path_type=Path),
              default=DEFAULT_OUT, show_default=True)
@click.option('--dates', default=16, show_default=True,
              help='Evenly spaced dates scored per series.')
@click.option('--mavic-crowns', default=32, show_default=True)
@click.option('--phantom-crowns', default=16, show_default=True,
              help='Crown-dates sampled from the 24782016 phantom release.')
@click.option('--phantom-ext-crowns', default=16, show_default=True,
              help='Crown-dates sampled from the C3KW2X phantom release.')
@click.option('--threshold', default=0.5, show_default=True,
              help='Minimum mean P(flowering) for a crown-date to be sampled.')
@click.option('--decimate', default=16, show_default=True,
              help='Resolution divisor for scoring crowns (~0.75 m at 16).')
@click.option('--per-sheet', default=16, show_default=True)
@click.option('--seed', default=1, show_default=True)
def sample(outdir, dates, mavic_crowns, phantom_crowns, phantom_ext_crowns,
           threshold, decimate, per_sheet, seed):
    """Sample flowering-labelled crowns and render contact sheets.

    Phantom crowns come from the locally aligned classifications.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    crowns = load_crowns()
    groups = [(key, n, sample_source(crowns, key, pu.classification_files(key),
                                     dates, n, threshold, decimate, seed))
              for key, n in (('mavic', mavic_crowns),
                             ('phantom-24782016', phantom_crowns),
                             ('phantom-C3KW2X', phantom_ext_crowns)) if n > 0]
    pd.concat([df for _, _, df in groups]).to_csv(outdir / 'spotcheck_sample.csv',
                                                 index=False)
    for key, _, df in groups:
        if key == 'mavic':
            for k, start in enumerate(range(0, len(df), per_sheet), 1):
                contact_sheet(crowns, df.iloc[start:start + per_sheet],
                              outdir / f'mavic_flowering_{k}.png')
        elif key == 'phantom-24782016':
            contact_sheet(crowns, df, outdir / 'phantom_flowering_reference.png')
        else:
            contact_sheet(crowns, df, outdir / 'phantom_c3kw2x_flowering.png')


@cli.command()
@click.argument('name')
@click.argument('picks', nargs=-1, required=True)
@click.option('--outdir', type=click.Path(file_okay=False, path_type=Path),
              default=DEFAULT_OUT, show_default=True)
def zoom(name, picks, outdir):
    """Render crown-dates at native resolution as NAME.png.

    Each pick is YYYY-MM-DD:CROWN, or YYYY-MM-DD:CROWN:SERIES (e.g.
    2024-03-06:198:mavic) when a date is in more than one series.
    """
    df = pd.read_csv(outdir / 'spotcheck_sample.csv', parse_dates=['date'])
    crowns = load_crowns()
    fig, axes = plt.subplots(1, len(picks), figsize=(5.2 * len(picks), 5.4),
                             squeeze=False)
    for ax, pick in zip(axes[0], picks):
        d, c, *series = pick.split(':')
        match = df[(df['date'] == pd.Timestamp(d)) & (df['crown'] == int(c))]
        if series:
            match = match[match['source'] == series[0]]
        if match.empty:
            raise click.ClickException(f'{pick} is not in spotcheck_sample.csv')
        if len(match) > 1:
            raise click.ClickException(f'{pick} is in several series; append :SERIES')
        r = match.iloc[0]
        geom = crowns.geometry.iloc[int(c)]
        b = crown_bounds(geom, 0, 3)
        img, t = read_crop(pu.drone_ortho_for(r['cls']), b, [1, 2, 3])
        ax.imshow(to_display(img), extent=pu.extent(img, t))
        outline(ax, geom, 0.8)
        ax.set_xlim(b[0], b[2])
        ax.set_ylim(b[1], b[3])
        ax.set_axis_off()
        label = pu.SERIES[r['source']][0]
        ax.set_title(f"{label} {d} crown {c}  mean P={r['p_flower']:.2f}",
                     fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / f'{name}.png', dpi=120, bbox_inches='tight')
    click.echo(f"Wrote {outdir / f'{name}.png'}")


if __name__ == '__main__':
    cli()
