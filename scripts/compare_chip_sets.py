#!/usr/bin/env python
"""Compare two copies of a Planet chip set on the same grids: geometric shift,
sharpness and per-band correlation, chip by chip.

Written to check that the MODIS-calibrated chips (build_calibrated_chips.py:
the scene's SR-product clip, offset-calibrated and resampled onto the TOA
chip's grid) line up with the TOA chips the drone labels were coregistered to.
A shift or extra blur in the calibrated copy would misplace the labels and
could cancel a real radiometric gain.

For every <name>_4band.tif present in both directories, on pixels that are
valid in both and OCM-clear (the `.ocm.png` sidecar of the first set):

* ``dy_m``, ``dx_m``, ``shift_m``: the whole-chip shift of set B relative to
  set A in the standardized band (--band): an integer phase-correlation start
  refined by masked Lucas-Kanade (least squares on the image gradients, valid
  pixels only, iterated with cubic warping). Plain phase correlation is biased
  toward zero here, because invalid and cloudy pixels sit at the same place in
  both copies. Positive dx: B's content must move east to match A; positive
  dy: it must move south.
* ``win_shift_m``: the median magnitude of the same shift over --win px
  windows, a check on the whole-chip value.
* ``sharp_ratio``: Tenengrad over the band's variance in B divided by the same
  in A (both on the clear pixels away from invalid edges). Below 1: B is
  blurrier.
* ``r_<band>``: Pearson correlation of A and B for each band.

Typical usage:
    python scripts/compare_chip_sets.py \\
        /Volumes/Earth03/flower/20261012_full_label_extended2_mkl_x4_coreg_4band_stretch_stats_curated \\
        /Volumes/Earth03/flower/20261012_full_label_extended2_mkl_x4_coreg_4band_cal_curated \\
        results/chip_sets_toa_vs_cal.csv

``--self-test SHIFT_PX`` instead shifts each chip of set A by a known amount
(SHIFT_PX south and SHIFT_PX west) and compares it with itself, to check that
the shift is recovered; set B is then ignored.
"""
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import rasterio as rio
from scipy import ndimage
from skimage.registration import phase_cross_correlation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_planet_image_maskrcnn import _load_clear_mask  # noqa: E402

BANDS = ('blue', 'green', 'red', 'nir')  # chip band order


def read_chip(path):
    with rio.open(path) as src:
        return src.read().astype(np.float32), abs(src.transform.a)


def standardized(band, valid):
    out = np.zeros_like(band)
    v = band[valid]
    out[valid] = (v - v.mean()) / (v.std() + 1e-6)
    return out


def shift_px(a, b, valid, n_iter=20, tol=1e-3):
    """(dy, dx) that moves b's content onto a, in pixels."""
    d, _, _ = phase_cross_correlation(standardized(a, valid), standardized(b, valid),
                                      normalization=None)
    for _ in range(n_iter):
        inner = ndimage.binary_erosion(valid, iterations=int(np.abs(d).max()) + 3)
        if inner.sum() < 100:
            return np.full(2, np.nan)
        bw = ndimage.shift(b, d, order=3, mode='nearest')
        A, B = standardized(a, inner), standardized(bw, inner)
        gy, gx = np.gradient(B)
        # shift(B, delta) ~ B - delta_y * gy - delta_x * gx
        J = -np.stack([gy[inner], gx[inner]], axis=1)
        delta = np.linalg.lstsq(J, (A - B)[inner], rcond=None)[0]
        d = d + delta
        if np.abs(delta).max() < tol:
            break
    return d


def window_shifts(a, b, valid, win, min_valid=0.8):
    h, w = a.shape
    for r0 in range(0, h - win + 1, win):
        for c0 in range(0, w - win + 1, win):
            sl = (slice(r0, r0 + win), slice(c0, c0 + win))
            if valid[sl].mean() >= min_valid:
                yield shift_px(a[sl], b[sl], valid[sl])


def sharpness(band, valid):
    inner = ndimage.binary_erosion(valid, iterations=3)
    gx, gy = ndimage.sobel(band, 1), ndimage.sobel(band, 0)
    return (gx ** 2 + gy ** 2)[inner].mean() / band[inner].var()


def compare(a, b, valid, band_idx, win, res):
    dy, dx = shift_px(a[band_idx], b[band_idx], valid)
    wins = [np.hypot(*s) for s in window_shifts(a[band_idx], b[band_idx], valid, win)]
    row = {'dy_m': dy * res, 'dx_m': dx * res, 'shift_m': np.hypot(dy, dx) * res,
           'win_shift_m': np.median(wins) * res if wins else np.nan, 'n_win': len(wins),
           'sharp_ratio': sharpness(b[band_idx], valid) / sharpness(a[band_idx], valid),
           'valid_frac': valid.mean()}
    for i, name in enumerate(BANDS):
        row[f'r_{name}'] = np.corrcoef(a[i][valid], b[i][valid])[0, 1]
    return row


@click.command()
@click.argument('set_a', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument('set_b', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument('out_csv', type=click.Path(dir_okay=False, path_type=Path))
@click.option('--band', default='red', type=click.Choice(BANDS), show_default=True)
@click.option('--win', default=256, show_default=True, help='Window size (px).')
@click.option('--self-test', 'self_test', type=float, default=None, metavar='SHIFT_PX',
              help='Compare each chip of SET_A with itself shifted by SHIFT_PX '
                   '(SET_B is then ignored).')
def main(set_a, set_b, out_csv, band, win, self_test):
    band_idx = BANDS.index(band)
    rows = []
    for tif in sorted(set_a.glob('*_4band.tif')):
        a, res = read_chip(tif)
        if self_test is not None:
            b = np.stack([ndimage.shift(x, (self_test, -self_test), order=3, mode='nearest')
                          for x in a])
        else:
            other = set_b / tif.name
            if not other.exists():
                continue
            b, res_b = read_chip(other)
            if a.shape != b.shape or not np.isclose(res, res_b):
                raise click.ClickException(f'{tif.name}: grids differ')
        valid = (a > 0).all(axis=0) & (b > 0).all(axis=0) & _load_clear_mask(str(tif), a.shape[1:])
        if self_test is not None:
            m = int(np.ceil(abs(self_test))) + 2
            valid[:m], valid[-m:], valid[:, :m], valid[:, -m:] = False, False, False, False
        if valid.mean() < 0.2:
            continue
        rows.append({'chip': tif.name, **compare(a, b, valid, band_idx, win, res)})
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    q = df[['dy_m', 'dx_m', 'shift_m', 'win_shift_m', 'sharp_ratio']
           + [f'r_{b}' for b in BANDS]].quantile([0.1, 0.5, 0.9])
    with pd.option_context('display.float_format', '{:.3f}'.format, 'display.width', 160):
        click.echo(f'{len(df)} chips compared ({band}); quantiles over chips:')
        click.echo(q.to_string())
    click.echo(f'wrote {out_csv}')


if __name__ == '__main__':
    main()
