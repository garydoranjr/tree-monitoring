#!/usr/bin/env python
"""Test-time radiometric sensitivity of temporal-CV Mask R-CNN folds.

Re-scores each fold's held-out chips (both windows, as eval_temporal_cv.py)
with an additive per-band offset applied to the raw chip before the model's
own input stretch. The offset is k times the across-scene SD of the fitted
Planet-MODIS offsets (results/planet_calibration/planet_scene_offsets.csv, own-fit scenes), i.e.
the size of the per-scene error that calibration removes, with the same sign
in every band (haze-like). k = 0 is the unperturbed score.

Offsets are in surface-reflectance units. Calibrated chips store reflectance
x 10^4, so the offset is added as k * sd * 10^4 (--units reflectance). TOA
chips store radiance counts, so with --units toa each band's offset is
converted with the chip's own counts-per-reflectance gain: the slope of the
TOA band against the same band of the calibrated copy (--cal-chip-dir) over
clear pixels. Valid pixels stay at least 1 so they never turn into nodata.

If a model's skill barely moves under offsets as large as those calibration
removes, calibration cannot help that model. A model that degrades sharply
is one whose input stretch passes scene-level errors on, as the fixed stretch
does for borrowed-offset scenes.

Typical usage (TOA folds, on the Mac; CUDA is much faster):
    python scripts/radiometric_sensitivity.py \\
        --chip-dir /Volumes/Earth03/flower/20261012_full_label_extended2_mkl_x4_coreg_4band_stretch_stats_curated \\
        --cal-chip-dir /Volumes/Earth03/flower/20261012_full_label_extended2_mkl_x4_coreg_4band_cal_curated \\
        --folds config/temporal_folds_20261012.csv \\
        --run-template '/Volumes/Earth03/flower/202610_temporal/fold{fold}' \\
        --units toa --device cpu --k -1,0,1 --out-csv sensitivity_toa.csv
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must precede torch import
from pathlib import Path

import click
import numpy as np
import pandas as pd
import rasterio as rio
import torch

from eval_temporal_cv import DAY_BINS, Accumulator, day_bin, load_fold_model, score_fold
from train_planet_image_maskrcnn import PlanetMaskRCNNDataset, _load_clear_mask
from util import calibration_results, offset_sd, select_device

CAL = calibration_results()
BANDS = ('blue', 'green', 'red', 'nir')  # chip band order


def toa_gain(toa_path, cal_dir, n=20000, seed=0):
    """Per-band TOA counts per unit reflectance, from the calibrated copy."""
    with rio.open(toa_path) as src:
        toa = src.read().astype(np.float64)
    with rio.open(Path(cal_dir) / Path(toa_path).name) as src:
        cal = src.read().astype(np.float64) * 1e-4
    valid = ((toa > 0).all(0) & (cal > 0).all(0)
             & _load_clear_mask(str(toa_path), toa.shape[1:]))
    idx = np.flatnonzero(valid)
    idx = np.random.default_rng(seed).choice(idx, min(n, idx.size), replace=False)
    return np.array([np.polyfit(cal[b].ravel()[idx], toa[b].ravel()[idx], 1)[0]
                     for b in range(len(BANDS))])


class Perturb:
    """raw_transform for PlanetMaskRCNNDataset: add offset * gain per band."""

    def __init__(self, offset, units, cal_dir=None):
        self.offset, self.units, self.cal_dir = offset, units, cal_dir
        self.gains = {}

    def gain(self, path):
        if self.units == 'reflectance':
            return np.full(len(BANDS), 1e4)
        if path not in self.gains:
            self.gains[path] = toa_gain(path, self.cal_dir)
        return self.gains[path]

    def __call__(self, arr, path):
        if not np.any(self.offset):
            return arr
        valid = (arr > 0).all(axis=-1)
        out = arr.copy()
        out[valid] = np.maximum(out[valid] + self.offset * self.gain(path), 1)
        return out


@click.command()
@click.option('--chip-dir', required=True, type=click.Path(exists=True, file_okay=False))
@click.option('--folds', 'folds_csv', required=True, type=click.Path(exists=True))
@click.option('--run-template', required=True,
              help="Run directory per fold, with '{fold}' for the fold number.")
@click.option('--epoch', default=12, show_default=True)
@click.option('--units', required=True, type=click.Choice(['toa', 'reflectance']))
@click.option('--cal-chip-dir', default=None, type=click.Path(exists=True, file_okay=False),
              help='Calibrated copy of the chips, for --units toa gains.')
@click.option('--k', 'ks', default='-1,-0.5,0,0.5,1', show_default=True,
              help='Offset multiples of the across-scene offset SD.')
@click.option('--offsets-csv', default=str(CAL / 'planet_scene_offsets.csv'),
              show_default=True)
@click.option('--score-thresh', default=0.5, type=float)
@click.option('--iou-thresh', default=0.5, type=float)
@click.option('--device', 'device_name', default='auto', show_default=True,
              type=click.Choice(['auto', 'cpu', 'cuda', 'mps']),
              help="'auto' picks CUDA, then MPS, then CPU. MPS hit GPU "
                   'command-buffer errors and a float64 failure on the Mac, so '
                   'use cpu there.')
@click.option('--out-csv', required=True, type=click.Path(dir_okay=False))
def main(chip_dir, folds_csv, run_template, epoch, units, cal_chip_dir, ks, offsets_csv,
         score_thresh, iou_thresh, device_name, out_csv):
    if units == 'toa' and cal_chip_dir is None:
        raise click.UsageError('--units toa needs --cal-chip-dir')
    ks = [float(k) for k in ks.split(',')]
    sd = offset_sd(offsets_csv)
    click.echo('offset SD per band (reflectance): '
               + ', '.join(f'{b} {v:.4f}' for b, v in zip(BANDS, sd)))
    folds_df = pd.read_csv(folds_csv)
    folds_df['day_bin'] = folds_df['days_to_nearest_train'].apply(day_bin)
    chip_info = folds_df.set_index('chip').to_dict('index')
    folds = sorted(folds_df['fold'].unique())
    device = select_device(device_name)

    perturbs = {k: Perturb(k * sd, units, cal_chip_dir) for k in ks}
    accs = {k: {'pooled': Accumulator(device),
                **{f'bin:{b}': Accumulator(device) for b, _, _ in DAY_BINS},
                **{f'fold:{f}': Accumulator(device) for f in folds}} for k in ks}
    for fold in folds:
        path = os.path.join(run_template.format(fold=fold), f'epoch_{epoch:03d}.pth')
        model, params, predict = load_fold_model(path, device)
        held_out = sorted(folds_df.loc[folds_df['fold'] == fold, 'chip'])
        if sorted(params['test_chips']) != held_out:
            raise click.ClickException(f"fold {fold}: the run's test chips differ")
        if (params.get('stretch') is None) != (units == 'toa'):
            raise click.ClickException(f'fold {fold}: --units {units} does not match the '
                                       "checkpoint's stretch")
        for k in ks:
            dataset = PlanetMaskRCNNDataset(
                chip_dir, split=('left', 'right'), size=params['size'],
                min_instance_size=params['min_instance_size'],
                use_ocm_masks=params['use_ocm_masks'],
                channel_kinds=params.get('channel_kinds', ('red', 'green', 'blue')),
                chip_names=held_out, stretch=params.get('stretch'),
                raw_transform=perturbs[k],
            )
            score_fold(model, dataset, chip_info, device, params['use_ocm_masks'],
                       score_thresh, iou_thresh, accs[k], predict=predict)
            p = accs[k][f'fold:{fold}'].compute()
            click.echo(f'fold {fold} k {k:+.2f}: map_50 {p["test_map/map_50"]:.4f} '
                       f'iou {p["test_iou"]:.4f}')
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    rows = []
    for k in ks:
        for key, acc in accs[k].items():
            if acc.chips:
                r = acc.compute()
                rows.append({'k': k, 'group': key, 'n_chips': r['n_chips'],
                             **{m: r[m] for m in ('test_map/map_50', 'test_iou',
                                                  'test_event/precision',
                                                  'test_event/recall')}})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    pooled = df[df.group == 'pooled'].drop(columns='group').set_index('k')
    click.echo('\npooled held-out metrics by offset multiple k:')
    click.echo(pooled.to_string(float_format=lambda v: f'{v:.4f}'))
    click.echo(f'wrote {out_csv}')


if __name__ == '__main__':
    main()
