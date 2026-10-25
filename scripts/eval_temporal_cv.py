#!/usr/bin/env python
"""Score a blocked temporal cross-validation of the Planet Mask R-CNN.

Each fold's run (``train_planet_image_maskrcnn.py --folds CSV --fold K``)
is scored on its own held-out chips (both 512 x 512 windows) at every epoch
of a fixed grid. Metrics are pooled over all folds -- one IoU / mAP /
event-count accumulator across every held-out chip -- rather than averaged
per fold, since a fold holds only ~20-25 chips. The same metrics are also
broken down by fold and by the chip's distance in days to the nearest
training chip (the `days_to_nearest_train` column of the fold CSV), which is
the quantity the deployed model is extrapolated over.

The epoch with the best pooled --select-metric is the training length for
the final full-record model (``--fold all``). See
docs/planet_temporal_generalization.md.

Example:

    python scripts/eval_temporal_cv.py \\
        --chip-dir /local/gdoran_flower/20261012_full_..._curated \\
        --folds config/temporal_folds_20261012.csv \\
        --run-template 'runs/202610_temporal_fold{fold}' \\
        --epochs 1,2,3,5,8,12,17,25,40 --ocm-masks --min-instance-size 64 \\
        --json-out temporal_cv.json --chip-csv temporal_cv_chips.csv
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must precede torch import
import json

import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

# Importing compare_maskrcnn_runs also re-exports the custom model classes
# into __main__, which unpickling the checkpoints needs.
import compare_maskrcnn_runs  # noqa: F401
from train_planet_image_maskrcnn import (
    PlanetMaskRCNNDataset,
    classify_instances_torch,
    collate_fn,
    filtered_predictions,
    select_device,
)

# Distance bins in days to the nearest training chip, as (label, lo, hi)
# inclusive. Beyond 30 days is outside the interpolation envelope.
DAY_BINS = (('0-3', 0, 3), ('4-7', 4, 7), ('8-14', 8, 14),
            ('15-30', 15, 30), ('>30', 31, 10**6))


def load_fold_model(path, device):
    """(model, params, predict) for a fold checkpoint of either kind: a pickled
    Mask R-CNN from train_planet_image_maskrcnn.py, scored with
    filtered_predictions, or a DINOv3 segmenter from
    train_planet_image_dinov3.py ('kind': 'dinov3'), scored with its
    predict_instances (connected components, same output format)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get('kind') == 'dinov3':
        from train_planet_image_dinov3 import model_from_checkpoint, predict_instances
        return model_from_checkpoint(ckpt, device), ckpt['params'], predict_instances
    model = ckpt['model'].to(device)
    model.eval()
    return model, ckpt.get('params', {}), filtered_predictions


def day_bin(days):
    for label, lo, hi in DAY_BINS:
        if lo <= days <= hi:
            return label
    raise ValueError(f'negative day distance {days}')


class Accumulator:
    """Pooled binary IoU (clear pixels only), mask mAP and greedy-match
    event counts over any subset of evaluated windows."""

    def __init__(self, device):
        self.map = MeanAveragePrecision(iou_type='segm').to(device)
        self.inter = self.union = 0
        self.tp = self.fp = self.fn = 0
        self.chips = set()

    def update(self, chip, pred, gt, row):
        self.map.update([pred], [gt])
        self.inter += row['inter_px']
        self.union += row['union_px']
        self.tp += row['tp']
        self.fp += row['fp']
        self.fn += row['fn']
        self.chips.add(chip)

    def compute(self):
        out = {'n_chips': len(self.chips),
               'test_iou': self.inter / self.union if self.union else float('nan')}
        for k, v in self.map.compute().items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                out[f'test_map/{k}'] = v.item()
        tp, fp, fn = self.tp, self.fp, self.fn
        out.update({
            'test_event/precision': tp / (tp + fp) if tp + fp else float('nan'),
            'test_event/recall': tp / (tp + fn) if tp + fn else float('nan'),
            'test_event/accuracy': tp / (tp + fp + fn) if tp + fp + fn else float('nan'),
            'test_event/tp': tp, 'test_event/fp': fp, 'test_event/fn': fn,
        })
        return out


@torch.no_grad()
def score_fold(model, dataset, chip_info, device, use_ocm_masks,
               score_thresh, iou_thresh, accumulators,
               predict=filtered_predictions):
    """Score one fold's held-out windows, updating every accumulator that
    the window belongs to, and return one row per window."""
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=collate_fn)
    rows = []
    for idx, (imgs, targets) in enumerate(loader):
        file_idx, window = dataset.items[idx]
        chip = os.path.basename(dataset.img_files[file_idx])
        info = chip_info[chip]
        imgs = [img.to(device) for img in imgs]
        pred = predict(
            model, imgs, targets, device, score_thresh=score_thresh,
            use_ocm_masks=use_ocm_masks,
        )[0]
        tgt = targets[0]
        gt = {'masks': tgt['masks'].to(device).bool(),
              'labels': tgt['labels'].to(device),
              'boxes': tgt['boxes'].to(device)}

        h, w = imgs[0].shape[-2:]
        pred_union = (pred['masks'].any(dim=0) if pred['masks'].shape[0]
                      else torch.zeros((h, w), dtype=torch.bool, device=device))
        gt_union = (gt['masks'].any(dim=0) if gt['masks'].shape[0]
                    else torch.zeros((h, w), dtype=torch.bool, device=device))
        if use_ocm_masks:
            clear = tgt['clear_mask'].to(device).bool()
            pred_union &= clear
            gt_union &= clear
        cls = classify_instances_torch(gt['masks'], pred['masks'],
                                       pred['scores'], iou_thresh=iou_thresh)
        row = {
            'chip': chip, 'window': window, 'fold': info['fold'],
            'planet_date': info['planet_date'],
            'days_to_nearest_train': info['days_to_nearest_train'],
            'day_bin': info['day_bin'],
            'n_gt': int(gt['masks'].shape[0]),
            'n_pred': int(pred['masks'].shape[0]),
            'tp': cls['pred_labels'].count('TP'),
            'fp': cls['pred_labels'].count('FP'),
            'fn': cls['gt_labels'].count('FN'),
            'inter_px': int((pred_union & gt_union).sum().item()),
            'union_px': int((pred_union | gt_union).sum().item()),
        }
        for key in ('pooled', f'fold:{info["fold"]}', f'bin:{info["day_bin"]}'):
            accumulators[key].update(chip, pred, gt, row)
        rows.append(row)
    return rows


@click.command()
@click.option('--chip-dir', required=True,
              type=click.Path(exists=True, file_okay=False))
@click.option('--folds', 'folds_csv', required=True,
              type=click.Path(exists=True, dir_okay=False),
              help='Fold CSV from make_temporal_folds.py, the same one the '
                   'runs were trained with.')
@click.option('--run-template', required=True,
              help="Run directory per fold, with '{fold}' standing for the "
                   "fold number, e.g. 'runs/temporal_fold{fold}'.")
@click.option('--epochs', required=True,
              help='Comma-separated epochs to score; each fold run must have '
                   'epoch_NNN.pth for every one.')
@click.option('--min-instance-size', default=None, type=int,
              help="Defaults to the checkpoint's own setting.")
@click.option('--ocm-masks/--no-ocm-masks', 'use_ocm_masks', default=None,
              help="Defaults to the checkpoint's own setting.")
@click.option('--score-thresh', default=0.5, type=float,
              help='Mask binarization threshold, as in evaluate().')
@click.option('--iou-thresh', default=0.5, type=float)
@click.option('--select-metric', default='test_map/map_50',
              help='Pooled metric (maximized) that picks the epoch.')
@click.option('--json-out', required=True, type=click.Path(dir_okay=False))
@click.option('--chip-csv', default=None, type=click.Path(dir_okay=False),
              help='Per-window rows (every epoch) for later breakdowns.')
def main(chip_dir, folds_csv, run_template, epochs, min_instance_size,
         use_ocm_masks, score_thresh, iou_thresh, select_metric, json_out,
         chip_csv):
    epochs = [int(e) for e in epochs.split(',') if e.strip()]
    folds_df = pd.read_csv(folds_csv)
    folds_df['day_bin'] = folds_df['days_to_nearest_train'].apply(day_bin)
    chip_info = folds_df.set_index('chip').to_dict('index')
    folds = sorted(folds_df['fold'].unique())

    paths = {
        (k, e): os.path.join(run_template.format(fold=k), f'epoch_{e:03d}.pth')
        for k in folds for e in epochs
    }
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise click.ClickException(
            f'{len(missing)} checkpoint(s) missing, e.g. {missing[:3]}'
        )

    device = select_device()
    per_epoch = {}
    all_rows = []
    settings = None
    for e in epochs:
        accs = {'pooled': Accumulator(device)}
        accs.update({f'fold:{k}': Accumulator(device) for k in folds})
        accs.update({f'bin:{b}': Accumulator(device) for b, _, _ in DAY_BINS})
        for k in folds:
            model, params, predict = load_fold_model(paths[(k, e)], device)
            if params.get('folds') is None or str(params.get('fold')) != str(k):
                raise click.ClickException(
                    f'{paths[(k, e)]} was not trained as fold {k} '
                    f"(params fold={params.get('fold')!r})"
                )
            held_out = sorted(folds_df.loc[folds_df['fold'] == k, 'chip'])
            if sorted(params['test_chips']) != held_out:
                raise click.ClickException(
                    f"fold {k}: the run's test chips differ from {folds_csv}"
                )
            run_settings = {
                'min_instance_size': (min_instance_size
                                      if min_instance_size is not None
                                      else params['min_instance_size']),
                'use_ocm_masks': (use_ocm_masks if use_ocm_masks is not None
                                  else params['use_ocm_masks']),
                'size': params['size'],
            }
            if settings is None:
                settings = run_settings
            elif run_settings != settings:
                raise click.ClickException(
                    f'fold {k} epoch {e} settings {run_settings} differ from '
                    f'earlier runs {settings}'
                )
            dataset = PlanetMaskRCNNDataset(
                chip_dir, split=('left', 'right'), size=settings['size'],
                min_instance_size=settings['min_instance_size'],
                use_ocm_masks=settings['use_ocm_masks'],
                channel_kinds=params.get('channel_kinds',
                                         ('red', 'green', 'blue')),
                chip_names=held_out, stretch=params.get('stretch'),
            )
            rows = score_fold(model, dataset, chip_info, device,
                              settings['use_ocm_masks'], score_thresh,
                              iou_thresh, accs, predict=predict)
            all_rows.extend({'epoch': e, **r} for r in rows)
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        per_epoch[e] = {
            'pooled': accs['pooled'].compute(),
            'by_fold': {str(k): accs[f'fold:{k}'].compute() for k in folds},
            'by_day_bin': {b: accs[f'bin:{b}'].compute()
                           for b, _, _ in DAY_BINS
                           if accs[f'bin:{b}'].chips},
        }
        p = per_epoch[e]['pooled']
        print(f"epoch {e:3d}: map_50 {p['test_map/map_50']:.4f}  "
              f"map {p['test_map/map']:.4f}  iou {p['test_iou']:.4f}  "
              f"P/R {p['test_event/precision']:.3f}/"
              f"{p['test_event/recall']:.3f}")

    if select_metric not in per_epoch[epochs[0]]['pooled']:
        raise click.ClickException(
            f'--select-metric {select_metric!r} is not a pooled metric; '
            f"available: {sorted(per_epoch[epochs[0]]['pooled'])}"
        )
    best_epoch = max(epochs,
                     key=lambda e: per_epoch[e]['pooled'][select_metric])
    print(f'\nBest pooled {select_metric}: '
          f"{per_epoch[best_epoch]['pooled'][select_metric]:.4f} "
          f'at epoch {best_epoch}\n')
    print(f'By days to nearest training chip, epoch {best_epoch}:')
    print(pd.DataFrame(per_epoch[best_epoch]['by_day_bin']).T[[
        'n_chips', 'test_map/map_50', 'test_iou', 'test_event/precision',
        'test_event/recall',
    ]].to_string(float_format=lambda v: f'{v:.4f}'))
    print(f'\nBy fold, epoch {best_epoch}:')
    print(pd.DataFrame(per_epoch[best_epoch]['by_fold']).T[[
        'n_chips', 'test_map/map_50', 'test_iou', 'test_event/precision',
        'test_event/recall',
    ]].to_string(float_format=lambda v: f'{v:.4f}'))

    with open(json_out, 'w') as fh:
        json.dump({
            'chip_dir': chip_dir, 'folds': folds_csv,
            'run_template': run_template, 'epochs': epochs,
            'score_thresh': score_thresh, 'iou_thresh': iou_thresh,
            **settings,
            'day_bins': [list(b) for b in DAY_BINS],
            'select_metric': select_metric, 'best_epoch': best_epoch,
            'per_epoch': {str(e): v for e, v in per_epoch.items()},
        }, fh, indent=2)
    print(f'\nWrote {json_out}')
    if chip_csv:
        pd.DataFrame(all_rows).to_csv(chip_csv, index=False)
        print(f'Wrote {chip_csv}')


if __name__ == '__main__':
    main()
