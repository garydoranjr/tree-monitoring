#!/usr/bin/env python
"""Spatial (east/west) evaluation of the PlanetScope Mask R-CNN: metrics, PR and ROC curves.

Re-scores one checkpoint on the head-to-head benchmark: the right (east)
512 x 512 windows of the 56 base chips
(20260706_full_label_application_x4_coreg_4band_stretch_stats_curated), with
the settings of compare_maskrcnn_runs.py (--ocm-masks, --min-instance-size 64,
mask threshold 0.5, IoU 0.5). It keeps every detection and its score so that:

  instance PR    detections sorted by score, each a TP if it matches an unmatched
                 ground-truth crown at mask IoU >= 0.5
                 (classify_instances_torch), recall over all ground-truth crowns
  pixel ROC/PR   per-pixel score = the highest detection score whose binarized
                 mask covers the pixel (0 elsewhere), against the ground-truth
                 crown union, on clear (OCM) pixels only

The summary metrics (mask IoU, mAP, and precision/recall over every detection
the model returns, i.e. above its 0.05 box score threshold) come from train_planet_image_maskrcnn.evaluate() and match the
sweep in figs/20261020_updates/headtohead_sweep_202610_mkl.json.

Typical usage:
    python scripts/plot_spatial_eval.py
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import json
import sys
from pathlib import Path

import click
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_maskrcnn_runs as cmr  # noqa: E402  (re-exports pickled classes)
import train_planet_image_maskrcnn as tp  # noqa: E402

FLOWER = Path('/Volumes/Earth03/flower')
DEFAULT_OUT = FLOWER / 'figs' / '20261103_presentation'
CHIPS = FLOWER / '20260706_full_label_application_x4_coreg_4band_stretch_stats_curated'
CHECKPOINT = FLOWER / 'maskrcnn_results/20260824_maskrcnn_out_min064/epoch_012.pth'
C_MAIN = '#4c72b0'


def collect(model, loader, device, mask_thresh=0.5, iou_thresh=0.5):
    """Per-detection (score, is_tp), number of GT crowns, and pixel (score, label) samples."""
    import torch
    det_scores, det_tp, n_gt = [], [], 0
    pix_s, pix_y = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [i.to(device) for i in imgs]
            preds = tp.filtered_predictions(model, imgs, targets, device,
                                            score_thresh=mask_thresh, use_ocm_masks=True)
            for img, pred, tgt in zip(imgs, preds, targets):
                gt = tgt['masks'].to(device).bool()
                n_gt += gt.shape[0]
                if pred['masks'].shape[0]:
                    cls = tp.classify_instances_torch(gt, pred['masks'], pred['scores'],
                                                      iou_thresh=iou_thresh)
                    det_scores += pred['scores'].cpu().tolist()
                    det_tp += [lab == 'TP' for lab in cls['pred_labels']]
                h, w = img.shape[-2:]
                score = torch.zeros((h, w), device=device)
                for m, s in zip(pred['masks'], pred['scores']):
                    score = torch.where(m & (s > score), s, score)
                clear = tgt['clear_mask'].to(device).bool()
                label = gt.any(dim=0) if gt.shape[0] else torch.zeros_like(clear)
                pix_s.append(score[clear].cpu().numpy())
                pix_y.append(label[clear].cpu().numpy())
    return (np.array(det_scores), np.array(det_tp, bool), n_gt,
            np.concatenate(pix_s), np.concatenate(pix_y))


@click.command()
@click.option('--outdir', type=click.Path(file_okay=False, path_type=Path),
              default=DEFAULT_OUT, show_default=True)
@click.option('--checkpoint', type=click.Path(exists=True, path_type=Path),
              default=CHECKPOINT, show_default=True)
@click.option('--size', nargs=2, type=float, default=(12.5, 4.6), show_default=True)
@click.option('--device', 'device_name', default='cpu', show_default=True,
              help="Torch device. Inference on Apple 'mps' is not deterministic for this "
                   'model (two passes over the same chips matched different crowns), so the '
                   'default is cpu.')
def main(outdir, checkpoint, size, device_name):
    import torch
    import torchmetrics
    from torch.utils.data import DataLoader
    from torchmetrics.detection import MeanAveragePrecision
    from sklearn.metrics import (auc, average_precision_score, precision_recall_curve,
                                 roc_auc_score, roc_curve)
    device = torch.device(device_name)
    torch.manual_seed(0)
    model, params, _ = cmr.load_checkpoint(str(checkpoint), device)
    ds, used = cmr.build_subset(str(CHIPS), None, 512, params.get('min_instance_size', 64),
                                True, params.get('channel_kinds', ('red', 'green', 'blue')),
                                stretch=params.get('stretch'))
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=tp.collate_fn)
    iou_m = torchmetrics.JaccardIndex(task='binary').to(device)
    map_m = MeanAveragePrecision(iou_type='segm').to(device)
    iou, mp, ev = tp.evaluate(model, loader, device, iou_m, map_m, score_thresh=0.5,
                              iou_thresh=0.5, use_ocm_masks=True)
    click.echo(f'{len(used)} chips: IoU {iou:.3f}  mAP@50 {mp["map_50"]:.3f}  '
               f'P {ev["precision"]:.3f}  R {ev["recall"]:.3f}')
    scores, is_tp, n_gt, ps, py = collect(model, loader, device)

    # instance PR from the detection list
    order = np.argsort(-scores)
    ctp = np.cumsum(is_tp[order])
    prec = ctp / np.arange(1, len(order) + 1)
    rec = ctp / n_gt
    ap_inst = float(np.sum(np.diff(np.concatenate([[0], rec])) *
                           np.maximum.accumulate(prec[::-1])[::-1]))
    # the summary precision/recall count every detection the model returns
    # (its box score threshold, 0.05): the end of the PR curve
    op = (rec[-1], prec[-1])
    # pixel ROC / PR
    fpr, tpr, _ = roc_curve(py, ps)
    pauc = roc_auc_score(py, ps)
    pp, pr, _ = precision_recall_curve(py, ps)
    pap = average_precision_score(py, ps)

    fig = plt.figure(figsize=size)
    W, H = size
    # (a) summary metrics at the 0.5 operating point
    ax = fig.add_axes([0.1, 0.16, 0.21, 0.72])
    names = ['mask IoU', 'precision', 'recall', 'mAP@50', 'mAP@[.5:.95]', 'mAR@100']
    vals = [iou, ev['precision'], ev['recall'], float(mp['map_50']), float(mp['map']),
            float(mp['mar_100'])]
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=C_MAIN, height=0.6)
    for yi, v in zip(y, vals):
        ax.text(v + 0.01, yi, f'{v:.2f}', va='center', fontsize=9)
    ax.set_yticks(y, names, fontsize=9.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel('score', fontsize=10)
    ax.tick_params(axis='x', labelsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('Summary', fontsize=11, loc='left')

    # (b) instance-level PR
    ax = fig.add_axes([0.39, 0.16, 0.26, 0.72])
    ax.plot(rec, prec, color=C_MAIN, lw=2)
    ax.plot(op[0], op[1], 'o', color=C_MAIN, mec='k', ms=7, zorder=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('recall (crowns)', fontsize=10)
    ax.set_ylabel('precision (detections)', fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.set_title('Crown detections (IoU ≥ 0.5)', fontsize=11, loc='left')
    ax.text(0.97, 0.95, f'AP {ap_inst:.2f}', transform=ax.transAxes, ha='right', va='top',
            fontsize=10, bbox=dict(fc='white', ec='0.8', pad=3))

    # (c) pixel ROC with PR inset
    ax = fig.add_axes([0.72, 0.16, 0.26, 0.72])
    ax.plot(fpr, tpr, color=C_MAIN, lw=2)
    ax.plot([0, 1], [0, 1], color='0.5', ls=':', lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('false positive rate (pixels)', fontsize=10)
    ax.set_ylabel('true positive rate (pixels)', fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.set_title('Crown pixels', fontsize=11, loc='left')
    ax.text(0.97, 0.05, f'AUROC {pauc:.2f}\npixel AP {pap:.2f}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10, bbox=dict(fc='white', ec='0.8', pad=3))
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'spatial_eval.{ext}', dpi=300)
    plt.close(fig)

    out = dict(checkpoint=str(checkpoint), n_chips=len(used), n_gt=int(n_gt),
               n_detections=int(len(scores)), test_iou=iou, precision=ev['precision'],
               recall=ev['recall'], tp=ev['tp'], fp=ev['fp'], fn=ev['fn'],
               map_50=float(mp['map_50']), map=float(mp['map']), map_75=float(mp['map_75']),
               mar_100=float(mp['mar_100']), instance_ap=ap_inst,
               max_recall=float(rec[-1]) if len(rec) else 0.0,
               pixel_auroc=pauc, pixel_ap=pap, pixel_positive_rate=float(py.mean()),
               n_pixels=int(len(py)))
    (outdir / 'materials/spatial_eval.json').write_text(json.dumps(out, indent=1))
    click.echo(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
