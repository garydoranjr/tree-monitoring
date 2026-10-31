#!/usr/bin/env python
"""Spatial (east/west) evaluation of the PlanetScope Mask R-CNN: metrics and curves.

Re-scores one checkpoint on the head-to-head benchmark: the right (east)
512 x 512 windows of the 56 base chips
(20260706_full_label_application_x4_coreg_4band_stretch_stats_curated), with
the settings of compare_maskrcnn_runs.py (--ocm-masks, --min-instance-size 64,
mask threshold 0.5, IoU 0.5). One inference pass is cached
(materials/spatial_eval_cache.npz); everything else is computed from it:

  summary        mask IoU, COCO mAP and crown precision/recall over every
                 detection the model returns (box score >= 0.05), from
                 train_planet_image_maskrcnn.evaluate(); these match the
                 cluster sweep figs/20261020_updates/headtohead_sweep_202610_mkl.json
  detection PR   detections ranked by score, each a TP if it matches an
                 unmatched ground-truth crown at mask IoU >= 0.5
  crown ROC      every crown of the drone flight's dated crown map
                 (crowns_20240803), moved by the scene's AROSICS shift as the
                 labels were, that lies in a test window (>= 90% of it inside,
                 >= 90% clear, >= 64 px). Label: >= 50% of its pixels in the
                 ground-truth event mask. Score: the mean over its pixels of the
                 highest detection score whose mask covers the pixel (0 if none).
  pixel ROC      the same pixel score against the ground-truth mask, clear
                 pixels only (in the JSON only)

Inference on Apple 'mps' is not deterministic for this model (two passes over
the same chips matched different crowns), so it runs on CPU (~15 min).

Typical usage:
    python scripts/plot_spatial_eval.py              # uses the cache if present
    python scripts/plot_spatial_eval.py --rerun      # recompute it
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_maskrcnn_runs as cmr  # noqa: E402  (re-exports pickled classes)
import train_planet_image_maskrcnn as tp  # noqa: E402

FLOWER = Path('/Volumes/Earth03/flower')
DEFAULT_OUT = FLOWER / 'figs' / '20261103_presentation'
CHIPS = FLOWER / '20260706_full_label_application_x4_coreg_4band_stretch_stats_curated'
CHECKPOINT = FLOWER / 'maskrcnn_results/20260824_maskrcnn_out_min064/epoch_012.pth'
# dated crown outlines: phantom dates, then the mavic dates (2024-03-18 outlines carried over)
DATED_CROWNS = (FLOWER / 'crowns_20240803/BCI_50ha_crownmap_timeseries.shp',
                FLOWER / 'results/crowns_mavic_dates.gpkg')
WINDOW = 512
C_MAIN = '#4c72b0'


def run_inference(checkpoint, device_name, cache):
    """Summary metrics plus, per test window, the detections and score/label/clear maps."""
    import torch
    import torchmetrics
    from torch.utils.data import DataLoader
    from torchmetrics.detection import MeanAveragePrecision
    device = torch.device(device_name)
    torch.manual_seed(0)
    model, params, _ = cmr.load_checkpoint(str(checkpoint), device)
    ds, used = cmr.build_subset(str(CHIPS), None, WINDOW, params.get('min_instance_size', 64),
                                True, params.get('channel_kinds', ('red', 'green', 'blue')),
                                stretch=params.get('stretch'))
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=tp.collate_fn)
    iou_m = torchmetrics.JaccardIndex(task='binary').to(device)
    map_m = MeanAveragePrecision(iou_type='segm').to(device)
    iou, mp, ev = tp.evaluate(model, loader, device, iou_m, map_m, score_thresh=0.5,
                              iou_thresh=0.5, use_ocm_masks=True)
    summary = dict(test_iou=iou, precision=ev['precision'], recall=ev['recall'],
                   tp=ev['tp'], fp=ev['fp'], fn=ev['fn'], map_50=float(mp['map_50']),
                   map=float(mp['map']), map_75=float(mp['map_75']),
                   mar_100=float(mp['mar_100']))
    click.echo(f'{len(used)} chips: ' + json.dumps(summary))
    det_scores, det_tp, n_gt = [], [], 0
    score_maps, gt_maps, clear_maps = [], [], []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [i.to(device) for i in imgs]
            preds = tp.filtered_predictions(model, imgs, targets, device, score_thresh=0.5,
                                            use_ocm_masks=True)
            for img, pred, tgt in zip(imgs, preds, targets):
                gt = tgt['masks'].to(device).bool()
                n_gt += gt.shape[0]
                if pred['masks'].shape[0]:
                    cls = tp.classify_instances_torch(gt, pred['masks'], pred['scores'],
                                                      iou_thresh=0.5)
                    det_scores += pred['scores'].cpu().tolist()
                    det_tp += [lab == 'TP' for lab in cls['pred_labels']]
                h, w = img.shape[-2:]
                score = torch.zeros((h, w), device=device)
                for m, s in zip(pred['masks'], pred['scores']):
                    score = torch.where(m & (s > score), s, score)
                clear = tgt['clear_mask'].to(device).bool()
                label = gt.any(dim=0) if gt.shape[0] else torch.zeros_like(clear)
                score_maps.append(score.cpu().numpy().astype(np.float16))
                gt_maps.append(label.cpu().numpy())
                clear_maps.append(clear.cpu().numpy())
    np.savez_compressed(cache, chips=np.array(used), det_scores=np.array(det_scores),
                        det_tp=np.array(det_tp, bool), n_gt=n_gt,
                        score=np.stack(score_maps), gt=np.stack(gt_maps),
                        clear=np.stack(clear_maps), summary=json.dumps(summary),
                        checkpoint=str(checkpoint))


def load_dated_crowns():
    import geopandas as gpd
    crowns = pd.concat([gpd.read_file(p) for p in DATED_CROWNS], ignore_index=True)
    crowns['tag'] = pd.to_numeric(crowns.tag, errors='coerce')
    return crowns.drop_duplicates(['tag', 'date', 'area'])


def crown_table(chips, windows, score, gt, clear, chip_dir=CHIPS, min_px=64):
    """One row per (test window, crown): event label, scores, size.

    chips/windows name each window ('left' or 'right'); score/gt/clear are its
    per-pixel best detection score, ground-truth event mask and OCM-clear mask.
    Crowns are that window's drone-flight outlines from the dated crown maps,
    moved by the scene's AROSICS shift as the labels were; a crown is kept if
    >= 90% of it lies in the window, >= 90% of that is clear, and >= min_px.
    """
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    log = pd.DataFrame(json.loads((Path(chip_dir) / 'coreg_log.json').read_text()))
    log = log[log.coreg_ok].drop_duplicates('scene').set_index('scene')
    crowns = load_dated_crowns()
    rows = []
    for k, (chip, window) in enumerate(zip(chips, windows)):
        rec = log.loc[str(chip).replace('.tif', '')]
        day = pd.Series([rec.label]).str.extract(r'(\d{4}_\d{2}_\d{2})')[0].iloc[0]
        with rasterio.open(Path(chip_dir) / str(chip)) as src:
            t, h, w = src.transform, src.height, src.width
        r0, _, c0, _ = tp._split_window(h, w, str(window), WINDOW)
        wt = t * Affine.translation(c0, r0)
        c = crowns[crowns.date == day]
        if c.empty:
            raise click.ClickException(f'no dated crown outlines for {day} ({chip})')
        # the labels were moved by the scene's AROSICS shift; move the outlines too
        geoms = c.geometry.translate(rec.x_shift_m, rec.y_shift_m).values
        sc, g_, cl = score[k].astype(np.float32), gt[k], clear[k]
        ids = rasterize(((g, i + 1) for i, g in enumerate(geoms)), out_shape=(WINDOW, WINDOW),
                        transform=wt, fill=0, dtype='int32').ravel()
        n = len(geoms) + 1
        n_in = np.bincount(ids, minlength=n)
        n_clear = np.bincount(ids, weights=cl.ravel(), minlength=n)
        n_gt = np.bincount(ids, weights=g_.ravel(), minlength=n)
        s_sum = np.bincount(ids, weights=sc.ravel(), minlength=n)
        n_pred = np.bincount(ids, weights=(sc > 0).ravel(), minlength=n)
        s_max = np.zeros(n)
        np.maximum.at(s_max, ids, sc.ravel())
        full = np.concatenate([[0], np.array([g.area for g in geoms]) / abs(t.a * t.e)])
        for i in np.flatnonzero(n_in[1:]) + 1:
            if n_in[i] < min_px or n_in[i] < 0.9 * full[i] or n_clear[i] < 0.9 * n_in[i]:
                continue
            rows.append(dict(chip=str(chip), window=str(window), date=day,
                             tag=c.tag.iloc[i - 1], px=int(n_in[i]),
                             gt_frac=n_gt[i] / n_in[i], pred_frac=n_pred[i] / n_in[i],
                             score_mean=s_sum[i] / n_in[i], score_max=s_max[i]))
    df = pd.DataFrame(rows)
    df['event'] = df.gt_frac >= 0.5
    return df


@click.command()
@click.option('--outdir', type=click.Path(file_okay=False, path_type=Path),
              default=DEFAULT_OUT, show_default=True)
@click.option('--checkpoint', type=click.Path(exists=True, path_type=Path),
              default=CHECKPOINT, show_default=True)
@click.option('--rerun', is_flag=True, help='Recompute the inference cache.')
@click.option('--size', nargs=2, type=float, default=(12.5, 4.6), show_default=True)
@click.option('--device', 'device_name', default='cpu', show_default=True)
def main(outdir, checkpoint, rerun, size, device_name):
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    cache = outdir / 'materials/spatial_eval_cache.npz'
    if rerun or not cache.exists():
        run_inference(checkpoint, device_name, cache)
    z = np.load(cache, allow_pickle=False)
    summary = json.loads(str(z['summary']))
    scores, is_tp, n_gt = z['det_scores'], z['det_tp'], int(z['n_gt'])

    # detection PR
    order = np.argsort(-scores)
    ctp = np.cumsum(is_tp[order])
    prec = ctp / np.arange(1, len(order) + 1)
    rec = ctp / n_gt
    ap_det = float(np.sum(np.diff(np.concatenate([[0], rec]))
                          * np.maximum.accumulate(prec[::-1])[::-1]))
    # pixel ROC (clear pixels; JSON only)
    clear = z['clear']
    ps, py = z['score'][clear].astype(np.float32), z['gt'][clear]
    pixel_auroc, pixel_ap = roc_auc_score(py, ps), average_precision_score(py, ps)
    # crown ROC
    crowns = crown_table(z['chips'], ['right'] * len(z['chips']), z['score'], z['gt'],
                         z['clear'])
    cy, cs = crowns.event.values, crowns.score_mean.values
    fpr, tpr, _ = roc_curve(cy, cs)
    crown_auroc, crown_ap = roc_auc_score(cy, cs), average_precision_score(cy, cs)

    fig = plt.figure(figsize=size)
    # summary metrics
    ax = fig.add_axes([0.1, 0.16, 0.21, 0.72])
    names = ['mask IoU', 'precision', 'recall', 'mAP@50', 'mAP@[.5:.95]', 'mAR@100']
    vals = [summary['test_iou'], summary['precision'], summary['recall'], summary['map_50'],
            summary['map'], summary['mar_100']]
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=C_MAIN, height=0.6)
    for yi, v in zip(y, vals):
        ax.text(v + 0.01, yi, f'{v:.2f}', va='center', fontsize=9)
    ax.set_yticks(y, names, fontsize=9.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel('score', fontsize=10)
    ax.set_title('Summary', fontsize=11, loc='left')
    # detection PR
    ax2 = fig.add_axes([0.39, 0.16, 0.26, 0.72])
    ax2.plot(rec, prec, color=C_MAIN, lw=2)
    ax2.plot(rec[-1], prec[-1], 'o', color=C_MAIN, mec='k', ms=7, zorder=5)
    ax2.set_xlabel('recall (crowns)', fontsize=10)
    ax2.set_ylabel('precision (detections)', fontsize=10)
    ax2.set_title('Crown detections (IoU ≥ 0.5)', fontsize=11, loc='left')
    ax2.text(0.97, 0.95, f'AP {ap_det:.2f}', transform=ax2.transAxes, ha='right', va='top',
             fontsize=10, bbox=dict(fc='white', ec='0.8', pad=3))
    # crown ROC
    ax3 = fig.add_axes([0.72, 0.16, 0.26, 0.72])
    ax3.plot(fpr, tpr, color=C_MAIN, lw=2)
    ax3.plot([0, 1], [0, 1], color='0.5', ls=':', lw=1)
    ax3.set_xlabel('false positive rate (crowns)', fontsize=10)
    ax3.set_ylabel('true positive rate (crowns)', fontsize=10)
    ax3.set_title('Event vs no-event crowns', fontsize=11, loc='left')
    ax3.text(0.97, 0.05, f'AUROC {crown_auroc:.2f}', transform=ax3.transAxes, ha='right',
             va='bottom', fontsize=10, bbox=dict(fc='white', ec='0.8', pad=3))
    for a in (ax, ax2, ax3):
        a.tick_params(labelsize=9)
        a.spines[['top', 'right']].set_visible(False)
    for a in (ax2, ax3):
        a.set_xlim(0, 1)
        a.set_ylim(0, 1.02)
        a.grid(alpha=0.3)
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'spatial_eval.{ext}', dpi=300)
    plt.close(fig)

    out = dict(summary, checkpoint=str(z['checkpoint']), n_chips=int(len(z['chips'])),
               n_gt=n_gt, n_detections=int(len(scores)), detection_ap=ap_det,
               detection_max_recall=float(rec[-1]), pixel_auroc=pixel_auroc,
               pixel_ap=pixel_ap, pixel_positive_rate=float(py.mean()),
               n_crowns=int(len(crowns)), n_crown_events=int(cy.sum()),
               crown_positive_rate=float(cy.mean()), crown_auroc=crown_auroc,
               crown_ap=crown_ap,
               crown_auroc_max_score=roc_auc_score(cy, crowns.score_max.values),
               crowns_unscored_frac=float((cs == 0).mean()),
               crown_events_unscored_frac=float((cs[cy] == 0).mean()))
    (outdir / 'materials/spatial_eval.json').write_text(json.dumps(out, indent=1))
    crowns.to_csv(outdir / 'materials/spatial_eval_crowns.csv', index=False)
    click.echo(f'Wrote {outdir}/spatial_eval.png/.pdf\n' + json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
