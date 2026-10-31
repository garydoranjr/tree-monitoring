#!/usr/bin/env python
"""Temporal-CV evaluation of the PlanetScope Mask R-CNN: metrics and curves.

The companion of plot_spatial_eval.py for the first temporal cross-validation
(config/temporal_folds_20261012.csv): each fold's epoch-12 model
(202610_temporal/fold{k}/epoch_012.pth) scores its own held-out chips, both
512 x 512 windows, as eval_temporal_cv.py does (OCM masks, min instance size
64, mask threshold 0.5, IoU 0.5). Everything is pooled over the six folds.
One inference pass is cached (materials/temporal_eval_cache.npz).

temporal_eval.{png,pdf}        summary metrics, detection precision-recall and
                               crown-level ROC, as in spatial_eval
temporal_eval_by_days.{png,pdf} the same measures by the held-out chip's
                               distance in days to the nearest training chip

Crown-level items, labels and scores follow plot_spatial_eval.crown_table.

Typical usage:
    python scripts/plot_temporal_eval.py              # uses the cache if present
    python scripts/plot_temporal_eval.py --rerun      # recompute it (~40 min on CPU)
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
import eval_temporal_cv as etc  # noqa: E402  (re-exports pickled classes)
import plot_spatial_eval as pse  # noqa: E402
import train_planet_image_maskrcnn as tp  # noqa: E402

FLOWER = pse.FLOWER
DEFAULT_OUT = pse.DEFAULT_OUT
REPO = Path(__file__).resolve().parent.parent
CHIPS = FLOWER / '20261012_full_label_extended2_mkl_x4_coreg_4band_stretch_stats_curated'
FOLDS = REPO / 'config/temporal_folds_20261012.csv'
RUNS = FLOWER / '202610_temporal'
EPOCH = 12
C_MAIN = pse.C_MAIN


def run_inference(device_name, cache):
    import torch
    device = torch.device(device_name)
    torch.manual_seed(0)
    folds = pd.read_csv(FOLDS)
    folds['day_bin'] = folds.days_to_nearest_train.map(etc.day_bin)
    chip_info = folds.set_index('chip').to_dict('index')
    accs = {'pooled': etc.Accumulator(device)}
    for b, _, _ in etc.DAY_BINS:
        accs[f'bin:{b}'] = etc.Accumulator(device)
    for k in sorted(folds.fold.unique()):
        accs[f'fold:{k}'] = etc.Accumulator(device)
    chips, windows, det_scores, det_tp, det_bin = [], [], [], [], []
    n_gt_bin = {b: 0 for b, _, _ in etc.DAY_BINS}
    score_maps, gt_maps, clear_maps = [], [], []
    for k in sorted(folds.fold.unique()):
        model, params, predict = etc.load_fold_model(
            RUNS / f'fold{k}' / f'epoch_{EPOCH:03d}.pth', device)
        held = sorted(folds.chip[folds.fold == k])
        ds = tp.PlanetMaskRCNNDataset(
            str(CHIPS), split=('left', 'right'), size=params['size'],
            min_instance_size=params['min_instance_size'],
            use_ocm_masks=params['use_ocm_masks'],
            channel_kinds=params.get('channel_kinds', ('red', 'green', 'blue')),
            chip_names=held, stretch=params.get('stretch'))
        loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                             collate_fn=tp.collate_fn)
        with torch.no_grad():
            for idx, (imgs, targets) in enumerate(loader):
                file_idx, window = ds.items[idx]
                chip = os.path.basename(ds.img_files[file_idx])
                info = chip_info[chip]
                imgs = [i.to(device) for i in imgs]
                pred = predict(model, imgs, targets, device, score_thresh=0.5,
                               use_ocm_masks=True)[0]
                tgt = targets[0]
                gt = {'masks': tgt['masks'].to(device).bool(),
                      'labels': tgt['labels'].to(device), 'boxes': tgt['boxes'].to(device)}
                h, w = imgs[0].shape[-2:]
                clear = tgt['clear_mask'].to(device).bool()
                gt_union = (gt['masks'].any(dim=0) if gt['masks'].shape[0]
                            else torch.zeros((h, w), dtype=torch.bool, device=device))
                pred_union = (pred['masks'].any(dim=0) if pred['masks'].shape[0]
                              else torch.zeros((h, w), dtype=torch.bool, device=device))
                cls = tp.classify_instances_torch(gt['masks'], pred['masks'], pred['scores'],
                                                  iou_thresh=0.5)
                row = {'tp': cls['pred_labels'].count('TP'),
                       'fp': cls['pred_labels'].count('FP'),
                       'fn': cls['gt_labels'].count('FN'),
                       'inter_px': int((pred_union & gt_union & clear).sum().item()),
                       'union_px': int(((pred_union | gt_union) & clear).sum().item())}
                for key in ('pooled', f'fold:{info["fold"]}', f'bin:{info["day_bin"]}'):
                    accs[key].update(chip, pred, gt, row)
                det_scores += pred['scores'].cpu().tolist()
                det_tp += [lab == 'TP' for lab in cls['pred_labels']]
                det_bin += [info['day_bin']] * len(cls['pred_labels'])
                n_gt_bin[info['day_bin']] += int(gt['masks'].shape[0])
                score = torch.zeros((h, w), device=device)
                for m, s in zip(pred['masks'], pred['scores']):
                    score = torch.where(m & (s > score), s, score)
                chips.append(chip)
                windows.append(window)
                score_maps.append(score.cpu().numpy().astype(np.float16))
                gt_maps.append(gt_union.cpu().numpy())
                clear_maps.append(clear.cpu().numpy())
        click.echo(f'fold {k}: {len(held)} chips scored')
        del model
    metrics = {key: a.compute() for key, a in accs.items()}
    np.savez_compressed(cache, chips=np.array(chips), windows=np.array(windows),
                        det_scores=np.array(det_scores), det_tp=np.array(det_tp, bool),
                        det_bin=np.array(det_bin), n_gt_bin=json.dumps(n_gt_bin),
                        score=np.stack(score_maps), gt=np.stack(gt_maps),
                        clear=np.stack(clear_maps), metrics=json.dumps(metrics))


def det_pr(scores, is_tp, n_gt):
    order = np.argsort(-scores)
    ctp = np.cumsum(is_tp[order])
    prec = ctp / np.arange(1, len(order) + 1)
    rec = ctp / n_gt
    ap = float(np.sum(np.diff(np.concatenate([[0], rec]))
                      * np.maximum.accumulate(prec[::-1])[::-1]))
    return prec, rec, ap


@click.command()
@click.option('--outdir', type=click.Path(file_okay=False, path_type=Path),
              default=DEFAULT_OUT, show_default=True)
@click.option('--rerun', is_flag=True, help='Recompute the inference cache.')
@click.option('--device', 'device_name', default='cpu', show_default=True)
def main(outdir, rerun, device_name):
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    cache = outdir / 'materials/temporal_eval_cache.npz'
    if rerun or not cache.exists():
        run_inference(device_name, cache)
    z = np.load(cache, allow_pickle=False)
    metrics = json.loads(str(z['metrics']))
    pooled = metrics['pooled']
    n_gt_bin = json.loads(str(z['n_gt_bin']))
    scores, is_tp, det_bin = z['det_scores'], z['det_tp'], z['det_bin']
    prec, rec, ap_det = det_pr(scores, is_tp, sum(n_gt_bin.values()))
    crowns = pse.crown_table(z['chips'], z['windows'], z['score'], z['gt'], z['clear'],
                             chip_dir=CHIPS)
    folds = pd.read_csv(FOLDS).set_index('chip')
    crowns['days'] = crowns.chip.map(folds.days_to_nearest_train)
    crowns['day_bin'] = crowns.days.map(etc.day_bin)
    cy, cs = crowns.event.values, crowns.score_mean.values
    fpr, tpr, _ = roc_curve(cy, cs)
    crown_auroc, crown_ap = roc_auc_score(cy, cs), average_precision_score(cy, cs)

    # ---- main figure, as spatial_eval ----
    fig = plt.figure(figsize=(12.5, 4.6))
    ax = fig.add_axes([0.1, 0.16, 0.21, 0.72])
    names = ['mask IoU', 'precision', 'recall', 'mAP@50', 'mAP@[.5:.95]', 'mAR@100']
    vals = [pooled['test_iou'], pooled['test_event/precision'], pooled['test_event/recall'],
            pooled['test_map/map_50'], pooled['test_map/map'], pooled['test_map/mar_100']]
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=C_MAIN, height=0.6)
    for yi, v in zip(y, vals):
        ax.text(v + 0.01, yi, f'{v:.2f}', va='center', fontsize=9)
    ax.set_yticks(y, names, fontsize=9.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel('score', fontsize=10)
    ax.set_title('Summary', fontsize=11, loc='left')
    ax2 = fig.add_axes([0.39, 0.16, 0.26, 0.72])
    ax2.plot(rec, prec, color=C_MAIN, lw=2)
    ax2.plot(rec[-1], prec[-1], 'o', color=C_MAIN, mec='k', ms=7, zorder=5)
    ax2.set_xlabel('recall (crowns)', fontsize=10)
    ax2.set_ylabel('precision (detections)', fontsize=10)
    ax2.set_title('Crown detections (IoU ≥ 0.5)', fontsize=11, loc='left')
    ax2.text(0.97, 0.95, f'AP {ap_det:.2f}', transform=ax2.transAxes, ha='right', va='top',
             fontsize=10, bbox=dict(fc='white', ec='0.8', pad=3))
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
        fig.savefig(outdir / f'temporal_eval.{ext}', dpi=300)
    plt.close(fig)

    # ---- by distance to training data ----
    bins = [b for b, _, _ in etc.DAY_BINS if f'bin:{b}' in metrics
            and metrics[f'bin:{b}']['n_chips']]
    by = []
    for b in bins:
        m = metrics[f'bin:{b}']
        sel = det_bin == b
        _, r_b, ap_b = det_pr(scores[sel], is_tp[sel], n_gt_bin[b])
        cb = crowns[crowns.day_bin == b]
        by.append(dict(bin=b, chips=m['n_chips'], iou=m['test_iou'],
                       precision=m['test_event/precision'], recall=m['test_event/recall'],
                       map_50=m['test_map/map_50'], detection_ap=ap_b,
                       crown_auroc=roc_auc_score(cb.event, cb.score_mean)
                       if cb.event.nunique() == 2 else np.nan,
                       crowns=len(cb), crown_events=int(cb.event.sum())))
    by = pd.DataFrame(by)
    fig = plt.figure(figsize=(12.5, 4.6))
    ax = fig.add_axes([0.07, 0.16, 0.58, 0.72])
    show = [('mask IoU', 'iou', '#4c72b0'), ('precision', 'precision', '#dd8452'),
            ('recall', 'recall', '#55a868'), ('mAP@50', 'map_50', '#8172b3'),
            ('crown AUROC', 'crown_auroc', '0.25')]
    x = np.arange(len(by))
    width = 0.16
    for j, (lab, col, color) in enumerate(show):
        ax.bar(x + (j - 2) * width, by[col], width=width * 0.92, color=color, label=lab)
    ax.set_xticks(x, [f'{b} days\n({c} chips)' for b, c in zip(by.bin, by.chips)],
                  fontsize=9.5)
    ax.set_xlabel('distance to the nearest training chip', fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel('score', fontsize=10)
    ax.legend(loc='upper right', ncol=5, fontsize=8.5, frameon=False,
              bbox_to_anchor=(1.0, 1.1))
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', labelsize=9)
    ax4 = fig.add_axes([0.72, 0.16, 0.26, 0.72])
    cmap = plt.get_cmap('viridis')
    for j, b in enumerate(bins):
        cb = crowns[crowns.day_bin == b]
        if cb.event.nunique() == 2:
            f_, t_, _ = roc_curve(cb.event, cb.score_mean)
            ax4.plot(f_, t_, color=cmap(j / max(len(bins) - 1, 1)), lw=1.8, label=f'{b} d')
    ax4.plot([0, 1], [0, 1], color='0.5', ls=':', lw=1)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1.02)
    ax4.set_xlabel('false positive rate (crowns)', fontsize=10)
    ax4.set_ylabel('true positive rate (crowns)', fontsize=10)
    ax4.legend(loc='lower right', fontsize=8.5, frameon=False, title='days to training',
               title_fontsize=8.5)
    ax4.grid(alpha=0.3)
    ax4.spines[['top', 'right']].set_visible(False)
    ax4.tick_params(labelsize=9)
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'temporal_eval_by_days.{ext}', dpi=300)
    plt.close(fig)

    out = dict(pooled=pooled, detection_ap=ap_det, detection_max_recall=float(rec[-1]),
               n_detections=int(len(scores)), n_crowns=int(len(crowns)),
               n_crown_events=int(cy.sum()), crown_positive_rate=float(cy.mean()),
               crown_auroc=crown_auroc, crown_ap=crown_ap,
               crown_auroc_max_score=roc_auc_score(cy, crowns.score_max.values),
               crowns_unscored_frac=float((cs == 0).mean()),
               crown_events_unscored_frac=float((cs[cy] == 0).mean()),
               by_days=by.to_dict('records'),
               by_fold={k: v for k, v in metrics.items() if k.startswith('fold:')})
    (outdir / 'materials/temporal_eval.json').write_text(json.dumps(out, indent=1,
                                                                    default=float))
    crowns.to_csv(outdir / 'materials/temporal_eval_crowns.csv', index=False)
    click.echo(json.dumps({k: v for k, v in out.items() if k != 'by_fold'}, indent=1,
                          default=float))


if __name__ == '__main__':
    main()
