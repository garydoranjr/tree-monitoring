#!/usr/bin/env python
"""Precision/recall as a function of a post-hoc crown-size threshold.

Consumes the caches written by `maskrcnn_size_sweep_stats.py` -- one per
Mask R-CNN model, where the models differ only in the `--min-instance-size`
filter applied to the ground truth during training -- and sweeps a *post-hoc*
minimum-area threshold S applied at metric time.

Because each cache stores the full instance-mask IoU matrix alongside the
predicted and ground-truth areas, TP/FP/FN at any S follow from a submatrix
selection plus a re-run of the same greedy matcher used during training. No
inference is repeated here.

Two views are produced:

* cumulative -- keep instances with ``area > S``, so each model's curve begins
  at its own training threshold (S can never drop below it);
* binned -- keep instances with ``lo < area <= hi`` over octave-wide bands,
  which isolates how the model does on each crown size rather than letting
  abundant small crowns dominate a running total.
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import csv

import click
import numpy as np
import matplotlib.pyplot as plt

from train_planet_image_maskrcnn import _greedy_match

# Chips are 0.75 m/px (EPSG:32617), so one pixel of crown area is 0.5625 m^2.
M2_PER_PIXEL = 0.75 * 0.75

# Octave bands for the binned view; the last is open-ended.
BIN_EDGES = [16, 32, 64, 128, 256, 512, np.inf]


def load_cache(path):
    """Read one stats npz into per-chip arrays plus scalar metadata."""
    z = np.load(path)
    n_pred = z['n_pred']
    n_gt = z['n_gt']
    off = z['iou_offset']
    p_ends = np.cumsum(n_pred)
    p_starts = p_ends - n_pred
    g_ends = np.cumsum(n_gt)
    g_starts = g_ends - n_gt

    chips = []
    for c in range(len(n_pred)):
        chips.append({
            'pred_area': z['pred_area'][p_starts[c]:p_ends[c]],
            'pred_score': z['pred_score'][p_starts[c]:p_ends[c]],
            'gt_area': z['gt_area'][g_starts[c]:g_ends[c]],
            'iou': z['iou_flat'][off[c]:off[c + 1]].reshape(
                int(n_pred[c]), int(n_gt[c])
            ),
        })
    return {
        'chips': chips,
        'min_instance_size': int(z['min_instance_size']),
        'epoch': int(z['epoch']),
        'iou_thresh': float(z['iou_thresh']),
        'baseline': (
            int(z['baseline_tp']), int(z['baseline_fp']), int(z['baseline_fn'])
        ),
        'gt_area_all': z['gt_area'],
        'path': path,
    }


def tally(cache, lo, hi, iou_thresh):
    """Pool TP/FP/FN over chips, keeping only instances with lo < area <= hi.

    Both predictions and ground truth are filtered, so the counts describe a
    world in which objects outside the size band simply do not exist. Deleting
    rows/columns preserves score order, so greedy matching on the submatrix is
    well defined."""
    tp = fp = fn = 0
    n_pred_kept = n_gt_kept = 0
    for ch in cache['chips']:
        pi = np.flatnonzero((ch['pred_area'] > lo) & (ch['pred_area'] <= hi))
        gi = np.flatnonzero((ch['gt_area'] > lo) & (ch['gt_area'] <= hi))
        n_pred_kept += pi.size
        n_gt_kept += gi.size
        cls = _greedy_match(
            ch['iou'][np.ix_(pi, gi)], ch['pred_score'][pi],
            iou_thresh=iou_thresh,
        )
        tp += cls['pred_labels'].count('TP')
        fp += cls['pred_labels'].count('FP')
        fn += cls['gt_labels'].count('FN')

    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall,
        'n_pred_kept': n_pred_kept, 'n_gt_kept': n_gt_kept,
    }


def sweep(cache, thresholds, iou_thresh):
    """Cumulative sweep: metrics vs a lower bound on instance area."""
    rows = []
    for s in thresholds:
        if s < cache['min_instance_size']:
            continue  # a post-hoc filter looser than training is meaningless
        r = tally(cache, s, np.inf, iou_thresh)
        r['threshold'] = int(s)
        rows.append(r)
    return rows


def binned(cache, iou_thresh):
    """Size-stratified metrics: match once, then split the outcomes by area.

    Matching is done with only the training threshold as a floor, then each
    ground-truth crown is attributed to a band by *its own* area and each
    prediction by *its own* area. Recall in a band is therefore the fraction of
    crowns that size which were found by any prediction, and precision the
    fraction of predictions that size which hit something.

    Restricting predictions and ground truth to the *same* band instead would
    discard every match that straddles a band edge -- 31% of all true positives
    for these models -- and would understate recall accordingly."""
    floor = cache['min_instance_size']
    edges = np.array(BIN_EDGES, dtype=float)
    n_band = len(edges) - 1
    gt_tot = np.zeros(n_band, int)
    gt_hit = np.zeros(n_band, int)
    pr_tot = np.zeros(n_band, int)
    pr_hit = np.zeros(n_band, int)

    for ch in cache['chips']:
        pi = np.flatnonzero(ch['pred_area'] > floor)
        gi = np.flatnonzero(ch['gt_area'] > floor)
        cls = _greedy_match(
            ch['iou'][np.ix_(pi, gi)], ch['pred_score'][pi],
            iou_thresh=iou_thresh,
        )
        # np.digitize with the octave edges maps an area to its band index.
        p_band = np.digitize(ch['pred_area'][pi], edges) - 1
        g_band = np.digitize(ch['gt_area'][gi], edges) - 1
        for k, lab in enumerate(cls['pred_labels']):
            b = p_band[k]
            if 0 <= b < n_band:
                pr_tot[b] += 1
                pr_hit[b] += lab == 'TP'
        for k, lab in enumerate(cls['gt_labels']):
            b = g_band[k]
            if 0 <= b < n_band:
                gt_tot[b] += 1
                gt_hit[b] += lab == 'TP'

    rows = []
    for b in range(n_band):
        if edges[b] < floor:
            continue  # band partly removed by the training filter
        rows.append({
            'lo': edges[b], 'hi': edges[b + 1],
            'tp': int(gt_hit[b]), 'fp': int(pr_tot[b] - pr_hit[b]),
            'fn': int(gt_tot[b] - gt_hit[b]),
            'recall': gt_hit[b] / gt_tot[b] if gt_tot[b] else np.nan,
            'precision': pr_hit[b] / pr_tot[b] if pr_tot[b] else np.nan,
            'n_gt_kept': int(gt_tot[b]), 'n_pred_kept': int(pr_tot[b]),
        })
    return rows


def check(cache, rows):
    """Compare S == the training threshold against the unfiltered baseline.

    At that S the ground-truth filter is a no-op (nothing survived training at
    or below it), so TP+FN must equal the ground-truth count exactly. TP alone
    need *not* match: the symmetric filter also drops predictions of area <= S,
    and an undersized prediction can still have matched a crown of area in
    (S, 2S] -- IoU >= 0.5 forces inter >= gt_area/2 and inter <= pred_area, so
    gt_area <= 2*pred_area. Those matches become FN, which is intended."""
    t = cache['min_instance_size']
    b_tp, b_fp, b_fn = cache['baseline']
    at_t = next(r for r in rows if r['threshold'] == t)
    assert at_t['tp'] + at_t['fn'] == b_tp + b_fn, (
        f"ground-truth count changed at S={t}: "
        f"{at_t['tp'] + at_t['fn']} vs {b_tp + b_fn}"
    )
    print(
        f"  min{t:<4d} n_gt={b_tp + b_fn:<5d} baseline tp/fp/fn="
        f"{b_tp}/{b_fp}/{b_fn}  ->  at S={t}: "
        f"{at_t['tp']}/{at_t['fp']}/{at_t['fn']}"
        f"   ({-(at_t['tp'] - b_tp)} TPs lost to undersized predictions, "
        f"{-(at_t['fp'] - b_fp)} FPs dropped)"
    )


PX_TICKS = [16, 32, 64, 128, 256, 512, 1024]


def _style_axis(ax, xlabel):
    """Log2 pixel axis with a matching physical-area axis along the top."""
    ax.set_xscale('log', base=2)
    ax.set_xlim(14, 1200)
    ax.set_xticks(PX_TICKS)
    ax.set_xticklabels([str(t) for t in PX_TICKS])
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.set_xlabel(xlabel)
    # The secondary axis inherits the parent's log scale, so its ticks must be
    # placed and formatted explicitly or matplotlib falls back to decade ticks
    # and labels the whole span "10^1".
    sec = ax.secondary_xaxis(
        'top', functions=(lambda s: s * M2_PER_PIXEL,
                          lambda a: a / M2_PER_PIXEL),
    )
    sec.set_xscale('log', base=2)
    sec.set_xticks([t * M2_PER_PIXEL for t in PX_TICKS])
    sec.set_xticklabels([f'{t * M2_PER_PIXEL:g}' for t in PX_TICKS],
                        fontsize=8)
    sec.set_xlabel('crown area (m$^2$)', fontsize=9)


def plot(caches, sweeps, bins_, outputfile):
    fig, axs = plt.subplots(nrows=3, figsize=(9, 13.5))
    cmap = plt.get_cmap('viridis', len(caches))

    for i, (cache, rows, brows) in enumerate(zip(caches, sweeps, bins_)):
        color = cmap(i)
        label = f"train filter {cache['min_instance_size']} px"
        s = np.array([r['threshold'] for r in rows])

        for ax, key in ((axs[0], 'precision'), (axs[1], 'recall')):
            y = np.array([r[key] for r in rows], dtype=float)
            ax.plot(s, y, color=color, lw=1.6, label=label)
            # Mark where each curve necessarily begins: its training threshold.
            ax.plot(s[:1], y[:1], 'o', color=color, ms=7, zorder=5)

        # Open-ended last band has no geometric center; place it an octave in.
        centers = np.array([
            r['lo'] * 2 ** 0.5 if np.isinf(r['hi'])
            else np.sqrt(r['lo'] * r['hi'])
            for r in brows
        ])
        axs[2].plot(centers, [r['recall'] for r in brows], '-o', color=color,
                    lw=1.6, ms=5, label=label)
        axs[2].plot(centers, [r['precision'] for r in brows], '--s',
                    color=color, lw=1.2, ms=4, alpha=0.75)

    _style_axis(axs[0], 'post-hoc minimum crown area (px)')
    _style_axis(axs[1], 'post-hoc minimum crown area (px)')
    _style_axis(axs[2], 'crown area band, geometric center (px)')

    axs[0].set_ylabel('precision')
    axs[1].set_ylabel('recall')
    axs[2].set_ylabel('precision / recall')
    axs[0].set_title('Precision vs post-hoc minimum crown area', fontsize=10)
    axs[1].set_title('Recall vs post-hoc minimum crown area', fontsize=10)
    axs[2].set_title(
        'Size-stratified: recall by ground-truth area (solid circles),\n'
        'precision by predicted area (dashed squares)', fontsize=10,
    )
    for ax in axs:
        ax.legend(fontsize=7, loc='upper left', framealpha=0.9)

    fig.tight_layout()
    fig.savefig(outputfile, dpi=150, bbox_inches='tight')
    print(f"Wrote {outputfile}")


@click.command()
@click.argument('statfiles', nargs=-1, required=True)
@click.argument('outputfile')
@click.option('--iou-thresh', default=None, type=float,
              help='IoU for a true positive; defaults to the cached value.')
@click.option('--csv', 'csvfile', default=None,
              help='Also write the swept counts and metrics as tidy CSV.')
def main(statfiles, outputfile, iou_thresh, csvfile):
    caches = sorted(
        (load_cache(f) for f in statfiles),
        key=lambda c: c['min_instance_size'],
    )
    thresholds = np.unique(
        np.round(2 ** np.arange(4, 10.01, 0.25)).astype(int)
    )

    print("Consistency check at S == training threshold:")
    sweeps, bins_ = [], []
    for c in caches:
        it = iou_thresh if iou_thresh is not None else c['iou_thresh']
        rows = sweep(c, thresholds, it)
        check(c, rows)
        sweeps.append(rows)
        bins_.append(binned(c, it))

    # Ground truth above any given area is the same set regardless of which
    # (lower) training filter produced the cache, since raising the area cut
    # only ever removes instances. Verifying it confirms that comparing models
    # at a fixed S is apples-to-apples.
    ref = np.sort(caches[-1]['gt_area_all'])
    cut = caches[-1]['min_instance_size']
    for c in caches[:-1]:
        other = np.sort(c['gt_area_all'][c['gt_area_all'] > cut])
        status = 'ok' if np.array_equal(other, ref) else 'MISMATCH'
        print(
            f"  GT areas > {cut} px: min{c['min_instance_size']} vs "
            f"min{cut} -> {status} ({other.size} vs {ref.size})"
        )

    plot(caches, sweeps, bins_, outputfile)

    if csvfile:
        with open(csvfile, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['view', 'min_instance_size', 'epoch', 'lo', 'hi',
                        'tp', 'fp', 'fn', 'precision', 'recall',
                        'n_pred_kept', 'n_gt_kept'])
            for c, rows, brows in zip(caches, sweeps, bins_):
                for r in rows:
                    w.writerow(['cumulative', c['min_instance_size'],
                                c['epoch'], r['threshold'], '',
                                r['tp'], r['fp'], r['fn'],
                                f"{r['precision']:.6f}", f"{r['recall']:.6f}",
                                r['n_pred_kept'], r['n_gt_kept']])
                for r in brows:
                    w.writerow(['binned', c['min_instance_size'], c['epoch'],
                                r['lo'], r['hi'], r['tp'], r['fp'], r['fn'],
                                f"{r['precision']:.6f}", f"{r['recall']:.6f}",
                                r['n_pred_kept'], r['n_gt_kept']])
        print(f"Wrote {csvfile}")


if __name__ == '__main__':
    main()
