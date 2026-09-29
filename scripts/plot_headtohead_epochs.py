#!/usr/bin/env python
"""Head-to-head Mask R-CNN training-set comparison over training epochs.

Consumes the JSON written by `compare_maskrcnn_runs.py --json-out`, where every
checkpoint in the sweep was scored on one pinned chip set, so curves from runs
that used different `imagedir`s are directly comparable. Run labels are
expected as ``<run>_e<epoch>`` (e.g. ``base_e012``).

Three views are produced:

* mask mAP@50 and binary IoU vs epoch, with each run's peak marked -- the
  quantity that motivated the comparison, and the epoch a best-checkpoint
  selector would pick;
* precision and recall vs epoch on shared axes, which shows that the late-epoch
  collapse is recall being traded away for precision rather than uniform decay;
* the precision/recall trajectory itself, where each run traces an arc and
  training time becomes a path parameter rather than an axis.
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import json
import re
from collections import defaultdict

import click
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

LABEL_RE = re.compile(r'^(?P<run>.+)_e(?P<epoch>\d+)$')

# Epochs to annotate on the P/R trajectory. Early epochs crowd together at high
# recall, so only a readable subset gets a callout.
ANNOT_EPOCHS = (1, 3, 5, 12, 25, 60, 200)


def load(path):
    """Group a sweep JSON into {run: sorted-by-epoch list of rows}."""
    with open(path) as fh:
        blob = json.load(fh)

    runs = defaultdict(list)
    for label, row in blob['results'].items():
        m = LABEL_RE.match(label)
        if m is None:
            raise click.ClickException(
                f'run label {label!r} is not <run>_e<epoch>; cannot place it '
                'on an epoch axis'
            )
        runs[m['run']].append({**row, 'epoch_n': int(m['epoch'])})

    for rows in runs.values():
        rows.sort(key=lambda r: r['epoch_n'])
    return blob, dict(runs)


def series(rows, key):
    return (np.array([r['epoch_n'] for r in rows], dtype=float),
            np.array([r[key] for r in rows], dtype=float))


def plot(blob, runs, display, outputfile):
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(13, 9.5))
    ax_map, ax_iou, ax_pr, ax_traj = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
    colors = {'base': '#1f77b4', 'ext': '#d62728'}
    traj_scatter = []

    # Stagger the peak callouts per run so they don't land on top of each other
    # or on the curves, which peak at nearby epochs.
    annot_offsets = {'base': (8, 10), 'ext': (8, -26)}

    for run, rows in sorted(runs.items()):
        color = colors.get(run, None)
        label = display.get(run, run)

        # Top row: the selection metrics, each with its peak marked, since the
        # whole point is that the reported final epoch is far from the peak.
        for ax, key in ((ax_map, 'test_map/map_50'), (ax_iou, 'test_iou')):
            x, y = series(rows, key)
            ax.plot(x, y, '-o', color=color, lw=1.6, ms=4, label=label)
            i = int(np.nanargmax(y))
            ax.plot(x[i], y[i], '*', color=color, ms=16, zorder=5,
                    markeredgecolor='white', markeredgewidth=0.6)
            ax.annotate(f'peak {y[i]:.3f} @ep{int(x[i])}',
                        xy=(x[i], y[i]),
                        xytext=annot_offsets.get(run, (6, -22)),
                        textcoords='offset points', fontsize=7.5, color=color,
                        fontweight='bold')

        # Bottom left: precision and recall share an axis (both in [0,1]);
        # linestyle separates the metrics, color the runs.
        x, p = series(rows, 'test_event/precision')
        _, r = series(rows, 'test_event/recall')
        ax_pr.plot(x, p, '-o', color=color, lw=1.6, ms=4,
                   label=f'{label} precision')
        ax_pr.plot(x, r, '--s', color=color, lw=1.3, ms=4, alpha=0.8,
                   label=f'{label} recall')

        # Bottom right: the same data as a trajectory, which makes the
        # precision-for-recall trade legible as a single arc per run. Epoch is
        # encoded in the marker fill on a shared log norm so both runs decode
        # against the one colorbar.
        ax_traj.plot(r, p, '-', color=color, lw=1.4, alpha=0.7, label=label)
        sc = ax_traj.scatter(r, p, c=x, cmap='viridis', s=46, zorder=4,
                             edgecolors=color, linewidths=1.1,
                             norm=LogNorm(vmin=1, vmax=200))
        for xi, ri, pi in zip(x, r, p):
            if int(xi) in ANNOT_EPOCHS:
                ax_traj.annotate(f'ep{int(xi)}', xy=(ri, pi), xytext=(6, 5),
                                 textcoords='offset points', fontsize=7,
                                 color=color)
        traj_scatter.append(sc)

    for ax in (ax_map, ax_iou, ax_pr):
        ax.set_xscale('log')
        ax.set_xlabel('training epoch')
        ax.grid(alpha=0.3)
        ax.set_xlim(0.9, 230)

    # Headroom above the highest point so the peak callouts have somewhere to
    # sit without overlapping the curves.
    ax_map.set_ylabel('mask mAP@50')
    ax_map.set_ylim(0, 0.21)
    ax_map.set_title('Mask mAP@50 vs epoch (star = peak)', fontsize=10)
    ax_iou.set_ylabel('binary IoU')
    ax_iou.set_ylim(0, 0.37)
    ax_iou.set_title('Binary IoU vs epoch (star = peak)', fontsize=10)
    ax_pr.set_ylabel('precision / recall')
    ax_pr.set_ylim(0, 0.92)
    ax_pr.set_title(
        'Crown-level precision (solid circles) and recall (dashed squares):\n'
        'late epochs trade recall away for precision', fontsize=10,
    )
    ax_traj.set_xlabel('recall')
    ax_traj.set_ylabel('precision')
    ax_traj.grid(alpha=0.3)
    ax_traj.set_title(
        'Precision-recall trajectory (marker fill = epoch)', fontsize=10,
    )

    ax_map.legend(fontsize=8, loc='lower center')
    ax_iou.legend(fontsize=8, loc='lower center')
    ax_pr.legend(fontsize=7, loc='upper left', ncol=2, framealpha=0.95)
    ax_traj.legend(fontsize=8, loc='lower left')

    if traj_scatter:
        cb = fig.colorbar(traj_scatter[0], ax=ax_traj, pad=0.02)
        cb.set_label('training epoch', fontsize=8)
        cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Mask R-CNN training-set head-to-head on {blob['n_chips']} shared "
        f"chips (test/right crops, min_instance_size="
        f"{blob['min_instance_size']}, OCM={blob['use_ocm_masks']})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outputfile, dpi=150, bbox_inches='tight')
    print(f'Wrote {outputfile}')


def report(runs, display):
    """Print peak-vs-final per run and the per-epoch gap between two runs."""
    for run, rows in sorted(runs.items()):
        print(f'\n{display.get(run, run)}:')
        for key in ('test_map/map_50', 'test_iou'):
            x, y = series(rows, key)
            i = int(np.nanargmax(y))
            print(f'  {key:18s} peak={y[i]:.4f} @ep{int(x[i]):<4d} '
                  f'final={y[-1]:.4f} (final/peak={y[-1] / y[i]:.2f})')

    if set(runs) == {'base', 'ext'}:
        print('\nper-epoch map_50, ext - base:')
        b = {r['epoch_n']: r['test_map/map_50'] for r in runs['base']}
        e = {r['epoch_n']: r['test_map/map_50'] for r in runs['ext']}
        for ep in sorted(set(b) & set(e)):
            d = e[ep] - b[ep]
            print(f'  ep{ep:<4d} {b[ep]:.4f} -> {e[ep]:.4f}  '
                  f'{d:+.4f} {"ext better" if d > 0 else "base better"}')


@click.command()
@click.argument('statfile')
@click.argument('outputfile')
@click.option('--label', 'labels', multiple=True, metavar='RUN=TEXT',
              help='Legend text for a run prefix, e.g. base="20260706 (56 '
                   'chips)". Repeatable.')
def main(statfile, outputfile, labels):
    display = {}
    for spec in labels:
        if '=' not in spec:
            raise click.ClickException(f'--label must be RUN=TEXT, got {spec!r}')
        run, text = spec.split('=', 1)
        display[run] = text

    blob, runs = load(statfile)
    print(f"Loaded {sum(len(v) for v in runs.values())} checkpoints across "
          f"{len(runs)} run(s), scored on {blob['n_chips']} chips")
    report(runs, display)
    plot(blob, runs, display, outputfile)


if __name__ == '__main__':
    main()
