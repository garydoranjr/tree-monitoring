#!/usr/bin/env python
"""Side-by-side summary of several compare_classifications_to_gt.py runs.

Each RUN is a name=directory pair holding gt_vs_ml_joined.csv from one run
(e.g. phantom, mavic, mavic before colour transfer). Metrics are recomputed
from the joined tables so every run is scored the same way, with
crown-bootstrap intervals, and runs whose name contains "mavic" are also
scored on the mavic-only dates (after the last phantom flight). Runs whose
joined table has a size_bin column are also scored per crown-size bin on
the segmentation==good rows. Writes OUTDIR/summary.md, metrics.csv
(by_size.csv when there are size bins) and roc_pr.png/.pdf.

Typical usage:
    E=/Volumes/Earth03/flower/figs/20261013_updates/paulino_eval
    python scripts/summarize_label_evals.py $E phantom=$E/phantom \
        phantom_heldout=$E/phantom_heldout mavic=$E/mavic \
        mavic_perband=$E/mavic_perband
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_classifications_to_gt import binary_metrics, bootstrap_ci  # noqa: E402

LAST_PHANTOM = pd.Timestamp('2024-03-18')
SHARED_DATES = ('2024-03-06', '2024-03-18')
TARGETS = {
    'flowering': ('flowering_probability', lambda d: d[d['isFlowering'].isin(['yes', 'no'])],
                  lambda d: (d['isFlowering'] == 'yes').astype(int)),
    'deciduous (leafing<50)': ('deciduous_probability', lambda d: d.dropna(subset=['leafing']),
                               lambda d: (d['leafing'] < 50).astype(int)),
}


def load_run(path):
    df = pd.read_csv(Path(path) / 'gt_vs_ml_joined.csv')
    df['dt'] = pd.to_datetime(df['date'], format='%Y_%m_%d')
    return df


def score(df, target, n_boot):
    col, select, label = TARGETS[target]
    sub = select(df)
    y, s = label(sub).to_numpy(), sub[col].to_numpy()
    m = binary_metrics(y, s, 0.5)
    if m is None:
        return None, None
    ci = bootstrap_ci(y, s, sub['tag'].to_numpy(), 0.5, n_boot)
    row = {'n': m['n'], 'n_pos': m['n_pos'], 'n_crowns': int(sub['tag'].nunique()),
           **{k: m[k] for k in ('auroc', 'auprc', 'precision', 'recall')}, **ci}
    return row, (y, s)


def fmt(r, k):
    v, lo, hi = r[k], r[f'{k}_lo'], r[f'{k}_hi']
    return '–' if np.isnan(v) else f'{v:.2f} [{lo:.2f}–{hi:.2f}]'


@click.command()
@click.argument('outdir', type=click.Path(file_okay=False, path_type=Path))
@click.argument('runs', nargs=-1, required=True)
@click.option('--n-boot', default=1000, show_default=True)
@click.option('--labels-name', default='Paulino', show_default=True,
              help='Whose labels, for the titles.')
def main(outdir, runs, n_boot, labels_name):
    runs = dict(r.split('=', 1) for r in runs)
    data = {name: load_run(path) for name, path in runs.items()}
    rows, curves = [], {}
    for name, df in data.items():
        subsets = {name: df}
        if 'mavic' in name:
            subsets[f'{name}, mavic-only dates'] = df[df['dt'] > LAST_PHANTOM]
        for label, sub in subsets.items():
            for target in TARGETS:
                r, c = score(sub, target, n_boot)
                if r is None:
                    continue
                rows.append({'run': label, 'target': target,
                             'dates': f'{sub["dt"].min():%Y-%m-%d}..{sub["dt"].max():%Y-%m-%d}',
                             **r})
                if label == name:
                    curves[(name, target)] = c
    metrics = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(outdir / 'metrics.csv', index=False)

    size_rows = []
    for name, df in data.items():
        if 'size_bin' not in df:
            continue
        good = df[df['segmentation'] == 'good']
        order = good.groupby('size_bin')['crown_area'].min().sort_values().index
        for b in order:
            for target in TARGETS:
                r, _ = score(good[good['size_bin'] == b], target, n_boot)
                if r is not None:
                    size_rows.append({'run': name, 'target': target, 'size_bin': b, **r})
    by_size = pd.DataFrame(size_rows)
    if len(by_size):
        by_size.to_csv(outdir / 'by_size.csv', index=False)

    # Same polygons on the two dates both drones flew.
    shared = []
    pha = next((d for n, d in data.items() if n.startswith('phantom') and 'heldout' not in n), None)
    for name, df in data.items():
        if 'mavic' not in name or pha is None:
            continue
        keys = ['tag', 'date']
        j = pha[pha['dt'].isin(pd.to_datetime(SHARED_DATES))].merge(
            df[keys + ['flowering_probability', 'deciduous_probability']], on=keys,
            suffixes=('_phantom', '_mavic'))
        for fl, g in j.groupby('isFlowering'):
            shared.append({'run': name, 'label': fl, 'n': len(g),
                           'P(flower) phantom': g['flowering_probability_phantom'].median(),
                           'P(flower) mavic': g['flowering_probability_mavic'].median()})
    shared = pd.DataFrame(shared)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for c, target in enumerate(TARGETS):
        for name in runs:
            if (name, target) not in curves:
                continue
            y, s = curves[(name, target)]
            r = metrics[(metrics.run == name) & (metrics.target == target)].iloc[0]
            fpr, tpr, _ = roc_curve(y, s)
            axes[0, c].plot(fpr, tpr, label=f'{name}: AUROC {r.auroc:.2f} (n_pos={r.n_pos})')
            p, rc, _ = precision_recall_curve(y, s)
            axes[1, c].plot(rc, p, label=f'{name}: AUPRC {r.auprc:.2f}')
        axes[0, c].plot([0, 1], [0, 1], color='#bbbbbb', lw=0.8)
        axes[0, c].set(xlabel='false positive rate', ylabel='true positive rate',
                       title=f'{target}: ROC')
        axes[1, c].set(xlabel='recall', ylabel='precision', title=f'{target}: precision-recall')
        for a in axes[:, c]:
            a.legend(fontsize=8)
            a.grid(alpha=0.3)
    fig.suptitle(f'Drone classifier vs {labels_name} crown labels (checked rows; crown-mean probability)')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'roc_pr.{ext}', dpi=130)

    lines = [f'# Drone classifier vs {labels_name} labels', '',
             'Crown-mean probability from the crown_classifications nc4s, joined on (tag, date). '
             'Metrics at threshold 0.5. Intervals are 95% crown-bootstrap '
             f'({n_boot} resamples). "maybe" flowering labels are excluded.', '']
    for target in TARGETS:
        lines += [f'## {target}', '',
                  '| run | dates | n | positives | crowns | AUROC | AUPRC | precision | recall |',
                  '|---|---|---|---|---|---|---|---|---|']
        for _, r in metrics[metrics.target == target].iterrows():
            lines.append(f'| {r.run} | {r.dates} | {r.n} | {r.n_pos} | {r.n_crowns} | '
                         f'{fmt(r, "auroc")} | {fmt(r, "auprc")} | {fmt(r, "precision")} | '
                         f'{fmt(r, "recall")} |')
        lines.append('')
    if len(by_size):
        lines += ['## By crown area of the dated outline (segmentation==good)', '']
        for target in TARGETS:
            lines += [f'### {target}', '',
                      '| run | crown area | n | positives | crowns | AUROC | AUPRC | precision | recall |',
                      '|---|---|---|---|---|---|---|---|---|']
            for _, r in by_size[by_size.target == target].iterrows():
                lines.append(f'| {r.run} | {r.size_bin} | {r.n} | {r.n_pos} | {r.n_crowns} | '
                             f'{fmt(r, "auroc")} | {fmt(r, "auprc")} | {fmt(r, "precision")} | '
                             f'{fmt(r, "recall")} |')
            lines.append('')
    if len(shared):
        lines += ['## Shared dates (2024-03-06, 2024-03-18): median P(flowering) on the same crowns', '',
                  '| run | label | n | P(flower) phantom | P(flower) mavic |', '|---|---|---|---|---|']
        lines += [f'| {r.run} | {r.label} | {r.n} | {r["P(flower) phantom"]:.3f} | '
                  f'{r["P(flower) mavic"]:.3f} |' for _, r in shared.iterrows()]
        lines.append('')
    (outdir / 'summary.md').write_text('\n'.join(lines))
    click.echo('\n'.join(lines))


if __name__ == '__main__':
    main()
