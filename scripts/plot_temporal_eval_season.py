#!/usr/bin/env python
"""Seasonality of the temporal-CV skill of the PlanetScope Mask R-CNN, in the
presentation style of plot_temporal_eval.py.

Restyles plot_temporal_cv_detail.py's _season figure and the dry/wet table in
docs/planet_temporal_generalization.md. Per-chip counts come from
eval_temporal_cv.py's temporal_cv_windows.csv at epoch 12 (both windows
summed); crown-level scores from plot_temporal_eval.py's
materials/temporal_eval_crowns.csv. Dry season is December to April.

temporal_eval_season.{png,pdf}          per-chip IoU, precision and recall by
                                        month, colored by days to training,
                                        with values pooled over each month
temporal_eval_season_by_days.{png,pdf}  pooled measures by season and days to
                                        training, and crown-level ROC by season

Typical usage:
    python scripts/plot_temporal_eval_season.py
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
import plot_spatial_eval as pse  # noqa: E402
from plot_temporal_cv_detail import POOL_FNS, chip_table  # noqa: E402

WINDOWS = pse.FLOWER / '202610_temporal/temporal_cv_windows.csv'
EPOCH = 12
DAY_BINS = ['0-3', '4-7', '8-14', '15-30', '>30']
DRY_MONTHS = (12, 1, 2, 3, 4)
C_MAIN = pse.C_MAIN
WET_SHADE = '#e9eef5'
METRICS = [('iou', 'mask IoU'), ('precision', 'precision'), ('recall', 'recall')]


def season(month):
    return np.where(np.isin(month, DRY_MONTHS), 'dry', 'wet')


def save(fig, outdir, name):
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'{name}.{ext}', dpi=300)
    plt.close(fig)
    click.echo(f'Wrote {outdir / name}.png/.pdf')


@click.command()
@click.option('--outdir', type=click.Path(file_okay=False, path_type=Path),
              default=pse.DEFAULT_OUT, show_default=True)
def main(outdir):
    from sklearn.metrics import roc_auc_score, roc_curve
    chips = chip_table(pd.read_csv(WINDOWS), EPOCH)
    chips['month'] = chips.planet_date.dt.month
    chips['season'] = season(chips.month)
    ev = json.loads((outdir / 'materials/temporal_eval.json').read_text())['pooled']
    # The cluster per-window counts must match this session's pooled metrics.
    for col, key in (('iou', 'test_iou'), ('precision', 'test_event/precision'),
                     ('recall', 'test_event/recall')):
        assert abs(POOL_FNS[col](chips) - ev[key]) < 1e-3, (col, POOL_FNS[col](chips), ev[key])
    crowns = pd.read_csv(outdir / 'materials/temporal_eval_crowns.csv')
    crowns['month'] = crowns.chip.map(chips.set_index('chip').planet_date).dt.month
    crowns['season'] = season(crowns.month)

    # ---- per chip by month ----
    starts = pd.date_range('2001-01-01', periods=13, freq='MS').dayofyear.to_numpy().copy()
    starts[-1] = 366
    mids = (starts[:-1] + starts[1:]) / 2
    doy = chips.planet_date.dt.dayofyear.values
    cmap = plt.get_cmap('viridis')
    bin_color = {b: cmap(j / (len(DAY_BINS) - 1)) for j, b in enumerate(DAY_BINS)}
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 6), sharex=True)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.88, hspace=0.12)
    rng = np.random.default_rng(0)
    for ax, (col, label) in zip(axes, METRICS):
        ax.axvspan(starts[4], starts[11], color=WET_SHADE, zorder=0, lw=0)
        for b in DAY_BINS:
            sel = chips.day_bin == b
            ax.scatter(doy[sel], chips[col][sel], s=22, color=bin_color[b], ec='white',
                       lw=0.5, zorder=3, label=f'{b} d')
        for m, c in chips.groupby('month'):
            ax.hlines(POOL_FNS[col](c), starts[m - 1] + 1.5, starts[m] - 1.5, color='k',
                      lw=2.2, zorder=4, label='pooled over month' if m == 1 else None)
        ax.set_ylabel(label, fontsize=10)
        ax.set_ylim(0, 0.85 if col == 'recall' else 0.55)
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=9)
        for s in starts[1:-1]:
            ax.axvline(s, color='0.85', lw=0.6, zorder=1)
    counts = chips.month.value_counts()
    for m in range(1, 13):
        axes[0].text(mids[m - 1], 1.02, f'{counts.get(m, 0)}', transform=axes[0].get_xaxis_transform(),
                     ha='center', va='bottom', fontsize=8.5, color='0.3')
    axes[0].text(-0.01, 1.02, 'chips', transform=axes[0].transAxes, ha='right', va='bottom',
                 fontsize=8.5, color='0.3')
    axes[0].text(mids[7], 0.97, 'wet season', transform=axes[0].get_xaxis_transform(),
                 ha='center', va='top', fontsize=9.5, color='0.35', style='italic')
    axes[-1].set_xticks(mids, pd.date_range('2001-01-01', periods=12, freq='MS').strftime('%b'))
    axes[-1].tick_params(axis='x', length=0, labelsize=9.5)
    axes[-1].set_xlim(1, 366)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=len(l), fontsize=9, frameon=False,
               title='days to the nearest training chip', title_fontsize=9,
               bbox_to_anchor=(0.53, 1.0), columnspacing=1.4, handletextpad=0.3,
               markerscale=1.6)
    save(fig, outdir, 'temporal_eval_season')

    # ---- by season and days to training ----
    groups = []
    for s, bins in (('dry', [('≤14', ['0-3', '4-7', '8-14']), ('15-30', ['15-30']),
                             ('>30', ['>30'])]),
                    ('wet', [('≤14', ['0-3', '4-7', '8-14']), ('15-30', ['15-30']),
                             ('>30', ['>30'])])):
        for name, members in bins:
            c = chips[(chips.season == s) & chips.day_bin.isin(members)]
            if not len(c):
                continue
            cr = crowns[(crowns.season == s) & crowns.day_bin.isin(members)]
            groups.append(dict(season=s, days=name, chips=len(c),
                               **{k: f(c) for k, f in POOL_FNS.items()},
                               crown_auroc=roc_auc_score(cr.event, cr.score_mean)
                               if cr.event.nunique() == 2 else np.nan,
                               crowns=len(cr), crown_events=int(cr.event.sum())))
    groups = pd.DataFrame(groups)
    fig = plt.figure(figsize=(12.5, 4.6))
    ax = fig.add_axes([0.07, 0.23, 0.58, 0.66])
    show = [('mask IoU', 'iou', '#4c72b0'), ('precision', 'precision', '#dd8452'),
            ('recall', 'recall', '#55a868'), ('crown AUROC', 'crown_auroc', '0.25')]
    x = np.arange(len(groups)) + np.where(groups.season == 'wet', 0.5, 0)
    width = 0.19
    for j, (lab, col, color) in enumerate(show):
        ax.bar(x + (j - 1.5) * width, groups[col], width=width * 0.92, color=color, label=lab)
    wet_x = x[groups.season == 'wet']
    ax.axvspan(wet_x.min() - 0.5, wet_x.max() + 0.5, color=WET_SHADE, zorder=0, lw=0)
    ax.set_xticks(x, [f'{d} days\n({c} chips)' for d, c in zip(groups.days, groups.chips)],
                  fontsize=9.5)
    for s in ('dry', 'wet'):
        xs = x[groups.season == s]
        ax.text(xs.mean(), -0.17, f'{s} season ({"Dec-Apr" if s == "dry" else "May-Nov"})',
                transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=10.5,
                fontweight='bold')
    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel('score', fontsize=10)
    ax.legend(loc='upper right', ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(1.0, 1.1))
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(axis='x', length=0)
    ax4 = fig.add_axes([0.72, 0.23, 0.26, 0.66])
    roc = {}
    for s, color in (('dry', '#c08a1e'), ('wet', '#2a78d6')):
        cs = crowns[crowns.season == s]
        f_, t_, _ = roc_curve(cs.event, cs.score_mean)
        roc[s] = dict(auroc=roc_auc_score(cs.event, cs.score_mean), crowns=len(cs),
                      events=int(cs.event.sum()))
        ax4.plot(f_, t_, color=color, lw=2,
                 label=f'{s} season  (AUROC {roc[s]["auroc"]:.2f})')
    ax4.plot([0, 1], [0, 1], color='0.5', ls=':', lw=1)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1.02)
    ax4.set_xlabel('false positive rate (crowns)', fontsize=10)
    ax4.set_ylabel('true positive rate (crowns)', fontsize=10)
    ax4.legend(loc='lower right', fontsize=9, frameon=False)
    ax4.grid(alpha=0.3)
    ax4.spines[['top', 'right']].set_visible(False)
    ax4.tick_params(labelsize=9)
    save(fig, outdir, 'temporal_eval_season_by_days')

    monthly = chips.groupby('month').apply(
        lambda c: pd.Series({'chips': len(c), **{k: f(c) for k, f in POOL_FNS.items()},
                             'median_days': c.days_to_nearest_train.median()}),
        include_groups=False)
    by_season = chips.groupby('season').apply(
        lambda c: pd.Series({'chips': len(c), **{k: f(c) for k, f in POOL_FNS.items()},
                             'median_days': c.days_to_nearest_train.median()}),
        include_groups=False)
    out = dict(by_month=monthly.reset_index().to_dict('records'),
               by_season=by_season.reset_index().to_dict('records'),
               by_season_days=groups.to_dict('records'), crown_roc_by_season=roc)
    (outdir / 'materials/temporal_eval_season.json').write_text(
        json.dumps(out, indent=1, default=float))
    click.echo(monthly.round(3).to_string())
    click.echo(by_season.round(3).to_string())
    click.echo(groups.round(3).to_string())
    click.echo(json.dumps(roc, indent=1))


if __name__ == '__main__':
    main()
