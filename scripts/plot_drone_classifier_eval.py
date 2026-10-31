#!/usr/bin/env python
"""Precision-recall of the drone crown classifier, and how much hand labelling it replaced.

Scores are crown-mean probabilities from results/crown_classifications.nc4
(phantom), joined to two label sets:

  test split   the geographic test split (vicente_202509/geo_folds.csv) of
               Vicente's training labels (flowering_dataset.gpkg), which the
               deployed models (results/models/drone_*_geo_models) never saw
  new labels   Vicente's June 2026 labels (vicente_20260618), joined by
               compare_classifications_to_gt.py --exclude-train-chips (crown-
               dates inside a same-date training chip dropped); pass its
               gt_vs_ml_joined.csv with --new-labels

Targets follow generate_drone_image_masks.py: flowering if isFlowering is
'yes' ('maybe' dropped), deciduous if leafing < 50. New labels are restricted
to segmentation == 'good', where the crown outline matches the crown.

Writes drone_classifier_pr.{png,pdf} and drone_classifier_roc.{png,pdf}, and prints the labelling counts for the
slide (also written to materials/drone_classifier_counts.json).

Typical usage:
    python scripts/compare_classifications_to_gt.py --exclude-train-chips --no-visualize \\
        --output-dir $OUT/materials/drone_eval_heldout
    python scripts/plot_drone_classifier_eval.py
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
from sklearn.metrics import (average_precision_score, precision_recall_curve, roc_auc_score,
                             roc_curve)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_study_area_figure as sa  # noqa: E402

FLOWER = sa.FLOWER
TRAIN_LABELS = FLOWER / 'vicente_202509/flowering_dataset.gpkg'
GEO_FOLDS = FLOWER / 'vicente_202509/geo_folds.csv'
THRESHOLD = 0.5  # compare_classifications_to_gt.py default
TASKS = {'flowering': 'flowering_probability', 'deciduous': 'deciduous_probability'}
SETS = {'test split': '#4c72b0', 'new labels (held out)': '#dd8452'}


def training_labels():
    """Vicente's training-set crown-dates with split, targets and nc4 tag."""
    import geopandas as gpd
    g = gpd.read_file(TRAIN_LABELS, ignore_geometry=True).drop(columns='tag')  # empty here
    g = g.merge(pd.read_csv(GEO_FOLDS), on='polygon_id', how='left')
    g['uuid'] = g.polygon_id.str.split('_').str[0]
    cm = gpd.read_file(sa.DATED_CROWNS, ignore_geometry=True)
    cm['tag'] = pd.to_numeric(cm.tag, errors='coerce')
    cm = cm.dropna(subset=['tag']).drop_duplicates(['GlobalID', 'date'])
    g = g.merge(cm[['GlobalID', 'date', 'tag']], left_on=['uuid', 'date'],
                right_on=['GlobalID', 'date'], how='left')
    flower = g.isFlowering.fillna('no')
    g['flowering'] = np.where(flower == 'yes', 1, np.where(flower == 'maybe', np.nan, 0))
    g['deciduous'] = (g.leafing < 50).astype(float)
    return g


def attach_scores(df, nc4):
    import xarray as xr
    with xr.open_dataset(nc4) as ds:
        for col in TASKS.values():
            s = ds[col].to_series()
            idx = pd.MultiIndex.from_arrays([df.tag.astype('Int64'),
                                             pd.to_datetime(df.date, format='%Y_%m_%d')])
            df[col] = s.reindex(idx).values
    return df


def classified_counts():
    """Crown-dates the classifier labelled, and its event calls, phantom + mavic."""
    import xarray as xr
    out = {'crown_dates': 0, 'flowering': 0, 'deciduous': 0, 'crowns': 0, 'dates': 0}
    tags = set()
    for key in ('phantom', 'mavic'):
        with xr.open_dataset(sa.NC4[key]) as ds:
            fp = ds.flowering_probability.values
            dp = ds.deciduous_probability.values
            ok = ~np.isnan(fp)
            out['crown_dates'] += int(ok.sum())
            out['flowering'] += int((fp[ok] >= THRESHOLD).sum())
            out['deciduous'] += int((dp[ok] >= THRESHOLD).sum())
            out['dates'] += ds.sizes['date']
            tags |= set(ds.tag.values[ok.any(axis=1)].tolist())
    out['crowns'] = len(tags)
    return out


@click.command()
@click.option('--outdir', type=click.Path(file_okay=False, path_type=Path),
              default=sa.DEFAULT_OUT, show_default=True)
@click.option('--new-labels', type=click.Path(exists=True, path_type=Path),
              default=sa.DEFAULT_OUT / 'materials/drone_eval_heldout/gt_vs_ml_joined.csv',
              show_default=True, help='compare_classifications_to_gt.py joined CSV.')
@click.option('--size', nargs=2, type=float, default=(8.3, 6.0), show_default=True,
              help='Figure size in inches (two thirds of a 12.5 in slide width).')
@click.option('--dpi', default=300, show_default=True)
def main(outdir, new_labels, size, dpi):
    tr = attach_scores(training_labels(), sa.NC4['phantom'])
    test = tr[tr.split == 'test']
    new = pd.read_csv(new_labels)
    new = new[new.segmentation == 'good'].copy()
    fl = new.isFlowering.fillna('no')
    new['flowering'] = np.where(fl == 'yes', 1, np.where(fl == 'maybe', np.nan, 0))
    new['deciduous'] = (new.leafing < 50).astype(float)
    sets = {'test split': test, 'new labels (held out)': new}

    fig, axes = plt.subplots(1, 2, figsize=size, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.2, top=0.93, wspace=0.08)
    summary = {}
    for ax, (task, col) in zip(axes, TASKS.items()):
        for name, df in sets.items():
            d = df.dropna(subset=[task, col])
            y, s = d[task].values.astype(int), d[col].values
            p, r, _ = precision_recall_curve(y, s)
            ap = average_precision_score(y, s)
            pred = s >= THRESHOLD
            prec = (y[pred] == 1).mean() if pred.any() else np.nan
            rec = pred[y == 1].mean()
            ax.plot(r, p, color=SETS[name], lw=2, drawstyle='steps-post')
            ax.plot(rec, prec, 'o', color=SETS[name], mec='k', ms=7, zorder=5)
            ax.axhline(y.mean(), color=SETS[name], lw=1, ls=':')
            summary[f'{task} | {name}'] = dict(n=len(y), positives=int(y.sum()), ap=ap,
                                               precision=prec, recall=rec,
                                               crowns=int(d.tag.nunique()))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel('recall', fontsize=11)
        ax.set_title(task, fontsize=12)
        ax.grid(alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=9.5)
        # AP box in the empty part of each panel
        bx, by = (0.42, 0.62) if task == 'flowering' else (0.03, 0.04)
        ax.text(bx, by, '\n'.join(f"{n}: AP {summary[f'{task} | {n}']['ap']:.2f}"
                                   for n in sets), transform=ax.transAxes, fontsize=9.5,
                va='bottom', bbox=dict(fc='white', ec='0.8', pad=3))
    axes[0].set_ylabel('precision', fontsize=11)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, lw=2, label=n) for n, c in SETS.items()]
    handles += [Line2D([], [], marker='o', color='0.4', mec='k', ls='none', ms=7,
                       label=f'threshold {THRESHOLD}'),
                Line2D([], [], color='0.4', ls=':', lw=1, label='positive rate (chance)')]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.53, 0.0))
    name = 'drone_classifier_pr'
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'{name}.{ext}', dpi=dpi)
    plt.close(fig)
    click.echo(f'Wrote {outdir / name}.png/.pdf')

    # ROC curves, same layout
    fig, axes = plt.subplots(1, 2, figsize=size, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.2, top=0.93, wspace=0.08)
    for ax, (task, col) in zip(axes, TASKS.items()):
        for name, df in sets.items():
            d = df.dropna(subset=[task, col])
            y, s = d[task].values.astype(int), d[col].values
            fpr, tpr, _ = roc_curve(y, s)
            pred = s >= THRESHOLD
            ax.plot(fpr, tpr, color=SETS[name], lw=2)
            ax.plot(pred[y == 0].mean(), pred[y == 1].mean(), 'o', color=SETS[name],
                    mec='k', ms=7, zorder=5)
            summary[f'{task} | {name}']['auroc'] = roc_auc_score(y, s)
        ax.plot([0, 1], [0, 1], color='0.5', lw=1, ls=':')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel('false positive rate', fontsize=11)
        ax.set_title(task, fontsize=12)
        ax.grid(alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=9.5)
        ax.text(0.97, 0.04, '\n'.join(f"{n}: AUROC {summary[f'{task} | {n}']['auroc']:.3f}"
                                      for n in sets), transform=ax.transAxes, fontsize=9.5,
                ha='right', va='bottom', bbox=dict(fc='white', ec='0.8', pad=3))
    axes[0].set_ylabel('true positive rate (recall)', fontsize=11)
    handles = [Line2D([], [], color=c, lw=2, label=n) for n, c in SETS.items()]
    handles += [Line2D([], [], marker='o', color='0.4', mec='k', ls='none', ms=7,
                       label=f'threshold {THRESHOLD}'),
                Line2D([], [], color='0.5', ls=':', lw=1, label='chance')]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.53, 0.0))
    name = 'drone_classifier_roc'
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'{name}.{ext}', dpi=dpi)
    plt.close(fig)
    click.echo(f'Wrote {outdir / name}.png/.pdf')

    # labelling effort
    trn = tr[tr.split == 'train']
    cls = classified_counts()
    counts = dict(
        hand_labels_total=len(tr), hand_labels_train=len(trn), hand_labels_test=len(test),
        hand_label_crowns=int(tr.uuid.nunique()), hand_label_dates=int(tr.date.nunique()),
        hand_flowering_train=int((trn.flowering == 1).sum()),
        hand_deciduous_train=int((trn.deciduous == 1).sum()),
        hand_flowering_total=int((tr.flowering == 1).sum()),
        hand_deciduous_total=int((tr.deciduous == 1).sum()),
        classified_crown_dates=cls['crown_dates'], classified_crowns=cls['crowns'],
        classified_dates=cls['dates'], classified_flowering=cls['flowering'],
        classified_deciduous=cls['deciduous'],
        new_label_rows_heldout_good=len(new), metrics=summary)
    counts['pct_hand_labelled_train'] = 100 * counts['hand_labels_train'] / cls['crown_dates']
    counts['pct_hand_labelled_total'] = 100 * counts['hand_labels_total'] / cls['crown_dates']
    counts['pct_flowering_hand_train'] = 100 * counts['hand_flowering_train'] / cls['flowering']
    counts['pct_deciduous_hand_train'] = 100 * counts['hand_deciduous_train'] / cls['deciduous']
    (outdir / 'materials/drone_classifier_counts.json').write_text(
        json.dumps(counts, indent=1, default=float))
    click.echo(json.dumps({k: v for k, v in counts.items() if k != 'metrics'}, indent=1,
                          default=float))
    click.echo(pd.DataFrame(summary).T.round(3).to_string())


if __name__ == '__main__':
    main()
