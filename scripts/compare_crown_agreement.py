#!/usr/bin/env python
"""Compare two temporal Planet Mask R-CNN apply runs (e.g. TOA against
MODIS-calibrated) by their per-crown agreement with the drone crown
classifications, on the crown-scenes both runs observe.

Uses the rules of plot_crown_species_comparison.py: a drone crown is an event
when 1 - (1 - p_flowering) * (1 - p_deciduous) > 0.5; a Planet crown is an
event when at least --pos-frac of its clear pixels lie under a detection
scoring at least the apply run's --det-thresh. Only crowns of at least
--min-crown-px Planet pixels are used, and only crown-scenes at least
--min-clear clear in both runs.

Planet scenes are matched to the drone flight they were coregistered to, using
scenes that are coregistered and 3 to --match-days days from that flight. They
are split into two strata: in the envelope (30 days or less from a training
chip) and outside it.

Writes, with the given output prefix:

- <prefix>.csv: per stratum and species (the --n-species species with the
  most drone event crown-dates, other species, all species), precision,
  recall and the share of crown-scenes called an event, for each run.
- <prefix>_rules.csv: per stratum and run, precision, recall and share
  called for alternative Planet rules, and the AUROC of pos_frac and
  max_score against the drone label.
- <prefix>.png/.pdf: precision, recall and event share by species for both
  runs, one row of panels per stratum.

It prints the same tables, the all-species results by year and by whether
the scene's calibration offset was fitted or borrowed (--offsets), Cohen's kappa
between the two runs' calls, and scene-bootstrap 95 % intervals of the
all-species precision and recall differences.

Typical usage:
    python scripts/compare_crown_agreement.py \\
        /Volumes/Earth03/flower/202610_temporal_apply_cal/crown_agreement_toa_vs_cal \\
        --toa-dir /Volumes/Earth03/flower/202610_temporal_apply \\
        --cal-dir /Volumes/Earth03/flower/202610_temporal_apply_cal
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from plot_crown_species_comparison import (APPLY, CROWN_COLS, DRONE, GRID,
                                           INK, NC4S, agreement, load_drone,
                                           save)

RUNS = ('toa', 'cal')
LABELS = {'toa': 'TOA, per-image stretch',
          'cal': 'MODIS-calibrated, fixed stretch'}
COLORS = {'toa': '#2a78d6', 'cal': '#1baf7a'}  # as in temporal_cv_toa_vs_cal.png
STRATA = {'in': 'in envelope (30 d or less from a training chip)',
          'out': 'outside envelope (more than 30 d from a training chip)'}
OFFSETS = os.path.join(os.path.dirname(__file__), '..', 'results',
                       'planet_scene_offsets.csv')
RULES = (('pos_frac', 0.10), ('pos_frac', 0.25), ('pos_frac', 0.50),
         ('max_score', 0.50))


def load_run(apply_dir, min_clear):
    """Crown-scenes of one apply run with enough clear pixels, with the
    scene's drone_date, and every crown's size."""
    crowns = pd.read_csv(f'{apply_dir}/crown_timeseries.csv',
                         usecols=CROWN_COLS + ['max_score'],
                         parse_dates=['date'])
    scenes = pd.read_csv(f'{apply_dir}/scene_summary.csv',
                         usecols=['scene', 'drone_date'],
                         parse_dates=['drone_date'])
    crown_px = crowns.groupby('tag')['crown_px'].first()
    # A few tags are shared by several polygons; number them so the two runs
    # pair polygon by polygon.
    crowns['poly'] = crowns.groupby(['scene', 'tag']).cumcount()
    crowns = crowns[crowns['crown_clear_frac'] >= min_clear]
    crowns = crowns.merge(scenes, on='scene', how='left')
    crowns[['pos_frac', 'max_score']] = crowns[['pos_frac', 'max_score']].fillna(0)
    return crowns, crown_px


def paired(toa, cal):
    """Crown-scenes present in both runs; scene metadata from the TOA run
    (both runs share the coregistration), model outputs suffixed by run."""
    keys = ['scene', 'tag', 'poly']
    out = ['pos_frac', 'max_score']
    return toa.merge(cal[keys + out], on=keys, suffixes=('_toa', '_cal'),
                     validate='one_to_one')


def with_call(m, run, col, thresh):
    return m.assign(event_planet=m[f'{col}_{run}'] >= thresh)


def scene_bootstrap(m, pos_frac, n_boot, seed=0):
    """95 % intervals of cal - toa precision and recall, resampling scenes."""
    g = pd.DataFrame({'scene': m['scene'], 'd': m['event_drone']})
    for run in RUNS:
        e = m[f'pos_frac_{run}'] >= pos_frac
        g[f'tp_{run}'] = e & m['event_drone']
        g[f'np_{run}'] = e
    per = g.groupby('scene').sum().to_numpy(dtype=float)  # d, tp, np per run
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), size=(n_boot, len(per)))
    s = per[idx].sum(axis=1)
    d, tp_t, np_t, tp_c, np_c = s.T
    dp = tp_c / np_c - tp_t / np_t
    dr = (tp_c - tp_t) / d
    return np.percentile(dp, [2.5, 97.5]), np.percentile(dr, [2.5, 97.5])


def by_group(m, key, name, pos_frac, n_boot=0):
    """All-species precision and recall of both runs per group of m, with
    scene-bootstrap 95 % intervals of cal - toa when n_boot."""
    rows = []
    for k, g in m.groupby(key):
        r = {name: k, 'scenes': g['scene'].nunique(),
             'drone_events': int(g['event_drone'].sum())}
        for run in RUNS:
            a = agreement(with_call(g, run, 'pos_frac', pos_frac), [])
            r[f'precision_{run}'] = a['precision'].iloc[-1]
            r[f'recall_{run}'] = a['recall'].iloc[-1]
        if n_boot:
            ci_p, ci_r = scene_bootstrap(g, pos_frac, n_boot)
            r['d_precision_95'] = f'[{ci_p[0]:+.3f}, {ci_p[1]:+.3f}]'
            r['d_recall_95'] = f'[{ci_r[0]:+.3f}, {ci_r[1]:+.3f}]'
        rows.append(r)
    return pd.DataFrame(rows)


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe)


def plot(tables, prefix):
    cols = (('precision', 'precision'), ('recall', 'recall'),
            ('share', 'share of crown-scenes called an event'))
    n = len(next(iter(tables.values())))
    fig, axes = plt.subplots(len(tables), 3, sharex='col',
                             figsize=(13, 0.33 * n * len(tables) + 1.6),
                             gridspec_kw={'width_ratios': (1, 1, 1)})
    for row, (stratum, t) in zip(axes, tables.items()):
        y = np.arange(len(t))[::-1]
        for ax, (col, xlabel) in zip(row, cols):
            if col == 'share':
                ax.scatter(t['drone_share'], y, s=40, color=DRONE,
                           edgecolor='white', lw=1, label='drone label', zorder=3)
            vals = [t[f'{col if col != "share" else "planet_share"}_{run}']
                    for run in RUNS]
            ax.hlines(y, np.fmin(*vals), np.fmax(*vals), color=GRID, lw=2,
                      zorder=2)
            for run, v in zip(RUNS, vals):
                ax.scatter(v, y, s=40, color=COLORS[run], edgecolor='white',
                           lw=1, label=LABELS[run], zorder=3)
            ax.set_xlim(0, 1 if col != 'share' else None)
            if col == 'share':
                ax.set_xlim(left=0)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='x', color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            # Separate the two pooled rows from the per-species rows.
            ax.axhline(1.5, color=INK, lw=0.6)
            ax.set_yticks(y)
            ax.set_yticklabels([])
            if stratum == list(tables)[-1]:
                ax.set_xlabel(xlabel)
        row[0].set_yticklabels(
            [f'{s} (n={d:,})' for s, d in zip(t['species'], t['drone_events'])],
            fontsize=8.5)
        n_sc = t.attrs['n_scenes']
        row[0].set_title(f'{STRATA[stratum]}: {n_sc} scenes, '
                         f'{t.attrs["n_flights"]} flights', loc='left',
                         fontsize=10)
    handles, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='upper center', ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / fig.get_figheight()))
    out = f'{prefix}.png'
    fig.savefig(out, dpi=200)
    fig.savefig(f'{prefix}.pdf')
    plt.close(fig)
    print(f'Wrote {out}')


@click.command()
@click.argument('outputprefix', type=click.Path(dir_okay=False))
@click.option('--toa-dir', default=APPLY, show_default=True,
              type=click.Path(exists=True, file_okay=False))
@click.option('--cal-dir', default=f'{APPLY}_cal', show_default=True,
              type=click.Path(exists=True, file_okay=False))
@click.option('--nc4', 'nc4s', multiple=True, default=NC4S, show_default=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option('--n-species', default=12, show_default=True)
@click.option('--pos-frac', default=0.25, show_default=True,
              help='Planet event: share of clear crown pixels under a detection.')
@click.option('--min-clear', default=0.5, show_default=True,
              help='Minimum clear fraction of a Planet crown, in both runs.')
@click.option('--min-crown-px', default=64, show_default=True,
              help='Minimum crown size in 0.75 m Planet pixels.')
@click.option('--match-days', default=7, show_default=True,
              help='Maximum days from Planet scene to drone flight.')
@click.option('--offsets', default=OFFSETS, show_default=True,
              type=click.Path(exists=True, dir_okay=False),
              help='Per-scene calibration offsets (fit or borrowed).')
@click.option('--n-boot', default=2000, show_default=True,
              help='Scene-bootstrap resamples for the difference intervals.')
def main(outputprefix, toa_dir, cal_dir, nc4s, n_species, pos_frac, min_clear,
         min_crown_px, match_days, offsets, n_boot):
    drone = load_drone(nc4s)
    toa, crown_px = load_run(toa_dir, min_clear)
    cal, _ = load_run(cal_dir, min_clear)
    n_scenes = {run: df['scene'].nunique() for run, df in zip(RUNS, (toa, cal))}
    both = paired(toa, cal)
    print(f"paired crown-scenes: {len(both):,} in {both['scene'].nunique()} "
          f"scenes (toa {n_scenes['toa']}, cal {n_scenes['cal']} scenes "
          'with any clear crown)')

    big = crown_px.index[crown_px >= min_crown_px]
    drone = drone[drone['tag'].isin(big)]
    both = both[both['tag'].isin(big)]
    both['species'] = both['tag'].map(
        drone.drop_duplicates('tag').set_index('tag')['species'])
    both = both.dropna(subset=['species'])

    # Species ranked as in plot_crown_species_comparison.py, over the TOA
    # run's span.
    period = drone['date'].between(toa['date'].min(), toa['date'].max())
    ranked = drone[period].groupby('species')['event'].sum()
    species = ranked.sort_values(ascending=False).index[:n_species].tolist()

    near = both[both['coreg_ok'] & (both['days_to_drone'] <= match_days)]
    matched = near.merge(
        drone.rename(columns={'date': 'drone_date', 'event': 'event_drone'})[
            ['tag', 'drone_date', 'event_drone']],
        on=['tag', 'drone_date'])
    source = pd.read_csv(offsets).groupby('scene_id')['source'].first()
    source = source.where(source == 'fit', 'borrowed')
    matched['offset'] = matched['scene'].str.removesuffix('_4band').map(source)
    strata = {'in': matched[matched['in_envelope']],
              'out': matched[~matched['in_envelope']]}

    tables, rules, rows = {}, [], []
    for stratum, m in strata.items():
        t = None
        for run in RUNS:
            a = agreement(with_call(m, run, 'pos_frac', pos_frac), species)
            a = a.rename(columns={c: f'{c}_{run}' for c in
                                  ('planet_events', 'precision', 'recall',
                                   'planet_share')})
            t = a if t is None else t.merge(
                a[['species'] + [c for c in a if c.endswith(f'_{run}')]],
                on='species')
        t.attrs.update(n_scenes=m['scene'].nunique(),
                       n_flights=m['drone_date'].nunique())
        tables[stratum] = t
        rows.append(t.assign(stratum=stratum))

        for run in RUNS:
            for col, thresh in RULES:
                e = m[f'{col}_{run}'] >= thresh
                tp = (e & m['event_drone']).sum()
                rules.append({'stratum': stratum, 'run': run,
                              'rule': f'{col} >= {thresh:.2f}',
                              'precision': tp / e.sum(),
                              'recall': tp / m['event_drone'].sum(),
                              'share': e.mean()})
            for col in ('pos_frac', 'max_score'):
                rules.append({'stratum': stratum, 'run': run,
                              'rule': f'AUROC {col}',
                              'auroc': roc_auc_score(m['event_drone'],
                                                     m[f'{col}_{run}'])})

    table = pd.concat(rows, ignore_index=True)
    cols = ['stratum', 'species', 'crown_scenes', 'drone_events', 'drone_share']
    cols += [f'{c}_{run}' for c in ('planet_events', 'precision', 'recall',
                                    'planet_share') for run in RUNS]
    table[cols].to_csv(f'{outputprefix}.csv', index=False)
    rules = pd.DataFrame(rules)
    rules.to_csv(f'{outputprefix}_rules.csv', index=False)

    with pd.option_context('display.width', 200, 'display.max_columns', 30):
        for stratum, m in strata.items():
            t = tables[stratum]
            print(f"\n{STRATA[stratum]}: {len(m):,} crown-scenes, "
                  f"{t.attrs['n_scenes']} scenes, {t.attrs['n_flights']} flights")
            print(t[cols[1:]].round(3).to_string(index=False))
            calls = [m[f'pos_frac_{run}'] >= pos_frac for run in RUNS]
            ci_p, ci_r = scene_bootstrap(m, pos_frac, n_boot)
            print(f'kappa between runs {kappa(*calls):.3f}; calls differ on '
                  f'{(calls[0] != calls[1]).mean():.2%} of crown-scenes')
            print(f'cal - toa, scene bootstrap 95 %: precision '
                  f'[{ci_p[0]:+.3f}, {ci_p[1]:+.3f}], recall '
                  f'[{ci_r[0]:+.3f}, {ci_r[1]:+.3f}]')
            for name, key in (('year', m['date'].dt.year),
                              ('offset', m['offset'])):
                print(by_group(m, key, name, pos_frac,
                               n_boot if name == 'offset' else 0).round(3)
                      .to_string(index=False))
        print('\nPlanet rules:')
        print(rules.round(3).to_string(index=False))

    plot(tables, outputprefix)


if __name__ == '__main__':
    main()
