#!/usr/bin/env python
"""Planet-drone agreement and phenological timing at the level of single trees.

Follows two suggestions from John Burley (2026-11-17): compare the Planet and
drone series across scales (whole plot, species, individual crown), and test
whether a tree that is early relative to its conspecifics in one year is
early in other years.

  scales       Pearson correlation of the Planet event series with the drone
               series for the whole plot, per species and per crown; whether
               a crown's Planet series follows its own drone series more
               closely than a conspecific's; how the correlation grows with
               the number of crowns averaged
  scales-html  interactive version of the species correlation heatmap: each
               cell shows the scatter plot and the time series behind it
  repeat       peak date of flowering and of leaf loss per tree and season,
               and its repeatability between years

The matched crown-flight tables are the ones cached by
`drone_planet_phenology.py series` (read from --tables).
"""
import json
import warnings
from pathlib import Path

import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import drone_planet_phenology as dp  # noqa: E402
from drone_planet_phenology import (  # noqa: E402
    C_DECID, C_FLOWER, C_PLANET, FLOWER, GRID, INK, TYPE_NAME, TYPES, _corr,
    boot_ci, build_tables, correlate_crowns, event_windows, save, short,
    species_pairs, style)

DEFAULT_OUT = FLOWER / 'figs' / '20261118_burley'
TABLES = FLOWER / 'figs' / '20261117_updates'
PLOT = 'All crowns'
dp_PHANTOM_END = pd.Timestamp('2024-03-18')   # last phantom flight used
SCRIPT = 'scripts/individual_phenology.py'
# The nine cells of the species heatmap: drone series and subset of flights.
CELLS = [('event', 'all'), ('event', 'dry'), ('event', 'wet'),
         ('flower', 'all'), ('flower', 'window'), ('flower', 'pure'),
         ('decid', 'all'), ('decid', 'window'), ('decid', 'pure')]
CELL_NAME = {
    ('event', 'all'): 'either, all', ('event', 'dry'): 'either, dry',
    ('event', 'wet'): 'either, wet', ('flower', 'all'): 'flowering, all',
    ('flower', 'window'): 'flowering, in season',
    ('flower', 'pure'): 'flowering, in season, no leaf loss',
    ('decid', 'all'): 'leaf loss, all',
    ('decid', 'window'): 'leaf loss, in season',
    ('decid', 'pure'): 'leaf loss, in season, no flowering'}
PROB = {'event': 'p_event', 'flower': 'p_flower', 'decid': 'p_decid'}
C_TYPE = {'event': INK, 'flower': C_FLOWER, 'decid': C_DECID}


def write_caption(outdir, name, command, caption, notes=''):
    text = (f'# {name}\n\nMade by `{SCRIPT} {command}`.\n\n'
            f'## Caption\n\n{caption}\n')
    if notes:
        text += f'\n## Notes\n\n{notes}\n'
    (outdir / f'{name}_caption.md').write_text(text)


def cell_key(ty, sub, pre):
    return f'{pre}_{ty}' if sub == 'all' else f'{pre}_{ty}_{sub}'


# ------------------------------------------------------------ unit series

def cell_masks(t, win):
    """Flights of a unit's table used by each heatmap cell (None where the
    event type has no season in the drone record)."""
    month, dry = t['month'].values, t['dry'].values
    other = {'flower': 'decid', 'decid': 'flower'}
    out = {}
    for ty, sub in CELLS:
        if sub == 'all':
            m = np.ones(len(t), bool)
        elif sub == 'dry':
            m = dry.copy()
        elif sub == 'wet':
            m = ~dry
        else:
            if win[ty] is None:
                out[(ty, sub)] = None
                continue
            m = np.isin(month, win[ty]['months'])
            if sub == 'pure' and win[other[ty]] is not None:
                m &= ~np.isin(month, win[other[ty]]['core'])
        out[(ty, sub)] = m
    return out


def correlate_units(sp_pairs, drone, kind, n_boot, rng, min_flights=20):
    """One row per unit (species, or the whole plot): the correlation of the
    Planet share with each drone share over each subset of flights, with a
    year-month block bootstrap interval. Returns the table and, per unit,
    the flights each cell used."""
    pre = 'rho' if kind == 'spearman' else 'r'
    rows, masks = [], {}
    for unit, t in sp_pairs.groupby('species'):
        if len(t) < min_flights or t['event'].sum() == 0:
            continue
        t = t.sort_values('flight').reset_index(drop=True)
        win = event_windows(drone, unit)
        mk = cell_masks(t, win)
        row = {'species': unit, 'n_flights': len(t),
               'n_crowns': int(t['n'].median())}
        for (ty, sub), m in mk.items():
            key = cell_key(ty, sub, pre)
            row[key] = row[f'{key}_lo'] = row[f'{key}_hi'] = np.nan
            row[f'{key}_n'] = 0 if m is None else int(m.sum())
            if m is None:
                continue
            s = t[m]
            row[key] = _corr(s[ty].values, s['planet'].values, kind)
            if np.isfinite(row[key]) and n_boot:
                row[f'{key}_lo'], row[f'{key}_hi'] = boot_ci(
                    s, ty, 'planet', n_boot, rng, kind)
        rows.append(row)
        masks[unit] = (t, mk)
    return pd.DataFrame(rows), masks


def unit_means(pairs, min_crowns=5):
    """Per unit and flight, the same quantity that is correlated per crown:
    mean drone probabilities and mean Planet score over the unit's crowns."""
    g = pairs.groupby(['species', 'flight'])
    t = g[['p_event', 'p_flower', 'p_decid', 'score']].mean()
    t['n'] = g.size()
    return t[t['n'] >= min_crowns].reset_index()


def correlate_means(means, units, min_flights=20):
    rows = []
    for unit, t in means.groupby('species'):
        if unit not in units or len(t) < min_flights:
            continue
        rows.append({'species': unit, 'n_flights': len(t),
                     **{f'r_{ty}': _corr(t[PROB[ty]].values, t['score'].values,
                                         'pearson') for ty in TYPES}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------- crown analyses

def pearson_cols(a, B, min_n=15):
    """Pearson of vector `a` with each column of `B` over the rows where
    both are finite; NaN below `min_n` rows or without variation."""
    ok = np.isfinite(a)[:, None] & np.isfinite(B)
    n = ok.sum(0)
    with warnings.catch_warnings(), np.errstate(invalid='ignore', divide='ignore'):
        warnings.simplefilter('ignore')
        A = np.where(ok, a[:, None], np.nan)
        Bm = np.where(ok, B, np.nan)
        A = A - np.nanmean(A, 0)
        Bm = Bm - np.nanmean(Bm, 0)
        den = np.sqrt(np.nansum(A * A, 0) * np.nansum(Bm * Bm, 0))
        r = np.nansum(A * Bm, 0) / den
    r[(n < min_n) | ~(den > 0)] = np.nan
    return r


def crown_pivots(pairs, min_pairs=15):
    """Flight-by-crown matrices of the paired values, for the crowns with at
    least `min_pairs` paired flights."""
    n = pairs.groupby('tag').size()
    p = pairs[pairs['tag'].isin(n.index[n >= min_pairs])]
    piv = {c: p.pivot(index='flight', columns='tag', values=c)
           for c in ('score', 'p_event', 'p_flower', 'p_decid')}
    species = p.drop_duplicates('tag').set_index('tag')['species']
    return piv, species.reindex(piv['score'].columns)


def own_vs_others(crowns, piv, species, rng, n_hetero=40, min_n=15):
    """For each crown: the correlation of its Planet series with (a) the
    drone series of each other crown of its species (mean over them), (b)
    the mean drone series of the other crowns of its species, (c) drone
    series of crowns of other species; and (d) the correlation of its Planet
    and drone departures from the means of its other conspecifics."""
    S, E = piv['score'].values, piv['p_event'].values
    tags = piv['score'].columns.values
    col = {t: i for i, t in enumerate(tags)}
    sp = species.values
    rows = []
    for tag in crowns['tag']:
        i = col[tag]
        same = np.nonzero((sp == sp[i]) & (tags != tag))[0]
        diff = np.nonzero(sp != sp[i])[0]
        row = {'tag': tag, 'n_conspecific': len(same)}
        if len(same):
            rc = pearson_cols(S[:, i], E[:, same], min_n)
            row['r_conspecific'] = np.nanmean(rc) if np.isfinite(rc).any() else np.nan
            row['n_conspecific_used'] = int(np.isfinite(rc).sum())
        if len(same) >= 3:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                cnt = np.isfinite(E[:, same]).sum(1)
                eo = np.where(cnt >= 3, np.nanmean(E[:, same], 1), np.nan)
                so = np.where(cnt >= 3, np.nanmean(S[:, same], 1), np.nan)
            row['r_species_mean'] = pearson_cols(S[:, i], eo[:, None], min_n)[0]
            row['r_departure'] = pearson_cols(S[:, i] - so, (E[:, i] - eo)[:, None],
                                              min_n)[0]
        pick = rng.choice(diff, min(n_hetero, len(diff)), replace=False)
        rh = pearson_cols(S[:, i], E[:, pick], min_n)
        row['r_other_species'] = np.nanmean(rh) if np.isfinite(rh).any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


AGG_SIZES = (1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 400, 800, 1600)


def aggregation_curve(S, E, rng, n_draw=200, min_flights=20):
    """Correlation of the mean Planet score with the mean drone event
    probability over random sets of m crowns, for growing m."""
    n = S.shape[1]
    rows = []
    for m in [s for s in AGG_SIZES if s < n] + [n]:
        rs = []
        for _ in range(1 if m == n else n_draw):
            idx = rng.choice(n, m, replace=False)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ok = np.isfinite(S[:, idx]).sum(1) >= max(1, (m + 1) // 2)
                x = np.nanmean(E[:, idx], 1)[ok]
                y = np.nanmean(S[:, idx], 1)[ok]
            if ok.sum() >= min_flights:
                rs.append(_corr(x, y, 'pearson'))
        rs = np.array(rs, float)
        rs = rs[np.isfinite(rs)]
        if len(rs):
            q = np.percentile(rs, [25, 50, 75])
            rows.append({'m': m, 'r': q[1], 'r_q25': q[0], 'r_q75': q[2],
                         'n_draws': len(rs)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- figures

def plot_heatmap_pearson(tab, outdir):
    cols = [cell_key(ty, sub, 'r') for ty, sub in CELLS]
    m = tab[cols].values.astype(float).T
    fig, ax = plt.subplots(figsize=(0.235 * len(tab) + 2.6, 6.2))
    im = ax.imshow(m, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
    for (i, j), v in np.ndenumerate(m):
        if np.isfinite(v):
            ax.text(j, i, f'{v:.2f}'.replace('0.', '.'), ha='center', va='center',
                    fontsize=5.5, rotation=90, color='w' if abs(v) > 0.6 else INK)
    ax.set_yticks(range(len(CELLS)), [CELL_NAME[c] for c in CELLS], fontsize=8)
    labels = [f'{s if s == PLOT else short(s)} ({int(n)})'
              for s, n in zip(tab['species'], tab['n_crowns'])]
    ax.set_xticks(range(len(tab)), labels, fontsize=7.5, rotation=60,
                  ha='right', rotation_mode='anchor')
    for lab, s in zip(ax.get_xticklabels(), tab['species']):
        lab.set_style('normal' if s == PLOT else 'italic')
        lab.set_weight('bold' if s == PLOT else 'normal')
    ax.tick_params(length=0)
    for y in (2.5, 5.5):
        ax.axhline(y, color='w', lw=3)
    ax.axvline(0.5, color='w', lw=3)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01,
                 label='Pearson correlation of Planet with drone share')
    save(fig, outdir, 'phenology_corr_species_pearson')


def plot_distribution(crowns, sp_means, plot_means, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4), sharey=True)
    rng = np.random.default_rng(1)
    for ax, ty in zip(axes, ('event', 'flower', 'decid')):
        ccol = 'r' if ty == 'event' else f'r_{ty}'
        groups = [crowns[ccol].dropna().values, sp_means[f'r_{ty}'].dropna().values]
        for x, v, ms in zip((0, 1), groups, (2.5, 4)):
            ax.plot(x + rng.uniform(-0.22, 0.22, len(v)), v, 'o', ms=ms,
                    color=C_TYPE[ty], alpha=0.25 if x == 0 else 0.55, mew=0)
            q = np.percentile(v, [25, 50, 75])
            ax.plot([x - 0.3, x + 0.3], [q[1], q[1]], color=INK, lw=2)
            ax.plot([x, x], [q[0], q[2]], color=INK, lw=1)
            ax.text(x + 0.33, q[1], f'{q[1]:.2f}', va='center', fontsize=8)
        pv = plot_means[f'r_{ty}'].iloc[0]
        ax.plot(2, pv, 'D', ms=8, color=C_TYPE[ty], mec=INK)
        ax.text(2.2, pv, f'{pv:.2f}', va='center', fontsize=8)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks([0, 1, 2], [f'individual\ncrowns\n(n = {len(groups[0])})',
                                  f'species\n(n = {len(groups[1])})',
                                  'whole plot'], fontsize=8)
        ax.set_xlim(-0.5, 2.6)
        ax.set_title(f'drone {TYPE_NAME[ty]}', fontsize=10, loc='left')
        style(ax)
    axes[0].set_ylabel('Pearson correlation of Planet with drone')
    axes[0].set_ylim(-1, 1)
    save(fig, outdir, 'scale_corr_distribution')


def plot_species_vs_crowns(crowns, sp_means, outdir, min_crowns=3):
    g = crowns.groupby('species')['r'].agg(['median', 'size'])
    t = sp_means.set_index('species').join(g, how='inner')
    t = t[t['size'] >= min_crowns]
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([-0.4, 1], [-0.4, 1], color=INK, lw=0.8)
    ax.scatter(t['median'], t['r_event'], s=8 + 4 * t['size'], color=C_PLANET,
               alpha=0.55, lw=0)
    gain = (t['r_event'] - t['median'])
    top = t['r_event'].sort_values().index[-2:]
    for s, off, ha in ([(s, (5, 3), 'left') for s in gain.sort_values().index[-4:]]
                       + [(top[1], (-6, 5), 'right'), (top[0], (7, -9), 'left')]):
        ax.annotate(short(s), (t.loc[s, 'median'], t.loc[s, 'r_event']),
                    fontsize=7, style='italic', xytext=off, ha=ha,
                    textcoords='offset points')
    ax.set_xlabel('median correlation of the species\' individual crowns')
    ax.set_ylabel('correlation of the species means')
    ax.set_xlim(-0.4, 1)
    ax.set_ylim(-0.4, 1)
    ax.set_aspect('equal')
    style(ax)
    save(fig, outdir, 'scale_corr_species_vs_crowns')
    return t


def plot_own_vs_conspecific(c, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6),
                             gridspec_kw={'width_ratios': [1, 1.15]})
    ax = axes[0]
    d = c.dropna(subset=['r', 'r_conspecific'])
    ax.plot([-0.5, 1], [-0.5, 1], color=INK, lw=0.8)
    ax.plot(d['r_conspecific'], d['r'], 'o', ms=3, color=C_PLANET, alpha=0.35, mew=0)
    ax.set_xlabel('correlation with the drone series of other\ncrowns of the same species (mean)')
    ax.set_ylabel('correlation with the crown\'s own drone series')
    ax.set_xlim(-0.5, 1)
    ax.set_ylim(-0.5, 1)
    ax.set_aspect('equal')
    style(ax)
    ax = axes[1]
    cols = [('r', 'own\ncrown'), ('r_species_mean', 'mean of\nother\nconspecifics'),
            ('r_conspecific', 'another\nconspecific'),
            ('r_other_species', 'crown of\nanother\nspecies'),
            ('r_departure', 'departures\nfrom the\nconspecific\nmean')]
    rng = np.random.default_rng(1)
    for x, (col, _) in enumerate(cols):
        v = c[col].dropna().values
        ax.plot(x + rng.uniform(-0.22, 0.22, len(v)), v, 'o', ms=2.2,
                color=C_PLANET if x < 4 else C_FLOWER, alpha=0.25, mew=0)
        q = np.percentile(v, [25, 50, 75])
        ax.plot([x - 0.3, x + 0.3], [q[1], q[1]], color=INK, lw=2)
        ax.plot([x, x], [q[0], q[2]], color=INK, lw=1)
        ax.text(x + 0.32, q[1], f'{q[1]:.2f}', va='center', fontsize=8)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(len(cols)), [n for _, n in cols], fontsize=8)
    ax.set_xlim(-0.5, len(cols) - 0.3)
    ax.set_ylabel('Pearson correlation of a crown\'s Planet series')
    ax.set_ylim(-1, 1)
    style(ax)
    save(fig, outdir, 'scale_corr_own_vs_conspecific')


def plot_aggregation(agg, outdir):
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    sp = agg[agg['species'] != PLOT]
    for s, t in sp.groupby('species'):
        ax.plot(t['m'], t['r'], color='#9a9a9a', lw=0.8, alpha=0.7)
    med = sp.groupby('m')['r'].agg(['median', 'size'])
    med = med[med['size'] == sp['species'].nunique()]
    ax.plot(med.index, med['median'], 'o-', color=INK, lw=2, ms=4,
            label='median over species')
    t = agg[agg['species'] == PLOT]
    ax.plot(t['m'], t['r'], 's-', color=C_PLANET, lw=2, ms=4,
            label='crowns drawn from all species')
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
                  ['1', '2', '5', '10', '20', '50', '100', '200', '500', '1000', '2000'])
    ax.set_xlabel('number of crowns averaged')
    ax.set_ylabel('Pearson correlation of Planet with drone')
    ax.axhline(0, color=INK, lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    style(ax)
    save(fig, outdir, 'scale_corr_aggregation')


FACTORS = [('area_m2', 'crown area (m²)', [36, 50, 100, 200, 400, 800, 3000]),
           ('n_events', 'drone events among paired flights', [3, 6, 12, 24, 48, 200]),
           ('illumination', 'crown illumination index', [1.5, 2.5, 3.5, 4.5, 5.5]),
           ('liana', 'liana load index', [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])]


def plot_crown_factors(c, outdir):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    rows = []
    for ax, (col, label, bins) in zip(axes, FACTORS):
        cut = pd.cut(c[col], bins, right=False)
        for k, (b, g) in enumerate(c.groupby(cut, observed=False)):
            v = g['r'].dropna().values
            if len(v) < 5:
                continue
            q = np.percentile(v, [25, 50, 75])
            ax.plot([k, k], [q[0], q[2]], color=C_PLANET, lw=6, alpha=0.35,
                    solid_capstyle='butt')
            ax.plot(k, q[1], 'o', color=C_PLANET, ms=6)
            ax.text(k, -0.37, f'{len(v)}', ha='center', fontsize=7, color='#666')
            rows.append({'factor': col, 'lo': b.left, 'hi': b.right, 'n': len(v),
                         'r_q25': q[0], 'r_median': q[1], 'r_q75': q[2]})
        integer = col in ('illumination', 'liana')
        ax.set_xticks(range(len(bins) - 1),
                      [f'{int(bins[i] + 0.5)}' if integer else
                       f'{bins[i]:g}–{bins[i + 1]:g}' for i in range(len(bins) - 1)],
                      fontsize=7.5, rotation=0 if integer else 30)
        ax.set_xlabel(label)
        ax.axhline(0, color=INK, lw=0.8)
        style(ax)
    axes[0].set_ylabel('Pearson correlation of Planet with\ndrone, per crown (median, quartiles)')
    axes[0].set_ylim(-0.42, 1)
    save(fig, outdir, 'scale_corr_crown_factors')
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- cli

@click.group()
def cli():
    pass


def scale_tables(T, source, kind, n_boot, rng, min_crowns=5):
    """Unit (plot + species) and crown tables for one Planet source."""
    pairs, drone = T['pairs'][source], T['drone']
    sp = species_pairs(pairs, min_crowns)
    pl = species_pairs(pairs.assign(species=PLOT), min_crowns)
    stab, smask = correlate_units(sp, drone, kind, n_boot, rng)
    ptab, pmask = correlate_units(pl, drone.assign(species=PLOT), kind, n_boot, rng)
    pre = 'rho' if kind == 'spearman' else 'r'
    stab = stab[stab['n_crowns'] >= 8].sort_values(f'{pre}_event', ascending=False)
    smask = {s: smask[s] for s in stab['species']}
    return stab, ptab, {**pmask, **smask}


@cli.command()
@click.option('--outdir', default=str(DEFAULT_OUT), show_default=True)
@click.option('--tables', default=str(TABLES), show_default=True,
              help='Directory whose materials/ holds the matched tables.')
@click.option('--source', default='mrcnn', show_default=True)
@click.option('--n-boot', default=500, show_default=True)
@click.option('--check-spearman', is_flag=True,
              help='Recompute with Spearman and compare with the tables of '
                   'drone_planet_phenology.py correlate, then stop.')
def scales(outdir, tables, source, n_boot, check_spearman):
    """Correlate Planet with the drone for the plot, species and crowns."""
    outdir = Path(outdir)
    mat = outdir / 'materials'
    mat.mkdir(parents=True, exist_ok=True)
    T = build_tables(Path(tables), 10)
    pairs = T['pairs'][source]
    if check_spearman:
        stab, _, _ = scale_tables(T, source, 'spearman', 0, np.random.default_rng(0))
        old = pd.read_csv(Path(tables) / 'materials' / f'phenology_corr_species_{source}.csv')
        m = stab.merge(old, on='species', suffixes=('', '_old'))
        for ty, sub in CELLS:
            k = cell_key(ty, sub, 'rho')
            d = (m[k] - m[f'{k}_old']).abs().max()
            click.echo(f'{k}: {m[k].notna().sum()} species, max |diff| {d:.2e}')
        new = correlate_crowns(pairs)
        oldc = pd.read_csv(Path(tables) / 'materials' / f'phenology_corr_crowns_{source}.csv')
        mc = new.merge(oldc, on='tag', suffixes=('', '_old'))
        click.echo(f'crowns: {len(new)} vs {len(oldc)}; max |diff| '
                   f'{(mc["rho"] - mc["rho_old"]).abs().max():.2e}')
        return
    rng = np.random.default_rng(0)
    stab, ptab, _ = scale_tables(T, source, 'pearson', n_boot, rng)
    stab.to_csv(mat / 'corr_species.csv', index=False)
    ptab.to_csv(mat / 'corr_plot.csv', index=False)
    plot_heatmap_pearson(pd.concat([ptab, stab], ignore_index=True), outdir)

    # The same quantity at every scale: mean probability against mean score.
    sp_means = correlate_means(unit_means(pairs), set(stab['species']))
    plot_means = correlate_means(unit_means(pairs.assign(species=PLOT)), {PLOT})
    sp_means.to_csv(mat / 'corr_species_means.csv', index=False)

    crowns = correlate_crowns(pairs, kind='pearson')
    piv, species = crown_pivots(pairs)
    crowns = crowns.merge(own_vs_others(crowns, piv, species, rng), on='tag')
    # A crown Planet never detected has no correlation of its own to compare.
    other = ['r_conspecific', 'r_species_mean', 'r_departure', 'r_other_species']
    crowns.loc[crowns['r'].isna(), other] = np.nan
    crowns = crowns.merge(dp.crown_meta(), on='tag', how='left')
    crowns.to_csv(mat / 'corr_crowns.csv', index=False)

    plot_distribution(crowns, sp_means, plot_means, outdir)
    svc = plot_species_vs_crowns(crowns, sp_means, outdir)
    plot_own_vs_conspecific(crowns, outdir)
    plot_crown_factors(crowns, outdir).to_csv(mat / 'corr_crowns_by_factor.csv',
                                              index=False)

    S, E = piv['score'].values, piv['p_event'].values
    aggs = [aggregation_curve(S, E, rng).assign(species=PLOT)]
    counts = species.value_counts()
    for s in counts.index[counts >= 20]:
        if s not in set(stab['species']):
            continue
        j = np.nonzero(species.values == s)[0]
        aggs.append(aggregation_curve(S[:, j], E[:, j], rng).assign(species=s))
    agg = pd.concat(aggs, ignore_index=True)
    agg.to_csv(mat / 'corr_aggregation.csv', index=False)
    plot_aggregation(agg, outdir)

    def q(v):
        v = pd.Series(v).dropna()
        return {'n': int(len(v)), 'median': float(v.median()),
                'q25': float(v.quantile(0.25)), 'q75': float(v.quantile(0.75)),
                'share_ge_0.5': float((v >= 0.5).mean())}
    both = crowns.dropna(subset=['r', 'r_conspecific'])
    dep = crowns['r_departure'].dropna()
    summary = {
        'source': source,
        'flights': int(pairs['flight'].nunique()),
        'first_flight': str(pairs['flight'].min().date()),
        'last_flight': str(pairs['flight'].max().date()),
        'shares': {
            'plot': {cell_key(ty, sub, 'r'): float(ptab[cell_key(ty, sub, 'r')].iloc[0])
                     for ty, sub in CELLS},
            'plot_ci_event': [float(ptab['r_event_lo'].iloc[0]),
                              float(ptab['r_event_hi'].iloc[0])],
            'species': {cell_key(ty, sub, 'r'): q(stab[cell_key(ty, sub, 'r')])
                        for ty, sub in CELLS}},
        'means': {
            'plot': {ty: float(plot_means[f'r_{ty}'].iloc[0]) for ty in TYPES},
            'species': {ty: q(sp_means[f'r_{ty}']) for ty in TYPES},
            'crown': {'event': q(crowns['r']), 'flower': q(crowns['r_flower']),
                      'decid': q(crowns['r_decid'])}},
        'crowns_tested': int(len(crowns)),
        'crowns_never_detected': int(crowns['r'].isna().sum()),
        'crowns_species': int(crowns.loc[crowns['r'].notna(), 'species'].nunique()),
        'own_vs_conspecific': {
            'n': int(len(both)),
            'own': q(both['r']), 'conspecific': q(both['r_conspecific']),
            'species_mean': q(crowns['r_species_mean']),
            'other_species': q(crowns['r_other_species']),
            'departure': q(dep),
            'share_own_gt_conspecific': float((both['r'] > both['r_conspecific']).mean()),
            'share_departure_gt_0': float((dep > 0).mean())},
        'species_vs_crowns': {
            'n_species': int(len(svc)),
            'share_species_gt_median_crown': float((svc['r_event'] > svc['median']).mean()),
            'median_gain': float((svc['r_event'] - svc['median']).median())},
        'aggregation_species_median': {
            int(m): float(v) for m, v in
            agg[agg['species'] != PLOT].groupby('m')['r'].median().items()},
        'aggregation_n_species': int(agg.loc[agg['species'] != PLOT, 'species'].nunique()),
        'aggregation_plot': {int(m): float(v) for m, v in
                             agg[agg['species'] == PLOT].set_index('m')['r'].items()}}
    (mat / 'corr_scales_summary.json').write_text(json.dumps(summary, indent=1))
    click.echo(json.dumps(summary, indent=1))


# ---------------------------------------------------------- interactive page

def _r(v, nd=3):
    """JSON-safe rounded list (NaN to null)."""
    v = np.asarray(v, float)
    return [None if not np.isfinite(x) else round(float(x), nd) for x in v]


def _dates(v):
    return [str(d)[:10] for d in pd.DatetimeIndex(v)]


def page_data(T, source, n_boot, rng):
    pairs, drone, planet = T['pairs'][source], T['drone'], T['planet'][source]
    stab, ptab, masks = scale_tables(T, source, 'pearson', n_boot, rng)
    tab = pd.concat([ptab, stab], ignore_index=True).set_index('species')
    lo = planet['date'].min()
    seen = drone[(drone['date'] >= lo) & drone['tag'].isin(planet['tag'])]
    units = []
    for unit, (t, mk) in masks.items():
        d = seen if unit == PLOT else seen[seen['species'] == unit]
        g = d.groupby('date')[['flower', 'decid', 'event']].mean()
        cells = []
        for ty, sub in CELLS:
            key = cell_key(ty, sub, 'r')
            m = mk[(ty, sub)]
            v = tab.loc[unit, key]
            if m is None or not np.isfinite(v):
                cells.append(None)
                continue
            cells.append({'r': round(float(v), 3), 'n': int(m.sum()),
                          'ci': _r([tab.loc[unit, f'{key}_lo'],
                                    tab.loc[unit, f'{key}_hi']], 2),
                          'idx': np.nonzero(m)[0].tolist()})
        units.append({
            'name': unit, 'label': unit if unit == PLOT else short(unit),
            'n_crowns': int(tab.loc[unit, 'n_crowns']),
            'flight': _dates(t['flight']), 'n': t['n'].astype(int).tolist(),
            'flower': _r(t['flower'], 5), 'decid': _r(t['decid'], 5),
            'event': _r(t['event'], 5), 'planet': _r(t['planet'], 5),
            'sd': _dates(g.index), 'sflower': _r(g['flower']),
            'sdecid': _r(g['decid']), 'sevent': _r(g['event']),
            'cells': cells})
    # Crowns.
    crowns = correlate_crowns(pairs, kind='pearson')
    crowns = crowns[crowns['r'].notna()]
    ddates = np.sort(seen['date'].unique())
    sdates = np.sort(planet['date'].unique())
    pg = dict(tuple(pairs.groupby('tag')))
    dg = dict(tuple(seen.groupby('tag')))
    sg = dict(tuple(planet.groupby('tag')))
    clist = []
    for _, c in crowns.iterrows():
        tag = int(c['tag'])
        p = pg[tag].sort_values('flight')
        d = dg[tag].sort_values('date')
        s = sg[tag].sort_values('date')
        clist.append({
            'tag': tag, 'sp': c['species'], 'area': round(float(c['area_m2'])),
            'n': int(c['n']), 'r': _r([c['r'], c['r_flower'], c['r_decid']]),
            'f': np.searchsorted(ddates, p['flight'].values).tolist(),
            'pf': (p['p_flower'] * 10000).round().astype(int).tolist(),
            'pd': (p['p_decid'] * 10000).round().astype(int).tolist(),
            'sc': (p['score'] * 10000).round().astype(int).tolist(),
            'ns': p['n_scenes'].astype(int).tolist(),
            'di': np.searchsorted(ddates, d['date'].values).tolist(),
            'df': (d['p_flower'] * 100).round().astype(int).tolist(),
            'dd': (d['p_decid'] * 100).round().astype(int).tolist(),
            'si': np.searchsorted(sdates, s['date'].values).tolist(),
            'ss': (s['score'] * 100).round().astype(int).tolist()})
    return {'cols': [CELL_NAME[c] for c in CELLS], 'types': [c[0] for c in CELLS],
            'units': units, 'ddates': _dates(ddates), 'sdates': _dates(sdates),
            'crowns': clist, 'source': dp.SOURCE_NAME[source],
            'colors': {'flower': C_FLOWER, 'decid': C_DECID, 'event': INK,
                       'planet': C_PLANET, 'grid': GRID, 'ink': INK}}


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Planet–drone correlation explorer</title>
<style>
html,body{height:100%;margin:0}
body{font:14px/1.4 -apple-system,BlinkMacSystemFont,"Helvetica Neue",Arial,sans-serif;
  color:#1a1a1a;background:#fff;display:flex;flex-direction:column}
header{padding:8px 16px;border-bottom:1px solid #e6e6e6;display:flex;gap:18px;
  align-items:center;flex-wrap:wrap}
h1{font-size:16px;margin:0;font-weight:600}
.grp{display:flex;gap:0}
.grp button{font:inherit;font-size:13px;padding:3px 10px;border:1px solid #1a1a1a;
  background:#fff;color:#1a1a1a;cursor:pointer;margin-left:-1px}
.grp button.on{background:#1a1a1a;color:#fff}
.hint{color:#666;font-size:13px}
.wrap{flex:1;display:flex;min-height:0}
#left{flex:0 0 44%;overflow:auto;border-right:1px solid #e6e6e6}
#right{flex:1;display:flex;flex-direction:column;min-width:0}
#info{padding:8px 16px 0;font-size:13.5px;min-height:44px}
#info b{font-weight:600}
#info .pin{color:#2a78d6;font-weight:600}
#scatter{flex:1 1 52%;min-height:0}
#series{flex:1 1 48%;min-height:0}
</style></head><body>
<header>
 <h1>Planet–drone correlation explorer</h1>
 <div class="grp" id="view">
  <button data-v="species" class="on">Plot and species</button>
  <button data-v="trees">Individual trees</button></div>
 <div class="grp" id="type" style="display:none">
  <button data-t="0" class="on">either event</button>
  <button data-t="1">flowering</button>
  <button data-t="2">leaf loss</button></div>
 <span class="hint">Hover a cell to see the points behind its correlation.
  Click to pin it; click again to release.</span>
</header>
<div class="wrap">
 <div id="left"><div id="heat"></div></div>
 <div id="right"><div id="info"></div><div id="scatter"></div><div id="series"></div></div>
</div>
<script>__PLOTLY__</script>
<script>
const D = __DATA__;
const C = D.colors, GRAY = '#b5b5b5';
const TYPE_KEYS = ['event', 'flower', 'decid'];
const TYPE_NAME = {event: 'either event', flower: 'flowering', decid: 'leaf loss'};
const RDBU = [[0,'#67001f'],[0.125,'#b2182b'],[0.25,'#d6604d'],[0.375,'#f4a582'],
  [0.5,'#f7f7f7'],[0.625,'#92c5de'],[0.75,'#4393c3'],[0.875,'#2166ac'],[1,'#053061']];
const CFG = {displayModeBar: false, responsive: true};
let view = 'species', ttype = 0, pinned = null, treeRows = null;

function pearson(x, y) {
  const n = x.length; if (n < 2) return NaN;
  let mx = 0, my = 0; for (let i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
  mx /= n; my /= n; let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) { const a = x[i]-mx, b = y[i]-my; sxy += a*b; sxx += a*a; syy += b*b; }
  return sxy / Math.sqrt(sxx * syy);
}
function fitLine(x, y) {
  const n = x.length; let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
  mx /= n; my /= n; let sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) { sxy += (x[i]-mx)*(y[i]-my); sxx += (x[i]-mx)**2; }
  const b = sxx ? sxy / sxx : 0; return [my - b*mx, b];
}
function axis(title, extra) {
  return Object.assign({title: {text: title, font: {size: 12}}, gridcolor: C.grid,
    zeroline: false, linecolor: C.ink, ticks: 'outside', tickfont: {size: 11}}, extra || {});
}
function fmt(v) { return (v >= 0 ? '' : '−') + Math.abs(v).toFixed(2); }

// ------------------------------------------------------------ heatmaps
function drawSpeciesHeat() {
  const z = D.units.map(u => u.cells.map(c => c ? c.r : null));
  const y = D.units.map(u => `${u.label} (${u.n_crowns})`);
  const text = z.map(r => r.map(v => v === null ? '' : fmt(v).replace('0.', '.')));
  Plotly.newPlot('heat', [{type: 'heatmap', z: z, x: D.cols.map((c, i) => i), y: y,
    text: text, texttemplate: '%{text}', textfont: {size: 10},
    colorscale: RDBU, zmin: -1, zmax: 1, xgap: 1, ygap: 1, hoverinfo: 'none',
    colorbar: {title: {text: 'Pearson r', side: 'right'}, thickness: 10, len: 0.35, y: 1, yanchor: 'top'}}],
   {height: 100 + 19 * y.length, margin: {l: 150, r: 60, t: 80, b: 10},
    xaxis: {side: 'top', tickvals: D.cols.map((c, i) => i),
      ticktext: D.cols.map(c => c.replace(/, /g, ',<br>')),
      tickangle: 0, tickfont: {size: 9.5}, fixedrange: true},
    yaxis: {autorange: 'reversed', tickfont: {size: 11}, fixedrange: true},
    shapes: [2.5, 5.5].map(x => ({type: 'line', x0: x, x1: x, yref: 'paper', y0: 0, y1: 1,
      line: {color: '#fff', width: 4}})).concat([{type: 'line', xref: 'paper', x0: 0, x1: 1,
      y0: 0.5, y1: 0.5, line: {color: '#fff', width: 4}}])}, CFG);
  bind(p => ({kind: 'species', u: p.pointIndex[0], c: p.pointIndex[1]}));
}
function buildTreeRows() {
  const by = {};
  D.crowns.forEach((c, i) => { (by[c.sp] = by[c.sp] || []).push(i); });
  const med = a => { const s = a.slice().sort((p, q) => p - q); return s[Math.floor(s.length / 2)]; };
  treeRows = Object.keys(by).map(sp => {
    const idx = by[sp].sort((a, b) => D.crowns[b].r[0] - D.crowns[a].r[0]);
    return {sp: sp, idx: idx, med: med(idx.map(i => D.crowns[i].r[0]))};
  }).sort((a, b) => b.med - a.med);
}
function drawTreeHeat() {
  if (!treeRows) buildTreeRows();
  const w = Math.max(...treeRows.map(r => r.idx.length));
  const z = treeRows.map(r => { const row = new Array(w).fill(null);
    r.idx.forEach((i, k) => { row[k] = D.crowns[i].r[ttype]; }); return row; });
  const cd = treeRows.map(r => { const row = new Array(w).fill(null);
    r.idx.forEach((i, k) => { row[k] = i; }); return row; });
  const y = treeRows.map(r => { const p = r.sp.split(' ');
    return `${p.length > 1 ? p[0][0] + '. ' + p.slice(1).join(' ') : r.sp} (${r.idx.length})`; });
  Plotly.newPlot('heat', [{type: 'heatmap', z: z, y: y, customdata: cd,
    colorscale: RDBU, zmin: -1, zmax: 1, xgap: 1, ygap: 1, hoverinfo: 'none', hoverongaps: false,
    colorbar: {title: {text: 'Pearson r', side: 'right'}, thickness: 10, len: 0.25, y: 1, yanchor: 'top'}}],
   {height: 90 + 15 * y.length, margin: {l: 150, r: 60, t: 50, b: 10},
    xaxis: {side: 'top', title: {text: 'crowns of the species, ordered by correlation with either event',
      font: {size: 12}}, showticklabels: false, fixedrange: true},
    yaxis: {autorange: 'reversed', tickfont: {size: 10}, fixedrange: true}}, CFG);
  bind(p => p.customdata === null || p.customdata === undefined ? null
       : {kind: 'tree', i: p.customdata});
}
function bind(toSel) {
  const el = document.getElementById('heat');
  el.on('plotly_hover', ev => { if (pinned) return; const s = toSel(ev.points[0]); if (s) show(s); });
  el.on('plotly_click', ev => { const s = toSel(ev.points[0]); if (!s) return;
    const same = pinned && JSON.stringify(pinned) === JSON.stringify(s);
    pinned = same ? null : s; show(s);
    history.replaceState(null, '', pinned ? '#' + (s.kind === 'tree' ? `tree=${D.crowns[s.i].tag}`
      : `cell=${s.u},${s.c}`) : location.pathname); });
}
function mark(sel) {
  let shapes = (document.getElementById('heat').layout.shapes || []).filter(s => !s._sel);
  if (sel && sel.kind === 'species') shapes.push({_sel: 1, type: 'rect', x0: sel.c - 0.5, x1: sel.c + 0.5,
    y0: sel.u - 0.5, y1: sel.u + 0.5, line: {color: pinned ? C.planet : C.ink, width: 2.5}});
  if (sel && sel.kind === 'tree') {
    const row = treeRows.findIndex(r => r.idx.includes(sel.i)), k = treeRows[row].idx.indexOf(sel.i);
    shapes.push({_sel: 1, type: 'rect', x0: k - 0.5, x1: k + 0.5, y0: row - 0.5, y1: row + 0.5,
      line: {color: pinned ? C.planet : C.ink, width: 2.5}});
  }
  Plotly.relayout('heat', {shapes: shapes});
}
function show(sel) { mark(sel); if (sel.kind === 'species') showSpecies(sel.u, sel.c); else showTree(sel.i); }

// ---------------------------------------------------------- right panels
function showSpecies(ui, ci) {
  const U = D.units[ui], cell = U.cells[ci], ty = D.types[ci], col = C[ty];
  const info = document.getElementById('info');
  const name = ui === 0 ? `<b>${U.name}</b>` : `<b><i>${U.name}</i></b>`;
  if (!cell) {
    info.innerHTML = `${name} · ${D.cols[ci]}: not computed (this event type has no season in the ` +
      `drone record, or the subset has fewer than 8 flights or no variation).`;
    Plotly.purge('scatter'); Plotly.purge('series'); return;
  }
  const inc = new Array(U.flight.length).fill(false); cell.idx.forEach(i => { inc[i] = true; });
  const x = U[ty], y = U.planet;
  const pick = (a, f) => a.filter((v, i) => inc[i] === f);
  const xi = pick(x, true), yi = pick(y, true);
  const r = pearson(xi, yi), [a, b] = fitLine(xi, yi);
  const xr = [Math.min(...xi), Math.max(...xi)];
  const ci95 = cell.ci[0] === null ? '' : `, 95% interval ${fmt(cell.ci[0])} to ${fmt(cell.ci[1])}`;
  info.innerHTML = `${name} · ${D.cols[ci]} · <b>r = ${fmt(cell.r)}</b> over ${cell.n} of ` +
    `${U.flight.length} flights${ci95} · median ${U.n_crowns} crowns per flight` +
    (pinned ? ' · <span class="pin">pinned</span>' : '') +
    `<br><span class="hint">Recomputed in the page from the highlighted points: r = ${fmt(r)}.</span>`;
  const ht = '%{text}<br>drone %{x:.3f}<br>Planet %{y:.3f}<extra></extra>';
  Plotly.react('scatter', [
    {x: pick(x, false), y: pick(y, false), text: pick(U.flight, false), mode: 'markers', type: 'scatter',
     name: 'flights not used', hovertemplate: ht,
     marker: {color: 'rgba(0,0,0,0)', size: 7, line: {color: GRAY, width: 1}}},
    {x: xi, y: yi, text: pick(U.flight, true), mode: 'markers', type: 'scatter',
     name: `flights used (${cell.n})`, hovertemplate: ht,
     marker: {color: C.planet, size: 8, opacity: 0.75, line: {color: '#fff', width: 0.5}}},
    {x: xr, y: xr.map(v => a + b * v), mode: 'lines', name: 'least-squares line',
     line: {color: C.ink, width: 1.2}, hoverinfo: 'skip'}],
   {margin: {l: 62, r: 16, t: 26, b: 62}, plot_bgcolor: '#fff',
    xaxis: axis(`drone share of crowns: ${TYPE_NAME[ty]}`, {rangemode: 'tozero'}),
    yaxis: axis('Planet share of clear crown-scenes called an event', {rangemode: 'tozero'}),
    legend: {orientation: 'h', y: 1.08, font: {size: 11}}, hovermode: 'closest'}, CFG);
  const other = ['flower', 'decid', 'event'].filter(k => k !== ty);
  const line = k => ({x: U.sd, y: U['s' + k], mode: 'lines', type: 'scatter',
    name: `drone ${TYPE_NAME[k]}`, line: {color: C[k], width: k === ty ? 2 : 1, dash: k === 'event' ? 'dot' : 'solid'},
    opacity: k === ty ? 1 : 0.55, hovertemplate: '%{x}<br>%{y:.3f}<extra></extra>'});
  const tr = [];
  (ty === 'event' ? ['flower', 'decid', 'event'] : [other[0] === 'event' ? other[1] : other[0], ty]).forEach(k => tr.push(line(k)));
  tr.push({x: pick(U.flight, false), y: pick(y, false), mode: 'markers', type: 'scatter',
    name: 'Planet, flights not used', marker: {color: 'rgba(0,0,0,0)', size: 6, line: {color: GRAY, width: 1}},
    hovertemplate: '%{x}<br>Planet %{y:.3f}<extra></extra>'});
  tr.push({x: pick(U.flight, true), y: xi, mode: 'markers', type: 'scatter',
    name: `drone ${TYPE_NAME[ty]}, flights used`, marker: {color: col, size: 6, symbol: 'diamond'},
    hovertemplate: '%{x}<br>drone %{y:.3f}<extra></extra>'});
  tr.push({x: pick(U.flight, true), y: yi, mode: 'markers', type: 'scatter',
    name: 'Planet, flights used', marker: {color: C.planet, size: 7, line: {color: '#fff', width: 0.5}},
    hovertemplate: '%{x}<br>Planet %{y:.3f}<extra></extra>'});
  Plotly.react('series', tr, {margin: {l: 62, r: 16, t: 26, b: 36}, plot_bgcolor: '#fff',
    xaxis: axis('', {type: 'date'}), yaxis: axis('share of crowns', {rangemode: 'tozero'}),
    legend: {orientation: 'h', y: 1.12, font: {size: 11}}, hovermode: 'closest'}, CFG);
}
function showTree(i) {
  const c = D.crowns[i], k = TYPE_KEYS[ttype], col = C[k];
  const pf = c.pf.map(v => v / 10000), pd = c.pd.map(v => v / 10000), y = c.sc.map(v => v / 10000);
  const x = k === 'flower' ? pf : k === 'decid' ? pd : pf.map((v, j) => 1 - (1 - v) * (1 - pd[j]));
  const fl = c.f.map(j => D.ddates[j]);
  const r = pearson(x, y), [a, b] = fitLine(x, y), xr = [Math.min(...x), Math.max(...x)];
  const rv = c.r[ttype];
  document.getElementById('info').innerHTML = `<b>Tag ${c.tag}</b>, <i>${c.sp}</i> · crown ${c.area} m² · ` +
    `drone ${TYPE_NAME[k]} probability against Planet score · <b>r = ${rv === null ? 'n/a' : fmt(rv)}</b> over ` +
    `${c.n} paired flights (either ${c.r[0] === null ? 'n/a' : fmt(c.r[0])}, flowering ` +
    `${c.r[1] === null ? 'n/a' : fmt(c.r[1])}, leaf loss ${c.r[2] === null ? 'n/a' : fmt(c.r[2])})` +
    (pinned ? ' · <span class="pin">pinned</span>' : '') +
    `<br><span class="hint">Recomputed in the page from the highlighted points: r = ${isFinite(r) ? fmt(r) : 'n/a'}.</span>`;
  Plotly.react('scatter', [
    {x: x, y: y, text: fl, mode: 'markers', type: 'scatter', name: `paired flights (${c.n})`,
     hovertemplate: '%{text}<br>drone %{x:.3f}<br>Planet %{y:.3f}<extra></extra>',
     marker: {color: C.planet, size: 8, opacity: 0.75, line: {color: '#fff', width: 0.5}}},
    {x: xr, y: xr.map(v => a + b * v), mode: 'lines', name: 'least-squares line',
     line: {color: C.ink, width: 1.2}, hoverinfo: 'skip'}],
   {margin: {l: 62, r: 16, t: 26, b: 62}, plot_bgcolor: '#fff',
    xaxis: axis(`drone ${TYPE_NAME[k]} probability of the crown at the flight`, {range: [-0.03, 1.03]}),
    yaxis: axis('Planet score: mean covered fraction, scenes within 10 d', {range: [-0.03, 1.03]}),
    legend: {orientation: 'h', y: 1.08, font: {size: 11}}, hovermode: 'closest'}, CFG);
  const dd = c.di.map(j => D.ddates[j]);
  Plotly.react('series', [
    {x: dd, y: c.df.map(v => v / 100), mode: 'lines', name: 'drone flowering probability',
     line: {color: C.flower, width: k === 'decid' ? 1 : 2}, opacity: k === 'decid' ? 0.55 : 1},
    {x: dd, y: c.dd.map(v => v / 100), mode: 'lines', name: 'drone leaf-loss probability',
     line: {color: C.decid, width: k === 'flower' ? 1 : 2}, opacity: k === 'flower' ? 0.55 : 1},
    {x: c.si.map(j => D.sdates[j]), y: c.ss.map(v => v / 100), mode: 'markers', name: 'Planet, single scenes',
     marker: {color: GRAY, size: 3.5}, hovertemplate: '%{x}<br>Planet %{y:.2f}<extra></extra>'},
    {x: fl, y: x, mode: 'markers', name: `drone ${TYPE_NAME[k]}, paired flights`,
     marker: {color: col, size: 6, symbol: 'diamond'}, hovertemplate: '%{x}<br>drone %{y:.3f}<extra></extra>'},
    {x: fl, y: y, mode: 'markers', name: 'Planet, mean within 10 d of a paired flight',
     marker: {color: C.planet, size: 7, line: {color: '#fff', width: 0.5}},
     hovertemplate: '%{x}<br>Planet %{y:.3f}<extra></extra>'}],
   {margin: {l: 62, r: 16, t: 26, b: 36}, plot_bgcolor: '#fff',
    xaxis: axis('', {type: 'date'}), yaxis: axis('probability or covered fraction', {range: [-0.03, 1.03]}),
    legend: {orientation: 'h', y: 1.14, font: {size: 11}}, hovermode: 'closest'}, CFG);
}

// --------------------------------------------------------------- controls
function setView(v) {
  view = v; pinned = null;
  document.querySelectorAll('#view button').forEach(b => b.classList.toggle('on', b.dataset.v === v));
  document.getElementById('type').style.display = v === 'trees' ? 'flex' : 'none';
  Plotly.purge('heat');
  if (v === 'species') { drawSpeciesHeat(); show({kind: 'species', u: 0, c: 0}); }
  else { drawTreeHeat(); show({kind: 'tree', i: treeRows[0].idx[0]}); }
}
document.querySelectorAll('#view button').forEach(b => b.onclick = () => setView(b.dataset.v));
document.querySelectorAll('#type button').forEach(b => b.onclick = () => {
  ttype = +b.dataset.t;
  document.querySelectorAll('#type button').forEach(o => o.classList.toggle('on', o === b));
  const keep = pinned; Plotly.purge('heat'); drawTreeHeat();
  show(keep || {kind: 'tree', i: treeRows[0].idx[0]});
});
// A pinned cell is kept in the address (#cell=row,column or #tree=tag).
(function () {
  const t = location.hash.match(/^#tree=(\d+)/), c = location.hash.match(/^#cell=(\d+),(\d+)/);
  const i = t ? D.crowns.findIndex(k => k.tag === +t[1]) : -1;
  if (t && i >= 0) { setView('trees'); pinned = {kind: 'tree', i: i}; show(pinned); }
  else if (c && D.units[+c[1]] && +c[2] < D.cols.length) {
    setView('species'); pinned = {kind: 'species', u: +c[1], c: +c[2]}; show(pinned);
  } else setView('species');
})();
</script></body></html>
'''


@cli.command('scales-html')
@click.option('--outdir', default=str(DEFAULT_OUT), show_default=True)
@click.option('--tables', default=str(TABLES), show_default=True)
@click.option('--source', default='mrcnn', show_default=True)
@click.option('--n-boot', default=500, show_default=True)
def scales_html(outdir, tables, source, n_boot):
    """Interactive correlation heatmap with the points behind each cell."""
    from plotly.offline import get_plotlyjs
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    T = build_tables(Path(tables), 10)
    data = page_data(T, source, n_boot, np.random.default_rng(0))
    blob = json.dumps(data, allow_nan=False, separators=(',', ':')).replace('</', '<\\/')
    html = PAGE.replace('__PLOTLY__', get_plotlyjs()).replace('__DATA__', blob)
    path = outdir / 'phenology_corr_interactive.html'
    path.write_text(html)
    click.echo(f'Wrote {path} ({path.stat().st_size / 1e6:.1f} MB; '
               f'{len(data["units"])} units, {len(data["crowns"])} crowns)')


# ------------------------------------------------- timing of single trees

HALF = 150          # days either side of the species peak that make a season
MAX_SHIFT = 120     # largest departure of a tree's peak from the species peak
SHIFTS = np.arange(-MAX_SHIFT, MAX_SHIFT + 1.0)
ESTIMATORS = ('fit', 'centroid', 'onset', 'mid')
EST_NAME = {'fit': 'fitted peak', 'centroid': 'centroid', 'onset': 'onset',
            'mid': 'midpoint of onset and end'}
WEEKLY = 10         # a season is weekly when its median flight spacing is at most this


def species_fits(drone, min_crowns=8, min_events=10):
    """All-years von Mises fit of each species' flowering and leaf-loss
    share, from the whole drone record."""
    rng = np.random.default_rng(0)
    rows = []
    for sp, d in drone.groupby('species'):
        if not sp or d['tag'].nunique() < min_crowns:
            continue
        for ty in ('flower', 'decid'):
            if d[ty].sum() < min_events:
                continue
            row = dp.fit_series(dp.Series(d, ty), 0, rng)
            rows.append({'species': sp, 'type': ty, 'resolved': dp.resolved(row),
                         **{k: row[k] for k in ('n_crowns', 'n_dates', 'events', 'base',
                                                'amp1', 'peak1', 'kappa1', 'fwhm1')}})
    return pd.DataFrame(rows)


def tree_peak(off, p, base, kappa, thresh=0.5):
    """Timing of one tree's event in one season, from its values `p` at
    `off` days from the species peak. None unless the tree has an event
    (p > thresh) that is bracketed by observations without one."""
    order = np.argsort(off)
    off, p = off[order], p[order]
    ev = p > thresh
    if len(off) < 4 or not ev.any():
        return None
    i0, i1 = int(np.argmax(ev)), int(len(ev) - 1 - np.argmax(ev[::-1]))
    if i0 == 0 or i1 == len(ev) - 1:
        return None
    out = {'n_obs': len(off), 'n_event': int(ev.sum()), 'pmax': float(p.max()),
           'spacing': float(np.median(np.diff(off))),
           'gap': float(max(off[i0] - off[i0 - 1], off[i1 + 1] - off[i1]))}
    # Fitted peak: von Mises bump of the species' width on the tree's own
    # baseline; amplitude in closed form at every candidate shift.
    d = p - base
    V = dp.vm(off[None, :], SHIFTS[:, None], kappa)
    a = np.clip((V * d).sum(1) / (V * V).sum(1), 0, 1 - base)
    sse = ((d[None, :] - a[:, None] * V) ** 2).sum(1)
    j = int(sse.argmin())
    ok = 0 < j < len(SHIFTS) - 1 and a[j] >= 0.2
    out.update(fit=SHIFTS[j] if ok else np.nan, amp=float(a[j]))
    w = np.clip(d - 0.1, 0, None) * np.gradient(off)
    out['centroid'] = float((w * off).sum() / w.sum()) if w.sum() > 0 else np.nan
    on = off[i0 - 1] + (thresh - p[i0 - 1]) / (p[i0] - p[i0 - 1]) * (off[i0] - off[i0 - 1])
    end = off[i1] + (p[i1] - thresh) / (p[i1] - p[i1 + 1]) * (off[i1 + 1] - off[i1])
    out.update(onset=float(on), end=float(end), mid=float((on + end) / 2),
               duration=float(end - on))
    return out


def tree_peaks(obs, col, fits, ty, source, thresh=0.5, base_q=0.1):
    """One row per tree and season. `obs` has tag, species, date and `col`;
    `fits` gives each species' all-years peak day and width for `ty`."""
    rows = []
    for sp, d in obs.groupby('species'):
        if sp not in fits.index:
            continue
        mu, kappa = fits.loc[sp, 'peak1'], fits.loc[sp, 'kappa1']
        base = d.groupby('tag')[col].quantile(base_q)
        for y in range(d['date'].dt.year.min() - 1, d['date'].dt.year.max() + 2):
            center = pd.Timestamp(year=y, month=1, day=1) + pd.Timedelta(days=mu - 1)
            off = (d['date'] - center).dt.days.values.astype(float)
            m = np.abs(off) <= HALF
            if m.sum() < 4:
                continue
            dm = d[m].assign(off=off[m])
            mavic = float((dm.drop_duplicates('date')['date'] > dp_PHANTOM_END).mean())
            for tag, t in dm.groupby('tag'):
                r = tree_peak(t['off'].values, t[col].values.astype(float),
                              float(base[tag]), kappa, thresh)
                if r is not None:
                    rows.append({'species': sp, 'type': ty, 'source': source,
                                 'season': center.year, 'center': center.date(),
                                 'tag': tag, 'mavic_frac': mavic, **r})
    return pd.DataFrame(rows)


def add_anomalies(pk, min_trees=5):
    """Each estimator's departure from the median of the species' trees in
    the same season (NaN in seasons with fewer than `min_trees` trees)."""
    keys = ['species', 'type', 'source', 'season']
    for est in ESTIMATORS:
        g = pk.groupby(keys)[est]
        pk[f'{est}_anom'] = np.where(g.transform('count') >= min_trees,
                                     pk[est] - g.transform('median'), np.nan)
    pk['season_spacing'] = pk.groupby(keys)['spacing'].transform('median')
    return pk


def icc1(vals, groups):
    """One-way intraclass correlation (repeatability) for unbalanced groups."""
    _, g = np.unique(groups, return_inverse=True)
    n = np.bincount(g)
    a, N = len(n), len(vals)
    if a < 2 or N <= a:
        return np.nan
    means = np.bincount(g, vals) / n
    msb = (n * (means - vals.mean()) ** 2).sum() / (a - 1)
    msw = ((vals - means[g]) ** 2).sum() / (N - a)
    k0 = (N - (n ** 2).sum() / N) / (a - 1)
    den = msb + (k0 - 1) * msw
    return (msb - msw) / den if den > 0 else np.nan


def repeatability(d, col, rng, n_boot=1000, n_perm=1000, min_trees=8, min_pair=8):
    """Repeatability of `col` among the trees of one species with at least
    two seasons: ICC with a tree-bootstrap interval, a permutation p value
    (values shuffled among trees within each season) and the correlations
    between pairs of seasons."""
    d = d[np.isfinite(d[col])]
    d = d[d.groupby('tag')[col].transform('count') >= 2]
    if d['tag'].nunique() < min_trees:
        return None, None
    v, tags, seas = d[col].values, d['tag'].values, d['season'].values
    icc = icc1(v, tags)
    by = {t: np.nonzero(tags == t)[0] for t in np.unique(tags)}
    keys = list(by)
    boot = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([by[keys[k]] for k in pick])
        grp = np.concatenate([np.full(len(by[keys[k]]), i) for i, k in enumerate(pick)])
        boot.append(icc1(v[idx], grp))
    sidx = [np.nonzero(seas == s)[0] for s in np.unique(seas)]
    perm = []
    for _ in range(n_perm):
        vp = v.copy()
        for i in sidx:
            vp[i] = v[rng.permutation(i)]
        perm.append(icc1(vp, tags))
    perm = np.array(perm)
    piv = d.pivot(index='tag', columns='season', values=col)
    pairs = []
    for i, a in enumerate(piv.columns):
        for b in piv.columns[i + 1:]:
            ok = piv[a].notna() & piv[b].notna()
            if ok.sum() >= min_pair:
                pairs.append({'season_a': a, 'season_b': b, 'n': int(ok.sum()),
                              'r': _corr(piv.loc[ok, a].values, piv.loc[ok, b].values,
                                         'pearson')})
    pairs = pd.DataFrame(pairs)
    raw = col.replace('_anom', '')
    ranks = d.groupby('season')[col].rank(pct=True).values
    row = {'n_trees': int(d['tag'].nunique()), 'n_obs': len(d),
           'n_seasons': int(d['season'].nunique()), 'icc': icc,
           'icc_rank': icc1(ranks, tags),
           'icc_lo': np.nanpercentile(boot, 2.5), 'icc_hi': np.nanpercentile(boot, 97.5),
           'null_hi': np.nanpercentile(perm, 95),
           'p_perm': float((np.sum(perm >= icc) + 1) / (len(perm) + 1)),
           'sd_within_season': float(d.groupby('season')[raw].std().median()),
           'median_gap': float(d['gap'].median()),
           'n_pairs': len(pairs),
           'r_pairs': (float(np.average(pairs['r'], weights=pairs['n']))
                       if len(pairs) and pairs['r'].notna().all() else np.nan)}
    return row, pairs


def repeat_table(pk, rng, subsets):
    rows, allpairs = [], []
    for (sp, ty, src), d in pk.groupby(['species', 'type', 'source']):
        for name, f in subsets.items():
            ds = d[f(d)]
            for est in ESTIMATORS:
                row, pairs = repeatability(ds, f'{est}_anom', rng)
                if row is None:
                    continue
                rows.append({'species': sp, 'type': ty, 'source': src,
                             'subset': name, 'estimator': est, **row})
                if len(pairs):
                    allpairs.append(pairs.assign(species=sp, type=ty, source=src,
                                                 subset=name, estimator=est))
    return (pd.DataFrame(rows),
            pd.concat(allpairs, ignore_index=True) if allpairs else pd.DataFrame())


def pair_rows(pk, col):
    """Every (tree, season a, season b) pair of anomalies, with the camera
    class of the two seasons."""
    d = pk[np.isfinite(pk[col])][['species', 'type', 'tag', 'season', 'mavic_frac', col]]
    m = d.merge(d, on=['species', 'type', 'tag'], suffixes=('_a', '_b'))
    m = m[m['season_a'] < m['season_b']].copy()

    def cam(f):
        return np.select([f <= 0.2, f >= 0.8], ['phantom', 'mavic'], 'mixed')
    ca, cb = cam(m['mavic_frac_a']), cam(m['mavic_frac_b'])
    m['cameras'] = np.where(ca == cb, ca, np.where((ca == 'mixed') | (cb == 'mixed'),
                                                   'mixed', 'phantom to mavic'))
    m.loc[(m['cameras'] == 'phantom'), 'cameras'] = 'phantom to phantom'
    m.loc[(m['cameras'] == 'mavic'), 'cameras'] = 'mavic to mavic'
    return m


def crown_positions():
    """Per tag: easting, northing and crown area from the 2022 crown map."""
    import geopandas as gpd
    g = gpd.read_file(dp.CROWNMAP_2022)
    g['tag'] = pd.to_numeric(g['tag'], errors='coerce')
    g = g[g['tag'].notna() & (g['tag'] != -9999)]
    g = g[~g['tag'].duplicated(keep=False)]
    c = g.geometry.centroid
    return pd.DataFrame({'tag': g['tag'].astype(int).values, 'x': c.x.values,
                         'y': c.y.values, 'crown_area_m2': g.geometry.area.values})


COVARS = [('log_area', 'crown area (log)'), ('illumination', 'illumination index'),
          ('liana', 'liana load'), ('dbh_mm', 'trunk diameter'),
          ('edge_m', 'distance to plot edge'), ('x', 'easting'), ('y', 'northing'),
          ('n_seasons', 'seasons observed'), ('neighbor', 'nearest conspecific\'s timing')]


def covariate_table(pk, col, rng, n_boot=1000):
    """Correlation of each tree's mean timing anomaly with crown properties,
    pooled over species after standardizing both within species."""
    d = pk[np.isfinite(pk[col])]
    d = d[d.groupby(['species', 'type', 'tag'])[col].transform('count') >= 2]
    t = d.groupby(['species', 'type', 'tag'])[col].agg(['mean', 'size']).reset_index()
    t = t.rename(columns={'mean': 'anom', 'size': 'n_seasons'})
    meta = dp.crown_meta().merge(crown_positions(), on='tag', how='outer')
    t = t.merge(meta, on='tag', how='left')
    t['log_area'] = np.log(t['crown_area_m2'])
    t['neighbor'] = np.nan
    for _, g in t.groupby(['species', 'type']):
        g = g.dropna(subset=['x', 'y'])
        if len(g) < 3:
            continue
        dist = np.hypot(g['x'].values[:, None] - g['x'].values[None, :],
                        g['y'].values[:, None] - g['y'].values[None, :])
        np.fill_diagonal(dist, np.inf)
        t.loc[g.index, 'neighbor'] = g['anom'].values[dist.argmin(1)]
        t.loc[g.index, 'neighbor_m'] = dist.min(1)
    grp = t.groupby(['species', 'type'])
    keep = grp['tag'].transform('count') >= 8
    z = t[keep].copy()
    cols = ['anom'] + [c for c, _ in COVARS]
    gz = z.groupby(['species', 'type'])
    for c in cols:
        sd = gz[c].transform('std')
        z[c] = (z[c] - gz[c].transform('mean')) / sd.where(sd > 0)
    rows = []
    for ty in ('flower', 'decid'):
        zt = z[z['type'] == ty]
        for c, label in COVARS:
            ok = zt[['anom', c]].dropna()
            if len(ok) < 20:
                continue
            x, y = ok['anom'].values, ok[c].values
            bs = [_corr(x[i], y[i], 'pearson') for i in
                  (rng.integers(0, len(x), len(x)) for _ in range(n_boot))]
            rows.append({'type': ty, 'covariate': c, 'label': label, 'n': len(ok),
                         'r': _corr(x, y, 'pearson'),
                         'lo': np.nanpercentile(bs, 2.5), 'hi': np.nanpercentile(bs, 97.5)})
    return pd.DataFrame(rows), t


def manual_obs(T):
    """Vicente's labels as per-tree series: flowering (0 or 1) and leaf loss
    (1 - leafing / 100), on the phantom flights."""
    m = dp.manual_table(T)
    sp = T['drone'].drop_duplicates('tag').set_index('tag')['species']
    return pd.DataFrame({
        'tag': m['tag'].values, 'date': pd.to_datetime(m['flight']).values,
        'species': m['tag'].map(sp).values,
        'p_flower': m['manual'].isin(['flowering', 'both']).astype(float).values,
        'p_decid': (1 - m['leafing'] / 100).clip(0, 1).values})


def selftest(rng):
    """Synthetic trees with a fixed timing offset (SD 15 d) plus year-to-year
    noise (SD 10 d): the true repeatability is 15^2 / (15^2 + 10^2) = 0.69."""
    out = []
    for spacing in (7, 30):
        rows = []
        kappa = 8.0
        dates = pd.date_range('2018-01-03', '2025-12-31', freq=f'{spacing}D')
        tree_off = rng.normal(0, 15, 60)
        for tag in range(60):
            for y in range(2018, 2026):
                peak = (pd.Timestamp(year=y, month=1, day=1) + pd.Timedelta(days=79)
                        + pd.Timedelta(days=float(tree_off[tag] + rng.normal(0, 10)
                                                  + 8 * np.sin(y))))
                t = (dates - peak).days.values.astype(float)
                m = np.abs(t) < 200
                p = 0.1 + 0.75 * dp.vm(t[m], 0, kappa) + rng.normal(0, 0.05, m.sum())
                rows.append(pd.DataFrame({'tag': tag, 'species': 'synthetic',
                                          'date': dates[m], 'p': np.clip(p, 0, 1)}))
        obs = pd.concat(rows).groupby(['tag', 'species', 'date'], as_index=False)['p'].max()
        fits = pd.DataFrame({'peak1': [80.0], 'kappa1': [kappa]}, index=['synthetic'])
        pk = add_anomalies(tree_peaks(obs, 'p', fits, 'decid', 'synthetic'))
        for est in ESTIMATORS:
            row, _ = repeatability(pk, f'{est}_anom', rng, 200, 200)
            out.append({'spacing_d': spacing, 'estimator': est, 'true_icc': 0.69,
                        **{k: row[k] for k in ('n_trees', 'n_obs', 'icc', 'icc_lo',
                                               'icc_hi', 'null_hi', 'p_perm')}})
    return pd.DataFrame(out)


# ------------------------------------------------------- timing figures

def _label(sp, ty):
    return f'{short(sp)}, {TYPE_NAME[ty]}'


def plot_repeatability(rt, outdir):
    d = rt[(rt['source'] == 'drone') & (rt['estimator'] == 'fit')]
    a = d[d['subset'] == 'all'].sort_values('icc')
    w = d[d['subset'] == 'weekly'].set_index(['species', 'type'])
    fig, ax = plt.subplots(figsize=(7.4, 0.27 * len(a) + 1.4))
    for y, (_, r) in enumerate(a.iterrows()):
        c = C_TYPE[r['type']]
        ax.plot([r['icc_lo'], r['icc_hi']], [y, y], color=c, lw=1.6)
        ax.plot(r['icc'], y, 'o', color=c, ms=6,
                mfc=c if r['p_perm'] < 0.05 else 'w')
        ax.plot([r['null_hi']] * 2, [y - 0.3, y + 0.3], color='#888', lw=1)
        key = (r['species'], r['type'])
        if key in w.index:
            ax.plot(w.loc[key, 'icc'], y, 'D', color=c, ms=4.5, mfc='none', mew=1)
        ax.text(1.03, y, f'{int(r["n_trees"])} trees, {int(r["n_seasons"])} seasons',
                va='center', fontsize=7, color='#555')
    ax.set_yticks(range(len(a)), [_label(s, t) for s, t in zip(a['species'], a['type'])],
                  fontsize=8, style='italic')
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlim(-0.5, 1.0)
    ax.set_ylim(-0.7, len(a) - 0.3)
    ax.set_xlabel('repeatability of peak timing between years\n(intraclass correlation)')
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker='o', color=C_FLOWER, lw=1.6, label='flowering'),
        Line2D([], [], marker='o', color=C_DECID, lw=1.6, label='leaf loss'),
        Line2D([], [], marker='o', color=INK, mfc='w', lw=0, label='not above shuffled trees'),
        Line2D([], [], marker='D', color=INK, mfc='none', lw=0, ms=4.5,
               label='weekly seasons only'),
        Line2D([], [], marker='|', color='#888', lw=0, ms=9, label='95th percentile, shuffled')],
        fontsize=7, frameon=False, loc='lower right')
    style(ax)
    save(fig, outdir, 'tree_timing_repeatability')


def plot_year_pairs(pk, rt, prs, outdir, n=6):
    d = rt[(rt['source'] == 'drone') & (rt['estimator'] == 'fit') & (rt['subset'] == 'all')]
    top = d.sort_values('n_trees', ascending=False).head(n)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.2))
    for ax, (_, r) in zip(axes.ravel(), top.iterrows()):
        p = prs[(prs['species'] == r['species']) & (prs['type'] == r['type'])
                & (prs['source'] == 'drone') & (prs['estimator'] == 'fit')
                & (prs['subset'] == 'all')]
        k = pk[(pk['species'] == r['species']) & (pk['type'] == r['type'])
               & (pk['source'] == 'drone')]
        piv = k.pivot(index='tag', columns='season', values='fit_anom')
        lim = 0
        for (_, q), col in zip(p.sort_values('n', ascending=False).head(3).iterrows(),
                               (INK, C_PLANET, '#b04fc4')):
            x, y = piv[q['season_a']], piv[q['season_b']]
            ax.plot(x, y, 'o', ms=4, color=col, alpha=0.55, mew=0,
                    label=f'{int(q["season_a"])} and {int(q["season_b"])}: '
                          f'r = {q["r"]:.2f} (n = {int(q["n"])})')
            lim = max(lim, np.nanmax(np.abs(np.r_[x[y.notna()], y[x.notna()]])))
        lim = np.ceil((lim + 5) / 10) * 10
        ax.plot([-lim, lim], [-lim, lim], color=INK, lw=0.6)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_title(f'{_label(r["species"], r["type"])}\nrepeatability {r["icc"]:.2f}',
                     fontsize=9, style='italic', loc='left')
        ax.legend(fontsize=6.5, frameon=False, loc='upper left')
        style(ax)
    for ax in axes[-1]:
        ax.set_xlabel('days from the species median, earlier season')
    for ax in axes[:, 0]:
        ax.set_ylabel('days from the species median, later season')
    fig.tight_layout()
    save(fig, outdir, 'tree_timing_year_pairs')
    return top


def plot_raster(pk, examples, outdir):
    fig, axes = plt.subplots(1, len(examples), figsize=(4.3 * len(examples), 6.2))
    cmap = plt.get_cmap('viridis')
    for ax, (sp, ty) in zip(np.atleast_1d(axes), examples):
        k = pk[(pk['species'] == sp) & (pk['type'] == ty) & (pk['source'] == 'drone')
               & pk['fit_anom'].notna()]
        k = k[k.groupby('tag')['fit_anom'].transform('count') >= 2]
        order = k.groupby('tag')['fit_anom'].mean().sort_values().index
        pos = {t: i for i, t in enumerate(order)}
        seasons = sorted(k['season'].unique())
        for t, g in k.groupby('tag'):
            ax.plot([g['fit_anom'].min(), g['fit_anom'].max()], [pos[t]] * 2,
                    color='#bbb', lw=0.8, zorder=1)
            ax.plot(g['fit_anom'].mean(), pos[t], '|', color=INK, ms=7, zorder=3)
        for i, s in enumerate(seasons):
            g = k[k['season'] == s]
            ax.scatter(g['fit_anom'], g['tag'].map(pos), s=12, zorder=2,
                       color=cmap(i / max(len(seasons) - 1, 1)), label=str(s), lw=0)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_yticks([])
        ax.set_ylim(-1, len(order))
        ax.set_xlabel('peak date, days from the species\nmedian of that season')
        ax.set_title(f'{_label(sp, ty)} ({len(order)} trees)', fontsize=9,
                     style='italic', loc='left')
        ax.legend(fontsize=6.5, frameon=False, ncol=2, loc='lower right',
                  handletextpad=0.1, columnspacing=0.6)
        style(ax)
    np.atleast_1d(axes)[0].set_ylabel('trees, ordered by mean departure')
    fig.tight_layout()
    save(fig, outdir, 'tree_timing_raster')


def plot_cross_source(m, rt, a, b, xlabel, ylabel, outdir, name):
    """Anomalies of the same tree-seasons from two sources, and the
    repeatability each source gives per species."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))
    ax = axes[0]
    lim = 100
    for ty in ('decid', 'flower'):
        d = m[m['type'] == ty]
        r = _corr(d[f'fit_anom_{a}'].values, d[f'fit_anom_{b}'].values, 'pearson')
        ax.plot(d[f'fit_anom_{a}'], d[f'fit_anom_{b}'], 'o', ms=2.5, color=C_TYPE[ty],
                alpha=0.35, mew=0, label=f'{TYPE_NAME[ty]}: r = {r:.2f} (n = {len(d)})')
    ax.plot([-lim, lim], [-lim, lim], color=INK, lw=0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    style(ax)
    ax = axes[1]
    d = rt[(rt['estimator'] == 'fit') & (rt['subset'] == 'common')]
    pv = d.pivot_table(index=['species', 'type'], columns='source', values='icc')
    if a in pv and b in pv:
        pv = pv.dropna(subset=[a, b])
        for ty in ('decid', 'flower'):
            q = pv[pv.index.get_level_values('type') == ty]
            ax.plot(q[a], q[b], 'o', ms=6, color=C_TYPE[ty], alpha=0.7, mew=0)
        for (sp, ty), q in pv.iterrows():
            ax.annotate(short(sp), (q[a], q[b]), fontsize=6.5, style='italic',
                        xytext=(3, 3), textcoords='offset points')
    ax.plot([-0.5, 1], [-0.5, 1], color=INK, lw=0.6)
    ax.axhline(0, color=INK, lw=0.6)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_xlim(-0.5, 1)
    ax.set_ylim(-0.5, 1)
    ax.set_aspect('equal')
    ax.set_xlabel(f'repeatability, {xlabel.split(",")[0]}')
    ax.set_ylabel(f'repeatability, {ylabel.split(",")[0]}')
    style(ax)
    fig.tight_layout()
    save(fig, outdir, name)


def plot_covariates(ct, outdir):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    labels = [l for _, l in COVARS]
    for k, ty in enumerate(('flower', 'decid')):
        d = ct[ct['type'] == ty].set_index('label').reindex(labels)
        y = np.arange(len(labels)) + (0.17 if k == 0 else -0.17)
        ax.errorbar(d['r'], y, xerr=[d['r'] - d['lo'], d['hi'] - d['r']], fmt='o',
                    color=C_TYPE[ty], ms=5, lw=1.4,
                    label=f'{TYPE_NAME[ty]} (n = {int(d["n"].max())} trees)')
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel('correlation with a tree\'s mean departure in peak date\n'
                  '(positive: later than conspecifics)')
    ax.legend(fontsize=8, frameon=False, loc='lower right')
    style(ax)
    save(fig, outdir, 'tree_timing_covariates')


@cli.command()
@click.option('--outdir', default=str(DEFAULT_OUT), show_default=True)
@click.option('--tables', default=str(TABLES), show_default=True)
@click.option('--self-test', 'self_test', is_flag=True,
              help='Run the estimators on synthetic trees of known repeatability.')
def repeat(outdir, tables, self_test):
    """Peak date per tree and season, and its repeatability between years."""
    outdir = Path(outdir)
    mat = outdir / 'materials'
    mat.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    if self_test:
        st = selftest(rng)
        st.to_csv(mat / 'tree_timing_selftest.csv', index=False)
        click.echo(st.round(3).to_string(index=False))
        return
    T = build_tables(Path(tables), 10)
    drone = T['drone']
    drone = drone[drone['species'].astype(bool)]
    fits_path = mat / 'tree_species_fits.csv'
    if fits_path.exists():
        sf = pd.read_csv(fits_path)
    else:
        sf = species_fits(drone)
        sf.to_csv(fits_path, index=False)
    click.echo(f'species fits: {len(sf)}, resolved {int(sf["resolved"].sum())}')
    planet = T['planet']['mrcnn'].groupby(['tag', 'species', 'date'],
                                          as_index=False)['score'].mean()
    manual = manual_obs(T)
    frames = []
    for ty in ('flower', 'decid'):
        f = sf[(sf['type'] == ty) & sf['resolved']].set_index('species')
        col = PROB[ty]
        frames.append(tree_peaks(drone, col, f, ty, 'drone'))
        frames.append(tree_peaks(manual, col, f, ty, 'manual',
                                 base_q=0.1 if ty == 'decid' else 0.0))
        frames.append(tree_peaks(planet, 'score', f, ty, 'planet', thresh=dp.POS_FRAC))
    pk = add_anomalies(pd.concat(frames, ignore_index=True))
    pk.to_csv(mat / 'tree_peaks.csv', index=False)
    click.echo(pk.groupby(['source', 'type']).agg(
        rows=('tag', 'size'), trees=('tag', 'nunique'), species=('species', 'nunique'),
        fitted=('fit_anom', 'count')).to_string())

    # Tree-seasons seen by two sources, for like-for-like repeatability.
    key = ['species', 'type', 'season', 'tag']
    dr = pk[pk['source'] == 'drone']
    cross = {}
    for src in ('planet', 'manual'):
        o = pk[pk['source'] == src]
        m = dr.merge(o, on=key, suffixes=('_drone', f'_{src}'))
        cross[src] = m.dropna(subset=['fit_anom_drone', f'fit_anom_{src}'])
        cross[src].to_csv(mat / f'tree_peaks_drone_vs_{src}.csv', index=False)
    subsets = {'all': lambda d: np.ones(len(d), bool),
               'weekly': lambda d: (d['season_spacing'] <= WEEKLY).values}
    rt, prs = repeat_table(pk, rng, subsets)
    # The same tree-seasons in both sources.
    extra = []
    for src, m in cross.items():
        idx = pd.MultiIndex.from_frame(m[key])
        both = pk[pk['source'].isin(['drone', src])
                  & pd.MultiIndex.from_frame(pk[key]).isin(idx)]
        r2, _ = repeat_table(both, rng, {'common': lambda d: np.ones(len(d), bool)})
        extra.append(r2.assign(common_with=src))
    rt = pd.concat([rt] + extra, ignore_index=True)
    rt.to_csv(mat / 'tree_repeatability.csv', index=False)
    prs.to_csv(mat / 'tree_repeatability_pairs.csv', index=False)

    main = rt[(rt['source'] == 'drone') & (rt['estimator'] == 'fit')
              & (rt['subset'] == 'all')]
    click.echo(main.sort_values('icc', ascending=False).round(3).to_string(index=False))
    plot_repeatability(rt, outdir)
    plot_year_pairs(pk, rt, prs, outdir)
    ex = []
    for sp, ty in (('Handroanthus guayacan', 'decid'), ('Jacaranda copaia', 'flower')):
        if ((main['species'] == sp) & (main['type'] == ty)).any():
            ex.append((sp, ty))
    best = main[(main['n_trees'] >= 15)].sort_values('icc', ascending=False)
    for _, r in best.iterrows():
        if (r['species'], r['type']) not in ex and len(ex) < 3:
            ex.append((r['species'], r['type']))
    plot_raster(pk, ex, outdir)
    for src, xl, yl, name in (
            ('planet', 'drone, days from the species median',
             'Planet, days from the species median', 'tree_timing_planet_vs_drone'),
            ('manual', 'drone classifier, days from the species median',
             'manual labels, days from the species median', 'tree_timing_manual_check')):
        plot_cross_source(cross[src], rt[rt.get('common_with') == src], 'drone', src,
                          xl, yl, outdir, name)
    ct, trees = covariate_table(dr, 'fit_anom', rng)
    ct.to_csv(mat / 'tree_timing_covariates.csv', index=False)
    trees.to_csv(mat / 'tree_mean_anomaly.csv', index=False)
    plot_covariates(ct, outdir)

    pr = pair_rows(dr, 'fit_anom')
    cams = []
    for (ty, cam), g in pr.groupby(['type', 'cameras']):
        cams.append({'type': ty, 'cameras': cam, 'n_pairs': len(g),
                     'n_trees': int(g['tag'].nunique()),
                     'r': _corr(g['fit_anom_a'].values, g['fit_anom_b'].values, 'pearson')})
    cams = pd.DataFrame(cams)
    cams.to_csv(mat / 'tree_timing_camera_pairs.csv', index=False)
    click.echo(cams.round(3).to_string(index=False))

    def med(d, c='icc'):
        return float(d[c].median()) if len(d) else None
    summary = {'species_fits': int(len(sf)), 'species_fits_resolved': int(sf['resolved'].sum())}
    for ty in ('flower', 'decid'):
        m = main[main['type'] == ty]
        wk = rt[(rt['source'] == 'drone') & (rt['estimator'] == 'fit')
                & (rt['subset'] == 'weekly') & (rt['type'] == ty)]
        summary[ty] = {
            'n_species': int(len(m)), 'median_icc': med(m),
            'n_above_null': int((m['p_perm'] < 0.05).sum()),
            'median_icc_rank': med(m, 'icc_rank'),
            'n_trees': int(m['n_trees'].sum()),
            'median_sd_within_season_d': med(m, 'sd_within_season'),
            'median_gap_d': med(m, 'median_gap'),
            'weekly_n_species': int(len(wk)), 'weekly_median_icc': med(wk),
            'weekly_n_above_null': int((wk['p_perm'] < 0.05).sum()),
            'by_estimator_median_icc': {
                est: med(rt[(rt['source'] == 'drone') & (rt['estimator'] == est)
                            & (rt['subset'] == 'all') & (rt['type'] == ty)])
                for est in ESTIMATORS}}
        for src, mm in cross.items():
            d = mm[mm['type'] == ty]
            summary[ty][f'drone_vs_{src}'] = {
                'n': int(len(d)), 'trees': int(d['tag'].nunique()),
                'species': int(d['species'].nunique()),
                'r': float(_corr(d['fit_anom_drone'].values,
                                 d[f'fit_anom_{src}'].values, 'pearson')) if len(d) >= 8 else None}
    (mat / 'tree_timing_summary.json').write_text(json.dumps(summary, indent=1))
    click.echo(json.dumps(summary, indent=1))


if __name__ == '__main__':
    cli()
