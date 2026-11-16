#!/usr/bin/env python
"""Do the Planet event series lead to the same phenology as the drone series?

The drone classifier gives every crown a flowering and a deciduous (leaf
loss) probability at each flight. The Planet models predict one merged
"event" (flowering and/or leafless), per crown and scene. This script puts
both on a common footing per species and asks how well they agree:

  series     build the matched tables (drone flights paired with the Planet
             scenes within a few days) and save them under OUTDIR/materials
  correlate  correlation of the Planet event share with the drone flowering,
             leaf-loss and union shares, per species, with the merged Planet
             signal decomposed by event type
  fit        phenology curve fits to both and comparison of their parameters
  evaluate   Planet skill by crown size and other crown properties

Planet sources:
  mrcnn      temporal Mask R-CNN apply run (crown_timeseries.csv), every
             coregistered scene
  mrcnn_env  the same, only scenes within 30 days of a training chip
  pse        crown PSE-LTAE temporal CV; every scene is taken from the fold
             whose training flights are farthest from it (held out)
"""
import glob
import json
import os
from pathlib import Path

import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import optimize, stats  # noqa: E402

from plot_crown_species_comparison import (  # noqa: E402
    APPLY, NC4S, load_drone)

FLOWER = Path('/Volumes/Earth03/flower')
DEFAULT_OUT = FLOWER / 'figs' / '20261117_updates'
PSE_CV = FLOWER / 'crown_series_20261029' / 'cv'
PX_M2 = 0.5625            # area of a 0.75 m apply-run pixel
MIN_AREA = 36.0           # m^2, the 64 px training minimum
MIN_CLEAR = 0.5
POS_FRAC = 0.25           # Mask R-CNN crown event: covered fraction
PSE_THRESH = 0.5
DRY_MONTHS = (12, 1, 2, 3, 4)
TYPES = ('flower', 'decid', 'event')
TYPE_NAME = {'flower': 'flowering', 'decid': 'leaf loss', 'event': 'either'}
SOURCES = ('mrcnn', 'mrcnn_env', 'pse')
SOURCE_NAME = {'mrcnn': 'Mask R-CNN', 'mrcnn_env': 'Mask R-CNN, ≤ 30 d of training',
               'pse': 'PSE-LTAE (held out)'}
INK, GRID = '#1a1a1a', '#e6e6e6'
C_FLOWER, C_DECID, C_PLANET = '#eb6834', '#1baf7a', '#2a78d6'


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, outdir, name):
    fig.savefig(outdir / f'{name}.png', dpi=200, bbox_inches='tight')
    fig.savefig(outdir / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    click.echo(f'Wrote {outdir / name}.png')


def write_caption(outdir, name, caption, notes=''):
    text = (f'# {name}\n\nMade by `scripts/drone_planet_phenology.py`.\n\n'
            f'## Caption\n\n{caption}\n')
    if notes:
        text += f'\n## Notes\n\n{notes}\n'
    (outdir / f'{name}_caption.md').write_text(text)


# ---------------------------------------------------------------- loading

def load_drone_crowns(nc4s=NC4S):
    """Drone crown-flights with flower / decid / event calls."""
    d = load_drone(nc4s, parts=True)
    d = d[d['tag'] != -9999].copy()
    d['flower'] = d['p_flower'] > 0.5
    d['decid'] = d['p_decid'] > 0.5
    d['p_event'] = 1 - (1 - d['p_flower']) * (1 - d['p_decid'])
    return d


def load_mrcnn(apply_dir=APPLY):
    """Mask R-CNN crown-scenes: clear, coregistered, one row per (scene, tag).
    Columns tag, scene, date, area_m2, score (covered fraction), event, env."""
    c = pd.read_csv(f'{apply_dir}/crown_timeseries.csv', parse_dates=['date'],
                    usecols=['scene', 'date', 'in_envelope', 'coreg_ok', 'tag',
                             'crown_px', 'crown_clear_frac', 'pos_frac',
                             'days_to_nearest_train'])
    c = c[(c['tag'] != -9999) & c['coreg_ok']
          & (c['crown_clear_frac'] >= MIN_CLEAR)]
    shared = c.loc[c.duplicated(['scene', 'tag'], keep=False), 'tag'].unique()
    c = c[~c['tag'].isin(shared)]
    return pd.DataFrame({
        'tag': c['tag'].values, 'scene': c['scene'].values,
        'date': c['date'].values, 'area_m2': c['crown_px'].values * PX_M2,
        'score': c['pos_frac'].fillna(0).values,
        'event': (c['pos_frac'].fillna(0) >= POS_FRAC).values,
        'env': c['in_envelope'].values.astype(bool),
        'days_train': c['days_to_nearest_train'].values})


def load_pse(cv_dir=PSE_CV):
    """PSE-LTAE crown-scenes, each scene from the fold whose training flights
    are farthest from it, so no prediction comes from a model that saw a
    label within `days_train` days."""
    runs = sorted(glob.glob(str(cv_dir / 'fold*' / 'predictions.csv')))
    frames, dist = [], []
    for k, path in enumerate(runs):
        with open(os.path.join(os.path.dirname(path), 'params.json')) as fh:
            train = np.array(json.load(fh)['train_flights'], dtype='M8[D]')
        p = pd.read_csv(path, parse_dates=['date'],
                        usecols=['scene', 'date', 'tag', 'area_m2',
                                 'clear_frac', 'coreg_ok', 'p_event'])
        sc = p[['scene', 'date']].drop_duplicates('scene')
        d = np.abs(sc['date'].values.astype('M8[D]')[:, None]
                   - train[None, :]).min(axis=1).astype(int)
        dist.append(pd.DataFrame({'scene': sc['scene'].values, 'fold': k,
                                  'days_train': d}))
        p['fold'] = k
        frames.append(p)
    dist = pd.concat(dist)
    best = dist.sort_values('days_train').drop_duplicates('scene', keep='last')
    p = pd.concat(frames).merge(best, on=['scene', 'fold'])
    p = p[(p['tag'] != -9999) & p['coreg_ok'] & (p['clear_frac'] >= MIN_CLEAR)]
    shared = p.loc[p.duplicated(['scene', 'tag'], keep=False), 'tag'].unique()
    p = p[~p['tag'].isin(shared)]
    return pd.DataFrame({
        'tag': p['tag'].values, 'scene': p['scene'].values,
        'date': p['date'].values, 'area_m2': p['area_m2'].values,
        'score': p['p_event'].values,
        'event': (p['p_event'] >= PSE_THRESH).values,
        'env': True, 'days_train': p['days_train'].values})


def load_planet(source):
    if source == 'pse':
        return load_pse()
    p = load_mrcnn()
    return p[p['env']] if source == 'mrcnn_env' else p


def pair_with_flights(planet, flights, window):
    """Every Planet crown-scene joined to each drone flight within `window`
    days of it, then reduced to one row per (tag, flight): the number of
    clear scenes, the mean score and whether any scene was an event."""
    sc = planet[['scene', 'date']].drop_duplicates('scene')
    f = np.array(sorted(flights), dtype='M8[ns]')
    dd = (sc['date'].values[:, None] - f[None, :]) / np.timedelta64(1, 'D')
    i, j = np.nonzero(np.abs(dd) <= window)
    link = pd.DataFrame({'scene': sc['scene'].values[i], 'flight': f[j]})
    m = planet.merge(link, on='scene')
    g = m.groupby(['tag', 'flight'])
    return pd.DataFrame({'n_scenes': g.size(), 'score': g['score'].mean(),
                         'k': g['event'].sum(),
                         'env': g['env'].mean()}).reset_index()


def build_tables(outdir, window=10, rebuild=False):
    """Matched crown and species tables for every Planet source (cached)."""
    mat = outdir / 'materials'
    mat.mkdir(parents=True, exist_ok=True)
    cache = mat / f'phenology_tables_w{window}.pkl'
    if cache.exists() and not rebuild:
        return pd.read_pickle(cache)
    drone = load_drone_crowns()
    out = {'drone': drone, 'planet': {}, 'pairs': {}}
    for src in SOURCES:
        p = load_planet(src)
        p = p[(p['area_m2'] >= MIN_AREA) & p['tag'].isin(drone['tag'])]
        sp = drone.drop_duplicates('tag').set_index('tag')['species']
        p = p.assign(species=p['tag'].map(sp).values)
        lo, hi = p['date'].min(), p['date'].max()
        flights = drone.loc[drone['date'].between(
            lo - pd.Timedelta(days=window), hi + pd.Timedelta(days=window)),
            'date'].unique()
        pr = pair_with_flights(p, flights, window)
        pr = pr.merge(drone.rename(columns={'date': 'flight'}),
                      on=['tag', 'flight'])
        pr['area_m2'] = pr['tag'].map(p.groupby('tag')['area_m2'].first())
        out['planet'][src], out['pairs'][src] = p, pr
        click.echo(f'{src}: {len(p):,} clear crown-scenes on '
                   f'{p["scene"].nunique()} scenes; {len(pr):,} crown-flights '
                   f'paired on {pr["flight"].nunique()} flights')
    pd.to_pickle(out, cache)
    return out


def species_pairs(pairs, min_crowns=5):
    """Per species and flight: the drone shares over the crowns Planet saw,
    and the Planet share of clear crown-scenes that are events."""
    g = pairs.groupby(['species', 'flight'])
    t = pd.DataFrame({
        'n': g.size(), 'flower': g['flower'].mean(), 'decid': g['decid'].mean(),
        'event': g['event'].mean(),
        'planet': g['k'].sum() / g['n_scenes'].sum(),
        'planet_any': g['k'].apply(lambda k: (k > 0).mean()),
        'score': g['score'].mean()}).reset_index()
    t = t[t['n'] >= min_crowns]
    t['month'] = t['flight'].dt.month
    t['dry'] = t['month'].isin(DRY_MONTHS)
    return t


# ------------------------------------------------------------ correlation

def event_windows(drone, species, pad=1, frac=0.25, min_events=10):
    """Months in which each event type happens in the drone record: the
    months whose mean share is at least `frac` of the peak month's, padded
    by `pad` months on each side. None when the species has fewer than
    `min_events` crown-flights of that type."""
    d = drone[drone['species'] == species]
    out = {}
    for t in ('flower', 'decid'):
        if d[t].sum() < min_events:
            out[t] = None
            continue
        by = d.groupby([d['date'].dt.year, d['date'].dt.month])[t].mean()
        clim = by.groupby(level=1).mean().reindex(range(1, 13), fill_value=0)
        core = set(clim.index[clim >= frac * clim.max()])
        months = set()
        for m in core:
            for k in range(-pad, pad + 1):
                months.add((m - 1 + k) % 12 + 1)
        out[t] = {'core': sorted(core), 'months': sorted(months)}
    return out


def _corr(x, y, kind='spearman'):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 8 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return (stats.spearmanr(x, y)[0] if kind == 'spearman'
            else stats.pearsonr(x, y)[0])


def boot_ci(t, xcol, ycol, n_boot, rng, kind='spearman'):
    """95% interval of a correlation, resampling year-months of flights."""
    blocks = (t['flight'].dt.year * 12 + t['flight'].dt.month).values
    ids = np.unique(blocks)
    idx = {b: np.nonzero(blocks == b)[0] for b in ids}
    x, y = t[xcol].values, t[ycol].values
    vals = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[b] for b in rng.choice(ids, len(ids))])
        vals.append(_corr(x[pick], y[pick], kind))
    vals = np.array(vals)
    if np.isfinite(vals).mean() < 0.8:
        return np.nan, np.nan
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def anomaly(t, col):
    """Departure from the month-of-year mean of the paired flights."""
    return t[col] - t.groupby('month')[col].transform('mean')


def partial_spearman(x, y, z):
    """Spearman correlation of x and y after removing rank-linear z."""
    r = [stats.rankdata(v) for v in (x, y, z)]
    if np.ptp(r[2]) == 0:
        return _corr(x, y)
    res = [v - np.polyval(np.polyfit(r[2], v, 1), r[2]) for v in r[:2]]
    return _corr(res[0], res[1], 'pearson')


def correlate_species(sp_pairs, drone, n_boot, rng):
    """One row per species: correlations of the Planet share with each drone
    share, overall and in subsets, and the event-type decomposition."""
    rows = []
    for species, t in sp_pairs.groupby('species'):
        if len(t) < 20 or t['event'].sum() == 0:
            continue
        row = {'species': species, 'n_flights': len(t),
               'n_crowns': int(t['n'].median()),
               'drone_event_share': t['event'].mean(),
               'planet_event_share': t['planet'].mean()}
        y = t['planet'].values
        for ty in TYPES:
            row[f'rho_{ty}'] = _corr(t[ty].values, y)
            row[f'r_{ty}'] = _corr(t[ty].values, y, 'pearson')
            lo, hi = boot_ci(t, ty, 'planet', n_boot, rng)
            row[f'rho_{ty}_lo'], row[f'rho_{ty}_hi'] = lo, hi
            for name, sub in (('dry', t[t['dry']]), ('wet', t[~t['dry']])):
                row[f'rho_{ty}_{name}'] = _corr(sub[ty].values,
                                                sub['planet'].values)
        row['r_anom_event'] = _corr(anomaly(t, 'event').values,
                                    anomaly(t, 'planet').values, 'pearson')
        # Event windows from the whole drone record of the species.
        win = event_windows(drone, species)
        other = {'flower': 'decid', 'decid': 'flower'}
        for ty in ('flower', 'decid'):
            w = win[ty]
            row[f'months_{ty}'] = '' if w is None else ' '.join(map(str, w['core']))
            row[f'rho_{ty}_window'] = row[f'rho_{ty}_pure'] = np.nan
            row[f'n_{ty}_window'] = 0
            if w is None:
                continue
            sub = t[t['month'].isin(w['months'])]
            row[f'n_{ty}_window'] = len(sub)
            row[f'rho_{ty}_window'] = _corr(sub[ty].values, sub['planet'].values)
            wo = win[other[ty]]
            if wo is not None:
                sub = sub[~sub['month'].isin(wo['core'])]
            row[f'rho_{ty}_pure'] = _corr(sub[ty].values, sub['planet'].values)
        if win['flower'] and win['decid']:
            a, b = set(win['flower']['core']), set(win['decid']['core'])
            row['window_overlap'] = len(a & b) / len(a | b)
        else:
            row['window_overlap'] = np.nan
        # Non-negative regression planet ~ a flower + b decid + c.
        A = np.column_stack([t['flower'], t['decid'], np.ones(len(t))])
        coef, _ = optimize.nnls(A, y)
        ss = ((y - y.mean()) ** 2).sum()
        row.update(a_flower=coef[0], b_decid=coef[1], intercept=coef[2],
                   r2_nnls=1 - ((y - A @ coef) ** 2).sum() / ss if ss else np.nan)
        row['prho_flower'] = partial_spearman(t['flower'].values, y, t['decid'].values)
        row['prho_decid'] = partial_spearman(t['decid'].values, y, t['flower'].values)
        rows.append(row)
    return pd.DataFrame(rows).sort_values('rho_event', ascending=False)


def cross_species(sp_pairs, species):
    """Spearman of Planet species A (rows) with drone species B (columns)
    over the flights both were observed on."""
    pl = sp_pairs.pivot(index='flight', columns='species', values='planet')[species]
    dr = sp_pairs.pivot(index='flight', columns='species', values='event')[species]
    m = np.full((len(species), len(species)), np.nan)
    for i, a in enumerate(species):
        for j, b in enumerate(species):
            m[i, j] = _corr(pl[a].values, dr[b].values)
    return pd.DataFrame(m, index=species, columns=species)


def correlate_crowns(pairs, min_pairs=15, min_events=3):
    """Per crown: Spearman of the Planet score with the drone event
    probability over its paired flights."""
    rows = []
    for tag, t in pairs.groupby('tag'):
        if len(t) < min_pairs or t['event'].sum() < min_events:
            continue
        rows.append({
            'tag': tag, 'species': t['species'].iloc[0],
            'area_m2': t['area_m2'].iloc[0], 'n': len(t),
            'n_events': int(t['event'].sum()),
            'n_flower': int(t['flower'].sum()), 'n_decid': int(t['decid'].sum()),
            'rho': _corr(t['p_event'].values, t['score'].values),
            'rho_flower': _corr(t['p_flower'].values, t['score'].values),
            'rho_decid': _corr(t['p_decid'].values, t['score'].values)})
    return pd.DataFrame(rows)


def short(name):
    g, _, s = name.partition(' ')
    return f'{g[0]}. {s}' if s else name


def plot_heatmap(tabs, outdir):
    cols = [('rho_event', 'either\nall'), ('rho_event_dry', 'either\ndry'),
            ('rho_event_wet', 'either\nwet'), ('r_anom_event', 'either\nanomaly'),
            ('rho_flower', 'flowering\nall'), ('rho_flower_window', 'flowering\nin season'),
            ('rho_flower_pure', 'flowering\nin season,\nno leaf loss'),
            ('rho_decid', 'leaf loss\nall'), ('rho_decid_window', 'leaf loss\nin season'),
            ('rho_decid_pure', 'leaf loss\nin season,\nno flowering')]
    order = tabs['mrcnn']['species'].tolist()
    srcs = ['mrcnn', 'pse']
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.27 * len(order) + 2.2),
                             sharey=True)
    for ax, src in zip(axes, srcs):
        t = tabs[src].set_index('species').reindex(order)
        m = t[[c for c, _ in cols]].values.astype(float)
        im = ax.imshow(m, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        for (i, j), v in np.ndenumerate(m):
            if np.isfinite(v):
                ax.text(j, i, f'{v:.2f}'.replace('0.', '.'), ha='center',
                        va='center', fontsize=6.5,
                        color='w' if abs(v) > 0.6 else INK)
        ax.xaxis.tick_top()
        ax.set_xticks(range(len(cols)), [n.replace('\n', ' ') for _, n in cols],
                      fontsize=7.5, rotation=50, ha='left')
        ax.set_yticks(range(len(order)),
                      [f'{short(s)} ({int(n)})' for s, n in
                       zip(order, tabs['mrcnn'].set_index('species')['n_crowns'])],
                      fontsize=7.5, style='italic')
        ax.set_title(SOURCE_NAME[src], fontsize=10, pad=95)
        for x in (3.5, 6.5):
            ax.axvline(x, color='w', lw=3)
    fig.colorbar(im, ax=axes, shrink=0.5, pad=0.01,
                 label='correlation of Planet with drone share')
    save(fig, outdir, 'phenology_corr_species')


def plot_cross(cross, outdir):
    sp = cross.index.tolist()
    fig, ax = plt.subplots(figsize=(0.33 * len(sp) + 3, 0.33 * len(sp) + 2.5))
    im = ax.imshow(cross.values, cmap='RdBu', vmin=-1, vmax=1)
    best = np.nanargmax(np.nan_to_num(cross.values, nan=-2), axis=1)
    for i, j in enumerate(best):
        ax.plot(j, i, 'o', ms=4, mfc='none', mec=INK)
    ax.set_xticks(range(len(sp)), [short(s) for s in sp], rotation=90,
                  fontsize=7.5, style='italic')
    ax.set_yticks(range(len(sp)), [short(s) for s in sp], fontsize=7.5,
                  style='italic')
    ax.set_xlabel('drone species')
    ax.set_ylabel('Planet species')
    fig.colorbar(im, ax=ax, shrink=0.6, label='Spearman correlation')
    save(fig, outdir, 'phenology_corr_cross_species')


def plot_examples(sp_pairs, planet, drone, species, outdir, name):
    fig, axes = plt.subplots(len(species), 1, figsize=(12.5, 1.9 * len(species)),
                             sharex=True)
    lo = planet['date'].min()
    for ax, s in zip(np.atleast_1d(axes), species):
        d = drone[(drone['species'] == s) & (drone['date'] >= lo)
                  & drone['tag'].isin(planet['tag'])]
        g = d.groupby('date')
        ax.plot(g['flower'].mean(), color=C_FLOWER, lw=1.3, label='drone flowering')
        ax.plot(g['decid'].mean(), color=C_DECID, lw=1.3, label='drone leaf loss')
        t = sp_pairs[sp_pairs['species'] == s]
        ax.plot(t['flight'], t['planet'], 'o', ms=3.5, color=C_PLANET,
                label='Planet event (scenes within 10 d of the flight)')
        style(ax)
        ax.set_ylabel('share of crowns', fontsize=8)
        ax.set_title(f'{s} ({int(t["n"].median())} crowns)', fontsize=9,
                     style='italic', loc='left')
    np.atleast_1d(axes)[0].legend(fontsize=8, ncol=3, frameon=False,
                                  loc='upper right')
    save(fig, outdir, name)


@click.group()
def cli():
    pass


@cli.command()
@click.option('--outdir', default=str(DEFAULT_OUT), show_default=True)
@click.option('--window', default=10, show_default=True,
              help='Days between a Planet scene and a drone flight to pair.')
def series(outdir, window):
    """Build and cache the matched tables."""
    build_tables(Path(outdir), window, rebuild=True)


@cli.command()
@click.option('--outdir', default=str(DEFAULT_OUT), show_default=True)
@click.option('--window', default=10, show_default=True)
@click.option('--n-boot', default=500, show_default=True)
@click.option('--min-crowns', default=5, show_default=True,
              help='Minimum crowns of a species seen by Planet at a flight.')
def correlate(outdir, window, n_boot, min_crowns):
    """Correlate the Planet and drone species series."""
    outdir = Path(outdir)
    mat = outdir / 'materials'
    T = build_tables(outdir, window)
    rng = np.random.default_rng(0)
    tabs, sps = {}, {}
    for src in SOURCES:
        sps[src] = species_pairs(T['pairs'][src], min_crowns)
        tabs[src] = correlate_species(sps[src], T['drone'], n_boot, rng)
        tabs[src].to_csv(mat / f'phenology_corr_species_{src}.csv', index=False)
        sps[src].to_csv(mat / f'phenology_species_pairs_{src}.csv', index=False)
        correlate_crowns(T['pairs'][src]).to_csv(
            mat / f'phenology_corr_crowns_{src}.csv', index=False)
    # Sensitivity of the headline correlation to the pairing window.
    sens = []
    for w in (5, 10, 15):
        Tw = build_tables(outdir, w)
        for src in ('mrcnn', 'pse'):
            sp_w = species_pairs(Tw['pairs'][src], min_crowns)
            for s, t in sp_w.groupby('species'):
                if len(t) >= 20:
                    sens.append({'window': w, 'source': src, 'species': s,
                                 'n_flights': len(t),
                                 'rho_event': _corr(t['event'].values,
                                                    t['planet'].values)})
    pd.DataFrame(sens).to_csv(mat / 'phenology_corr_window_sensitivity.csv',
                              index=False)

    main = tabs['mrcnn']
    keep = main[main['n_crowns'] >= 8]['species'].tolist()
    for src in SOURCES:
        tabs[src] = tabs[src][tabs[src]['species'].isin(keep)]
    plot_heatmap(tabs, outdir)
    cross = cross_species(sps['mrcnn'], keep)
    cross.to_csv(mat / 'phenology_corr_cross_species.csv')
    plot_cross(cross, outdir)
    v = cross.values
    rank_row = [(np.sum(v[i] > v[i, i]) + 1) for i in range(len(keep))
                if np.isfinite(v[i, i])]
    click.echo(f'cross-species: matched species ranks first for '
               f'{sum(r == 1 for r in rank_row)} of {len(rank_row)} Planet '
               f'series; median rank {np.median(rank_row):.0f} of {len(keep)}')
    ok = main[main['species'].isin(keep) & main['rho_event'].notna()]
    plot_examples(sps['mrcnn'], T['planet']['mrcnn'], T['drone'],
                  ok['species'].head(4).tolist(), outdir,
                  'phenology_series_best')
    # The species with the most drone flowering: where a flowering signal
    # in Planet would be easiest to see.
    sp = sps['mrcnn'][sps['mrcnn']['species'].isin(keep)]
    nfl = (sp['flower'] * sp['n']).groupby(sp['species']).sum()
    plot_examples(sps['mrcnn'], T['planet']['mrcnn'], T['drone'],
                  nfl.sort_values(ascending=False).index[:5].tolist(), outdir,
                  'phenology_series_flowering')
    summary = {}
    for src in SOURCES:
        t = tabs[src]
        summary[src] = {
            'n_species': int(len(t)),
            'median_rho': {ty: float(t[f'rho_{ty}'].median()) for ty in TYPES},
            'n_rho_event_ge_0.5': int((t['rho_event'] >= 0.5).sum()),
            'n_rho_flower_ge_0.5': int((t['rho_flower'] >= 0.5).sum()),
            'n_rho_decid_ge_0.5': int((t['rho_decid'] >= 0.5).sum()),
            'n_flower_window_tested': int(t['rho_flower_window'].notna().sum()),
            'n_flower_window_ge_0.5': int((t['rho_flower_window'] >= 0.5).sum()),
            'n_ci_excludes_0': int((t['rho_event_lo'] > 0).sum())}
    summary['cross_species'] = {
        'first': int(sum(r == 1 for r in rank_row)), 'of': len(rank_row),
        'median_rank': float(np.median(rank_row))}
    (mat / 'phenology_corr_summary.json').write_text(json.dumps(summary, indent=1))
    click.echo(json.dumps(summary, indent=1))


if __name__ == '__main__':
    cli()
