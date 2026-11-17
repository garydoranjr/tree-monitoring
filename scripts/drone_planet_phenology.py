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
            ('rho_event_wet', 'either\nwet'),
            ('rho_flower', 'flowering\nall'), ('rho_flower_window', 'flowering\nin season'),
            ('rho_flower_pure', 'flowering\nin season,\nno leaf loss'),
            ('rho_decid', 'leaf loss\nall'), ('rho_decid_window', 'leaf loss\nin season'),
            ('rho_decid_pure', 'leaf loss\nin season,\nno flowering')]
    t = tabs['mrcnn']
    m = t[[c for c, _ in cols]].values.astype(float).T
    fig, ax = plt.subplots(figsize=(0.235 * len(t) + 2.6, 6.2))
    im = ax.imshow(m, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
    for (i, j), v in np.ndenumerate(m):
        if np.isfinite(v):
            ax.text(j, i, f'{v:.2f}'.replace('0.', '.'), ha='center', va='center',
                    fontsize=5.5, rotation=90, color='w' if abs(v) > 0.6 else INK)
    ax.set_yticks(range(len(cols)), [n.replace('\n', ' ') for _, n in cols],
                  fontsize=8)
    ax.set_xticks(range(len(t)), [f'{short(s)} ({int(n)})' for s, n in
                                  zip(t['species'], t['n_crowns'])],
                  fontsize=7.5, style='italic', rotation=60, ha='right',
                  rotation_mode='anchor')
    ax.tick_params(length=0)
    for y in (2.5, 5.5):
        ax.axhline(y, color='w', lw=3)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01,
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


# -------------------------------------------------------------- curve fits

YEAR = 365.25
PHEN_CSV = FLOWER / 'data' / 'phenology.csv'


def circ_diff(a, b):
    """a - b in days on the annual circle, in [-182.6, 182.6)."""
    return (np.asarray(a, float) - b + YEAR / 2) % YEAR - YEAR / 2


def vm(t, mu, kappa):
    return np.exp(kappa * (np.cos(2 * np.pi * (t - mu) / YEAR) - 1))


def fwhm_days(kappa):
    """Width at half maximum of a von Mises bump (a full year if it never
    falls to half)."""
    c = 1 + np.log(0.5) / kappa
    return YEAR if c <= -1 else 2 * np.arccos(c) / (2 * np.pi) * YEAR


def _unpack(x, ncomp):
    base = 1 / (1 + np.exp(-x[0]))
    comps, room = [], 1 - base
    for i in range(ncomp):
        amp = room / (1 + np.exp(-x[1 + 3 * i]))
        comps.append((amp, x[2 + 3 * i] % YEAR, np.exp(x[3 + 3 * i])))
        room -= amp
    return base, comps


def curve(t, base, comps):
    return base + sum(a * vm(t, m, k) for a, m, k in comps)


def _nll(x, t, k, n, w, ncomp):
    base, comps = _unpack(x, ncomp)
    if any(not 0.7 <= c[2] <= 300 for c in comps):
        return 1e12
    p = np.clip(curve(t, base, comps), 1e-6, 1 - 1e-6)
    return -np.sum(w * (k * np.log(p) + (n - k) * np.log(1 - p)))


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_curve(t, k, n, w, ncomp=1, start=None):
    """Binomial maximum-likelihood fit of a baseline plus `ncomp` von Mises
    bumps to k events of n crowns at days of year t, with weights w.
    Returns (x, nll)."""
    rate = max(k.sum() / max(n.sum(), 1), 1e-3)
    starts = [start] if start is not None else []
    if start is None and ncomp == 1:
        for mu in np.arange(15, 365, 30.4):
            for kap in (2, 8, 30):
                starts.append([_logit(rate / 4), _logit(min(4 * rate, 0.9)),
                               mu, np.log(kap)])
    best = (None, np.inf)
    for x0 in starts:
        r = optimize.minimize(_nll, x0, args=(t, k, n, w, ncomp),
                              method='Nelder-Mead',
                              options={'xatol': 1e-3, 'fatol': 1e-6,
                                       'maxiter': 2000})
        if r.fun < best[1]:
            best = (r.x, r.fun)
    if best[0] is not None:   # polish: Nelder-Mead can stall on a ridge
        r = optimize.minimize(_nll, best[0], args=(t, k, n, w, ncomp),
                              method='Powell', options={'xtol': 1e-4, 'ftol': 1e-9})
        if r.fun < best[1]:
            best = (r.x, r.fun)
    return best


def fit_two(t, k, n, w, x1):
    """Two-bump fit started from the one-bump fit plus a second bump at
    each month in turn."""
    base, ((a, mu, kap),) = _unpack(x1, 1)
    best = (None, np.inf)
    for mu2 in np.arange(15, 365, 30.4):
        if abs(circ_diff(mu2, mu)) < 30:
            continue
        x0 = [x1[0], _logit(a / (1 - base)), mu, np.log(kap),
              _logit(0.3 * a / max(1 - base - a, 1e-3)), mu2, np.log(8)]
        r = fit_curve(t, k, n, w, 2, x0)
        if r[1] < best[1]:
            best = r
    return best


MIX_TYPES = ('flower', 'decid')


def _match(comps, peaks):
    """Order of the two components that puts each nearest its peak in
    `peaks` (summed circular distance)."""
    def cost(order):
        return sum(abs(circ_diff(comps[i][1], p)) for i, p in zip(order, peaks))
    return min(((0, 1), (1, 0)), key=cost)


def fit_drone_init(s, x1, drone_rows, n_boot, rng):
    """Two-bump fit of a merged Planet event series with one bump started
    at each drone event type's fit (flowering, leaf loss), the peaks free to
    move after that. Each fitted bump is labelled by the drone type whose
    peak it lies nearest, with a crown-bootstrap interval on its peak date,
    and the AIC difference against the one-bump fit `x1`."""
    k, n = s.counts()
    nll1 = _nll(x1, s.t, k, n, s.w, 1)
    base, ((amp, _, _),) = _unpack(x1, 1)
    drone = [_unpack(r['_x1'], 1)[1][0] for r in drone_rows]
    peaks = [c[1] for c in drone]
    best = (None, np.inf)
    for split in (0.5, 0.8, 0.2):
        a1, a2 = split * amp, (1 - split) * amp
        x0 = [x1[0], _logit(a1 / (1 - base))]
        x0 += [drone[0][1], np.log(np.clip(drone[0][2], 1, 200))]
        x0 += [_logit(a2 / max(1 - base - a1, 1e-3))]
        x0 += [drone[1][1], np.log(np.clip(drone[1][2], 1, 200))]
        r = fit_curve(s.t, k, n, s.w, 2, x0)
        if r[1] < best[1]:
            best = r
    xm, nll2 = best
    mbase, comps = _unpack(xm, 2)
    comps = [comps[i] for i in _match(comps, peaks)]
    out = {'mix_base': mbase, 'daic_mix': 2 * (nll1 - nll2) - 2 * 3, '_xmix': xm}
    for ty, (a, mu, kap) in zip(MIX_TYPES, comps):
        out.update({f'mix_{ty}_amp': a, f'mix_{ty}_peak': mu,
                    f'mix_{ty}_kappa': kap, f'mix_{ty}_fwhm': fwhm_days(kap),
                    f'mix_{ty}_peak_obs': _peak_obs(s, mu)})
    boot = []
    for _ in range(n_boot):
        tagw = np.bincount(rng.integers(0, len(s.tags), len(s.tags)),
                           minlength=len(s.tags)).astype(float)
        kb, nb = s.counts(tagw)
        ok = nb > 0
        if kb.sum() == 0:
            continue
        xb, _ = fit_curve(s.t[ok], kb[ok], nb[ok], s.w[ok], 2, xm)
        cb = _unpack(xb, 2)[1]
        boot.append([cb[i][1] for i in _match(cb, [c[1] for c in comps])])
    for i, ty in enumerate(MIX_TYPES):
        if len(boot) >= 0.8 * n_boot and n_boot:
            d = circ_diff([b[i] for b in boot], out[f'mix_{ty}_peak'])
            out[f'mix_{ty}_peak_lo'], out[f'mix_{ty}_peak_hi'] = np.percentile(d, [2.5, 97.5])
        else:
            out[f'mix_{ty}_peak_lo'] = out[f'mix_{ty}_peak_hi'] = np.nan
    return out


class Series:
    """Crown-level event observations of one species from one source, kept
    per crown so that crowns can be resampled."""

    def __init__(self, df, col):
        df = df[['tag', 'date', col]].dropna()
        self.dates = np.sort(df['date'].unique())
        self.tags = np.sort(df['tag'].unique())
        self.di = np.searchsorted(self.dates, df['date'].values)
        self.ti = np.searchsorted(self.tags, df['tag'].values)
        self.ev = df[col].values.astype(float)
        d = pd.DatetimeIndex(self.dates)
        self.t = d.dayofyear.values.astype(float)
        ym = d.year * 12 + d.month
        w = 1 / pd.Series(ym).map(pd.Series(ym).value_counts()).values
        self.w = w / w.mean()   # every year-month counts equally
        self.year, self.month = d.year.values, d.month.values

    def counts(self, tagw=None):
        w = np.ones(len(self.ti)) if tagw is None else tagw[self.ti]
        n = np.bincount(self.di, w, len(self.dates))
        k = np.bincount(self.di, w * self.ev, len(self.dates))
        return k, n


def describe(x, ncomp=1):
    base, comps = _unpack(x, ncomp)
    comps = sorted(comps, key=lambda c: -c[0])
    out = {'base': base}
    for i, (a, mu, kap) in enumerate(comps, 1):
        out.update({f'amp{i}': a, f'peak{i}': mu, f'kappa{i}': kap,
                    f'fwhm{i}': fwhm_days(kap)})
    return out


def fit_series(s, n_boot, rng, two=False):
    """Fit one series: one bump (with a crown-bootstrap interval on the
    peak date), and optionally two bumps with the AIC difference."""
    k, n = s.counts()
    x1, nll1 = fit_curve(s.t, k, n, s.w)
    row = describe(x1)
    row.update(n_dates=len(s.dates), n_crowns=len(s.tags), events=k.sum(),
               obs=n.sum(), peak_month_obs=_peak_obs(s, row['peak1']))
    peaks = []
    for _ in range(n_boot):
        tagw = np.bincount(rng.integers(0, len(s.tags), len(s.tags)),
                           minlength=len(s.tags)).astype(float)
        kb, nb = s.counts(tagw)
        ok = nb > 0
        if kb.sum() == 0:
            continue
        xb, _ = fit_curve(s.t[ok], kb[ok], nb[ok], s.w[ok], 1, x1)
        peaks.append(_unpack(xb, 1)[1][0][1])
    if len(peaks) >= 0.8 * n_boot and n_boot:
        d = circ_diff(peaks, row['peak1'])
        row['peak1_lo'], row['peak1_hi'] = np.percentile(d, [2.5, 97.5])
    else:
        row['peak1_lo'] = row['peak1_hi'] = np.nan
    if two:
        x2, nll2 = fit_two(s.t, k, n, s.w, x1)
        row['daic2'] = 2 * (nll1 - nll2) - 2 * 3   # > 0 favors two bumps
        if x2 is not None:
            for key, v in describe(x2, 2).items():
                row[f'two_{key}'] = v
    row['_x1'] = x1
    return row


def _peak_obs(s, peak):
    """Number of observation dates within 30 days of the fitted peak."""
    return int((np.abs(circ_diff(s.t, peak)) <= 30).sum())


def resolved(row, min_amp=0.03, min_events=10):
    """A fit that describes a real seasonal peak: enough events, an
    amplitude above `min_amp` and above the baseline, a season shorter
    between 10 days and half a year (a narrower bump is a single flight's
    spike at the width bound), and observations near the peak."""
    return bool(row['events'] >= min_events and row['amp1'] >= min_amp
                and row['amp1'] >= 2 * row['base'] and 10 <= row['fwhm1'] <= 183
                and row['peak_month_obs'] >= 3)


def mixed(row):
    """Whether the drone-initialized two-bump fit is selected (by AIC)."""
    return bool(row is not None and row.get('daic_mix', -np.inf) > 0)


def resolved_comp(row, ty, min_amp=0.03, min_events=10):
    """`resolved` for the `ty` bump of a selected two-bump fit."""
    if not mixed(row):
        return False
    a = row[f'mix_{ty}_amp']
    return bool(row['events'] >= min_events and a >= min_amp
                and a >= 2 * row['mix_base'] and 10 <= row[f'mix_{ty}_fwhm'] <= 183
                and row[f'mix_{ty}_peak_obs'] >= 3)


def yearly_peaks(s, x1, half=120, min_near=3):
    """Peak date and amplitude in each year, holding the baseline and the
    width of the all-years fit: one row per year with observations on both
    sides of the peak."""
    base, ((amp, mu, kap),) = _unpack(x1, 1)
    k, n = s.counts()
    d = pd.DatetimeIndex(s.dates)
    rows = []
    for y in range(d.year.min() - 1, d.year.max() + 1):
        center = pd.Timestamp(year=y, month=1, day=1) + pd.Timedelta(days=mu - 1)
        off = (d - center).days.values.astype(float)
        m = np.abs(off) <= half
        if ((off[m] < 0).sum() < min_near or (off[m] > 0).sum() < min_near
                or (np.abs(off) <= 30).sum() < 2 or k[m].sum() < 3):
            continue

        def nll(x):
            a = 1 / (1 + np.exp(-x[0]))
            if abs(x[1]) > half:
                return 1e12
            p = np.clip(base + a * (1 - base) * vm(off[m], x[1], kap), 1e-6, 1 - 1e-6)
            return -np.sum(k[m] * np.log(p) + (n[m] - k[m]) * np.log(1 - p))
        best = min((optimize.minimize(nll, [_logit(amp), x0], method='Nelder-Mead')
                    for x0 in (-45, 0, 45)), key=lambda r: r.fun)
        rows.append({'year': y, 'shift': best.x[1],
                     'amp': (1 - base) / (1 + np.exp(-best.x[0])),
                     'n_dates': int(m.sum())})
    return rows


def trap_peaks():
    """Litter-trap flowering and fruiting peak day of year per species, from
    the AIC-best circular model of phenology.csv (mean in radians)."""
    from plot_ocm_observability import taxa
    p = pd.read_csv(PHEN_CSV)
    p = p[p['deltaAIC'] == 0].copy()
    p['species'] = p['sp'].str.upper().map(taxa())
    p['peak'] = (p['mean1'] % (2 * np.pi)) / (2 * np.pi) * YEAR
    p['fwhm'] = p['kappa1'].clip(lower=0.7).map(fwhm_days)
    return p.dropna(subset=['species']).pivot_table(
        index='species', columns='type', values=['peak', 'fwhm', 'kappa1'],
        aggfunc='first')


def monthly(s):
    """Year-month-balanced share by calendar month, for plotting."""
    k, n = s.counts()
    df = pd.DataFrame({'y': s.year, 'm': s.month, 'k': k, 'n': n})
    ym = df.groupby(['y', 'm']).sum()
    share = (ym['k'] / ym['n']).groupby(level='m').mean()
    return share.reindex(range(1, 13))


def plot_fit_curves(fits, series, species, outdir):
    n = len(species)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12.5, 2.3 * nrow), sharex=True)
    tt = np.arange(1, 366.0)
    mid = np.array([15, 46, 75, 106, 136, 167, 197, 228, 259, 289, 320, 350])
    spec = [('flower', C_FLOWER, 'drone flowering'), ('decid', C_DECID, 'drone leaf loss'),
            ('mrcnn', C_PLANET, 'Planet event (Mask R-CNN)')]
    for ax, sp in zip(axes.ravel(), species):
        for key, col, lab in spec:
            row = fits.get((sp, key))
            if row is None:
                continue
            ax.plot(mid, monthly(series[(sp, key)]).values, 'o', ms=3.5, color=col,
                    alpha=0.6)
            if key == 'mrcnn' and mixed(row):
                base, comps = _unpack(row['_xmix'], 2)
                ok = any(resolved_comp(row, ty) for ty in MIX_TYPES)
                ax.plot(tt, curve(tt, base, comps), color=col, lw=1.6,
                        ls='-' if ok else ':', label=lab)
                for ty, tcol in zip(MIX_TYPES, (C_FLOWER, C_DECID)):
                    a, mu = row[f'mix_{ty}_amp'], row[f'mix_{ty}_peak']
                    one = [(a, mu, row[f'mix_{ty}_kappa'])]
                    ax.plot(tt, curve(tt, base, one), color=col, lw=0.9, ls='--',
                            alpha=0.8, label='Planet bump (drone-initialized)'
                            if ty == 'flower' else None)
                    ax.plot([mu], [base + a], 'v', ms=5, color=tcol, mec=col, mew=0.6)
                continue
            base, comps = _unpack(row['_x1'], 1)
            ax.plot(tt, curve(tt, base, comps), color=col, lw=1.6,
                    ls='-' if resolved(row) else ':', label=lab)
        style(ax)
        ax.set_title(sp, fontsize=9, style='italic', loc='left')
        ax.set_xticks(mid, list('JFMAMJJASOND'), fontsize=8)
        ax.set_ylim(bottom=0)
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    for ax in axes[:, 0]:
        ax.set_ylabel('share of crowns', fontsize=8)
    handles = {}
    for ax in axes.ravel()[:n]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    axes.ravel()[0].legend(handles.values(), handles.keys(), fontsize=7, frameon=False)
    save(fig, outdir, 'phenology_fit_curves')


def plot_peak_scatter(cmp, outdir):
    panels = [('event', 'drone either event'), ('decid', 'drone leaf loss'),
              ('flower', 'drone flowering')]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    months = pd.date_range('2001-01-01', periods=12, freq='MS').dayofyear.values
    for ax, (ty, title) in zip(axes, panels):
        c = cmp[(cmp['type'] == ty) & cmp['both_resolved']]
        ax.plot([0, 366], [0, 0], color=INK, lw=0.8)
        ax.axhspan(-21, 21, color=GRID, alpha=0.6, lw=0)
        for src, mk, col in (('mrcnn', 'o', C_PLANET),):
            d = c[c['source'] == src]
            ax.errorbar(d['drone_peak'], d['dpeak'],
                        yerr=[-d['planet_lo'].clip(upper=0), d['planet_hi'].clip(lower=0)],
                        fmt=mk, ms=5, color=col, lw=0.7, capsize=0,
                        label=f'{SOURCE_NAME[src]} (n = {len(d)})')
        for r in c[(c['source'] == 'mrcnn') & (c['dpeak'].abs() > 21)].itertuples():
            ax.annotate(short(r.species), (r.drone_peak, r.dpeak), fontsize=6.5,
                        style='italic', xytext=(3, 3), textcoords='offset points')
        style(ax)
        ax.set_xticks(months, list('JFMAMJJASOND'))
        ax.set_xlim(0, 366)
        ax.set_ylim(-185, 185)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f'{title.replace("drone ", "")} peak date in the drone record')
        ax.legend(fontsize=7.5, frameon=False, loc='lower right')
    axes[0].set_ylabel('Planet peak date − drone peak date (days)')
    save(fig, outdir, 'phenology_fit_peaks')


def plot_year_shifts(ys, outdir):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    lim = 90
    ax.plot([-lim, lim], [-lim, lim], color=INK, lw=0.8)
    for sp, d in ys.groupby('species'):
        ax.plot(d['shift_drone'], d['shift_planet'], 'o', ms=4.5, alpha=0.8,
                label=short(sp))
    style(ax)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("drone: year's peak − all-years peak (days)")
    ax.set_ylabel("Planet: year's peak − all-years peak (days)")
    ax.legend(fontsize=6.5, frameon=False, ncol=2, prop={'style': 'italic', 'size': 6.5})
    save(fig, outdir, 'phenology_fit_year_shifts')


@cli.command()
@click.option('--outdir', default=str(DEFAULT_OUT), show_default=True)
@click.option('--n-boot', default=100, show_default=True)
@click.option('--min-crowns', default=8, show_default=True)
@click.option('--tolerance', default=21, show_default=True,
              help='Days within which two peak dates count as the same '
                   '(the median phantom flight interval).')
@click.option('--refit', is_flag=True, help='Ignore the cached fits.')
def fit(outdir, n_boot, min_crowns, tolerance, refit):
    """Fit phenology curves to the drone and Planet series and compare."""
    outdir = Path(outdir)
    mat = outdir / 'materials'
    T = build_tables(outdir, 10)
    rng = np.random.default_rng(0)
    drone = T['drone']
    planet = {src: T['planet'][src] for src in ('mrcnn', 'pse')}
    # Common crowns and period: crowns the Mask R-CNN run observed, from the
    # first Planet scene on.
    tags = set(planet['mrcnn']['tag'])
    start = planet['mrcnn']['date'].min()
    drone = drone[drone['tag'].isin(tags) & (drone['date'] >= start)]
    ncr = drone.groupby('species')['tag'].nunique()
    species = ncr.index[ncr >= min_crowns]
    fits, series = {}, {}
    cache = mat / f'phenology_fits_b{n_boot}.pkl'
    if cache.exists() and not refit:
        fits, series = pd.read_pickle(cache)
    for sp in ([] if fits else species):
        d = drone[drone['species'] == sp]
        srcs = {ty: Series(d, ty) for ty in TYPES}
        for src in planet:
            p = planet[src]
            srcs[src] = Series(p[(p['species'] == sp) & p['tag'].isin(tags)], 'event')
        for key, s in srcs.items():
            if s.counts()[0].sum() < 3:
                continue
            row = fit_series(s, n_boot, rng, two=key in ('event', 'mrcnn', 'pse'))
            row.update(species=sp, series=key)
            fits[(sp, key)], series[(sp, key)] = row, s
    pd.to_pickle((fits, series), cache)
    # Planet two-bump fits started from the drone flowering and leaf-loss
    # fits, where the drone resolves both seasons.
    mcache = mat / f'phenology_mix_fits_b{n_boot}.pkl'
    if mcache.exists() and not refit:
        mix = pd.read_pickle(mcache)
    else:
        mix = {}
        for sp in species:
            dr = [fits.get((sp, ty)) for ty in MIX_TYPES]
            if not all(r is not None and resolved(r) for r in dr):
                continue
            for src in planet:
                if (sp, src) in fits:
                    mix[(sp, src)] = fit_drone_init(
                        series[(sp, src)], fits[(sp, src)]['_x1'], dr, n_boot, rng)
        pd.to_pickle(mix, mcache)
    rows = []
    for key, row in fits.items():
        row.update(mix.get(key, {}))
        row['resolved'] = resolved(row)
        if key in mix:
            row['mix_selected'] = mixed(row)
            for ty in MIX_TYPES:
                row[f'mix_{ty}_resolved'] = resolved_comp(row, ty)
        rows.append({k: v for k, v in row.items() if not k.startswith('_')})
    ft = pd.DataFrame(rows)
    first = ['species', 'series', 'resolved', 'n_crowns', 'n_dates', 'events',
             'base', 'amp1', 'peak1', 'peak1_lo', 'peak1_hi', 'fwhm1']
    ft[first + [c for c in ft.columns if c not in first]].to_csv(
        mat / 'phenology_fits.csv', index=False)

    trap = trap_peaks()
    cmp_rows = []
    for sp in species:
        for src in planet:
            pr = fits.get((sp, src))
            if pr is None:
                continue
            for ty in TYPES:
                dr = fits.get((sp, ty))
                if dr is None:
                    continue
                # The flowering and leaf-loss rows compare against the matched
                # bump of the drone-initialized two-bump fit when AIC selects it.
                if ty in MIX_TYPES and mixed(pr):
                    model, pres = 'two_drone_init', resolved_comp(pr, ty)
                    pp = {f: pr[f'mix_{ty}_{f}']
                          for f in ('peak', 'peak_lo', 'peak_hi', 'fwhm', 'amp')}
                else:
                    model, pres = 'one', resolved(pr)
                    pp = {'peak': pr['peak1'], 'peak_lo': pr['peak1_lo'],
                          'peak_hi': pr['peak1_hi'], 'fwhm': pr['fwhm1'],
                          'amp': pr['amp1']}
                both = pres and resolved(dr)
                dpk = float(circ_diff(pp['peak'], dr['peak1']))
                lo = np.hypot(pp['peak_lo'], dr['peak1_hi'])
                hi = np.hypot(pp['peak_hi'], dr['peak1_lo'])
                cmp_rows.append({
                    'species': sp, 'source': src, 'type': ty, 'planet_model': model,
                    'drone_resolved': resolved(dr), 'planet_resolved': pres,
                    'both_resolved': both, 'drone_peak': dr['peak1'],
                    'planet_peak': pp['peak'], 'dpeak': dpk,
                    'planet_lo': pp['peak_lo'], 'planet_hi': pp['peak_hi'],
                    'dpeak_lo': dpk - lo, 'dpeak_hi': dpk + hi,
                    'drone_fwhm': dr['fwhm1'], 'planet_fwhm': pp['fwhm'],
                    'dfwhm': pp['fwhm'] - dr['fwhm1'],
                    'drone_amp': dr['amp1'], 'planet_amp': pp['amp'],
                    'amp_ratio': pp['amp'] / dr['amp1'],
                    'same_peak': both and abs(dpk) <= tolerance,
                    'planet_two_bumps': pr.get('daic2', -np.inf) > 10,
                })
    cmp = pd.DataFrame(cmp_rows)
    for ty, col in (('flw', 'trap_flower_peak'), ('frt', 'trap_fruit_peak')):
        if ('peak', ty) in trap.columns:
            cmp[col] = cmp['species'].map(trap[('peak', ty)])
    cmp.to_csv(mat / 'phenology_fit_compare.csv', index=False)

    # Year-by-year peak shifts where both see the season.
    ys = []
    for sp in species:
        dr, pr = fits.get((sp, 'event')), fits.get((sp, 'mrcnn'))
        if dr is None or pr is None or not (resolved(dr) and resolved(pr)):
            continue
        if abs(circ_diff(pr['peak1'], dr['peak1'])) > 45:
            continue
        a = pd.DataFrame(yearly_peaks(series[(sp, 'event')], dr['_x1']))
        b = pd.DataFrame(yearly_peaks(series[(sp, 'mrcnn')], pr['_x1']))
        if len(a) and len(b):
            m = a.merge(b, on='year', suffixes=('_drone', '_planet'))
            m = m[(m['amp_drone'] >= 0.05) & (m['amp_planet'] >= 0.03)]
            ys.append(m.assign(species=sp))
    ys = pd.concat(ys, ignore_index=True) if ys else pd.DataFrame()
    ys.to_csv(mat / 'phenology_fit_year_shifts.csv', index=False)

    summary = {'tolerance_days': tolerance, 'n_species': int(len(species))}
    for src in planet:
        for ty in TYPES:
            c = cmp[(cmp['source'] == src) & (cmp['type'] == ty)]
            b = c[c['both_resolved']]
            summary[f'{src}_{ty}'] = {
                'drone_resolved': int(c['drone_resolved'].sum()),
                'planet_resolved': int(c['planet_resolved'].sum()),
                'both_resolved': int(len(b)),
                'n_two_drone_init': int((c['planet_model'] == 'two_drone_init').sum()),
                'both_resolved_two_drone_init': int((b['planet_model'] == 'two_drone_init').sum()),
                'same_peak': int(b['same_peak'].sum()),
                'median_abs_dpeak': float(b['dpeak'].abs().median()) if len(b) else None,
                'median_dpeak': float(b['dpeak'].median()) if len(b) else None,
                'median_dfwhm': float(b['dfwhm'].median()) if len(b) else None,
                'median_amp_ratio': float(b['amp_ratio'].median()) if len(b) else None}
    # Does the drone flowering peak agree with the litter traps?
    fl = cmp[(cmp['source'] == 'mrcnn') & (cmp['type'] == 'flower')
             & cmp['drone_resolved'] & cmp['trap_flower_peak'].notna()]
    d = circ_diff(fl['drone_peak'], fl['trap_flower_peak'])
    summary['drone_flower_vs_trap'] = {
        'n': int(len(fl)), 'median_abs_days': float(np.median(np.abs(d))) if len(fl) else None,
        'within_30d': int((np.abs(d) <= 30).sum())}
    if len(ys) > 2:
        summary['year_shifts'] = {
            'n': int(len(ys)), 'n_species': int(ys['species'].nunique()),
            'pearson_r': float(stats.pearsonr(ys['shift_drone'], ys['shift_planet'])[0]),
            'median_abs_diff_days': float((ys['shift_planet'] - ys['shift_drone']).abs().median())}
    (mat / 'phenology_fit_summary.json').write_text(json.dumps(summary, indent=1))
    click.echo(json.dumps(summary, indent=1))

    plot_peak_scatter(cmp, outdir)
    main = cmp[(cmp['source'] == 'mrcnn') & (cmp['type'] == 'event')]
    order = main.sort_values('drone_amp', ascending=False)['species']
    show = [s for s in order if fits.get((s, 'event')) and resolved(fits[(s, 'event')])][:12]
    plot_fit_curves(fits, series, show, outdir)
    if len(ys):
        plot_year_shifts(ys, outdir)
    c = cmp[(cmp['source'] == 'mrcnn') & cmp['drone_resolved']]
    click.echo(c[['species', 'type', 'planet_resolved', 'drone_peak', 'planet_peak',
                  'dpeak', 'dpeak_lo', 'dpeak_hi', 'drone_fwhm', 'planet_fwhm',
                  'amp_ratio', 'same_peak', 'trap_flower_peak']]
               .round(2).to_string(index=False))


# -------------------------------------------------------------- evaluation

PRES = FLOWER / 'figs' / '20261103_presentation' / 'materials'
CROWNMAP_2022 = (FLOWER / 'stri/24784053/BCI_50ha_2022_09_29_crownmap_improved'
                 / 'BCI_50ha_2022_09_29_crownmap_improved.shp')
AVUELO = FLOWER / 'stri/avuelo/AVUELO_combined_crownmaps_2025.gpkg'
VICENTE = FLOWER / 'vicente_20260618' / 'vicente_20260618_labels.csv'
DATED_CROWNS = FLOWER / 'crowns_20240803' / 'BCI_50ha_crownmap_timeseries.shp'
SIZE_BINS = [36, 50, 100, 200, 400, 800, 1600]
EVAL_NAME = {'temporal': 'Mask R-CNN, temporal CV chips',
             'spatial': 'Mask R-CNN, spatial hold-out chips',
             'apply': 'Mask R-CNN, apply run vs drone',
             'pse': 'PSE-LTAE, temporal CV'}
EVAL_COLOR = {'temporal': '#2a78d6', 'spatial': '#8a8a8a', 'apply': '#7a4fd0',
              'pse': '#e8a13a'}


def kind_of(event, pf, pdc):
    return np.where(~np.asarray(event, bool), 'none',
                    np.where(np.asarray(pf) >= np.asarray(pdc), 'flowering',
                             'deciduous'))


def eval_tables(T):
    """Per-crown evaluation tables on one layout: unit (the flight, for
    resampling), tag, area_m2, kind (flowering / deciduous / none from the
    drone classifier), det (Planet call) and score."""
    out = {}
    a = pd.read_pickle(PRES / 'temporal_eval_crowns_species.pkl')
    days = pd.read_csv(PRES / 'temporal_eval_crowns.csv',
                       usecols=['chip', 'window', 'tag', 'days'])
    a = a.merge(days.drop_duplicates(['chip', 'window', 'tag']),
                on=['chip', 'window', 'tag'], how='left')
    a['p_event'] = 1 - (1 - a['flowering_probability']) * (1 - a['deciduous_probability'])
    out['temporal'] = a.rename(columns={'date': 'unit', 'score_max': 'score'})[
        ['unit', 'tag', 'area_m2', 'kind', 'det', 'score', 'gt_frac', 'p_event',
         'wet', 'days', 'latin']]

    b = pd.read_csv(PRES / 'spatial_eval_crowns.csv')
    d = T['drone'].assign(date=T['drone']['date'].dt.strftime('%Y_%m_%d'))
    b = b.merge(d[['tag', 'date', 'p_flower', 'p_decid']], on=['tag', 'date'],
                how='left')
    out['spatial'] = pd.DataFrame({
        'unit': b['date'], 'tag': b['tag'], 'area_m2': b['px'] * PX_M2,
        'kind': kind_of(b['event'], b['p_flower'].fillna(0), b['p_decid'].fillna(0)),
        'det': np.nan, 'score': b['score_max']})

    c = T['pairs']['mrcnn']
    out['apply'] = pd.DataFrame({
        'unit': c['flight'], 'tag': c['tag'], 'area_m2': c['area_m2'],
        'kind': kind_of(c['event'], c['p_flower'], c['p_decid']),
        'det': c['k'] / c['n_scenes'] >= 0.5, 'score': c['score']})

    e = pd.read_csv(PSE_CV / 'eval_pairs.csv')
    out['pse'] = pd.DataFrame({
        'unit': e['flight'], 'tag': e['tag'], 'area_m2': e['area'],
        'kind': kind_of(e['true'], e['p_flower'], e['p_decid']),
        'det': e['p'] >= PSE_THRESH, 'score': e['p']})
    return out


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, bool)
    if y.sum() < 10 or (~y).sum() < 10:
        return np.nan
    return roc_auc_score(y, s)


def size_stats(df):
    """Skill in each crown-area bin."""
    from sklearn.metrics import average_precision_score
    rows = []
    cut = pd.cut(df['area_m2'], SIZE_BINS, right=False)
    for b, g in df.groupby(cut, observed=False):
        ev = g['kind'] != 'none'
        row = {'bin': f'{int(b.left)}-{int(b.right)}', 'lo': b.left, 'hi': b.right,
               'n': len(g), 'n_flowering': int((g['kind'] == 'flowering').sum()),
               'n_deciduous': int((g['kind'] == 'deciduous').sum())}
        det = g['det'].astype(float)
        for kd in ('flowering', 'deciduous'):
            m = g['kind'] == kd
            row[f'recall_{kd}'] = det[m].mean() if m.sum() >= 20 else np.nan
            row[f'auroc_{kd}'] = _auc(g.loc[m | ~ev, 'kind'] == kd,
                                      g.loc[m | ~ev, 'score'])
        row['false_alarm'] = det[~ev].mean()
        row['precision'] = ev[det == 1].mean() if (det == 1).sum() >= 20 else np.nan
        row['auroc'] = _auc(ev, g['score'])
        row['ap'] = (average_precision_score(ev, g['score'])
                     if ev.sum() >= 10 else np.nan)
        row['prevalence'] = ev.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def size_stats_ci(df, n_boot, rng):
    """size_stats with 95% intervals from resampling flights."""
    base = size_stats(df)
    units = df['unit'].unique()
    groups = {u: g for u, g in df.groupby('unit')}
    cols = ['recall_flowering', 'recall_deciduous', 'false_alarm', 'auroc',
            'precision']
    boots = []
    for _ in range(n_boot):
        bs = pd.concat([groups[u] for u in rng.choice(units, len(units))])
        boots.append(size_stats(bs)[cols].values)
    boots = np.array(boots)
    with np.errstate(all='ignore'):
        lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    for j, c in enumerate(cols):
        base[f'{c}_lo'], base[f'{c}_hi'] = lo[:, j], hi[:, j]
    return base


def crown_meta():
    """Per tag: illumination, liana load, distance to the plot edge (m) and
    DBH (mm). Tags shared by several crowns are left out."""
    import geopandas as gpd
    g = gpd.read_file(CROWNMAP_2022)
    g['tag'] = pd.to_numeric(g['tag'], errors='coerce')
    g = g[g['tag'].notna() & (g['tag'] != -9999)]
    g = g[~g['tag'].duplicated(keep=False)]
    x0, y0, x1, y1 = g.total_bounds
    c = g.geometry.centroid
    edge = np.minimum.reduce([c.x - x0, x1 - c.x, c.y - y0, y1 - c.y])
    m = pd.DataFrame({'tag': g['tag'].astype(int).values,
                      'illumination': pd.to_numeric(g['iluminatio'], errors='coerce').values,
                      'liana': pd.to_numeric(g['lianas'], errors='coerce').values,
                      'edge_m': np.asarray(edge)})
    av = gpd.read_file(AVUELO, ignore_geometry=True)
    av = av[(av['Plot'] == 'p50') & (av['Dbh_mm'] > 0)]
    dbh = av.drop_duplicates('Tag').set_index('Tag')['Dbh_mm']
    m['dbh_mm'] = m['tag'].map(dbh)
    return m


FACTORS = [
    ('p_event', 'drone event probability', [0.5, 0.6, 0.7, 0.8, 0.9, 1.001]),
    ('gt_frac', 'share of crown labeled event', [0, 0.25, 0.5, 0.75, 0.9, 1.001]),
    ('illumination', 'crown illumination index', [1.5, 2.5, 3.5, 4.5, 5.5]),
    ('liana', 'liana load index', [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]),
    ('edge_m', 'distance to plot edge (m)', [0, 25, 50, 100, 150, 260]),
    ('dbh_mm', 'trunk diameter (mm)', [100, 300, 500, 700, 1000, 2600]),
    ('days', 'days to nearest training chip', [0, 15, 30, 60, 400]),
]


def factor_stats(ev):
    rows = []
    for col, label, bins in FACTORS:
        cut = pd.cut(ev[col], bins, right=False)
        for kd in ('flowering', 'deciduous'):
            e = ev[ev['kind'] == kd]
            g = e.groupby(cut[e.index], observed=False)['det'].agg(['size', 'mean'])
            for b, r in g.iterrows():
                rows.append({'factor': col, 'label': label, 'kind': kd,
                             'lo': b.left, 'hi': b.right, 'mid': e.loc[cut[e.index] == b, col].mean(),
                             'n': int(r['size']), 'recall': r['mean']})
    return pd.DataFrame(rows)


def detection_logit(ev, n_boot, rng):
    """Standardized logistic model of detection over event crown-dates;
    intervals from resampling flights."""
    from sklearn.linear_model import LogisticRegression
    feats = ['flowering', 'log_area', 'wet', 'far', 'p_event', 'gt_frac',
             'illumination', 'liana', 'log_edge']
    d = ev.assign(flowering=(ev['kind'] == 'flowering').astype(float),
                  log_area=np.log(ev['area_m2']), wet=ev['wet'].astype(float),
                  far=(ev['days'] > 30).astype(float),
                  log_edge=np.log(ev['edge_m'].clip(lower=1)))
    d = d.dropna(subset=feats + ['det'])
    mu, sd = d[feats].mean(), d[feats].std()

    def coef(df):
        X = ((df[feats] - mu) / sd).values
        m = LogisticRegression(C=10.0, max_iter=2000).fit(X, df['det'].astype(int))
        return m.coef_[0]
    base = coef(d)
    units = d['unit'].unique()
    groups = {u: g for u, g in d.groupby('unit')}
    bs = np.array([coef(pd.concat([groups[u] for u in rng.choice(units, len(units))]))
                   for _ in range(n_boot)])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return pd.DataFrame({'feature': feats, 'coef_per_sd': base, 'lo': lo, 'hi': hi,
                         'sd': sd.values, 'n': len(d)})


def manual_table(T):
    """Vicente's crown-date labels joined to the drone calls and to the
    Planet calls within 10 days of the same flight."""
    import geopandas as gpd
    from compare_classifications_to_gt import aggregate_gt_labels, load_gt
    gt, _ = aggregate_gt_labels(load_gt(str(VICENTE)))
    pairs = gpd.read_file(DATED_CROWNS, ignore_geometry=True, columns=['GlobalID', 'tag'])
    pairs = pairs.rename(columns={'GlobalID': 'uuid'}).drop_duplicates()
    pairs['tag'] = pairs['tag'].astype(int)
    pairs = pairs[~pairs['tag'].duplicated(keep=False) & ~pairs['uuid'].duplicated(keep=False)]
    gt = gt.merge(pairs, on='uuid')
    # Labels on segments marked bad are mostly tool defaults (leafing 0).
    gt = gt[gt['isFlowering'].isin(['yes', 'no']) & gt['leafing'].notna()
            & (gt['segmentation'] != 'bad')]
    fl, lf = gt['isFlowering'] == 'yes', gt['leafing'] < 50
    gt['manual'] = np.select([fl & lf, fl, lf], ['both', 'flowering', 'leafless'], 'none')
    out = gt[['tag', 'dt', 'manual', 'leafing', 'floweringIntensity', 'segmentation']]
    out = out.rename(columns={'dt': 'flight'})
    d = T['drone'].rename(columns={'date': 'flight'})
    out = out.merge(d[['tag', 'flight', 'flower', 'decid', 'event']], on=['tag', 'flight'])
    for src in ('mrcnn', 'pse'):
        p = T['pairs'][src][['tag', 'flight', 'k', 'n_scenes', 'area_m2']]
        p = p.assign(**{src: p['k'] > 0}).drop(columns=['k', 'n_scenes'])
        out = out.merge(p.rename(columns={'area_m2': f'area_{src}'}),
                        on=['tag', 'flight'], how='left')
    return out


def _rate_ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan, np.nan, np.nan, len(x)
    lo, hi = stats.binomtest(int(x.sum()), len(x)).proportion_ci(method='wilson')
    return x.mean(), lo, hi, len(x)


def plot_size(tabs, outdir):
    panels = [('recall_deciduous', 'leafless crowns detected'),
              ('recall_flowering', 'flowering crowns detected'),
              ('false_alarm', 'crowns with no event called an event'),
              ('auroc', 'AUROC, event vs no event')]
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.9))
    for ax, (col, title) in zip(axes, panels):
        for name, t in tabs.items():
            if t[col].notna().sum() == 0:
                continue
            mid = np.sqrt(t['lo'] * t['hi'])
            ax.plot(mid, t[col], 'o-', color=EVAL_COLOR[name], lw=1.6, ms=4,
                    label=EVAL_NAME[name])
            if f'{col}_lo' in t:
                ax.fill_between(mid, t[f'{col}_lo'], t[f'{col}_hi'],
                                color=EVAL_COLOR[name], alpha=0.13, lw=0)
        ax.set_xscale('log')
        ax.set_xticks([50, 100, 200, 400, 800], ['50', '100', '200', '400', '800'])
        ax.minorticks_off()
        ax.set_xlabel('crown area (m²)')
        ax.set_title(title, fontsize=9.5)
        ax.set_ylim(0.4 if col == 'auroc' else 0, 0.2 if col == 'false_alarm' else 1)
        style(ax)
    axes[3].legend(fontsize=7.5, frameon=False, loc='lower right')
    save(fig, outdir, 'planet_eval_crown_size')


def plot_factors(fs, outdir, factors=FACTORS, figname='planet_eval_factors'):
    ncol = min(len(factors), 4)
    nrow = -(-len(factors) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.1 * nrow),
                             squeeze=False)
    for ax, (col, label, _) in zip(axes.ravel(), factors):
        for kd, color, name in (('deciduous', C_DECID, 'leafless'),
                                ('flowering', C_FLOWER, 'flowering')):
            t = fs[(fs['factor'] == col) & (fs['kind'] == kd) & (fs['n'] >= 20)]
            ax.plot(t['mid'], t['recall'], 'o-', color=color, lw=1.6, ms=4, label=name)
            for r in t.itertuples():
                ax.annotate(str(r.n), (r.mid, r.recall), fontsize=6, color=color,
                            xytext=(0, 4), textcoords='offset points', ha='center')
        ax.set_xlabel(label)
        ax.set_ylim(0, 1.05)
        style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel('share of event crowns detected')
    axes[0, 0].legend(fontsize=8, frameon=False)
    for ax in axes.ravel()[len(factors):]:
        ax.axis('off')
    save(fig, outdir, figname)


def plot_manual(mt, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))
    series = [('drone', 'drone classifier', INK), ('mrcnn', 'Planet Mask R-CNN', C_PLANET)]
    ax = axes[0]
    classes = ['flowering', 'leafless', 'both', 'none']
    t = mt[mt['group'] == 'class'].set_index(['level', 'source'])
    for j, (src, name, col) in enumerate(series):
        y = [t.loc[(c, src), 'rate'] if (c, src) in t.index else np.nan for c in classes]
        lo = [t.loc[(c, src), 'lo'] if (c, src) in t.index else np.nan for c in classes]
        hi = [t.loc[(c, src), 'hi'] if (c, src) in t.index else np.nan for c in classes]
        x = np.arange(len(classes)) + (j - (len(series) - 1) / 2) * 0.3
        ax.bar(x, y, 0.28, color=col, label=name)
        ax.errorbar(x, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], fmt='none',
                    ecolor='0.35', lw=0.8)
    n = [int(t.loc[(c, 'mrcnn'), 'n']) if (c, 'mrcnn') in t.index else 0 for c in classes]
    ax.set_xticks(range(len(classes)),
                  [f'{c}\n(n = {k:,})' for c, k in zip(
                      ['flowering only', 'leafless only', 'both', 'neither'], n)], fontsize=8)
    ax.set_xlabel('manual label')
    ax.set_ylabel('share of crown-dates called an event')
    ax.legend(fontsize=8, frameon=False)
    style(ax)
    for ax, grp, xlabel in ((axes[1], 'leafing', 'manual leafing (%)'),
                            (axes[2], 'intensity', 'manual flowering intensity (%)')):
        for src, name, col in series:
            d = mt[(mt['group'] == grp) & (mt['source'] == src)]
            ax.plot(d['mid'], d['rate'], 'o-', color=col, lw=1.6, ms=4, label=name)
            ax.fill_between(d['mid'], d['lo'], d['hi'], color=col, alpha=0.13, lw=0)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1)
        style(ax)
    save(fig, outdir, 'planet_eval_manual')


@cli.command()
@click.option('--outdir', default=str(DEFAULT_OUT), show_default=True)
@click.option('--n-boot', default=200, show_default=True)
def evaluate(outdir, n_boot):
    """Planet skill by crown size, crown properties and manual labels."""
    outdir = Path(outdir)
    mat = outdir / 'materials'
    T = build_tables(outdir, 10)
    rng = np.random.default_rng(0)
    E = eval_tables(T)
    tabs = {}
    for name, df in E.items():
        tabs[name] = size_stats_ci(df, n_boot, rng).assign(source=name)
    size = pd.concat(tabs.values(), ignore_index=True)
    size.to_csv(mat / 'planet_eval_crown_size.csv', index=False)
    plot_size({'temporal': tabs['temporal']}, outdir)
    click.echo(size[['source', 'bin', 'n', 'n_flowering', 'n_deciduous',
                     'recall_flowering', 'recall_deciduous', 'false_alarm',
                     'precision', 'auroc', 'auroc_flowering', 'auroc_deciduous',
                     'ap']].round(3).to_string(index=False))

    # Per-crown temporal correlation (correlate) against crown area.
    f = mat / 'phenology_corr_crowns_mrcnn.csv'
    if f.exists():
        cc = pd.read_csv(f)
        g = cc.groupby(pd.cut(cc['area_m2'], SIZE_BINS, right=False), observed=False)
        t = pd.DataFrame({'n_crowns': g.size(), 'median_rho': g['rho'].median(),
                          'median_rho_decid': g['rho_decid'].median(),
                          'median_rho_flower': g['rho_flower'].median()})
        t.to_csv(mat / 'planet_eval_crown_corr_by_size.csv')
        click.echo('\nper-crown Planet-drone correlation by crown area')
        click.echo(t.round(2).to_string())
        r = stats.spearmanr(cc['area_m2'], cc['rho'], nan_policy='omit')
        click.echo(f'Spearman of crown rho with area: {r[0]:.2f} (p = {r[1]:.1g})')

    ev = E['temporal'].merge(crown_meta(), on='tag', how='left')
    ev = ev[ev['kind'] != 'none']
    fs = factor_stats(ev)
    fs.to_csv(mat / 'planet_eval_factors.csv', index=False)
    plot_factors(fs, outdir)
    plot_factors(fs, outdir, [f for f in FACTORS if f[0] in ('p_event', 'gt_frac')],
                 'planet_eval_factors_label')
    lg = detection_logit(ev, min(n_boot, 100), rng)
    lg.to_csv(mat / 'planet_eval_detection_logit.csv', index=False)
    click.echo('\nlogistic model of detection (temporal CV event crown-dates)')
    click.echo(lg.round(3).to_string(index=False))

    m = manual_table(T)
    m.to_csv(mat / 'planet_eval_manual_pairs.csv', index=False)
    rows = []
    cols = {'drone': 'event', 'mrcnn': 'mrcnn', 'pse': 'pse'}
    seen = m[m['mrcnn'].notna()]
    for src, col in cols.items():
        base = m[m[col].notna()] if src == 'pse' else seen
        for c, g in base.groupby('manual'):
            r, lo, hi, n = _rate_ci(g[col])
            rows.append({'group': 'class', 'level': c, 'source': src, 'mid': np.nan,
                         'rate': r, 'lo': lo, 'hi': hi, 'n': n})
        nf = base[base['manual'].isin(['leafless', 'none'])]
        for b, g in nf.groupby(pd.cut(nf['leafing'], [0, 10, 25, 50, 75, 90, 100.01],
                                      right=False), observed=False):
            r, lo, hi, n = _rate_ci(g[col])
            rows.append({'group': 'leafing', 'level': str(b), 'source': src,
                         'mid': g['leafing'].mean(), 'rate': r, 'lo': lo, 'hi': hi, 'n': n})
        fo = base[base['manual'] == 'flowering']
        for b, g in fo.groupby(pd.cut(fo['floweringIntensity'], [0, 25, 50, 75, 100.01],
                                      right=False), observed=False):
            r, lo, hi, n = _rate_ci(g[col])
            rows.append({'group': 'intensity', 'level': str(b), 'source': src,
                         'mid': g['floweringIntensity'].mean(), 'rate': r, 'lo': lo,
                         'hi': hi, 'n': n})
    mt = pd.DataFrame(rows)
    mt.to_csv(mat / 'planet_eval_manual.csv', index=False)
    plot_manual(mt, outdir)
    click.echo('\nPlanet and drone calls against manual labels')
    click.echo(mt.round(3).to_string(index=False))
    # Drone flowering call specifically, for the flowering classes.
    fl = seen[seen['manual'].isin(['flowering', 'both'])]
    click.echo(f'drone flowering call on manual flowering: {fl["flower"].mean():.3f} '
               f'(n = {len(fl)}); drone leafless call on manual leafless: '
               f'{seen.loc[seen["manual"].isin(["leafless", "both"]), "decid"].mean():.3f}')


if __name__ == '__main__':
    cli()
