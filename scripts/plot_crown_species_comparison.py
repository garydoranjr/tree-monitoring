#!/usr/bin/env python
"""Compare the temporal Planet Mask R-CNN's per-crown calls between drone
flights (apply_planet_maskrcnn.py's crown_timeseries.csv) with the drone
crown classifications, aggregated by species.

The drone label is the training-mask rule of apply_drone_labels_coreg.py
(--mode both) applied to the per-crown mean probabilities of the nc4 tables:
a crown is an event (flowering or deciduous) when
1 - (1 - p_flowering) * (1 - p_deciduous) > 0.5. The classification rasters
are near constant inside a crown, so this matches the per-pixel rule. A
Planet crown is an event when at least --pos-frac of its clear pixels are
covered by an instance scoring at least the apply run's --det-thresh.

Only crowns of at least --min-crown-px Planet pixels (0.75 m), the training
--min-instance-size, are used on both sides, and only Planet crowns at least
--min-clear clear. Species are ranked by drone event crown-dates over the
apply period.

Writes, with the given output prefix (PNG + PDF each):

- <prefix>_timeseries: per species, the share of crowns that are events in
  each drone flight and in each Planet scene, 2020 to 2026.
- <prefix>_season: the same shares by calendar month, all years together,
  over the span of the in-envelope Planet scenes on both sides.
- <prefix>_agreement: per species, crown-level precision and recall of the
  Planet call against the drone label of the nearest flight, and the share
  of crowns each calls an event, for scenes --match-days or less from it.

It also writes <prefix>_agreement.csv.
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import click
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

INK = '#1a1a1a'
GRID = '#e6e6e6'
DRONE = INK
PLANET = '#2a78d6'
OUTSIDE = '#b0b0b0'
PRECISION = '#eb6834'
RECALL = '#1baf7a'

RESULTS = '/Volumes/Earth03/flower/results'
NC4S = (f'{RESULTS}/crown_classifications.nc4',
        f'{RESULTS}/crown_classifications_mavic.nc4')
APPLY = '/Volumes/Earth03/flower/202610_temporal_apply'
# Phantom and mavic both flew 2024-03-06 to 03-18; the phantom record is used
# up to its last flight, as in the training-chip builds.
PHANTOM_END = pd.Timestamp('2024-03-18')

CROWN_COLS = ['scene', 'date', 'days_to_drone', 'in_envelope', 'coreg_ok',
              'tag', 'crown_px', 'crown_clear_frac', 'pos_frac']


def load_drone(nc4s):
    """Long table of drone crown-dates: tag, date, species, event."""
    frames = []
    for path in nc4s:
        ds = xr.open_dataset(path)
        p = 1 - (1 - ds.flowering_probability) * (1 - ds.deciduous_probability)
        df = p.to_dataframe('p').reset_index().dropna(subset=['p'])
        species = pd.Series(ds.species.values, index=ds.tag.values)
        df['species'] = df['tag'].map(species)
        df['mavic'] = 'mavic' in os.path.basename(path)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[~(df['mavic'] & (df['date'] <= PHANTOM_END))]
    df['event'] = df['p'] > 0.5
    return df[['tag', 'date', 'species', 'event']]


def load_planet(apply_dir, min_clear, pos_frac):
    """Planet crown-scenes with enough clear pixels, plus the scene's
    drone_date, and the event call."""
    crowns = pd.read_csv(f'{apply_dir}/crown_timeseries.csv',
                         usecols=CROWN_COLS, parse_dates=['date'])
    scenes = pd.read_csv(f'{apply_dir}/scene_summary.csv',
                         usecols=['scene', 'drone_date'],
                         parse_dates=['drone_date'])
    crown_px = crowns.groupby('tag')['crown_px'].first()
    crowns = crowns[crowns['crown_clear_frac'] >= min_clear]
    crowns = crowns.merge(scenes, on='scene', how='left')
    crowns['event'] = crowns['pos_frac'].fillna(0) >= pos_frac
    return crowns, crown_px


def share(df, keys):
    g = df.groupby(keys)['event']
    return pd.DataFrame({'share': g.mean(), 'n': g.size()}).reset_index()


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, prefix, suffix):
    out = f'{prefix}_{suffix}.png'
    fig.savefig(out, dpi=200)
    fig.savefig(f'{prefix}_{suffix}.pdf')
    plt.close(fig)
    print(f'Wrote {out}')


def species_label(name, n):
    return f'{name}\n({n} crowns)'


def plot_timeseries(drone, planet, species, n_crowns, min_obs, prefix):
    t0 = pd.Timestamp(planet['date'].min()) - pd.Timedelta(days=15)
    t1 = pd.Timestamp(planet['date'].max()) + pd.Timedelta(days=15)
    d = share(drone[drone['date'].between(t0, t1)], ['species', 'date'])
    p = planet.groupby(['species', 'date', 'in_envelope'])['event'].agg(
        share='mean', n='size').reset_index()

    fig, axes = plt.subplots(len(species), 1, sharex=True,
                             figsize=(13, 1.35 * len(species) + 1))
    for ax, sp in zip(axes, species):
        ps = p[(p['species'] == sp) & (p['n'] >= min_obs)]
        for inside, color, label in ((False, OUTSIDE, 'Planet, > 30 d from training'),
                                     (True, PLANET, 'Planet scene')):
            q = ps[ps['in_envelope'] == inside]
            ax.scatter(q['date'], q['share'], s=9, color=color, lw=0,
                       label=label, zorder=2)
        ds = d[d['species'] == sp]
        ax.plot(ds['date'], ds['share'], color=DRONE, lw=1.2, marker='o',
                ms=3, label='drone flight', zorder=3)
        ax.set_ylim(bottom=0)
        ax.set_ylabel(species_label(sp, n_crowns[sp]), rotation=0,
                      ha='right', va='center', fontsize=8.5)
        style(ax)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[-1].set_xlim(t0, t1)
    axes[-1].set_xlabel('date')
    fig.supylabel('share of crowns with an event (flowering and/or deciduous)', x=0.005)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(l) for l in ('drone flight', 'Planet scene',
                                       'Planet, > 30 d from training')]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               frameon=False, loc='upper center', ncol=3)
    fig.tight_layout(rect=(0.01, 0, 1, 1 - 0.35 / fig.get_figheight()))
    save(fig, prefix, 'timeseries')


def plot_season(drone, planet, species, n_crowns, prefix):
    # Both sides cover the span of the in-envelope Planet scenes.
    p = planet[planet['in_envelope']]
    t0, t1 = pd.Timestamp(p['date'].min()), pd.Timestamp(p['date'].max())
    d = drone[drone['date'].between(t0, t1)].assign(
        month=lambda x: x['date'].dt.month)
    p = p.assign(month=lambda x: x['date'].dt.month)
    d = share(d, ['species', 'month'])
    p = share(p, ['species', 'month'])
    months = np.arange(1, 13)

    ncol = 4
    nrow = int(np.ceil(len(species) / ncol))
    fig, axes = plt.subplots(nrow, ncol, sharex=True, figsize=(13, 2.4 * nrow + 0.6))
    for ax, sp in zip(axes.flat, species):
        for df, color, label in ((d, DRONE, 'drone flights'),
                                 (p, PLANET, 'Planet scenes')):
            s = df[df['species'] == sp].set_index('month')['share'].reindex(months)
            ax.plot(months, s, color=color, lw=1.6, marker='o', ms=3.5,
                    label=label)
        ax.set_title(species_label(sp, n_crowns[sp]).replace('\n', ' '),
                     fontsize=9, style='italic', loc='left')
        ax.set_ylim(bottom=0)
        style(ax)
    for ax in axes.flat[len(species):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xticks(months)
        ax.set_xticklabels([m[0] for m in pd.date_range(
            '2001-01-01', periods=12, freq='MS').strftime('%b')])
    fig.supylabel('share of crowns with an event (flowering and/or deciduous)')
    fig.supxlabel(f'calendar month (all years, {t0:%Y-%m} to {t1:%Y-%m})')
    fig.legend(*axes.flat[0].get_legend_handles_labels(), frameon=False,
               loc='upper center', ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
    save(fig, prefix, 'season')


def agreement(matched, species):
    rows = []
    groups = [(sp, matched[matched['species'] == sp]) for sp in species]
    groups.append(('other species', matched[~matched['species'].isin(species)]))
    groups.append(('all species', matched))
    for name, m in groups:
        tp = (m['event_planet'] & m['event_drone']).sum()
        n_p, n_d = m['event_planet'].sum(), m['event_drone'].sum()
        rows.append({
            'species': name, 'crown_scenes': len(m),
            'drone_events': int(n_d), 'planet_events': int(n_p),
            'precision': tp / n_p if n_p else np.nan,
            'recall': tp / n_d if n_d else np.nan,
            'drone_share': n_d / len(m), 'planet_share': n_p / len(m),
        })
    return pd.DataFrame(rows)


def plot_agreement(table, n_scenes, match_days, prefix):
    y = np.arange(len(table))[::-1]
    fig, (ax_pr, ax_sh) = plt.subplots(1, 2, sharey=True, figsize=(11, 6.5))
    for col, color, label in (('precision', PRECISION, 'precision'),
                              ('recall', RECALL, 'recall')):
        ax_pr.scatter(table[col], y, s=48, color=color, edgecolor='white',
                      lw=1, label=label, zorder=3)
    ax_pr.hlines(y, table[['precision', 'recall']].min(axis=1),
                 table[['precision', 'recall']].max(axis=1), color=GRID,
                 lw=2, zorder=2)
    ax_pr.set_xlim(0, 1)
    ax_pr.set_xlabel('Planet call against drone label, per crown-scene')
    ax_pr.legend(frameon=False, loc='lower right')
    for col, color, label in (('drone_share', DRONE, 'drone label'),
                              ('planet_share', PLANET, 'Planet call')):
        ax_sh.scatter(table[col], y, s=48, color=color, edgecolor='white',
                      lw=1, label=label, zorder=3)
    ax_sh.hlines(y, table[['drone_share', 'planet_share']].min(axis=1),
                 table[['drone_share', 'planet_share']].max(axis=1),
                 color=GRID, lw=2, zorder=2)
    ax_sh.set_xlim(left=0)
    ax_sh.set_xlabel('share of crown-scenes called an event')
    ax_sh.legend(frameon=False, loc='lower right')
    ax_pr.set_yticks(y)
    ax_pr.set_yticklabels([f"{s} (n={d:,} drone events)" for s, d in
                           zip(table['species'], table['drone_events'])],
                          fontsize=8.5)
    for ax in (ax_pr, ax_sh):
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        # Separate the two pooled rows from the per-species rows.
        ax.axhline(1.5, color=INK, lw=0.6)
    fig.suptitle(f'{n_scenes} Planet scenes 3-{match_days} days from a drone '
                 'flight, against that flight', fontsize=10, x=0.01,
                 ha='left')
    fig.tight_layout()
    save(fig, prefix, 'agreement')


@click.command()
@click.argument('outputprefix', type=click.Path(dir_okay=False))
@click.option('--apply-dir', default=APPLY, show_default=True,
              type=click.Path(exists=True, file_okay=False))
@click.option('--nc4', 'nc4s', multiple=True, default=NC4S, show_default=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option('--n-species', default=12, show_default=True)
@click.option('--pos-frac', default=0.25, show_default=True,
              help='Planet event: share of clear crown pixels under a detection.')
@click.option('--min-clear', default=0.5, show_default=True,
              help='Minimum clear fraction of a Planet crown.')
@click.option('--min-crown-px', default=64, show_default=True,
              help='Minimum crown size in 0.75 m Planet pixels.')
@click.option('--min-obs', default=5, show_default=True,
              help='Minimum observed crowns of a species to plot a Planet scene.')
@click.option('--match-days', default=7, show_default=True,
              help='Maximum days from Planet scene to drone flight for agreement.')
def main(outputprefix, apply_dir, nc4s, n_species, pos_frac, min_clear,
         min_crown_px, min_obs, match_days):
    drone = load_drone(nc4s)
    planet, crown_px = load_planet(apply_dir, min_clear, pos_frac)
    big = crown_px.index[crown_px >= min_crown_px]
    drone = drone[drone['tag'].isin(big)]
    planet = planet[planet['tag'].isin(big)]
    planet['species'] = planet['tag'].map(
        drone.drop_duplicates('tag').set_index('tag')['species'])
    planet = planet.dropna(subset=['species'])

    period = drone['date'].between(planet['date'].min(), planet['date'].max())
    ranked = drone[period].groupby('species')['event'].sum()
    species = ranked.sort_values(ascending=False).index[:n_species].tolist()
    n_crowns = drone.groupby('species')['tag'].nunique()

    plot_timeseries(drone, planet, species, n_crowns, min_obs, outputprefix)
    plot_season(drone, planet, species, n_crowns, outputprefix)

    near = planet[planet['coreg_ok'] & planet['in_envelope']
                  & (planet['days_to_drone'] <= match_days)]
    matched = near.merge(
        drone.rename(columns={'date': 'drone_date'})[
            ['tag', 'drone_date', 'event']],
        on=['tag', 'drone_date'], suffixes=('_planet', '_drone'))
    table = agreement(matched, species)
    table.to_csv(f'{outputprefix}_agreement.csv', index=False)
    plot_agreement(table, matched['scene'].nunique(), match_days, outputprefix)
    print(f"{len(matched):,} matched crown-scenes, "
          f"{matched['drone_date'].nunique()} flights")
    print(table.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
