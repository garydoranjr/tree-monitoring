#!/usr/bin/env python
"""Figures showing why weekly drone flights miss short phenological events.

Outputs go to OUTDIR (default /Volumes/Earth03/flower/figs/20261103_presentation).

Subcommands:

  durations      Three panels:
                 (a) BCI litter traps, 2020-09 .. 2024-09: for canopy species
                     with crowns in the 50ha crown map, the share of trap-years
                     whose flowers fell within a single weekly census
                     (individual_trap_analysis.py events, event_length == 7).
                 (b) The chance that an event of a given length contains at
                     least one observation, from the real mavic flight dates and
                     from the dates each crown was clear in Planet, over the
                     mavic record (2024-03-06 .. 2026-01-20).
                 (c) Anacardium excelsum crowns: the mavic flights on which the
                     drone classifier called the crown deciduous (P > 0.5).
  missed-event   One Handroanthus guayacan crown (tag 5058) that flowered
                 between two weekly phantom flights (2023-03-28, 2023-04-04):
                 Planet tiles across the week, drone tiles on the flight dates.

The trap events come from the observability pipeline (Snakefile rules
get_annual_trap_data_flower and individual_trap_analysis); pass the CSV with
--trap-events.

Typical usage:
    python scripts/plot_phenology_cadence_figures.py durations \\
        --trap-events results/sp_flower_counts_annual_stats.csv
    python scripts/plot_phenology_cadence_figures.py missed-event
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
import plot_update_figures as pu  # noqa: E402
import plot_study_area_figure as sa  # noqa: E402

FLOWER = pu.FLOWER
DEFAULT_OUT = sa.DEFAULT_OUT
TAXA = FLOWER / 'data/Condit_ValidBCITaxa.tab'
CROWN_SERIES = FLOWER / 'crown_series_20261029/crown_pixel_series.npz'
MAVIC_SPAN = ('2024-03-06', '2026-01-20')
C_PLANET = '#4c72b0'
C_BOTH = '0.15'
C_HIGHLIGHT = '#e3a400'


def save(fig, outdir, name, dpi=300):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'{name}.{ext}', dpi=dpi)
    plt.close(fig)
    click.echo(f'Wrote {outdir / name}.png/.pdf')


@click.group()
def cli():
    pass


out_option = click.option('--outdir', type=click.Path(file_okay=False, path_type=Path),
                          default=DEFAULT_OUT, show_default=True)


# --------------------------------------------------------------------------
# durations
# --------------------------------------------------------------------------
def trap_latin():
    """Old 4-letter trap codes and 6-letter mnemonics -> Latin name."""
    tx = pd.read_csv(TAXA, sep='\t')
    out = {}
    for r in tx.itertuples():
        for c in (r.past_code, r.Mnemonic):
            if isinstance(c, str):
                for cc in c.replace(';', ',').split(','):
                    out.setdefault(cc.strip().upper(), r.Latin)
    return out


def single_week_share(events_csv, min_events, min_crowns):
    """Per canopy species: share of trap-years with flowers in one census."""
    import xarray as xr
    e = pd.read_csv(events_csv)
    e['latin'] = e.species.map(trap_latin())
    with xr.open_dataset(sa.NC4['phantom']) as ds:
        crowns = ds.species.to_series().value_counts()
    g = (e.dropna(subset=['latin']).groupby('latin').event_length
         .agg(n='size', one=lambda s: (s == 7).mean(), median='median'))
    g['crowns'] = g.index.map(crowns).fillna(0).astype(int)
    return g[(g.n >= min_events) & (g.crowns >= min_crowns)].sort_values('one', ascending=False)


def capture_curves(lengths, n_crowns=400, seed=0):
    """P(an event of L days holds >= 1 observation) over the mavic record."""
    t0, t1 = map(pd.Timestamp, MAVIC_SPAN)
    days = pd.date_range(t0, t1)

    def capture(obs):
        c = np.concatenate([[0], np.cumsum(obs)])
        return np.array([((c[np.arange(len(days) - l + 1) + l]
                           - c[np.arange(len(days) - l + 1)]) > 0).mean() for l in lengths])

    drone = days.isin(pd.DatetimeIndex(pu.drone_dates('mavic')))
    z = np.load(CROWN_SERIES, allow_pickle=True)
    sdate = pd.to_datetime(pd.DataFrame(json.loads(str(z['scenes']))).date).values
    # a crown counts as observed when nearly all of its pixels are clear
    ok = (z['obs_n_clear'] >= 0.9 * z['obs_n_px']) & (z['obs_n_px'] >= 8)
    obs = pd.DataFrame({'crown': z['obs_crown'][ok], 'date': sdate[z['obs_scene'][ok]]})
    obs = obs[(obs.date >= t0) & (obs.date <= t1)].drop_duplicates()
    crowns = np.random.default_rng(seed).choice(obs.crown.unique(), n_crowns, replace=False)
    planet, both = [], []
    by_crown = obs.groupby('crown').date
    for c in crowns:
        o = days.isin(pd.DatetimeIndex(by_crown.get_group(c)))
        planet.append(capture(o))
        both.append(capture(o | drone))
    return pd.DataFrame({'L': lengths, 'drone': capture(drone), 'planet': np.mean(planet, 0),
                         'both': np.mean(both, 0)})


def deciduous_flights(species):
    """mavic flight dates with P(deciduous) > 0.5, per crown of `species`."""
    import xarray as xr
    with xr.open_dataset(sa.NC4['mavic']) as ds:
        sp = ds.species.to_series()
        dp = ds.deciduous_probability.sel(tag=sp[sp == species].index).to_pandas()
    return dp


@cli.command()
@out_option
@click.option('--trap-events', type=click.Path(exists=True, path_type=Path),
              default=DEFAULT_OUT / 'materials/sp_flower_counts_annual_stats.csv',
              show_default=True, help='individual_trap_analysis.py output for flowers.')
@click.option('--n-species', default=10, show_default=True)
def durations(outdir, trap_events, n_species):
    """Trap flowering durations, capture probability and Anacardium leaf loss."""
    import matplotlib.dates as mdates
    share = single_week_share(trap_events, min_events=15, min_crowns=3).head(n_species)
    lengths = np.arange(1, 29)
    cap = capture_curves(lengths)
    cap.to_csv(outdir / 'materials/capture_probability.csv', index=False)
    share.to_csv(outdir / 'materials/trap_single_week_share.csv')
    dp = deciduous_flights('Anacardium excelsum')

    fig = plt.figure(figsize=(12.5, 4.8))
    ax_a = fig.add_axes([0.155, 0.13, 0.19, 0.74])
    ax_b = fig.add_axes([0.42, 0.13, 0.25, 0.74])
    ax_c = fig.add_axes([0.73, 0.13, 0.255, 0.74])

    # (a) trap flowering confined to one weekly census
    y = np.arange(len(share))[::-1]
    colors = [C_HIGHLIGHT if s == 'Handroanthus guayacan' else '0.6' for s in share.index]
    ax_a.barh(y, 100 * share.one, color=colors)
    ax_a.set_yticks(y, [f'$\\it{{{s.split()[0]}}}$ $\\it{{{s.split()[1]}}}$' for s in share.index],
                    fontsize=9)
    for yi, r in zip(y, share.itertuples()):
        ax_a.text(100 * r.one + 1, yi, f'{r.n}', va='center', fontsize=7.5, color='0.35')
    ax_a.set_xlim(0, 75)
    ax_a.set_xlabel('% of trap-years with flowers\nin only one weekly census', fontsize=9)
    ax_a.set_title('(a) Flowering in litter traps', loc='left', fontsize=10.5)
    ax_a.spines[['top', 'right']].set_visible(False)

    # (b) capture probability
    ax_b.axvspan(1, 2, color=C_HIGHLIGHT, alpha=0.25, lw=0)
    ax_b.text(1.25, 4, '$\\it{H.\\ guayacan}$ flowers\nseen in Planet\n(1–2 days)', fontsize=7.5,
              va='bottom', ha='left', color='#8a6400')
    ax_b.plot(cap.L, 100 * cap.drone, color=pu.C_MAVIC, lw=2, label='weekly drone flights')
    ax_b.plot(cap.L, 100 * cap.planet, color=C_PLANET, lw=2, label='Planet (crown clear)')
    ax_b.plot(cap.L, 100 * cap.both, color=C_BOTH, lw=2, ls='--', label='drone + Planet')
    ax_b.set_xlim(1, 28)
    ax_b.set_ylim(0, 102)
    ax_b.set_xlabel('event length (days)', fontsize=9)
    ax_b.set_ylabel('% of events observed at least once', fontsize=9)
    ax_b.set_title('(b) Chance of seeing an event', loc='left', fontsize=10.5)
    ax_b.legend(loc='lower right', fontsize=8, frameon=False)
    ax_b.grid(alpha=0.3)
    ax_b.spines[['top', 'right']].set_visible(False)

    # (c) Anacardium leaf loss in the weekly mavic record
    flights = pd.DatetimeIndex(dp.columns)
    order = dp.max(axis=1).sort_values().index
    for i, tag in enumerate(order):
        ax_c.plot(flights, np.full(len(flights), i), '|', color='0.8', ms=5)
        hit = flights[dp.loc[tag].values > 0.5]
        ax_c.plot(hit, np.full(len(hit), i), 's', color='#8c510a', ms=4.5)
    ax_c.set_yticks([])
    ax_c.set_ylabel(f'{len(order)} $\\it{{Anacardium\\ excelsum}}$ crowns', fontsize=9)
    ax_c.set_xlim(pd.Timestamp('2024-10-15'), pd.Timestamp('2026-01-31'))
    ax_c.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax_c.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax_c.tick_params(axis='x', labelsize=8)
    ax_c.set_title('(c) Brief leaf loss, weekly flights', loc='left', fontsize=10.5)
    ax_c.plot([], [], '|', color='0.6', label='drone flight')
    ax_c.plot([], [], 's', color='#8c510a', ms=4.5, label='classified leafless')
    ax_c.set_ylim(-0.8, len(order) + 1.2)
    ax_c.legend(loc='upper left', fontsize=7.5, frameon=False, ncol=2, borderaxespad=0.1)
    ax_c.spines[['top', 'right', 'left']].set_visible(False)
    save(fig, outdir, 'phenology_event_durations')


# --------------------------------------------------------------------------
# missed-event
# --------------------------------------------------------------------------
MISSED = dict(
    tag=5058, species='Handroanthus guayacan', series='phantom',
    drone_dates=('2023-03-28', '2023-04-04'),
    planet=('20230328_144849_31_241a_4band', '20230331_145345_86_24c2_4band',
            '20230401_154503_82_2413_4band', '20230402_145653_20_24c8_4band',
            '20230404_154320_37_2413_4band'),
    # for comparison: another guayacan that flowered on a flight day
    compare=dict(tag=3825, drone_date='2023-03-21', planet='20230320_145416_05_24c7_4band'),
)


@cli.command('missed-event')
@out_option
@click.option('--half', default=30.0, show_default=True,
              help='Half-width of each tile in metres.')
def missed_event(outdir, half):
    """A guayacan flowering burst between two weekly drone flights."""
    import geopandas as gpd
    ev = MISSED
    tag = ev['tag']
    outline = sa.dated_crown(tag, ev['drone_dates'][1])
    bounds = sa.crown_bounds(outline, 0, min_half=half)
    planet = [(pd.Timestamp(s[:8]), sa.scene_info(s)) for s in ev['planet']]
    drone = [pd.Timestamp(d) for d in ev['drone_dates']]
    cols = sorted({d for d, _ in planet} | set(drone))

    n = len(cols)
    tile, gap, sep = 1.78, 0.1, 0.45
    left = 0.55
    fig_w = left + (n + 1) * tile + (n - 1) * gap + sep + 0.1
    bottom, top = 0.5, 0.75
    fig = plt.figure(figsize=(fig_w, bottom + 2 * tile + 0.25 + top))

    def axes(i, row):
        x = left + i * (tile + gap) + (sep - gap if i == n else 0)
        y = bottom + (1 - row) * (tile + 0.25)
        return fig.add_axes([x / fig_w, y / fig.get_figheight(), tile / fig_w,
                             tile / fig.get_figheight()])

    def pad(b, m=15):
        # read Planet beyond the tile so the shifted image still fills it
        return (b[0] - m, b[1] - m, b[2] + m, b[3] + m)

    def finish(ax, title=None, geom=None, box=None):
        geom, box = geom or outline, box or bounds
        gpd.GeoSeries([geom]).boundary.plot(ax=ax, color=sa.C_CROWN, lw=1.0)
        ax.set_xlim(box[0], box[2])
        ax.set_ylim(box[1], box[3])
        ax.set_aspect('equal')
        sa.bare(ax)
        if title:
            ax.set_title(title, fontsize=10, pad=3)

    for i, d in enumerate(cols):
        for day, s in planet:
            if day == d:
                ax = axes(i, 0)
                arr, ext = sa.read_window(s['sr'], sa.shifted_bounds(pad(bounds), s['shift']),
                                          [1, 2, 3])
                lo, hi = sa.scene_stretch(s['sr'])
                ax.imshow(sa.planet_rgb(arr, lo, hi), extent=sa.unshift_extent(ext, s['shift']),
                          interpolation='nearest')
                finish(ax, f'{d:%a %d %b %Y}' if i == 0 else f'{d:%a %d %b}')
                if i == 0:
                    sa.add_scalebar(ax, 20, '20 m', loc=(0.08, 0.07), lw=2.5)
        ax = axes(i, 1)
        if d in drone:
            arr, ext = sa.read_window(sa.drone_file(ev['series'], d), bounds, [1, 2, 3, 4],
                                      factor=4)
            ax.imshow(pu.rgb_display(arr), extent=ext)
            finish(ax)
            if i == 0:
                sa.add_scalebar(ax, 20, '20 m', loc=(0.08, 0.07), lw=2.5)
        else:
            ax.set_axis_off()
            ax.text(0.5, 0.5, 'no drone\nflight', transform=ax.transAxes, ha='center',
                    va='center', fontsize=9, color='0.55')
    # comparison column: a guayacan in flower on a flight day
    cmp = ev['compare']
    c_out = sa.dated_crown(cmp['tag'], cmp['drone_date'])
    c_box = sa.crown_bounds(c_out, 0, min_half=half)
    s = sa.scene_info(cmp['planet'])
    ax = axes(n, 0)
    arr, ext = sa.read_window(s['sr'], sa.shifted_bounds(pad(c_box), s['shift']), [1, 2, 3])
    lo, hi = sa.scene_stretch(s['sr'])
    ax.imshow(sa.planet_rgb(arr, lo, hi), extent=sa.unshift_extent(ext, s['shift']),
              interpolation='nearest')
    finish(ax, f'{pd.Timestamp(cmp["planet"][:8]):%a %d %b}', c_out, c_box)
    ax = axes(n, 1)
    arr, ext = sa.read_window(sa.drone_file(ev['series'], cmp['drone_date']), c_box,
                              [1, 2, 3, 4], factor=4)
    ax.imshow(pu.rgb_display(arr), extent=ext)
    finish(ax, f'{pd.Timestamp(cmp["drone_date"]):%a %d %b}', c_out, c_box)
    xc = left + n * (tile + gap) + sep - gap + tile / 2
    fig.text(xc / fig_w, (bottom + 2 * tile + 0.25 + 0.3) / fig.get_figheight(),
             'for comparison:\nanother crown, flight day', ha='center', va='bottom',
             fontsize=8.5, color='0.3')
    fig.text(0.25 / fig_w, (bottom + 1.5 * tile + 0.25) / fig.get_figheight(), 'PlanetScope',
             rotation=90, va='center', ha='center', fontsize=11)
    fig.text(0.25 / fig_w, (bottom + 0.5 * tile) / fig.get_figheight(), 'drone',
             rotation=90, va='center', ha='center', fontsize=11)
    sp = ev['species'].split()
    fig.text((left + (n * tile + (n - 1) * gap) / 2) / fig_w, 0.2 / fig.get_figheight(),
             f'One $\\it{{{sp[0]}}}$ $\\it{{{sp[1]}}}$ crown (cyan): yellow in Planet on '
             f'{cols[2]:%d %b %Y}, between drone flights on {drone[0]:%d %b} and {drone[1]:%d %b} '
             'that both show it without flowers', ha='center', fontsize=9.5)
    save(fig, outdir, 'missed_flowering_example')


if __name__ == '__main__':
    cli()
