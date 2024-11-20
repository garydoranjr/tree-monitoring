#!/usr/bin/env python
import click
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from calc_decid_resolution import (
    CadenceInterp, PCS, sample_cadence,
)

VMAX = 28.


def get_bounds(mean, sd, lb=PCS[0], ub=PCS[-1]):
    days = mean.dayofyear
    lb = norm.ppf(lb / 100., loc=days, scale=sd)
    ub = norm.ppf(ub / 100., loc=days, scale=sd)
    return lb, ub


def add_bound_dist(i, ax, lb, ub, marker, color='darkgreen'):
    ax.plot([lb, ub], [i, i], marker, color=color, markersize=3.)
    ax.plot([lb, ub], [i, i], '-', color=color, lw=2)

    if lb < 0:
        ax.plot([lb + 365., ub + 365.], [i, i], marker, color=color, markersize=3.)
        ax.plot([lb + 365., ub + 365.], [i, i], '-', color=color, lw=2)

    if ub > 365:
        ax.plot([lb - 365., ub - 365.], [i, i], marker, color=color, markersize=3.)
        ax.plot([lb - 365., ub - 365.], [i, i], '-', color=color, lw=2)


def make_cadence_plot(ax, cintp):
    dd = np.linspace(0, 365., 1024)[:-1]
    v = cintp(dd)

    V = np.column_stack(1024 * [
        np.linspace(0, VMAX, 1024)
    ])
    ax.imshow(
        V, extent=[0, 365., VMAX, 0.],
        cmap='Wistia', vmin=0, vmax=VMAX,
    )

    ax.plot(dd, v, 'k-', lw=2)

    ax.set_xticks([])
    ax.set_ylim(0, VMAX)
    ax.set_yticks(np.arange(0, VMAX + 1, 7.))
    ax.set_ylabel('Cadence (Days)', fontsize=16)
    ax.set_xlim(0, 365.)
    ax.set_aspect('auto')

    ticks = []
    for month in range(1, 13):
        ticks.append(pd.to_datetime(f'2021-{month:02d}-01').dayofyear)
    ax.set_xticks(ticks)
    ax.set_xticklabels([])
    ax.grid(color='gray', linestyle=':', linewidth=2)


def make_species_plot(ax, species, cintp):

    dd = np.linspace(0, 365., 1024)
    v = cintp(dd)
    V = np.vstack(1024 * [v])

    ax.imshow(
        V, extent=[0, 365., 0, 5 * len(species)],
        cmap='Wistia', vmin=0, vmax=VMAX,
    )

    i = 0
    sps = []
    for sp, (sta, end, _, _) in sorted(species.items())[::-1]:
        if i > 0: ax.plot([0, 365.], [5 * i, 5 * i], 'k:')
        y = (5 * i) + 2
        add_bound_dist(y, ax, *end, marker='^', color='darkgreen')
        y = (5 * i) + 3
        add_bound_dist(y, ax, *sta, marker='v', color='brown')
        i += 1
        sps.append(sp)

    ax.set_yticks([
        5 * i + 2.5 for i in range(len(sps))
    ])
    ax.set_yticklabels(sps)

    ticks = []
    for month in range(1, 13):
        ticks.append(pd.to_datetime(f'2021-{month:02d}-01').dayofyear)
    ax.set_xticks(ticks)
    ax.set_xticklabels([
        'Jan', 'Feb', 'Mar',
        'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep',
        'Oct', 'Nov', 'Dec',
    ])

    ax.set_xlim(0, 365.)
    ax.set_aspect('auto')


def add_dist(x, ax, pcs, marker, color='darkgreen'):
    ax.plot(len(pcs)*[x], pcs, marker, color=color, markersize=3.)
    ax.plot([x, x], [np.min(pcs), np.max(pcs)], '-', color=color, lw=2)


def make_estimate_plots(ax, species, cintp):

    i = 0
    sps = []
    for sp, (_, _, sta, end) in sorted(species.items()):
        if i > 0: ax.plot([5 * i, 5 * i], [0, VMAX], 'k:')
        sta_pcs = sample_cadence(cintp, *sta)
        end_pcs = sample_cadence(cintp, *end)
        y = (5 * i) + 2
        add_dist(y, ax, end_pcs, marker='^', color='darkgreen')
        y = (5 * i) + 3
        add_dist(y, ax, sta_pcs, marker='v', color='brown')
        ax.text(
            (5 * i) + 2.5, 0.95 * VMAX, sp,
            rotation=90, va='top', ha='center'
        )
        i += 1
        sps.append(sp)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, VMAX)
    ax.set_xlim(0, 5 * i)


@click.command()
@click.argument('cadencefile')
@click.argument('decidfile')
@click.argument('outputfile')
def main(cadencefile, decidfile, outputfile):

    cdata = np.load(cadencefile)
    med_idx = np.where(cdata['percentiles'] == 50.)[0][0]
    cintp = CadenceInterp(cdata['dates'], cdata['cadence'][:, med_idx])

    decid_df = pd.read_excel(decidfile)

    species = {}

    for i, row in decid_df.iterrows():
        spcode = row['spcode']
        start_mean = row['StartDate_mean']
        start_sd = row['StartDate_sd']
        end_mean = row['EndDate_mean']
        end_sd = row['EndDate_sd']

        if row['group'] == 'evergreen': continue

        # N/A check
        if pd.isna(spcode): continue
        if pd.isna(start_mean): continue
        if pd.isna(end_mean): continue
        if pd.isna(start_sd): continue
        if pd.isna(end_sd): continue

        species[spcode] = (
            get_bounds(start_mean, start_sd),
            get_bounds(end_mean, end_sd),
            (start_mean, start_sd),
            (end_mean, end_sd),
        )

    fig = plt.figure(constrained_layout=True, figsize=(14, 9))
    spec = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    ax_c = fig.add_subplot(spec[0, 0])
    ax_s = fig.add_subplot(spec[1:, 0])
    ax_i = fig.add_subplot(spec[0, 1:])

    make_cadence_plot(ax_c, cintp)
    make_species_plot(ax_s, species, cintp)
    make_estimate_plots(ax_i, species, cintp)

    fig.savefig(outputfile)

if __name__ == '__main__':
    main()
