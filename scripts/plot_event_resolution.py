#!/usr/bin/env python
import click
import numpy as np
import pandas as pd
from scipy.stats import vonmises
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from calc_decid_resolution import (
    CadenceInterp, PCS
)
from calc_phenology_resolution import sample_cadence

VMAX = 28.
INCLUDE = {
    'ALSB': 'ALSEBL',
    'ANAE': 'ANAEX',
    'APEM': 'APEIME',
    'ASTG': 'AST2GR',
    'CAVP': 'CAVAPL',
    'CEIP': 'CEIBPE',
    'CORA': 'CORDAL',
    'DIPP': 'DIPTPA',
    'GUAS': 'GUAPST',
    'HURC': 'HURACR',
    'JACC': 'JAC1CO',
    'LUE1': 'LUEHSE',
    'PLAP': 'PLA1PI',
    'PLAE': 'PLA2EL',
    'SPOM': 'SPONMO',
    'SPOR': 'SPONRA',
    'STEA': 'STERAP',
    'TABG': 'TAB1GU',
    'TABR': 'TAB1RO',
    'TERO': 'TERMOB',
    'ZANB': 'ZANTBE',
    'ZAN1': 'ZANTP1',
}


def get_bounds(mean, kappa, lb=PCS[0], ub=PCS[-1]):
    t = lambda x: ((365 * x) / (2 * np.pi))
    lb = vonmises.ppf(lb / 100., loc=mean, kappa=kappa)
    ub = vonmises.ppf(ub / 100., loc=mean, kappa=kappa)
    return t(lb), t(ub)


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
        V, extent=[0, 365., 0, 3 * len(species)],
        cmap='Wistia', vmin=0, vmax=VMAX,
    )

    i = 0
    sps = []
    for sp, (evt, _) in sorted(species.items())[::-1]:
        if i > 0: ax.plot([0, 365.], [3 * i, 3 * i], 'k:')
        y = (3 * i) + 1.5
        add_bound_dist(y, ax, *evt, marker='o', color='k')
        i += 1
        sps.append(sp)

    ax.set_yticks([
        (3 * i) + 1.5 for i in range(len(sps))
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
    for sp, (_, evt) in sorted(species.items()):
        if i > 0: ax.plot([3 * i, 3 * i], [0, VMAX], 'k:')
        evt_pcs = sample_cadence(cintp, *evt)
        y = (3 * i) + 1.5
        add_dist(y, ax, evt_pcs, marker='o', color='k')
        ax.text(
            (3 * i) + 1.5, 0.95 * VMAX, sp,
            rotation=90, va='top', ha='center'
        )
        i += 1
        sps.append(sp)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, VMAX)
    ax.set_xlim(0, 3 * i)


@click.command()
@click.argument('cadencefile')
@click.argument('phenofile')
@click.argument('outputfile')
def main(cadencefile, phenofile, outputfile):

    cdata = np.load(cadencefile)
    med_idx = np.where(cdata['percentiles'] == 50.)[0][0]
    cintp = CadenceInterp(cdata['dates'], cdata['cadence'][:, med_idx])

    pheno_df = pd.read_csv(phenofile)
    pheno_df = pheno_df.loc[pheno_df['sp'].isin(INCLUDE.keys())]
    pheno_df = pheno_df.loc[pheno_df['model'] == 'M2A']
    pheno_df = pheno_df.loc[pheno_df['type'] == 'frt']
    #pheno_df = pheno_df.loc[pheno_df['type'] == 'flw']
    #pheno_df = pheno_df.loc[pheno_df['deltaAIC'] == 0]

    species = {}

    for i, row in pheno_df.iterrows():
        spcode = INCLUDE[row['sp']]
        mean = row['mean1']
        kappa = row['kappa1']

        species[spcode] = (
            get_bounds(mean, kappa),
            (mean, kappa),
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
