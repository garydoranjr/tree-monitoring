#!/usr/bin/env python
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from calc_decid_resolution import CadenceInterp


VMAX = 49.


def get_doy(df, field='event_peak'):
    return np.array([
        float(p.dayofyear)
        for p in pd.to_datetime(df[field])
    ])


def make_cadence_plot(ax, cintp, df_s):
    dd = np.linspace(0, 365., 1024)[:-1]
    v = cintp(dd)

    ax.plot(dd, v, 'k-', lw=2)

    peak = get_doy(df_s)
    dl = np.minimum(df_s['event_length'], VMAX)
    ax.plot(peak, dl, 'ro')

    ax.set_xticks([])
    ax.set_ylim(0, VMAX)
    ax.set_yticks(np.arange(0, VMAX + 1, 7.))
    ax.set_ylabel('Cadence/Duration (Days)', fontsize=16)
    ax.set_xlim(0, 365.)
    ax.set_aspect('auto')

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
    ax.grid(color='gray', linestyle=':', linewidth=2)


def make_comparison_plot(ax, cintp, df_s):

    peak = get_doy(df_s)
    v = cintp(peak)
    #dl = df_s['event_length']
    dl = np.minimum(df_s['event_length'], VMAX)
    ax.plot([0, VMAX], [0, VMAX], 'k:')
    ax.plot(v, dl, 'ro')

    #ax.set_xticks([])
    ax.set_xlim(0, VMAX)
    ax.set_ylim(0, VMAX)
    ax.set_xticks(np.arange(0, VMAX + 1, 7.))
    ax.set_yticks(np.arange(0, VMAX + 1, 7.))
    ax.set_aspect('equal')
    ax.set_xlabel('Cadence (Days)', fontsize=16)
    ax.set_ylabel('Duration (Days)', fontsize=16)
    #ax.set_xlim(0, 365.)
    #ax.set_aspect('auto')

    #ticks = []
    #for month in range(1, 13):
    #    ticks.append(pd.to_datetime(f'2021-{month:02d}-01').dayofyear)
    #ax.set_xticks(ticks)
    #ax.set_xticklabels([
    #    'Jan', 'Feb', 'Mar',
    #    'Apr', 'May', 'Jun',
    #    'Jul', 'Aug', 'Sep',
    #    'Oct', 'Nov', 'Dec',
    #])
    #ax.grid(color='gray', linestyle=':', linewidth=2)


@click.command()
@click.argument('cadencefile')
@click.argument('eventfile')
@click.argument('outputfile')
def main(cadencefile, eventfile, outputfile):

    cdata = np.load(cadencefile)
    med_idx = np.where(cdata['percentiles'] == 50.)[0][0]
    cintp = CadenceInterp(cdata['dates'], cdata['cadence'][:, med_idx])

    df = pd.read_csv(eventfile)
    unique_species = sorted(np.unique(df['species']))
    #unique_species = sorted(SPECIES)

    figs = []
    for sp in unique_species:
        df_s = df.loc[df['species'] == sp]

        fig = plt.figure(figsize=(16, 7))
        ax = fig.add_subplot(121)
        ax.set_title(sp, fontsize=18)
        make_cadence_plot(ax, cintp, df_s)

        ax = fig.add_subplot(122)
        make_comparison_plot(ax, cintp, df_s)

        figs.append(fig)

    with PdfPages(outputfile) as pdf:
        for fig in figs:
            pdf.savefig(fig)


if __name__ == '__main__':
    main()
