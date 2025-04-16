#!/usr/bin/env python
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from fit_count_models import PoissonMixtureModel
from analyze_windowed_counts import shift_months
from event_summary_stats import get_doy, get_obs_probabilities


VMAX = 49.
DMIN = 1
DMAX = 49.


def get_doy(df, field='event_peak'):
    return np.array([
        float(p.dayofyear)
        for p in pd.to_datetime(df[field])
    ])


def make_cadence_plot(fig, ax, models, df_s):

    durations = np.linspace(DMIN, DMAX, 100)
    P = np.column_stack([
        pmm.capture_prob(durations)
        for pmm in models
    ])
    sep = pd.to_datetime(f'2021-09-01').dayofyear
    P = np.hstack([
        P[:, sep-1:], P[:, :sep-1]
    ])

    im = ax.imshow(P, extent=[0, 365, DMAX, DMIN], vmin=0, vmax=1)
    ax.set_ylim(DMIN, DMAX)
    ax.set_yticks(np.arange(7., DMAX + 1, 7.))
    ax.set_xlabel('Month', fontsize=16)
    ax.set_ylabel('Duration (Days)', fontsize=16)

    cbar = fig.colorbar(im)
    cbar.set_label('Probability of Observation', fontsize=16)

    ticks = []
    for month in range(1, 13):
        ticks.append(pd.to_datetime(f'2021-{month:02d}-01').dayofyear)
    labels = [
        'Jan', 'Feb', 'Mar',
        'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep',
        'Oct', 'Nov', 'Dec',
    ]

    ticks, labels = shift_months(sep, ticks, labels)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)

    ax.set_xlim(0, 365.)
    ax.set_aspect('auto')
    ax.grid(color='k', linestyle=':', linewidth=2)

    peak = get_doy(df_s)
    dl = np.minimum(df_s['event_length'], DMAX)
    probs = get_obs_probabilities(models, peak, df_s['event_length'])

    # Shift peaks
    idx_s = (peak >= sep)
    idx_e = (peak < sep)
    pnew = np.hstack([peak[idx_s], peak[idx_e]])
    pnew = (pnew - sep) % 365.

    ax.plot(pnew, dl, 'ro')

    return probs


def make_comparison_plot(ax, probs):

    ax.hist(probs, bins=np.linspace(0, 1, 21), density=True, histtype='step', ec='k')
    ymin, ymax = ax.get_ylim()
    avg = np.average(probs)
    ax.plot([avg, avg], [ymin, ymax], 'r-', lw=3)
    ax.set_ylim(ymin, ymax)

    ax.set_xlim(0, 1)
    ax.set_xlabel('Observation Probability', fontsize=16)
    ax.set_ylabel('Density', fontsize=16)


@click.command()
@click.argument('modelfile')
@click.argument('eventfile')
@click.argument('outputfile')
def main(modelfile, eventfile, outputfile):

    data = np.load(modelfile)
    models = PoissonMixtureModel.from_dict(data)

    df = pd.read_csv(eventfile)
    unique_species = sorted(np.unique(df['species']))
    #unique_species = unique_species[:10]

    figs = []
    for sp in unique_species:
        df_s = df.loc[df['species'] == sp]

        fig = plt.figure(figsize=(16, 7))
        ax = fig.add_subplot(121)
        ax.set_title(sp, fontsize=18)
        probs = make_cadence_plot(fig, ax, models, df_s)

        ax = fig.add_subplot(122)
        make_comparison_plot(ax, probs)

        figs.append(fig)

    with PdfPages(outputfile) as pdf:
        for fig in figs:
            pdf.savefig(fig)


if __name__ == '__main__':
    main()
