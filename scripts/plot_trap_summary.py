#!/usr/bin/env python
import click
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from fit_empirical_count_models import EmpiricalCountModel
from analyze_windowed_counts import shift_months
from event_summary_stats import get_doy, get_obs_probabilities


VMAX = 49.
DMIN = 1
DMAX = 49.
DAYS_PER_WEEK = 7.


def make_cadence_plot(fig, ax, cax, model, df_s):

    durations = np.linspace(DMIN, DMAX, 100)
    P = np.column_stack([
        model.capture_prob(i, durations)
        for i in np.arange(365)
    ])
    sep = pd.to_datetime(f'2021-09-01').dayofyear
    P = np.hstack([
        P[:, sep-1:], P[:, :sep-1]
    ])

    wmin = DMIN / DAYS_PER_WEEK
    wmax = DMAX / DAYS_PER_WEEK
    im = ax.imshow(P, extent=[0, 365, wmax, wmin], vmin=0, vmax=1)
    ax.set_ylim(wmin, wmax)
    ax.set_yticks(np.arange(1., wmax + 1, 1.))
    ax.set_xlabel('Month', fontsize=16)
    ax.set_ylabel('Duration (Weeks)', fontsize=16)

    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.yaxis.tick_left()
    cbar.ax.yaxis.set_label_position('left')
    cbar.set_label('Probability of Observing Event', fontsize=16)

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
    dl = np.minimum(df_s['event_length'], DMAX) / DAYS_PER_WEEK
    probs = get_obs_probabilities(model, peak, df_s['event_length'])

    # Shift peaks
    idx_s = (peak >= sep)
    idx_e = (peak < sep)
    pnew = np.hstack([peak[idx_s], peak[idx_e]])
    pnew = (pnew - sep) % 365.

    ax.plot(pnew, dl, 'ro')

    return probs


def make_comparison_plot(ax, probs):

    ax.hist(
        probs, bins=np.linspace(0, 1, 21), density=False,
        histtype='step', ec='k', orientation='horizontal',
    )
    xmin, xmax = ax.get_xlim()
    avg = np.average(probs)
    ax.plot([xmin, xmax], [avg, avg], 'r-', lw=3)
    ax.set_xlim(xmin, xmax)

    ax.set_ylim(0, 1)
    ax.tick_params(left=False, labelleft=False)
    ax.set_xlabel('Count', fontsize=16)


@click.command()
@click.argument('modelfile')
@click.argument('eventfile')
@click.argument('outputfile')
def main(modelfile, eventfile, outputfile):

    model = EmpiricalCountModel.load(modelfile)

    df = pd.read_csv(eventfile)
    unique_species = sorted(np.unique(df['species']))
    #unique_species = unique_species[:10]

    figs = []
    for sp in tqdm(unique_species, 'Plotting'):
        df_s = df.loc[df['species'] == sp]

        fig = plt.figure(figsize=(16, 7))
        gs_outer = fig.add_gridspec(1, 2, width_ratios=[10, 5], wspace=0.2)
        ax_cadence = fig.add_subplot(gs_outer[0, 0])
        gs_right = gs_outer[0, 1].subgridspec(
            1, 2, width_ratios=[0.5, 5], wspace=0.0,
        )
        ax_cbar = fig.add_subplot(gs_right[0, 0])
        ax_hist = fig.add_subplot(gs_right[0, 1])

        ax_cadence.set_title(sp, fontsize=18)
        probs = make_cadence_plot(fig, ax_cadence, ax_cbar, model, df_s)
        make_comparison_plot(ax_hist, probs)

        figs.append(fig)

    with PdfPages(outputfile) as pdf:
        for fig in tqdm(figs, 'Saving'):
            pdf.savefig(fig)


if __name__ == '__main__':
    main()
