#!/usr/bin/env python
import click
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import poisson
from scipy.optimize import minimize, LinearConstraint
from collections import defaultdict
import matplotlib.pyplot as plt


from fit_count_models import PoissonMixtureModel


def shift_months(start, ticks, labels):
    ticks = np.asarray(ticks)
    labels = np.asarray(labels)
    idx_s = (ticks >= start)
    idx_e = (ticks < start)
    tnew = np.hstack([ticks[idx_s], ticks[idx_e]])
    tnew = (tnew - start) % 365.
    lnew = np.hstack([labels[idx_s], labels[idx_e]])
    return tnew, lnew


@click.command()
@click.argument('countfile')
@click.argument('modelfile')
@click.argument('outputfile')
def main(countfile, modelfile, outputfile):
    data = np.load(modelfile)

    models = PoissonMixtureModel.from_dict(data)

    DMIN = 1
    DMAX = 28
    durations = np.linspace(DMIN, DMAX, 100)

    P = []
    for pmm in tqdm(models):
        P.append(pmm.capture_prob(durations))
    P = np.column_stack(P)
    sep = pd.to_datetime(f'2021-09-01').dayofyear
    P = np.hstack([
        P[:, sep-1:], P[:, :sep-1]
    ])

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(P, extent=[0, 365, DMAX, DMIN], vmin=0, vmax=1)
    ax.set_ylim(DMIN, DMAX)
    ax.set_yticks(np.arange(7., DMAX + 1, 7.))

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

    ax.set_xlabel('Month', fontsize=16)
    ax.set_ylabel('Event Duration (Days)', fontsize=16)

    cbar = fig.colorbar(im)
    cbar.set_label('Probability of Observation', fontsize=16)

    fig.savefig(outputfile)


if __name__ == '__main__':
    main()
