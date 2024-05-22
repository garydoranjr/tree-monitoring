#!/usr/bin/env python
import os
import click
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ecdf
from datetime import datetime
import matplotlib.pyplot as plt

from cloud_coverage_plot import load_data


def cumulative_average(pp, pcs):
    pcs = np.asarray(pcs)
    cavg = np.array([
        np.average(pcs[pcs >= p])
        if np.sum(pcs >= p) > 0 else 0
        for p in pp
    ])
    cavg[-1] = 1e-9
    return cavg


@click.command()
@click.argument('inputfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
def main(inputfile, outputfile):

    dates, pcs = load_data(inputfile)

    dist = ecdf(pcs)

    period = (np.max(dates) - np.min(dates)).days
    factor = period / len(pcs)

    fig, ax = plt.subplots(figsize=(8, 6))

    pp = np.linspace(0, 1, 101)
    sf = dist.sf.evaluate(pp)
    cavg = cumulative_average(pp, pcs)
    sf[0] = 1.0
    sf[-1] = 1e-9

    days_btw_obs = 1.0 / (factor * sf)
    days_btw_crn = 1.0 / (factor * cavg * sf)

    best_idx = np.argmin(days_btw_crn)
    best = pp[best_idx]
    best_crn = days_btw_crn[best_idx]

    ax.plot(pp, days_btw_crn, 'g-', zorder=10, label='Individual Crown')
    ax.plot(pp, days_btw_obs, 'k-', label='Full Observation')
    #ax.plot([best, best], [0, best_crn], 'r-', label=f'Optimum = {best:.2f}')
    ax.set_ylim(0, 4.0)
    ax.set_xlim(0, 1.0)
    ax.legend(loc='upper left', fontsize=14)
    ax.set_xlabel('Clear Fraction Threshold', fontsize=16)
    ax.set_ylabel('Average Days between Observations', fontsize=16)

    fig.savefig(outputfile, bbox_inches='tight')


if __name__ == '__main__':
    main()
