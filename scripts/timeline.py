#!/usr/bin/env python
import click
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


def load_dates(datefile):
    with open(datefile, 'r') as f:
        return [
            datetime.strptime(line.strip(), '%Y%m%d')
            for line in f
        ]


@click.command()
@click.argument('dronedatefile')
@click.argument('planetdatefile')
@click.argument('outputfile')
def main(dronedatefile, planetdatefile, outputfile):

    dronedates = load_dates(dronedatefile)
    planetdates = load_dates(planetdatefile)
    all_dates = dronedates + planetdates
    start = datetime(np.min(all_dates).year, 1, 1)
    end = datetime(np.max(all_dates).year, 6, 1)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    ax.set_xticks(
        pd.date_range(start, end, freq='YS'),
        range(start.year, end.year + 1)
    )

    it = zip(
        ('Drone', 'Planet Scope'),
        ('r-', 'b-'),
        (2.5, 5),
        (dronedates, planetdates)
    )

    for label, style, y, dates in list(it)[::-1]:
        for date in dates:
            ax.plot([date, date], [0, y], style, label=label)
            label = None

    ax.plot([start, end], [0, 0], 'k-', lw=3)

    ax.legend(loc='upper right', fontsize=16)

    ax.spines[["left", "top", "right", "bottom"]].set_visible(False)
    ax.spines[["bottom"]].set_position(("axes", 0.5))
    ax.yaxis.set_visible(False)
    ax.set_xlim(start, end)
    ax.set_ylim(-10, 10)
    ax.tick_params(axis='x', labelsize=16)

    fig.savefig(outputfile, bbox_inches='tight')


if __name__ == '__main__':
    main()
