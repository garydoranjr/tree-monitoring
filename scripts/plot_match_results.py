#!/usr/bin/env python
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


@click.command()
@click.argument('resultfiles', nargs=-1)
@click.argument('outputfile')
def main(resultfiles, outputfile):

    df = pd.concat([pd.read_csv(f) for f in resultfiles], ignore_index=True)

    df['displacement'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['error'] = np.sqrt(df['xerr']**2 + df['yerr']**2)

    bins = np.linspace(0, 14, 15)
    matches = df.groupby(pd.cut(df['displacement'], bins=bins), dropna=False, observed=False)['match_fraction'].mean()
    errors = df.groupby(pd.cut(df['displacement'], bins=bins), dropna=False, observed=False)['error'].mean()

    good = np.logical_not(np.logical_or(np.isnan(matches), np.isnan(errors)))

    fig, ax = plt.subplots(figsize=(12, 6))

    centers = 3 * ((bins[:-1] + bins[1:]) / 2.0)

    centers = centers[good]
    matches = matches[good]
    errors = errors[good]

    ax.plot(centers, matches, 'ko-')
    ax.set_ylabel('Success Fraction', fontsize=16)
    ax.set_xlabel('Initial Displacement (m)', fontsize=16)
    ax.set_ylim(-0.1, 1.1)

    ax2 = ax.twinx()

    ax2.plot(centers, 3*errors, 'ro-')
    ax2.set_ylabel('Alignment Error (m)', fontsize=16, color='red')

    plt.show()
    exit()

    fig.savefig(outputfile)


if __name__ == '__main__':
    main()
