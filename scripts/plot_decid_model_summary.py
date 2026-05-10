#!/usr/bin/env python
import click
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from fit_empirical_count_models import EmpiricalCountModel
from plot_trap_summary import make_cadence_plot, make_comparison_plot


@click.command()
@click.argument('modelfile')
@click.argument('decidfile')
@click.argument('outputfile')
def main(modelfile, decidfile, outputfile):

    model = EmpiricalCountModel.load(modelfile)

    df = pd.read_csv(decidfile).rename(columns={
        'decid_peak': 'event_peak',
        'decid_length': 'event_length',
    })
    unique_species = sorted(np.unique(df['species']))

    figs = []
    for sp in tqdm(unique_species, 'Plotting'):
        df_s = df.loc[df['species'] == sp]

        fig = plt.figure(figsize=(16, 7))
        ax = fig.add_subplot(121)
        ax.set_title(sp, fontsize=18)
        probs = make_cadence_plot(fig, ax, model, df_s)

        ax = fig.add_subplot(122)
        make_comparison_plot(ax, probs)

        figs.append(fig)

    with PdfPages(outputfile) as pdf:
        for fig in tqdm(figs, 'Saving'):
            pdf.savefig(fig)


if __name__ == '__main__':
    main()
