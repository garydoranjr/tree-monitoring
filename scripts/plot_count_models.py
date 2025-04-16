#!/usr/bin/env python
import click
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from fit_count_models import PoissonMixtureModel


def plot_model(model, counts, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(title, fontsize=18)
    cc = np.arange(61)
    pmf = model.pmf(cc)

    bins = np.arange(62)
    hist, edges = np.histogram(counts, bins, density=True)

    ax.plot(cc, pmf, 'r-', lw=3)
    ax.plot(cc, hist, 'ko')

    ax.set_xlabel('Count', fontsize=16)
    ax.set_ylabel('Probability Density', fontsize=16)
    ax.set_yticks([])

    return fig


@click.command()
@click.argument('inputfile')
@click.argument('modelfile')
@click.argument('outputfile')
def main(inputfile, modelfile, outputfile):
    data = np.load(inputfile)
    counts = data['counts']
    win_size = data['window_size']

    data = np.load(modelfile)
    models = PoissonMixtureModel.from_dict(data)


    figs = [
        plot_model(mi, ci, f'DOY = {i+1}')
        for i, (mi, ci) in tqdm(list(enumerate(zip(models, counts))), 'Plotting')
    ]

    with PdfPages(outputfile) as pdf:
        for fig in tqdm(figs, 'Saving'):
            pdf.savefig(fig)


if __name__ == '__main__':
    main()
