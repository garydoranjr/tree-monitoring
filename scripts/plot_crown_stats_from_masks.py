#!/usr/bin/env python
import click
import xarray as xr
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot(ds, outputfile):

    species = ds.species.values
    dates = ds.date.values

    figs = []

    for sp in tqdm(species, 'Plotting'):
        if len(sp) == 0: continue
        sp_da = ds.sel(species=sp)

        n_tags = int(sp_da['n_tags'])
        #if n_tags < 5: continue

        fig, ax = plt.subplots(figsize=(6, 4))
        figs.append(fig)

        ax.set_title(f'{sp} (n = {n_tags})', fontsize=16)
        ax.plot(dates, sp_da['flowering_probability_mean'],
            color='indigo', linestyle='-', marker='.',
            label='Flowering',
        )
        ax.plot(dates, sp_da['deciduous_probability_mean'],
            color='tan', linestyle='-', marker='.',
            label='Deciduousness',
        )
        ax.set_ylim(0, 1)
        ax.legend()

    with PdfPages(outputfile) as pdf:
        for fig in tqdm(figs, 'Saving'):
            pdf.savefig(fig)


@click.command()
@click.argument('statfile')
@click.argument('outputfile')
def main(statfile, outputfile):
    ds = xr.open_dataset(statfile)

    flowering_mean = (
        (ds["flowering_probability"] > 0.5)
        .groupby(ds["species"])
        .mean(dim="tag")
    )

    deciduous_mean = (
        (ds["deciduous_probability"] > 0.5)
        .groupby(ds["species"])
        .mean(dim="tag")
    )

    summary = xr.Dataset(
        {
            "flowering_probability_mean": flowering_mean,
            "deciduous_probability_mean": deciduous_mean,
        }
    )

    n_tags = (
        ds["species"]
        .groupby(ds["species"])
        .count(dim="tag")
    )
    summary["n_tags"] = n_tags

    plot(summary, outputfile)


if __name__ == '__main__':
    main()
