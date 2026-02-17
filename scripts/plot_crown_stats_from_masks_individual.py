#!/usr/bin/env python
import click
import numpy as np
import xarray as xr
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages


def plot(ds, outputfile):

    species = ds["species"].values
    species = [sp for sp in np.unique(species) if len(sp) > 0]

    dates = ds.date.values
    figs = []

    for sp in tqdm(species, 'Plotting'):

        sp_ds = ds.where(ds["species"] == sp, drop=True)

        tags = sp_ds.tag.values
        n_tags = len(tags)

        fig, ax = plt.subplots(figsize=(6, 4))
        figs.append(fig)

        ax.set_title(f'{sp} (n = {n_tags})', fontsize=16)

        for tag in tags:
            tag_ds = sp_ds.sel(tag=tag)

            ax.plot(
                dates,
                tag_ds["flowering_probability"] > 0.5,
                color="indigo",
                alpha=0.2,
                linewidth=1,
            )

            ax.plot(
                dates,
                tag_ds["deciduous_probability"] > 0.5,
                color="tan",
                alpha=0.2,
                linewidth=1,
            )

        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")

    with PdfPages(outputfile) as pdf:
        for fig in tqdm(figs, 'Saving'):
            pdf.savefig(fig)


def plot_grid(ds, outputfile):

    species_vals = ds["species"].values
    species = [sp for sp in np.unique(species_vals) if len(sp) > 0]

    figs = []

    for sp in tqdm(species, "Plotting (grid)"):

        sp_ds = ds.where(ds["species"] == sp, drop=True)

        tags = sp_ds.tag.values
        dates = sp_ds.date.values.astype("datetime64[ns]")
        n_tags = len(tags)

        if n_tags == 0:
            continue

        # ---- compute variable time widths ----
        # width of each cell is the time until the next observation
        dt = np.diff(dates).astype("timedelta64[s]").astype(float)
        dt = np.append(dt, dt[-1])  # repeat last interval
        x_edges = np.concatenate(([0], np.cumsum(dt)))
        total_width = x_edges[-1]

        fig, ax = plt.subplots(
            figsize=(8, 8),
        )
        figs.append(fig)

        ax.set_title(f"{sp} (n = {n_tags})", fontsize=16)

        # ---- draw cells ----
        for i, tag in enumerate(tags):
            tag_ds = sp_ds.sel(tag=tag)

            flowering = (tag_ds["flowering_probability"].values >= 0.5)
            deciduous = (tag_ds["deciduous_probability"].values >= 0.5)

            # vertical placement
            y_flower = 2 * i + 1
            y_decid = 2 * i

            for j in range(len(dates)):
                width = dt[j]
                x0 = x_edges[j]

                if flowering[j]:
                    ax.add_patch(
                        Rectangle(
                            (x0, y_flower),
                            width,
                            1,
                            facecolor="indigo",
                            edgecolor="none",
                        )
                    )

                if deciduous[j]:
                    ax.add_patch(
                        Rectangle(
                            (x0, y_decid),
                            width,
                            1,
                            facecolor="tan",
                            edgecolor="none",
                        )
                    )

            ax.plot([0, total_width], [2*i, 2*i], 'k-', alpha=0.1)

        # ---- axes formatting ----
        ax.set_xlim(0, total_width)
        ax.set_ylim(0, 2 * n_tags)

        ax.set_yticks([2 * i + 1 for i in range(n_tags)])
        ax.set_yticklabels(tags)

        ax.set_xlabel("Time")
        ax.set_ylabel("Individual")

        # ---- yearly x ticks ----
        years = np.array([d.astype("datetime64[Y]") for d in dates])
        unique_years, first_idx = np.unique(years, return_index=True)

        ax.set_xticks(x_edges[first_idx])
        ax.set_xticklabels(
            [str(y.astype("datetime64[Y]"))[:4] for y in unique_years],
            rotation=0,
        )

        # legend (manual)
        ax.add_patch(Rectangle((0, -2), 0, 0, color="indigo", label="Flowering"))
        ax.add_patch(Rectangle((0, -2), 0, 0, color="tan", label="Deciduous"))
        ax.legend(loc="upper right")

        ax.spines[["top", "right"]].set_visible(False)

    with PdfPages(outputfile) as pdf:
        for fig in tqdm(figs, "Saving"):
            pdf.savefig(fig)


@click.command()
@click.argument('statfile')
@click.argument('outputfile')
def main(statfile, outputfile):
    ds = xr.open_dataset(statfile)
    plot_grid(ds, outputfile)


if __name__ == '__main__':
    main()
