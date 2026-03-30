#!/usr/bin/env python
import os
import click
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import geopandas as gpd
from datetime import datetime
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def hist(x, title=None, yscale='linear'):
    fig, ax = plt.subplots(figsize=(16, 7))

    if title is not None:
        ax.set_title(title, fontsize=18)

    ax.hist(x, bins=np.linspace(0, 100, 11))

    ax.set_yscale(yscale)

    ax.set_ylabel('Count', fontsize=16)

    return fig


def pie(x, title=None):
    fig, ax = plt.subplots(figsize=(16, 7))

    if title is not None:
        ax.set_title(title, fontsize=18)

    counts = x.value_counts(normalize=True) * 100

    # Hide labels for slices below threshold
    threshold = 3  # percent
    labels = [
        name if pct >= threshold else "" 
        for name, pct in zip(counts.index, counts.values)
    ]

    ax.pie(counts, labels=labels, labeldistance=1.1)

    return fig


def fname_to_date(f):
    base = os.path.basename(f)
    datestr = "_".join(base.split("_")[2:5])
    return datetime.strptime(datestr, "%Y_%m_%d")


def get_drone_dates(dronedir):
    if dronedir is None: return []
    files = glob(os.path.join(dronedir, '*.tif'))
    return list(map(fname_to_date, files))


def plot_phenology_labels(df: pd.DataFrame, date_list: list[datetime]) -> plt.Figure:
    """
    Plot stacked bar counts of leafing > 0 / == 0 and isFlowering == 'yes' / 'no' per date,
    with vertical lines marking all dates in date_list.

    Args:
        df:        DataFrame with columns 'date' (str, 'YYYY_MM_DD'),
                   'leafing' (numeric), and 'isFlowering' (str).
        date_list: List of datetime objects to mark on the x-axis.

    Returns:
        A matplotlib Figure.
    """
    # --- Parse dates in the dataframe ---
    df = df.copy()
    df["date_parsed"] = pd.to_datetime(df["date"], format="%Y_%m_%d")

    # --- Aggregate counts per date ---
    grouped = df.groupby("date_parsed")
    counts = pd.DataFrame({
        "leafing_yes": grouped.apply(lambda g: (g["leafing"] > 0).sum()),
        "leafing_no":  grouped.apply(lambda g: (g["leafing"] == 0).sum()),
        "flowering_yes": grouped.apply(lambda g: (g["isFlowering"] == "yes").sum()),
        "flowering_no":  grouped.apply(lambda g: (g["isFlowering"] != "yes").sum()),
    })

    # --- Build figure ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    bar_width = pd.Timedelta(days=5)

    # Leafing subplot
    ax1.bar(counts.index, counts["leafing_yes"], width=bar_width, label="Leafing",  color="steelblue",     alpha=1.0)
    ax1.bar(counts.index, counts["leafing_no"],  width=bar_width, label="Not Leafing", color="lightsteelblue", alpha=1.0,
            bottom=counts["leafing_yes"])

    # Flowering subplot
    ax2.bar(counts.index, counts["flowering_yes"], width=bar_width, label="Flowering",     color="darkorange", alpha=1.0)
    ax2.bar(counts.index, counts["flowering_no"],  width=bar_width, label="Not Flowering", color="moccasin",   alpha=1.0,
            bottom=counts["flowering_yes"])

    # --- Vertical lines for every date in date_list ---
    for d in date_list:
        ax1.axvline(x=d, color="gray", linewidth=0.8, linestyle="--", alpha=0.25)
        ax2.axvline(x=d, color="gray", linewidth=0.8, linestyle="--", alpha=0.25)

    ax1.set_ylabel("Number of labels")
    ax1.set_title("Leafing")
    ax1.legend(loc="upper left")

    ax2.set_ylabel("Number of labels")
    ax2.set_title("Flowering")
    ax2.legend(loc="upper left")
    ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.suptitle("Phenology label counts over time", fontsize=13, y=0.97)
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()

    return fig


@click.command()
@click.argument('labelfile')
@click.argument('outputfile')
@click.option('-d', '--dronedir', default=None)
def main(labelfile, outputfile, dronedir):

    drone_dates = get_drone_dates(dronedir)

    labels = gpd.read_file(labelfile, layer='flowering_dataset')

    labels['isFlowering'] = labels['isFlowering'].apply(lambda v: 'no' if v is None else v)
    labels['floweringIntensity'] = labels['floweringIntensity'].apply(lambda v: 0.0 if np.isnan(v) else v)
    labels['isFruiting'] = labels['isFruiting'].apply(lambda v: 'no' if v is None else v)
    labels['newLeaves'] = labels['newLeaves'].apply(lambda v: 'no' if v is None else v)

    labels = labels.drop([
        'geometry',
        'score',
        'iou',
        'tag',
        'isFlowerin',
        'floweringI',
        'area',
    ], axis=1)


    figs = []

    figs.append(plot_phenology_labels(labels, drone_dates))

    figs.append(hist(labels['leafing'], title='Leafing', yscale='log'))
    figs.append(hist(labels['floweringIntensity'], title='Flowering Intensity'))

    figs.append(pie(labels['isFlowering'], title='Flowering?'))
    figs.append(pie(labels['isFruiting'], title='Fruiting?'))
    figs.append(pie(labels['newLeaves'], title='New Leaves?'))
    figs.append(pie(labels['latin'], title='Species'))

    with PdfPages(outputfile) as pdf:
        for fig in tqdm(figs, 'Saving'):
            pdf.savefig(fig)


if __name__ == '__main__':
    main()
