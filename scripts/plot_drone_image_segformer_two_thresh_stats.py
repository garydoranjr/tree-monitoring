#!/usr/bin/env python
import os
import click
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.metrics import (
    roc_curve, auc,
    precision_score, recall_score, accuracy_score,
)
import matplotlib.pyplot as plt


def metric_over_two_thresholds(
    df,
    frac_thresholds,
    metric_fn,
    label_col="label",
    frac_prefix="frac_above_",
    name=None,
):
    """
    Compute an sklearn metric over a grid of:
      1) confidence thresholds (from frac_above_X columns)
      2) frac_above value thresholds applied to those columns

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    frac_thresholds : sequence of float
        Thresholds applied to frac_above_X values (second threshold).
    metric_fn : callable
        sklearn-style metric: metric(y_true, y_pred)
    label_col : str
        Name of the true label column.
    frac_prefix : str
        Prefix used for frac_above columns.

    Returns
    -------
    xr.DataArray
        2-D array with dims:
          - frac_threshold
          - confidence_threshold
    """

    # Extract confidence thresholds from column names
    conf_cols = [
        c for c in df.columns if c.startswith(frac_prefix)
    ]
    conf_thresholds = np.array(
        [float(c.replace(frac_prefix, "")) for c in conf_cols]
    )

    # Ensure consistent ordering
    order = np.argsort(conf_thresholds)
    conf_thresholds = conf_thresholds[order]
    conf_cols = [conf_cols[i] for i in order]

    frac_thresholds = np.asarray(frac_thresholds)
    y_true = df[label_col].values

    values = np.zeros((len(frac_thresholds), len(conf_thresholds)))

    for i, t_frac in enumerate(frac_thresholds):
        for j, col in enumerate(conf_cols):
            y_pred = (df[col].values >= t_frac).astype(int)
            values[i, j] = metric_fn(y_true, y_pred)

    return xr.DataArray(
        values,
        dims=("frac_threshold", "confidence_threshold"),
        coords={
            "frac_threshold": frac_thresholds,
            "confidence_threshold": conf_thresholds,
        },
        name=(metric_fn.__name__ if name is None else name),
    )


def plot_metrics_over_thresholds(
    df,
    frac_thresholds,
    metric_fns,
    metric_fn_names=None,
    label_col="label",
    frac_prefix="frac_above_",
    figsize=(5, 4),
):
    """
    Compute and plot multiple sklearn metrics over two thresholds.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    frac_thresholds : sequence of float
        Thresholds applied to frac_above_X values.
    metric_fns : list of callables
        sklearn-style metric functions: metric(y_true, y_pred)
    metric_fn_names : list of str, optional
        Display names for metrics. Defaults to metric.__name__.
    label_col : str
        True label column name.
    frac_prefix : str
        Prefix for frac_above columns.
    figsize : tuple
        Size per subplot (width, height).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : array of matplotlib.axes.Axes
    dataarrays : list of xarray.DataArray
    """

    if metric_fn_names is None:
        metric_fn_names = [fn.__name__ for fn in metric_fns]

    dataarrays = [
        metric_over_two_thresholds(
            df,
            frac_thresholds,
            fn,
            label_col=label_col,
            frac_prefix=frac_prefix,
            name=name,
        )
        for fn, name in zip(metric_fns, metric_fn_names)
    ]

    n = len(dataarrays)
    fig, axs = plt.subplots(
        1, n, figsize=(figsize[0] * n, figsize[1]), squeeze=False
    )

    for ax, da, name in zip(axs[0], dataarrays, metric_fn_names):
        da.plot(
            ax=ax,
            vmin=0.0,
            vmax=1.0,
            add_colorbar=True,
        )
        ax.set_title(name)

    plt.tight_layout()
    return fig, axs[0], dataarrays


def precision_score_div0(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=0)


@click.command()
@click.argument('statfile')
@click.argument('outputfile')
def main(statfile, outputfile):

    df = pd.read_csv(statfile)

    fracs = np.linspace(0, 1, 101)

    fig, _, _ = plot_metrics_over_thresholds(
        df, fracs,
        [accuracy_score, precision_score_div0, recall_score],
        ['Accuracy', 'Precision', 'Recall'],
    )

    fig.savefig(outputfile)


if __name__ == '__main__':
    main()
