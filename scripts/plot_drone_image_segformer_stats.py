#!/usr/bin/env python
import os
import click
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, auc,
    precision_score, recall_score, accuracy_score,
)
import matplotlib.pyplot as plt


@click.command()
@click.argument('statfile')
@click.argument('outputfile')
def main(statfile, outputfile):

    df = pd.read_csv(statfile)
    print(df)

    # Extract arrays
    y_true = df["label"].values          # must be 0/1
    y_score = df["confidence"].values    # continuous confidence scores

    # ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Threshold for classification (optional)
    threshold = 0.5
    y_pred = (y_score >= threshold).astype(int)

    # Compute stats
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    # Plot ROC, sized so text stays legible at 3"x3" print size
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.plot(fpr, tpr, lw=2.0, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", lw=1.2, color="gray")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.tick_params(width=0.6, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    ax.text(
        0.97, 0.03,
        f"threshold = {threshold:.2f}\n"
        f"precision = {precision:.3f}\n"
        f"recall = {recall:.3f}\n"
        f"accuracy = {accuracy:.3f}",
        fontsize=8,
        ha="right", va="bottom",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", linewidth=0.5),
    )

    ax.legend(loc="upper left", frameon=False, handlelength=1.5, borderpad=0.3)
    fig.tight_layout(pad=0.3)
    fig.savefig(outputfile)


if __name__ == '__main__':
    main()
