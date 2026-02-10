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

    # Plot ROC
    fig = plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    # Add metrics as text
    plt.text(
        0.65, 0.25,
        f"threshold = {threshold:.2f}\n"
        f"precision = {precision:.3f}\n"
        f"recall = {recall:.3f}\n"
        f"accuracy = {accuracy:.3f}",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.8)
    )

    plt.legend()
    plt.tight_layout()
    fig.savefig(outputfile)


if __name__ == '__main__':
    main()
