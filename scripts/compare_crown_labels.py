#!/usr/bin/env python
import os
import json
import click
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from itertools import combinations
from shapely.ops import unary_union
import matplotlib.pyplot as plt

from match_crowns_to_labels import rectangle_from_top_left


def get_labels_from_user(labels, user):
    labels = labels.loc[labels['labeler'] == user]
    poly_labels = []
    for _, row in labels.iterrows():
        poly_labels.append(
            rectangle_from_top_left(
                row['top'], row['left'],
                row['width'], row['height'],
            )
        )
    return poly_labels


def compare_labels(unions, u1, u2):
    p1 = unions[u1]
    p2 = unions[u2]
    u = p1.union(p2).area
    p1only = p1.difference(p2).area
    p2only = p2.difference(p1).area
    p1andp2 = p1.intersection(p2).area
    return {
        'user1': u1,
        'user2': u2,
        'poly1': (p1only / u),
        'poly2': (p2only / u),
        'overlap': (p1andp2 / u),
    }


def pairwise_comparison(labels, filename):
    # Select relevant rows and drop column
    labels = labels.loc[labels['filename'] == filename]
    labels = labels.drop(columns=['filename'])

    labelers = labels['labeler'].unique()

    polygons = {
        l: get_labels_from_user(labels, l)
        for l in labelers
    }

    unions = { l: unary_union(p) for l, p in polygons.items() }

    return [
        compare_labels(unions, u1, u2)
        for u1, u2 in sorted(combinations(sorted(
            unions.keys(), key=lambda x: 'zzzzz' if x == 'sam2' else x
        ), 2))
    ]


@click.command()
@click.argument('labelfile')
@click.argument('outputfile')
def main(labelfile, outputfile):

    labels = pd.read_csv(labelfile)

    files = np.unique(labels['filename'])

    all_comparison = []
    for f in files:
        all_comparison += pairwise_comparison(labels, f)

    user_pairs = [
        (c['user1'], c['user2'])
        for c in all_comparison
    ]

    fractions = np.array([
        [c['poly1'], c['overlap'], c['poly2']]
        for c in all_comparison
    ])

    # Prepare bar positions
    n = len(user_pairs)
    y_pos = np.arange(n)

    # Split data for stacking
    user1_only = fractions[:, 0]
    shared = fractions[:, 1]
    user2_only = fractions[:, 2]

    # Plot setup
    fig, ax = plt.subplots(figsize=(8, 16))

    # First segment: user1_only
    bars1 = ax.barh(
        y_pos, user1_only,
        color='blue', alpha=0.7, label='Labeler 1 Only'
    )

    # Second segment: shared (stacked on top of user1_only)
    bars2 = ax.barh(
        y_pos, shared, left=user1_only,
        color='green', alpha=0.7, label='Shared'
    )

    # Third segment: user2_only (stacked on top of user1_only + shared)
    bars3 = ax.barh(
        y_pos, user2_only, left=user1_only + shared,
        color='gold', alpha=0.99, label='Labeler 2 Only'
    )

    # Left y-axis: user 1 labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([u1 for u1, u2 in user_pairs])
    ax.invert_yaxis()  # Optional: top-to-bottom ordering

    # Right y-axis: user 2 labels
    ax_right = ax.twinx()
    ax_right.set_yticks(y_pos)
    ax_right.set_yticklabels([u2 for u1, u2 in user_pairs])
    ax_right.set_ylim(ax.get_ylim())

    # Aesthetic settings
    ax.set_xlim(0, 1)
    ax.set_xlabel('Fraction')
    ax.set_title('Label Overlap Between User Pairs')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)

    plt.tight_layout()
    fig.savefig(outputfile)


if __name__ == '__main__':
    main()
