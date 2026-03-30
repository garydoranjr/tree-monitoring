#!/usr/bin/env python
import os
import click
import numpy as np
import pandas as pd
from tqdm import tqdm
import rasterio as rio
from collections import defaultdict
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt


def make_ext(fname, ext):
    base = os.path.splitext(fname)[0]
    return f'{base}.{ext}'


def plot_labeler_annotations(ax, ax_legend, annotations, labeler_colors=None, pad=5):
    """
    Plot rectangular annotations on top of an image for up to 3 labelers.

    Parameters:
    - ax: matplotlib axis
    - annotations: dict mapping labeler names to lists of rectangles.
                   Each rectangle is a tuple (x, y, width, height).
    - labeler_colors: optional dict mapping labeler names to colors.
                      If None, defaults to {'User1': 'cyan', 'User2': 'magenta', 'User3': 'yellow'}

    Returns:
    - The matplotlib axis with the plot.
    """


    if labeler_colors is None:
        default_colors = ['magenta', 'cyan', 'yellow']
        labeler_colors = {user: color for user, color in zip(annotations.keys(), default_colors)}

    legend_handles = {}

    total = len(annotations)
    start = total + 2
    
    for a, (user, rects) in enumerate(annotations.items()):
        color = labeler_colors.get(user, 'black')
        for i, (x, y, w, h) in enumerate(rects):
            rect = Rectangle(
                (x - pad, y - pad), w + 2*pad, h + 2*pad,
                linewidth=(start-a), edgecolor=color, facecolor='none', alpha=0.75
            )
            # Only add label to one rectangle per user for the legend
            label = user if user not in legend_handles else None
            handle = ax.add_patch(rect)
            if label:
                legend_handles[user] = handle

    # Place legend outside the image
    ax_legend.legend(
        handles=list(legend_handles.values()),
        labels=list(legend_handles.keys()),
        loc='upper left',
    )

    return ax



def plot_labels(labels, filename, imagedir, outputdir, exclude):

    imgfile = os.path.join(imagedir, make_ext(filename, 'tif'))
    with rio.open(imgfile) as f:
        image = np.transpose(f.read(), (1, 2, 0))

    # Select relevant rows and drop column
    labels = labels.loc[labels['filename'] == filename]
    labels = labels.drop(columns=['filename'])

    labelers = set(labels['labeler'].unique())
    if exclude and 'sam2' in labelers:
        labelers.remove('sam2')
        if len(labelers) == 0: return

    annotations = defaultdict(list)
    for labeler in labelers:
        subset = labels.loc[labels['labeler'] == labeler]
        for _, row in subset.iterrows():
            annotations[labeler].append((
                row['left'],
                row['top'],
                row['width'],
                row['height'],
            ))

    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
    fig.subplots_adjust(wspace=0.01)
    ax_img = fig.add_subplot(gs[0])

    ax_legend = fig.add_subplot(gs[1])
    ax_legend.axis('off')

    ax_img.imshow(image)
    ax_img.axis('off')  # Hide axes

    plot_labeler_annotations(ax_img, ax_legend, annotations)

    outfile = os.path.join(outputdir, make_ext(filename, 'jpg'))
    fig.savefig(outfile, dpi=300)


@click.command()
@click.argument('labelfile')
@click.argument('imagedir')
@click.argument('outputdir')
@click.option('-e', '--exclude', is_flag=True, default=False)
def main(labelfile, imagedir, outputdir, exclude):

    labels = pd.read_csv(labelfile)
    files = np.unique(labels['filename'])

    for f in tqdm(files):
        plot_labels(labels, f, imagedir, outputdir, exclude)


if __name__ == '__main__':
    main()
