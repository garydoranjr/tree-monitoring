#!/usr/bin/env python
import os
import click
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import rasterio as rio
from collections import defaultdict


def make_ext(fname, ext):
    base = os.path.splitext(fname)[0]
    return f'{base}.{ext}'


def extract_labels(labels, filename, imagedir, outputdir):

    imgfile = os.path.join(imagedir, make_ext(filename, 'tif'))
    with rio.open(imgfile) as f:
        image = np.transpose(f.read(), (1, 2, 0))

    # Select relevant rows and drop column
    labels = labels.loc[labels['filename'] == filename]
    labels = labels.drop(columns=['filename'])


    events = []
    non_events = []

    for i, row in labels.iterrows():
        bbox = (
            int(row['left']),
            int(row['top']),
            int(row['width']),
            int(row['height']),
        )
        if row['is_event']:
            events.append((i, bbox))
        else:
            non_events.append((i, bbox))

    event_dir = os.path.join(outputdir, 'events')
    nevnt_dir = os.path.join(outputdir, 'non_events')

    os.makedirs(event_dir, exist_ok=True)
    os.makedirs(nevnt_dir, exist_ok=True)

    def save_crops(crops, outdir):
        for j, (idx, bbox) in enumerate(crops):
            left, top, width, height = bbox
            right, bottom = left + width, top + height
            crop = image[top:bottom, left:right]  # (h, w, c)

            if crop.size == 0:
                continue  # skip invalid boxes

            pil_img = Image.fromarray(crop.astype(np.uint8))
            pil_img = pil_img.resize((30, 30), Image.BILINEAR)

            outfile = os.path.join(outdir, f"{filename}_{j:05d}.png")
            pil_img.save(outfile)

    save_crops(events, event_dir)
    save_crops(non_events, nevnt_dir)


@click.command()
@click.argument('labelfile')
@click.argument('imagedir')
@click.argument('outputdir')
def main(labelfile, imagedir, outputdir):

    labels = pd.read_csv(labelfile)
    files = np.unique(labels['filename'])

    for f in tqdm(files):
        extract_labels(labels, f, imagedir, outputdir)


if __name__ == '__main__':
    main()
