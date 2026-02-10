#!/usr/bin/env python
import os
import click
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from apply_drone_image_segformer import load_image, apply_model


@click.command()
@click.argument('modelfile')
@click.argument('imagedir')
@click.argument('labeldir')
@click.argument('maskdir')
@click.argument('foldfile')
@click.argument('outputfile')
def main(modelfile, imagedir, labeldir, maskdir, foldfile, outputfile):

    df = pd.read_csv(foldfile)
    ids = df.loc[df['split'] == 'test']['polygon_id'].values

    model = torch.load(modelfile, map_location=torch.device('cpu'))
    model.eval()
    model.to('cpu')

    stats = []
    for img_id in tqdm(ids, 'Applying Model'):
        img, x, mask = load_image(imagedir, maskdir, img_id)
        _, _, label = load_image(imagedir, labeldir, img_id)

        # Flip dimensions from PIL image size
        resize = img.size[::-1]

        conf = apply_model(model, x, resize)

        n = np.sum(mask)
        crown_conf = float(np.sum(np.multiply(mask, conf)) / n)
        crown_label = int(np.round(np.sum(np.multiply(mask, label)) / n))

        stats.append({
            'polygon_id': img_id,
            'label': crown_label,
            'confidence': crown_conf,
        })

    df = pd.DataFrame(stats)
    df.to_csv(outputfile, index=False)


if __name__ == '__main__':
    main()
