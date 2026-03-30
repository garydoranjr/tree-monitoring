#!/usr/bin/env python
import os
import glob
import click
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from glob import glob

from train_planet_image_segformer import get_split
from deploy_planet_image_segformer import load_image, apply_model


@click.command()
@click.argument('modelfile')
@click.argument('imagedir')
@click.argument('outputfile')
def main(modelfile, imagedir, outputfile):

    model = torch.load(modelfile, map_location=torch.device('cpu'))
    model.eval()
    model.to('cpu')

    imagefiles = glob(os.path.join(imagedir, '*rgb.png'))

    records = []
    for imgfile in tqdm(imagefiles, 'Applying Model'):

        img, x = load_image(imgfile, 'right')

        # Flip dimensions from PIL image size
        resize = img.size[::-1]

        conf = apply_model(model, x, resize)

        #fraction = np.average(conf >= max(np.quantile(conf.ravel(), 0.9), 0.5))
        fraction = np.average(conf >= 0.5)

        # Parse date from filename, e.g. 20210324_155225_84_2413_1B_AnalyticMS_rgb.png
        basename = os.path.basename(imgfile)
        date_str = basename.split('_')[0]
        date = pd.to_datetime(date_str, format='%Y%m%d')
 
        records.append({'date': date, 'fraction': fraction})

    df = pd.DataFrame(records).sort_values('date').reset_index(drop=True)
    df.to_csv(outputfile, index=False)


if __name__ == '__main__':
    main()
