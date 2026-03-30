#!/usr/bin/env python
import os
import glob
import click
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from glob import glob
from transformers import SegformerImageProcessor
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages

from train_planet_image_segformer import get_split


def load_image(imagefile, split, size=512):

    # Load image (ignore 4th band)
    img = Image.open(imagefile)
    img = np.array(img)[..., :3]  # take only RGB

    img, _ = get_split(img, img, split, size)

    img = Image.fromarray(img)

    processor = SegformerImageProcessor(
        do_resize=True, size=size, do_normalize=True,
    )

    encoded_inputs = processor(
        images=img,
        size=size,
        return_tensors="pt",
    )

    return img, encoded_inputs["pixel_values"]


def apply_model(model, x, resize):

    with torch.no_grad():
        output = model(x)

    logits = F.interpolate(
        output.logits, size=resize,
        mode="bilinear", align_corners=False,
    )

    conf = torch.sigmoid(logits.squeeze()[1]).numpy()

    return conf


def plot_results(img, conf):
    fig, axs = plt.subplots(ncols=3, figsize=(16, 6))

    def rm_ticks(ax):
        ax.set_xticks([])
        ax.set_yticks([])

    def clear(dummy):
        dummy.axis("off")

    axs[0].imshow(img)
    rm_ticks(axs[0])
    axs[0].set_title('Planet Image', fontsize=16)
    divider = make_axes_locatable(axs[0])
    dummy = divider.append_axes("right", size="5%", pad=0.1)
    clear(dummy)

    im = axs[1].imshow(conf, vmin=0, vmax=1, cmap='RdYlGn')
    rm_ticks(axs[1])
    axs[1].set_title('Model Confidence', fontsize=16)

    # Make a new axis to the right of the last subplot
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # Add colorbar into that axis
    cbar = fig.colorbar(im, cax=cax)

    #im = axs[2].imshow(conf >= max(np.quantile(conf.ravel(), 0.9), 0.5), cmap='gray')
    im = axs[2].imshow(conf >= 0.5, cmap='gray')
    rm_ticks(axs[2])
    axs[1].set_title('Model Confidence', fontsize=16)

    #axs[1].imshow(mask)
    #rm_ticks(axs[1])
    #axs[1].set_title('Ground Truth Mask', fontsize=16)
    #divider = make_axes_locatable(axs[1])
    #dummy = divider.append_axes("right", size="5%", pad=0.1)
    #clear(dummy)

    return fig


@click.command()
@click.argument('modelfile')
@click.argument('imagedir')
@click.argument('outputdir')
def main(modelfile, imagedir, outputdir):

    model = torch.load(modelfile, map_location=torch.device('cpu'))
    model.eval()
    model.to('cpu')

    imagefiles = glob(os.path.join(imagedir, '*rgb.png'))

    for imgfile in tqdm(imagefiles, 'Applying Model'):
        ofile = os.path.join(
            outputdir,
            os.path.splitext(os.path.basename(imgfile))[0] + '.jpg'
        )
        if os.path.exists(ofile): continue

        img, x = load_image(imgfile, 'right')

        # Flip dimensions from PIL image size
        resize = img.size[::-1]

        conf = apply_model(model, x, resize)

        fig = plot_results(img, conf)
        fig.savefig(ofile)


if __name__ == '__main__':
    main()
