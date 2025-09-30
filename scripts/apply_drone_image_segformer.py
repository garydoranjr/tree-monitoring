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
from transformers import SegformerImageProcessor
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages


def load_image(imagedir, maskdir, imageid, size=512):

    imagefile = os.path.join(imagedir, f'{imageid}.tif')
    maskfile = os.path.join(maskdir, f'{imageid}.tif')

    # Load image (ignore 4th band)
    img = Image.open(imagefile)
    img = np.array(img)[..., :3]  # take only RGB
    img = Image.fromarray(img)

    # Load mask (convert 255 → 1)
    mask = Image.open(maskfile)
    mask = np.array(mask)
    mask = (mask == 255).astype(np.uint8)

    processor = SegformerImageProcessor(
        do_resize=True, size=size, do_normalize=True,
    )

    encoded_inputs = processor(
        images=img,
        size=size,
        return_tensors="pt",
    )

    return img, encoded_inputs["pixel_values"], mask


def apply_model(model, x, resize):

    with torch.no_grad():
        output = model(x)

    logits = F.interpolate(
        output.logits, size=resize,
        mode="bilinear", align_corners=False,
    )

    conf = torch.sigmoid(logits.squeeze()[1]).numpy()

    return conf


def plot_results(img, mask, conf):
    fig, axs = plt.subplots(ncols=3, figsize=(16, 6))

    def rm_ticks(ax):
        ax.set_xticks([])
        ax.set_yticks([])

    def clear(dummy):
        dummy.axis("off")

    axs[0].imshow(img)
    rm_ticks(axs[0])
    axs[0].set_title('Drone Image', fontsize=16)
    divider = make_axes_locatable(axs[0])
    dummy = divider.append_axes("right", size="5%", pad=0.1)
    clear(dummy)

    im = axs[2].imshow(conf, vmin=0, vmax=1, cmap='RdYlGn')
    rm_ticks(axs[2])
    axs[2].set_title('Model Confidence', fontsize=16)

    # Make a new axis to the right of the last subplot
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # Add colorbar into that axis
    cbar = fig.colorbar(im, cax=cax)

    axs[1].imshow(mask)
    rm_ticks(axs[1])
    axs[1].set_title('Ground Truth Mask', fontsize=16)
    divider = make_axes_locatable(axs[1])
    dummy = divider.append_axes("right", size="5%", pad=0.1)
    clear(dummy)

    return fig


@click.command()
@click.argument('modelfile')
@click.argument('imagedir')
@click.argument('maskdir')
@click.argument('foldfile')
@click.argument('outputfile')
def main(modelfile, imagedir, maskdir, foldfile, outputfile):

    df = pd.read_csv(foldfile)
    ids = df.loc[df['split'] == 'test']['polygon_id'].values

    model = torch.load(modelfile, map_location=torch.device('cpu'))
    model.eval()
    model.to('cpu')

    figs = []
    for img_id in tqdm(ids, 'Applying Model'):
        img, x, mask = load_image(imagedir, maskdir, img_id)

        # Flip dimensions from PIL image size
        resize = img.size[::-1]

        conf = apply_model(model, x, resize)

        fig = plot_results(img, mask, conf)
        figs.append(fig)

    with PdfPages(outputfile) as pdf:
        for fig in tqdm(figs, 'Saving'):
            pdf.savefig(fig)


if __name__ == '__main__':
    main()
