#!/usr/bin/env python
import os
import glob
import click
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from transformers import SegformerImageProcessor
import matplotlib.pyplot as plt


@click.command()
@click.argument('modelfile')
@click.argument('imagefile')
@click.argument('foldfile')
def main(modelfile, imagefile, foldfile):

    # Load image (ignore 4th band)
    img = Image.open(imagefile)
    img = np.array(img)[..., :3]  # take only RGB
    img = Image.fromarray(img)

    size=512

    processor = SegformerImageProcessor(do_resize=True, size=size, do_normalize=True)

    encoded_inputs = processor(
        images=img,
        size=size,
        return_tensors="pt"
    )

    pixel_values = encoded_inputs["pixel_values"]  # (3,H,W)

    model = torch.load(modelfile)
    model.eval()
    model.to('cpu')
    with torch.no_grad():
        output = model(pixel_values)

    # Flip dimensions from PIL image size
    resize = img.size[::-1]

    logits = F.interpolate(
        output.logits, size=resize,
        mode="bilinear", align_corners=False,
    )
    fg = logits.squeeze()[1].numpy()

    fig, axs = plt.subplots(ncols=2)
    vmax = np.max(np.abs(fg))
    axs[0].imshow(fg, vmin=-vmax, vmax=vmax, cmap='RdYlGn')
    axs[1].imshow(img)
    plt.show()


    exit()


if __name__ == '__main__':
    main()
