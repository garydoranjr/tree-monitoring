#!/usr/bin/env python
import os
from glob import glob
import click
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torchvision import transforms as T
from torchvision.ops import masks_to_boxes
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from train_planet_image_maskrcnn import get_split, binary_mask_to_instances


def load_image_and_gt(imagefile, split, size=512, min_instance_size=4):
    maskfile = imagefile.replace('.png', '.mask.png')
    img = np.array(Image.open(imagefile))[..., :3]
    mask = np.array(Image.open(maskfile))
    mask = (mask == 255).astype(np.uint8)

    img_crop, mask_crop = get_split(img, mask, split, size)
    inst_masks = binary_mask_to_instances(
        mask_crop, min_instance_size=min_instance_size,
    )

    img_tensor = T.ToTensor()(Image.fromarray(img_crop))
    return img_crop, inst_masks, img_tensor


def overlay_instances(ax, img, masks, boxes, scores=None, cmap_name='tab20',
                      mask_alpha=0.45, edge_lw=1.2, label_scores=False):
    """Overlay each instance with a distinct color; draw bbox rectangle and
    (optional) score label above it."""
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    n = masks.shape[0]
    if n == 0:
        return
    cmap = plt.get_cmap(cmap_name, max(n, 1))
    for i in range(n):
        color = cmap(i % cmap.N)
        m = masks[i].astype(bool)
        rgba = np.zeros((*m.shape, 4), dtype=np.float32)
        rgba[m] = [color[0], color[1], color[2], mask_alpha]
        ax.imshow(rgba)
        x1, y1, x2, y2 = boxes[i]
        ax.add_patch(Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=edge_lw, edgecolor=color, facecolor='none',
        ))
        if label_scores and scores is not None:
            ax.text(
                x1, max(y1 - 2, 0), f'{scores[i]:.2f}',
                fontsize=6, color='white',
                bbox=dict(facecolor=color, alpha=0.7, pad=0.5,
                          edgecolor='none'),
            )


def plot_results(img, gt_masks, pred_masks, pred_boxes, pred_scores):
    fig, axs = plt.subplots(ncols=3, figsize=(18, 6))

    axs[0].imshow(img)
    axs[0].set_xticks([]); axs[0].set_yticks([])
    axs[0].set_title('Planet Image (right half)', fontsize=14)

    if gt_masks.shape[0] > 0:
        gt_boxes = masks_to_boxes(torch.from_numpy(gt_masks)).numpy()
    else:
        gt_boxes = np.zeros((0, 4))
    overlay_instances(axs[1], img, gt_masks, gt_boxes)
    axs[1].set_title(
        f'Ground Truth  (n={gt_masks.shape[0]})', fontsize=14,
    )

    overlay_instances(
        axs[2], img, pred_masks, pred_boxes, pred_scores,
        label_scores=True,
    )
    axs[2].set_title(
        f'Mask R-CNN  (n={pred_masks.shape[0]})', fontsize=14,
    )

    fig.tight_layout()
    return fig


@click.command()
@click.argument('modelfile')
@click.argument('imagedir')
@click.argument('outputdir')
@click.option('--score-thresh', default=0.5, type=float,
              help='Minimum score for a prediction to be rendered.')
@click.option('--mask-thresh', default=0.5, type=float,
              help='Threshold applied to soft Mask R-CNN mask logits.')
@click.option('--split', default='right',
              type=click.Choice(['left', 'right']),
              help='Which half of each tile to visualize (test=right).')
@click.option('--size', default=512, type=int)
def main(modelfile, imagedir, outputdir, score_thresh, mask_thresh, split,
         size):
    os.makedirs(outputdir, exist_ok=True)

    device = torch.device('cpu')
    ckpt = torch.load(modelfile, map_location=device)
    model = ckpt['model']
    min_instance_size = ckpt['params']['min_instance_size']
    model.eval()
    model.to(device)

    imagefiles = sorted(glob(os.path.join(imagedir, '*rgb.png')))
    imagefiles = [f for f in imagefiles if not f.endswith('.mask.png')]

    for imgfile in tqdm(imagefiles, desc='Applying Mask R-CNN'):
        ofile = os.path.join(
            outputdir,
            os.path.splitext(os.path.basename(imgfile))[0] + '.jpg',
        )
        if os.path.exists(ofile):
            continue

        img_np, gt_masks, img_tensor = load_image_and_gt(
            imgfile, split=split, size=size,
            min_instance_size=min_instance_size,
        )

        with torch.no_grad():
            output = model([img_tensor.to(device)])[0]

        scores = output['scores'].cpu().numpy()
        keep = scores >= score_thresh
        soft_masks = output['masks'][keep, 0].cpu().numpy()
        boxes = output['boxes'][keep].cpu().numpy()
        scores = scores[keep]
        pred_masks = (soft_masks >= mask_thresh).astype(np.uint8)

        fig = plot_results(img_np, gt_masks, pred_masks, boxes, scores)
        fig.savefig(ofile, dpi=120, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()
