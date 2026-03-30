#!/usr/bin/env python
import os
import click
import numpy as np
import pandas as pd
from tqdm import tqdm
import rasterio as rio
from rasterio.features import rasterize
import geopandas as gpd
from shapely.ops import transform
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from skimage.morphology import binary_erosion, binary_dilation, disk
from matplotlib.patches import Polygon, Rectangle
import matplotlib.pyplot as plt


script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_CONFIG = 'configs/sam2.1/sam2.1_hiera_l.yaml'
MODEL_FILE = os.path.join(script_dir, 'models', 'sam2.1_hiera_large.pt')
MIN_SIZE = 16


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
        color = np.array([1.0, 0, 0, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=5):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)


def show_masks(image, masks, scores, crown, point_coords=None, input_labels=None):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())

        poly = Polygon(
            crown.exterior.coords,
            facecolor='none',
            edgecolor='red',
        )
        plt.gca().add_patch(poly)

        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


def max_iou_with_translation(mask, polygon, max_offset=15):
    H, W = mask.shape

    # Rasterize polygon at native resolution
    poly_mask = rasterize(
        [(polygon, 1)],
        out_shape=mask.shape,
        fill=0,
        dtype=np.uint8
    )

    poly_bbox = np.argwhere(poly_mask)
    if poly_bbox.size == 0:
        return 0.0, (0, 0)

    y_min, x_min = poly_bbox.min(axis=0)
    y_max, x_max = poly_bbox.max(axis=0) + 1
    poly_crop = poly_mask[y_min:y_max, x_min:x_max]
    h, w = poly_crop.shape

    max_iou = 0.0
    best_offset = (0, 0)

    for dy in range(-max_offset, max_offset + 1):
        for dx in range(-max_offset, max_offset + 1):
            y_start = y_min + dy
            x_start = x_min + dx
            y_end = y_start + h
            x_end = x_start + w

            # Skip if out of bounds
            if y_start < 0 or x_start < 0 or y_end > H or x_end > W:
                continue

            mask_crop = mask[y_start:y_end, x_start:x_end]
            intersection = np.logical_and(poly_crop, mask_crop).sum()
            union = np.logical_or(poly_crop, mask_crop).sum()
            iou = intersection / union if union > 0 else 0.0

            if iou > max_iou:
                max_iou = iou
                best_offset = (dy, dx)

    return max_iou, best_offset


def select_masks(poly, masks, threshold=0.7):
    area = poly.area
    candidates = []
    for mask in masks:
        size = np.sum(mask)
        iou_bound = min(area, size) / max(area, size)
        if iou_bound < threshold: continue
        candidates.append(mask)

    if len(candidates) == 0: return None

    ious = np.array([
        max_iou_with_translation(c, poly)[0]
        for c in candidates
    ])
    best = np.argmax(ious)
    best_iou = ious[best]
    if best_iou < threshold: return None

    return candidates[best]


def clean_mask(mask, radius=2):
    selem = disk(radius)
    mask = binary_erosion(mask, selem)
    mask = binary_dilation(mask, selem)
    return mask


def to_bbox(mask):
    # Find non-zero indices
    rows, cols = np.where(mask)
    ymin, ymax = rows.min(), rows.max()
    xmin, xmax = cols.min(), cols.max()

    # Compute width and height
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    return xmin, ymin, width, height



def to_rectangle(mask):
    xmin, ymin, width, height = to_bbox(mask)
    # Create and add a rectangle patch
    rect = Rectangle(
        (xmin, ymin), width, height,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    return rect


@click.command()
@click.argument('inputfile')
@click.argument('shapefile')
@click.argument('outputfile')
def main(inputfile, shapefile, outputfile):

    with rio.open(inputfile) as f:
        image = np.transpose(f.read(), (1, 2, 0))
        tform = ~f.transform

    def map_to_pix(x, y, z=None):
        return tform * (x, y)

    crowns = gpd.read_file(shapefile)
    crowns = crowns.sort_values('area', ascending=False)

    model = build_sam2(MODEL_CONFIG, MODEL_FILE, device='cpu')
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(image)

    good_masks = []

    for _, crown in tqdm(list(crowns.iterrows())):
        pix = transform(map_to_pix, crown['geometry'])
        centroid = pix.centroid

        input_point = np.array([[centroid.x, centroid.y]])
        input_label = np.array([1])

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        best_mask = select_masks(pix, masks)
        if best_mask is not None:
            best_mask = clean_mask(best_mask)
            if np.sum(best_mask) > MIN_SIZE:
                good_masks.append(best_mask)

    entries = []
    for mask in good_masks:
        x, y, w, h = to_bbox(mask)
        entries.append({
            'filename': os.path.basename(inputfile),
            'labeler': 'sam2',
            'top': y,
            'left': x,
            'width': w,
            'height': h,
        })

    df = pd.DataFrame.from_dict(entries)
    df = df[['filename', 'labeler', 'top', 'left', 'width', 'height']]
    df.to_csv(outputfile, index=False)
    exit()


    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    for mask in good_masks:
        ax.add_patch(to_rectangle(mask))

    #show_mask(overall_mask, ax)

    plt.show()
    exit()


if __name__ == '__main__':
    main()
