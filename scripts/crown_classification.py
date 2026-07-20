#Import libraries/modules
import os
import re
import click
import torch
import rasterio
import numpy as np
from PIL import Image
from tqdm import tqdm
import geopandas as gpd
import torch.nn.functional as F
from transformers import SegformerImageProcessor
from rasterio.windows import Window
from rasterio.features import rasterize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def extract_centered_window(src, polygon, min_size=512, pixel_buffer=100):
    """
    Extract a raster window fully containing the polygon, with:
      • at least min_size x min_size pixels
      • at least pixel_buffer pixels around the polygon
      • centered on the polygon when expanding to min_size

    Parameters
    ----------
    src : rasterio.io.DatasetReader
        Open rasterio dataset.
    polygon : shapely.geometry.Polygon
        Polygon in the same CRS as the raster.
    min_size : int
        Minimum window size (pixels) on each side.
    pixel_buffer : int
        Extra pixels to include around the polygon.

    Returns
    -------
    window : rasterio.windows.Window
        The computed window.
    data : np.ndarray
        The extracted array `src.read(window=window)`.
    """

    # --- 1. Get polygon bounds in world coordinates ---
    minx, miny, maxx, maxy = polygon.bounds

    # --- 2. Convert world coordinates → pixel indices ---
    # rasterio.transform.rowcol uses (x, y)
    top_row, left_col = src.index(minx, maxy)
    bottom_row, right_col = src.index(maxx, miny)

    # Normalize row/col ordering
    row_min = min(top_row, bottom_row)
    row_max = max(top_row, bottom_row)
    col_min = min(left_col, right_col)
    col_max = max(left_col, right_col)

    # --- 3. Apply pixel buffer ---
    row_min -= pixel_buffer
    row_max += pixel_buffer
    col_min -= pixel_buffer
    col_max += pixel_buffer

    # --- 4. Compute size; expand to meet min_size (centered) ---
    height = row_max - row_min + 1
    width  = col_max - col_min + 1

    # Expand height
    if height < min_size:
        pad = (min_size - height) // 2
        row_min -= pad
        row_max += (min_size - height - pad)

    # Expand width
    if width < min_size:
        pad = (min_size - width) // 2
        col_min -= pad
        col_max += (min_size - width - pad)

    # --- 5. Clamp window to raster bounds ---
    row_min = max(row_min, 0)
    col_min = max(col_min, 0)
    row_max = min(row_max, src.height - 1)
    col_max = min(col_max, src.width - 1)

    # Final window parameters
    h = row_max - row_min + 1
    w = col_max - col_min + 1

    window = Window(col_off=col_min, row_off=row_min, width=w, height=h)

    # --- 6. Read and return ---
    data = src.read(window=window)

    return window, Image.fromarray(np.transpose(data, (1, 2, 0))[..., :3])


def polygon_mask(src, window, polygon):
    """
    Return a binary mask (H×W) for the given polygon inside the raster window.
    Uses rasterio.features.rasterize for speed.
    """

    # Window geometry
    height = int(window.height)
    width = int(window.width)

    # Get the transform for *just this window*
    window_transform = rasterio.windows.transform(window, src.transform)

    # Rasterize polygon into this window grid
    mask = rasterize(
        [(polygon, 1)],                   # list of (geometry, value)
        out_shape=(height, width),
        transform=window_transform,
        fill=0,                            # background
        dtype=np.uint8,
        all_touched=False                  # True includes edge-touching pixels
    )

    return mask


def preprocess(img, size=512):

    processor = SegformerImageProcessor(
        do_resize=True, size=size, do_normalize=True,
    )

    encoded_inputs = processor(
        images=img,
        size=size,
        return_tensors="pt",
    )

    return encoded_inputs["pixel_values"]


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


def save_window_geotiff(output_path, array, src, window):
    """
    Save a windowed array (H×W or C×H×W) as a GeoTIFF using
    the CRS and per-window transform from the source dataset.

    Parameters
    ----------
    output_path : str
        Path to output GeoTIFF.
    array : np.ndarray
        Single-band or multi-band array. Shape must be:
            (H, W) or (bands, H, W)
    src : rasterio.io.DatasetReader
        The open source raster.
    window : rasterio.windows.Window
        Window corresponding to the array area in the source raster.
    """

    # Ensure array has shape (bands, H, W)
    if array.ndim == 2:
        array = array[np.newaxis, ...]  # (1, H, W)

    bands, height, width = array.shape

    # Get transform for this window
    window_transform = rasterio.windows.transform(window, src.transform)

    # Write GeoTIFF
    profile = src.profile.copy()
    profile.update({
        "height": height,
        "width": width,
        "transform": window_transform,
        "count": bands,
        "dtype": array.dtype.name,
    })

    # Some profiles include tiling or compression; keep or modify as needed
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(array)


@click.command()
@click.argument('modelfile')
@click.argument('image_file')
@click.argument('shapefile_path')
@click.argument('output_dir')
def main(modelfile, image_file, shapefile_path, output_dir):

    model = torch.load(modelfile, weights_only=False, map_location=torch.device('cpu'))
    model.eval()
    model.to('cpu')

    shp = gpd.read_file(shapefile_path)

    # The shapefile is a timeseries with one row per crown per flight date.
    # Each image is a single-date orthomosaic (BCI_ava_YYYY_MM_DD_orthomosaic.tif),
    # so keep only the polygons whose date matches this image's date.
    m = re.search(r'\d{4}_\d{2}_\d{2}', os.path.basename(image_file))
    if m is None:
        raise ValueError(
            f"Could not parse a YYYY_MM_DD date from image filename: {image_file}"
        )
    date_token = m.group(0)
    shp = shp[shp['date'].dt.strftime('%Y_%m_%d') == date_token]
    print(f"Date {date_token}: {len(shp)} matching polygons")

    with rasterio.open(image_file) as src:
        for i, row in tqdm(shp.iterrows(), total=len(shp)):
            tag = row['tag']
            output_path = os.path.join(output_dir, f'{i:05d}_{tag}.tif')
            if os.path.exists(output_path): continue

            polygon = row['geometry']
            w, img = extract_centered_window(src, polygon)
            mask = polygon_mask(src, w, polygon)
            x = preprocess(img)

            # Flip dimensions from PIL image size
            resize = img.size[::-1]

            conf = apply_model(model, x, resize)

            save_window_geotiff(output_path, conf, src, w)


if __name__ == '__main__':
    main()
