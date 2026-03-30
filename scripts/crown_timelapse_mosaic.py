#!/usr/bin/env python
import os
import re
import math
import click
import numpy as np
import xarray as xr
import rasterio
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm
import geopandas as gpd
from PIL import Image, ImageDraw, ImageFont
from rasterio.windows import Window

DATE_PATTERN = re.compile(r'_(\d{4}_\d{2}_\d{2})_')

TAN = (210, 180, 140)
PURPLE = (138, 43, 226)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def get_date(path):
    m = DATE_PATTERN.search(os.path.basename(path))
    if m is None:
        raise ValueError(f"Could not parse date from {path}")
    return m.group(1)


def extract_centered_window(src, polygon, min_size=512, pixel_buffer=100):
    minx, miny, maxx, maxy = polygon.bounds

    top_row, left_col = src.index(minx, maxy)
    bottom_row, right_col = src.index(maxx, miny)

    row_min = min(top_row, bottom_row) - pixel_buffer
    row_max = max(top_row, bottom_row) + pixel_buffer
    col_min = min(left_col, right_col) - pixel_buffer
    col_max = max(left_col, right_col) + pixel_buffer

    height = row_max - row_min + 1
    width = col_max - col_min + 1

    if height < min_size:
        pad = (min_size - height) // 2
        row_min -= pad
        row_max += (min_size - height - pad)

    if width < min_size:
        pad = (min_size - width) // 2
        col_min -= pad
        col_max += (min_size - width - pad)

    row_min = max(row_min, 0)
    col_min = max(col_min, 0)
    row_max = min(row_max, src.height - 1)
    col_max = min(col_max, src.width - 1)

    window = Window(
        col_off=col_min,
        row_off=row_min,
        width=col_max - col_min + 1,
        height=row_max - row_min + 1
    )

    data = src.read(window=window)
    img = Image.fromarray(np.transpose(data, (1, 2, 0))[..., :3])

    return window, img


def draw_polygon_on_image(img, src, window, polygon,
                          outline=(255, 255, 0),
                          width=3):
    """
    Draw a polygon on a PIL image corresponding to a rasterio window.

    Parameters
    ----------
    img : PIL.Image
        Extracted image (from extract_centered_window).
    src : rasterio.DatasetReader
        Open raster source.
    window : rasterio.windows.Window
        Window used to extract img.
    polygon : shapely geometry
        Crown polygon in the same CRS as src.
    outline : tuple
        RGB color.
    width : int
        Line width.
    """

    draw = ImageDraw.Draw(img)

    window_transform = rasterio.windows.transform(window, src.transform)
    inv = ~window_transform

    def draw_single(poly):
        xy = []
        for x, y in poly.exterior.coords:
            col, row = inv * (x, y)
            xy.append((col, row))
        draw.line(xy, fill=outline, width=width, joint="curve")

    if isinstance(polygon, Polygon):
        draw_single(polygon)

    elif isinstance(polygon, MultiPolygon):
        for p in polygon.geoms:
            draw_single(p)

    return img


def annotate(img, tag, species, date, delicious, flowering):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    text = (
        f"Tag: {tag}\n"
        f"{species}\n"
        f"{date}\n"
        f"delicious: {delicious:.2f}\n"
        f"flowering: {flowering:.2f}"
    )

    # Estimate box size
    padding = 6
    lines = text.split("\n")
    line_height = font.getbbox("Ay")[3]
    box_width = max(font.getlength(line) for line in lines) + 2 * padding
    box_height = line_height * len(lines) + 2 * padding

    draw.rectangle(
        (5, 5, 5 + box_width, 5 + box_height),
        fill=(0, 0, 0, 160)
    )

    draw.multiline_text(
        (5 + padding, 5 + padding),
        text,
        fill=(255, 255, 255),
        font=font
    )

    w, h = img.size

    if delicious > 0.5:
        draw.rectangle((2, 2, w - 2, h - 2), outline=TAN, width=20)

    if flowering > 0.5:
        draw.rectangle((20, 20, w - 20, h - 20), outline=PURPLE, width=20)

    return img


def make_grid(images, ncols):
    w, h = images[0].size
    nrows = math.ceil(len(images) / ncols)

    canvas = Image.new(
        "RGB",
        (ncols * w, nrows * h),
        BLACK
    )

    for i, img in enumerate(images):
        r = i // ncols
        c = i % ncols
        canvas.paste(img, (c * w, r * h))

    return canvas


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

@click.command()
@click.argument('crownfile')
@click.argument('classification_nc')
@click.argument('image_files', nargs=-1)
@click.argument('tag')
@click.argument('output')
@click.option('--ncols', default=10, show_default=True)
def main(crownfile, classification_nc, image_files, tag, output, ncols):

    crowns = gpd.read_file(crownfile)
    crowns = crowns[crowns["tag"] == tag]

    if crowns.empty:
        raise ValueError(f"No crown with tag {tag}")

    species = crowns.iloc[0].get("latin", "unknown")

    ds = xr.open_dataset(classification_nc)

    images = []

    for img_path in tqdm(sorted(image_files)):
        date = get_date(img_path)

        row = crowns[crowns["date"] == date]
        if row.empty:
            continue

        date = date.replace('_', '-')

        delicious = float(ds["deciduous_probability"].sel(tag=float(tag), date=date))
        flowering = float(ds["flowering_probability"].sel(tag=float(tag), date=date))

        with rasterio.open(img_path) as src:
            window, img = extract_centered_window(src, row.geometry.values[0])

        img = draw_polygon_on_image(
            img,
            src,
            window,
            row.geometry.values[0],
            outline=(255, 0, 0),
            width=3
        )

        img = annotate(
            img,
            tag=tag,
            species=species,
            date=date,
            delicious=delicious,
            flowering=flowering
        )
        images.append(img)

    if not images:
        raise RuntimeError("No matching images found")

    mosaic = make_grid(images, ncols)
    mosaic.save(output)

    print(f"Saved {output}")


if __name__ == '__main__':
    main()

