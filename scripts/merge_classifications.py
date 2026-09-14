#!/usr/bin/env python
import os
import click
import rasterio
import rioxarray
import numpy as np
import xarray as xr
from tqdm import tqdm
from glob import glob


def gaussian_weight_mask(height, width, sigma):
    """
    Generate an unnormalized Gaussian weight mask for a tile.

    Parameters
    ----------
    height : int
        Number of rows in the tile.
    width : int
        Number of columns in the tile.
    sigma : float
        Standard deviation controlling spatial falloff.

    Returns
    -------
    weights : 2D ndarray (height, width)
        Weight mask with value 1.0 at the center, decreasing towards edges.
    """

    # Tile center
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0

    y = np.arange(height)[:, None]
    x = np.arange(width)[None, :]

    dy = y - cy
    dx = x - cx
    dist2 = dx * dx + dy * dy

    weights = np.exp(- dist2 / (2 * sigma * sigma))
    return weights.astype(np.float32)


def tile_window(src_transform, src_shape, tile_path, atol=1e-4):
    """
    Locate a tile on the source grid without resampling it.

    crown_classification.py cuts every tile straight out of the source raster,
    so tiles share the source CRS and resolution and start on whole-pixel
    offsets. When that holds, the tile can be slotted directly into place.

    Returns (row_off, col_off) or None when the tile is not grid-aligned, in
    which case the caller should fall back to reprojection.
    """
    height, width = src_shape

    with rasterio.open(tile_path) as t:
        tt = t.transform
        tile = t.read(1)
        crs_ok = t.crs == src_transform[1]

    st = src_transform[0]

    # Same resolution and orientation?
    if not (np.isclose(tt.a, st.a, rtol=1e-9) and np.isclose(tt.e, st.e, rtol=1e-9)):
        return None
    if not crs_ok:
        return None

    # Whole-pixel offset from the source origin?
    col_off = (tt.c - st.c) / st.a
    row_off = (tt.f - st.f) / st.e
    if not (np.isclose(col_off, round(col_off), atol=atol) and
            np.isclose(row_off, round(row_off), atol=atol)):
        return None

    row_off, col_off = int(round(row_off)), int(round(col_off))

    # Must fall inside the source grid.
    h, w = tile.shape
    if row_off < 0 or col_off < 0 or row_off + h > height or col_off + w > width:
        return None

    return row_off, col_off, tile


def mosaic_average_rioxarray(original_path, tile_paths, output_path, dtype=np.float32):
    """
    Mosaic thousands of tiles by streaming them one-by-one and averaging overlapping pixels.

    Tiles that already sit on the original grid are accumulated in place over
    just their own window; anything else falls back to .rio.reproject_match.
    Reprojecting a 512x512 tile onto a full 558 Mpx mosaic costs ~6 s and
    allocates the whole grid, so for the 50 ha mosaics the in-place path is the
    difference between minutes and hours per image.
    """

    # 1. Open original raster to define target grid
    original = rioxarray.open_rasterio(original_path).sel(band=[1, 2, 3])
    original = original.rio.write_nodata(None, inplace=False)
    original.attrs.pop("_FillValue", None)
    bands, height, width = original.shape

    with rasterio.open(original_path) as src:
        src_grid = (src.transform, src.crs)

    # 2. Allocate sum and count arrays
    sum_array = np.zeros((1, height, width), dtype=np.float64)
    count_array = np.zeros((1, height, width), dtype=np.float32)

    n_fast = 0

    # 3. Process each tile one by one
    for tile_path in tqdm(tile_paths, 'Loading'):

        # Fast path: drop the tile straight into its window on the source grid.
        placed = tile_window(src_grid, (height, width), tile_path)
        if placed is not None:
            row_off, col_off, tile_data = placed
            h, w = tile_data.shape

            weights = gaussian_weight_mask(h, w, w / 4.)
            valid = np.isfinite(tile_data)

            # Zero out invalid pixels rather than fancy-indexing the slice: a
            # chained index like sum_array[0, rows, cols][valid] += ... would
            # accumulate into a temporary copy and silently drop the update.
            contrib = np.where(valid, tile_data.astype(np.float64) * weights, 0.0)
            wcontrib = np.where(valid, weights, 0.0)

            sum_array[0, row_off:row_off + h, col_off:col_off + w] += contrib
            count_array[0, row_off:row_off + h, col_off:col_off + w] += wcontrib
            n_fast += 1
            continue

        # Slow path: the tile needs resampling onto the original grid. The
        # reprojected tile spans the full mosaic, so build the weight mask on
        # that grid too and let the valid mask pick out where the tile landed.
        tile = rioxarray.open_rasterio(tile_path)
        tile = tile.rio.write_nodata(None, inplace=False)
        tile.attrs.pop("_FillValue", None)

        # Align to original raster grid
        tile_aligned = tile.rio.reproject_match(original)

        # Convert to float32 and mask nodata
        tile_aligned = tile_aligned.astype(np.float32).where(tile_aligned.notnull())

        # Convert to numpy for accumulation
        tile_data = tile_aligned.data  # still a numpy array

        # Mask of valid (non-NaN) pixels
        valid = ~np.isnan(tile_data)

        # Weight by distance from the tile's own centre. The tile occupies a
        # bounding box within the full grid, so derive the mask over that box
        # and scatter it through `valid`.
        rows = np.any(valid[0], axis=1)
        cols = np.any(valid[0], axis=0)
        if not rows.any():
            continue
        r0, r1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
        c0, c1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])

        weights_full = np.zeros_like(tile_data, dtype=np.float32)
        weights_full[0, r0:r1, c0:c1] = gaussian_weight_mask(
            r1 - r0, c1 - c0, (c1 - c0) / 4.
        )

        # Accumulate
        sum_array[valid] += tile_data[valid] * weights_full[valid]
        count_array[valid] += weights_full[valid]

    print(f'Placed {n_fast}/{len(tile_paths)} tiles directly on the source grid')

    # 4. Compute final average
    avg_array = np.zeros_like(sum_array, dtype=dtype)
    nonzero = count_array > 0
    avg_array[nonzero] = (sum_array[nonzero] / count_array[nonzero]).astype(dtype)
    avg_array[~nonzero] = np.nan  # or 0 if you prefer
    
    # 5. Save result as GeoTIFF
    # Drop per-band metadata inherited from the source: the output is a
    # single confidence band, so the source's band names (the globus mosaics
    # label theirs 'Red'/'Green'/'Blue'/'Alpha') no longer describe it and
    # rioxarray refuses to write a long_name that outnumbers the bands.
    attrs = {k: v for k, v in original.attrs.items()
             if k not in ("long_name", "description")}

    avg_da = xr.DataArray(
        avg_array,
        dims=original.dims,
        coords={"y": original.y, "x": original.x, "band": [1]},
        attrs=attrs
    )
    avg_da.rio.write_crs(original.rio.crs, inplace=True)
    avg_da.rio.to_raster(output_path, dtype=dtype)


@click.command()
@click.argument('image_file')
@click.argument('classifications_dir')
@click.argument('output_file')
def main(image_file, classifications_dir, output_file):

    if os.path.exists(output_file):
        print('Output file already exists')
        return

    if not os.path.exists(classifications_dir):
        raise ValueError(f'Directory not found: "{classifications_dir}"')

    tile_paths = sorted(glob(os.path.join(classifications_dir, "*.tif")))

    mosaic_average_rioxarray(
        original_path=image_file,
        tile_paths=tile_paths,
        output_path=output_file,
    )


if __name__ == '__main__':
    main()
