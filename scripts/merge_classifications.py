#!/usr/bin/env python
import os
import click
import dask
import rioxarray
import numpy as np
import xarray as xr
from tqdm import tqdm
from glob import glob


def mosaic_average_rioxarray(original_path, tile_paths, output_path, dtype=np.float32):
    """
    Mosaic thousands of tiles by streaming them one-by-one and averaging overlapping pixels.
    
    Each tile is aligned to the original grid using .rio.reproject_match.
    """
    
    # 1. Open original raster to define target grid
    original = rioxarray.open_rasterio(original_path)
    bands, height, width = original.shape
    
    # 2. Allocate sum and count arrays
    sum_array = np.zeros((1, height, width), dtype=np.float64)
    count_array = np.zeros((1, height, width), dtype=np.float32)
    
    # 3. Process each tile one by one
    for tile_path in tqdm(tile_paths, 'Loading'):
        tile = rioxarray.open_rasterio(tile_path)
        
        # Align to original raster grid
        tile_aligned = tile.rio.reproject_match(original)
        
        # Convert to float32 and mask nodata
        tile_aligned = tile_aligned.astype(np.float32).where(tile_aligned.notnull())
        
        # Convert to numpy for accumulation
        tile_data = tile_aligned.data  # still a numpy array
        
        # Mask of valid (non-NaN) pixels
        valid = ~np.isnan(tile_data)
        
        # Accumulate
        sum_array[valid] += tile_data[valid]
        count_array[valid] += 1
    
    # 4. Compute final average
    avg_array = np.zeros_like(sum_array, dtype=dtype)
    nonzero = count_array > 0
    avg_array[nonzero] = (sum_array[nonzero] / count_array[nonzero]).astype(dtype)
    avg_array[~nonzero] = np.nan  # or 0 if you prefer
    
    # 5. Save result as GeoTIFF
    avg_da = xr.DataArray(
        avg_array,
        dims=original.dims,
        coords={"y": original.y, "x": original.x, "band": [1]},
        attrs=original.attrs
    )
    avg_da.rio.write_crs(original.rio.crs, inplace=True)
    avg_da.rio.to_raster(output_path, dtype=dtype)


@click.command()
@click.argument('image_file')
@click.argument('classifications_dir')
@click.argument('output_file')
def main(image_file, classifications_dir, output_file):

    tile_paths = sorted(glob(os.path.join(classifications_dir, "*.tif")))
    #tile_paths = tile_paths[:5]

    mosaic_average_rioxarray(
        original_path=image_file,
        tile_paths=tile_paths,
        output_path=output_file,
        #chunks={"x": 1024, "y": 1024}  # adjustable
    )


if __name__ == '__main__':
    main()
