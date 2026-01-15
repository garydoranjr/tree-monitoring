#!/usr/bin/env python
import sys
import click
import rioxarray
import xarray as xr
from pathlib import Path


def round_to_precision(da, precision=1e-2):
    """
    Round xarray Dataset or DataArray values to the nearest precision.

    Parameters
    ----------
    x : xr.DataArray
        Input data
    precision : float
        Rounding precision (e.g., 1e-3)

    Returns
    -------
    xr.Dataset or xr.DataArray
        Rounded data (same dtype as input)
    """
    if precision <= 0:
        raise ValueError("precision must be > 0")

    factor = 1.0 / precision
    return (da * factor).round() / factor


def process_pair(flower_tif, decid_tif, output_tif):
    """
    Load two single-band GeoTIFFs and write a two-band GeoTIFF.

    Band 1: flowering_probability
    Band 2: deciduous_probability
    """
    # Load inputs
    flower = rioxarray.open_rasterio(flower_tif)
    decid = rioxarray.open_rasterio(decid_tif)

    # Ensure single-band inputs
    if flower.sizes.get("band", 1) != 1:
        raise ValueError(f"{flower_tif} must be single-band")
    if decid.sizes.get("band", 1) != 1:
        raise ValueError(f"{decid_tif} must be single-band")

    # Drop band dimension
    flower = flower.squeeze("band", drop=True)
    decid = decid.squeeze("band", drop=True)

    # Check dimensions match
    if flower.dims != decid.dims or flower.shape != decid.shape:
        raise ValueError("Input rasters do not have matching dimensions")

    # Construct multi-band dataset
    ds = xr.Dataset(
        {
            "flowering_probability": round_to_precision(flower),
            "deciduous_probability": round_to_precision(decid),
        }
    )

    # Write CRS / transform info to all variables
    ds = ds.rio.write_crs(flower.rio.crs)
    ds = ds.rio.write_transform(flower.rio.transform())

    # Write to disk as a multi-band GeoTIFF
    ds.rio.to_raster(
        output_tif,
        driver="COG",
        compress="DEFLATE",
        blocksize=512,
        overview_resampling="nearest",
    )


@click.command()
@click.argument(
    "input_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
def main(input_dir: Path):
    """
    Process a single *_flower.tif and *_decid.tif pair in INPUT_DIR.
    """
    flower_files = list(input_dir.glob("*_flower.tif"))
    decid_files = list(input_dir.glob("*_decid.tif"))

    if len(flower_files) == 0:
        raise FileNotFoundError("Expected exactly one *_flower.tif, found none.")
    if len(decid_files) == 0:
        raise FileNotFoundError("Expected exactly one *_decid.tif, found none.")

    if len(flower_files) > 1:
        raise RuntimeError(
            f"Expected exactly one *_flower.tif, found {len(flower_files)}."
        )
    if len(decid_files) > 1:
        raise RuntimeError(
            f"Expected exactly one *_decid.tif, found {len(decid_files)}."
        )

    flower_file = flower_files[0]
    decid_file = decid_files[0]

    # Derive common prefix
    flower_suffix = "_flower.tif"
    decid_suffix = "_decid.tif"

    flower_prefix = flower_file.name[:-len(flower_suffix)]
    decid_prefix = decid_file.name[:-len(decid_suffix)]

    if flower_prefix != decid_prefix:
        raise ValueError(
            f"Filename prefixes do not match: "
            f"{flower_prefix} vs {decid_prefix}"
        )

    output_file = input_dir / f"{flower_prefix}_classifications.tif"

    if output_file.exists():
        click.echo(
            f"Output file already exists: {output_file}",
            err=True,
        )
        sys.exit(1)

    process_pair(
        flower_tif=flower_file,
        decid_tif=decid_file,
        output_tif=output_file,
    )


if __name__ == "__main__":
    main()
