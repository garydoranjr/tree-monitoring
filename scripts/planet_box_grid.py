"""Shared 3 m UTM grid over the BCNM Planet order box.

The box is the `geometry` of config/planet_search.yml (lon/lat), transformed to
EPSG:32617 and snapped outward to the 3 m PlanetScope ortho grid. Scenes are
warped onto this grid so that same-day scenes can be mosaicked and compared
with the static water mask pixel for pixel.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio as rio
import yaml
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject, transform_bounds

UTM = CRS.from_epsg(32617)
RES = 3.0
SEARCH_CONFIG = Path(__file__).resolve().parent.parent / 'config' / 'planet_search.yml'


def box_grid(search_config: Path = SEARCH_CONFIG) -> tuple[Affine, int, int]:
    """Return (transform, width, height) of the 3 m UTM grid over the order box."""
    with open(search_config) as f:
        ring = yaml.safe_load(f)['geometry']['coordinates'][0]
    lons, lats = zip(*ring)
    x0, y0, x1, y1 = transform_bounds('EPSG:4326', UTM, min(lons), min(lats),
                                      max(lons), max(lats))
    x0, y0 = math.floor(x0 / RES) * RES, math.floor(y0 / RES) * RES
    x1, y1 = math.ceil(x1 / RES) * RES, math.ceil(y1 / RES) * RES
    width, height = int(round((x1 - x0) / RES)), int(round((y1 - y0) / RES))
    return Affine(RES, 0, x0, 0, -RES, y1), width, height


def warp_to_box(path: Path, bands, transform: Affine, width: int, height: int,
                nodata, dtype, resampling=Resampling.nearest) -> np.ndarray:
    """Read `bands` (1-based list) of `path` onto the box grid; outside = nodata."""
    out = np.full((len(bands), height, width), nodata, dtype=dtype)
    with rio.open(path) as src:
        for i, b in enumerate(bands):
            reproject(rio.band(src, b), out[i], dst_transform=transform,
                      dst_crs=UTM, dst_nodata=nodata, resampling=resampling)
    return out
