#!/usr/bin/env python
"""Per-scene fraction of Barro Colorado Island with a clear view of the canopy.

For every OmniCloudMask (OCM) output under --ocm-dir, computes the fraction of
the fixed BCI polygon whose OCM class is Clear (class 0). Cloud, thin cloud,
shadow, NoData, and any island pixel not covered by the scene footprint all
count as *not clear* (missing == not clear). The denominator is the whole island
(a fixed pixel count), so partial-coverage strips correctly yield low fractions.

OCM band 1 class encoding (see scripts/cloud_mask_planet.py):
    0=Clear, 1=Thick Cloud, 2=Thin Cloud, 3=Cloud Shadow, 255=NoData

Scene name and acquisition datetime come from the filename stem (drop `_ocm`):
tokens are `YYYYMMDD_HHMMSS_<satid...>` (UTC time).

Output CSV columns:
    image_name, datetime_utc, fraction_clear, clear_pixels, island_pixels

Typical usage:
    python scripts/bci_clear_fraction.py
    python scripts/bci_clear_fraction.py \\
        --ocm-dir /Volumes/Earth03/flower/planet_bci_clipped/ocm \\
        --geometry /Volumes/Earth03/flower/whole_island/Barro_Colorado_Island.geojson \\
        --output /Volumes/Earth03/flower/whole_island/bci_clear_fraction.csv
"""
from __future__ import annotations

import csv
import json
import logging
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_geom
from tqdm import tqdm

log = logging.getLogger("bci_clear_fraction")

# ---------------------------------------------------------------------------
# Constants

OCM_CLEAR_CLASS = 0        # band-1 class value meaning "Clear" (canopy visible)
OCM_NODATA = 255           # band-1 NoData / off-footprint fill
GRID_RES_M = 3.0           # OCM native grid resolution (EPSG:32617, 3 m)
GRID_EPSG = 32617          # UTM 17N; all OCM scenes share this CRS

DEFAULT_OCM_DIR = Path("/Volumes/Earth03/flower/planet_bci_clipped/ocm")
DEFAULT_GEOMETRY = Path(
    "/Volumes/Earth03/flower/whole_island/Barro_Colorado_Island.geojson"
)
DEFAULT_OUTPUT = Path("/Volumes/Earth03/flower/whole_island/bci_clear_fraction.csv")

# `YYYYMMDD_HHMMSS_...` — first two tokens are date and UTC time (both 3- and
# 4-token scene ids match).
_SCENE_DT_RE = re.compile(r"^(\d{8})_(\d{6})_")


# ---------------------------------------------------------------------------
# Reference island grid


@dataclass(frozen=True)
class ReferenceGrid:
    mask: np.ndarray          # (H, W) bool: True where inside the island
    transform: rio.Affine     # maps pixel -> EPSG:32617 coords
    crs: rio.crs.CRS
    island_pixels: int        # fixed denominator = mask.sum()


def build_reference_grid(geometry_path: Path) -> ReferenceGrid:
    """Rasterize the island polygon onto a 3 m EPSG:32617 grid (fixed denominator)."""
    doc = json.loads(Path(geometry_path).read_text())
    feats = doc.get("features", [])
    if len(feats) != 1:
        raise ValueError(
            f"{geometry_path}: expected exactly 1 feature, got {len(feats)}"
        )
    src_crs = "EPSG:4326"
    if isinstance(doc.get("crs"), dict):
        name = doc["crs"].get("properties", {}).get("name")
        if name:
            src_crs = name

    dst_crs = rio.crs.CRS.from_epsg(GRID_EPSG)
    geom = transform_geom(src_crs, dst_crs, feats[0]["geometry"])

    xs: list[float] = []
    ys: list[float] = []

    def _walk(coords):
        if coords and isinstance(coords[0], (int, float)):
            xs.append(coords[0])
            ys.append(coords[1])
        else:
            for part in coords:
                _walk(part)

    _walk(geom["coordinates"])

    res = GRID_RES_M
    minx = math.floor(min(xs) / res) * res
    miny = math.floor(min(ys) / res) * res
    maxx = math.ceil(max(xs) / res) * res
    maxy = math.ceil(max(ys) / res) * res
    width = int(round((maxx - minx) / res))
    height = int(round((maxy - miny) / res))
    transform = from_origin(minx, maxy, res, res)

    mask = rasterize(
        [(geom, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)

    island_pixels = int(mask.sum())
    if island_pixels == 0:
        raise ValueError(f"{geometry_path}: island rasterized to 0 pixels")
    log.info(
        "reference grid %dx%d, island_pixels=%d (%.1f ha)",
        width, height, island_pixels, island_pixels * res * res / 1e4,
    )
    return ReferenceGrid(mask, transform, dst_crs, island_pixels)


# ---------------------------------------------------------------------------
# Per-scene clear fraction


def scene_clear_pixels(ocm_path: Path, ref: ReferenceGrid, dst: np.ndarray) -> int:
    """Reproject the scene's class band onto the island grid; count clear island px.

    `dst` is a reusable (H, W) uint8 buffer; it is filled with NODATA and then
    overwritten where the scene overlaps the island grid.
    """
    dst.fill(OCM_NODATA)
    with rio.open(ocm_path) as src:
        reproject(
            source=rio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            resampling=Resampling.nearest,
            src_nodata=OCM_NODATA,
            dst_nodata=OCM_NODATA,
        )
    clear = (dst == OCM_CLEAR_CLASS) & ref.mask
    return int(clear.sum())


def parse_scene(ocm_path: Path) -> tuple[str, str]:
    """Return (image_name, datetime_utc ISO-8601 Z) from an `*_ocm.tif` path."""
    stem = ocm_path.stem
    image_name = stem[:-4] if stem.endswith("_ocm") else stem
    m = _SCENE_DT_RE.match(image_name)
    if not m:
        raise ValueError(f"{ocm_path}: cannot parse date/time from {image_name!r}")
    dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return image_name, dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# CLI


@click.command(
    help="Fraction of BCI with a clear canopy view per OmniCloudMask scene.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option("--ocm-dir", type=click.Path(path_type=Path, file_okay=False),
              default=DEFAULT_OCM_DIR, show_default=True,
              help="Directory tree of <year>/*_ocm.tif OmniCloudMask outputs.")
@click.option("--geometry", type=click.Path(path_type=Path, dir_okay=False),
              default=DEFAULT_GEOMETRY, show_default=True,
              help="Single-feature GeoJSON of the island polygon.")
@click.option("--output", type=click.Path(path_type=Path, dir_okay=False),
              default=DEFAULT_OUTPUT, show_default=True,
              help="Output CSV path.")
@click.option("--glob", "pattern", default="*_ocm.tif", show_default=True,
              help="Filename pattern to match under --ocm-dir (recursive).")
@click.option("-v", "--verbose", is_flag=True, default=False)
def main(ocm_dir: Path, geometry: Path, output: Path, pattern: str,
         verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not ocm_dir.exists():
        click.echo(f"ocm-dir does not exist: {ocm_dir}", err=True)
        sys.exit(1)
    if not geometry.exists():
        click.echo(f"geometry does not exist: {geometry}", err=True)
        sys.exit(1)

    ref = build_reference_grid(geometry)

    # Recursive glob, skipping the `_tmp*` scratch dirs left by cloud_mask_planet.
    scenes = sorted(
        p for p in ocm_dir.rglob(pattern) if "_tmp" not in str(p)
    )
    if not scenes:
        click.echo(f"no files matched {pattern!r} under {ocm_dir}", err=True)
        sys.exit(1)
    click.echo(f"found {len(scenes)} scenes under {ocm_dir}")

    dst = np.empty(ref.mask.shape, dtype="uint8")
    rows: list[tuple[str, str, float, int, int]] = []
    n_errored = 0
    for scene in tqdm(scenes, desc="scoring", unit="scene"):
        try:
            image_name, dt_utc = parse_scene(scene)
            clear_px = scene_clear_pixels(scene, ref, dst)
            frac = clear_px / ref.island_pixels
            rows.append((image_name, dt_utc, frac, clear_px, ref.island_pixels))
        except Exception as exc:  # noqa: BLE001
            n_errored += 1
            log.exception("failed: %s: %s", scene, exc)

    rows.sort(key=lambda r: r[1])

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["image_name", "datetime_utc", "fraction_clear",
             "clear_pixels", "island_pixels"]
        )
        for image_name, dt_utc, frac, clear_px, island_px in rows:
            w.writerow([image_name, dt_utc, f"{frac:.6f}", clear_px, island_px])

    click.echo(f"wrote {len(rows)} rows to {output} ({n_errored} errored)")
    if n_errored:
        sys.exit(2)


if __name__ == "__main__":
    main()
