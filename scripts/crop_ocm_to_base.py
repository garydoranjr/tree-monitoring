#!/usr/bin/env python
"""Crop margin-extent OCM masks back to the base cutout grid.

OmniCloudMask requires >=32x32 px at its 10 m inference grid, so very small
AOIs (e.g. the AVA plot) must be masked at a padded/margin extent. This script
takes the margin OCM masks and crops each one to the exact pixel grid of the
paired base 4-band cutout, so the resulting masks align 1:1 with the base
`4band`/`rgb` cutouts.

For a given scene the base and margin cutouts are both `rasterio.mask` crops of
the same source Planet scene, so they share one pixel grid and the base extent
is a pixel-aligned subset of the margin extent. The crop is therefore a plain
integer-offset window read (no resampling), preserving band descriptions,
scales, and tags from the margin mask.

Typical usage:
    python scripts/crop_ocm_to_base.py
    python scripts/crop_ocm_to_base.py --force
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import rasterio as rio
from rasterio.windows import Window
from tqdm import tqdm

DEFAULT_BASE_DIR = Path("/Volumes/Earth03/flower/ava/planet/4band")
DEFAULT_OCM_DIR = Path("/Volumes/Earth03/flower/ava/planet_margin/ocm")
DEFAULT_OUTPUT_DIR = Path("/Volumes/Earth03/flower/ava/planet/ocm")
DEFAULT_GLOB = "*_4band.tif"


def _margin_ocm_for(base_path: Path, base_dir: Path, ocm_dir: Path) -> Path:
    """Map <base>/<year>/<prefix>_4band.tif -> <ocm>/<year>/<prefix>_ocm.tif."""
    rel = base_path.relative_to(base_dir)
    stem = base_path.stem
    if stem.endswith("_4band"):
        stem = stem[: -len("_4band")]
    return ocm_dir / rel.parent / f"{stem}_ocm.tif"


def _output_for(base_path: Path, base_dir: Path, output_dir: Path) -> Path:
    rel = base_path.relative_to(base_dir)
    stem = base_path.stem
    if stem.endswith("_4band"):
        stem = stem[: -len("_4band")]
    return output_dir / rel.parent / f"{stem}_ocm.tif"


def crop_one(base_path: Path, margin_ocm_path: Path, out_path: Path) -> None:
    """Window-read the margin OCM onto the base cutout's exact grid."""
    with rio.open(base_path) as base, rio.open(margin_ocm_path) as src:
        px = src.transform.a
        py = -src.transform.e
        col_off = round((base.transform.c - src.transform.c) / px)
        row_off = round((src.transform.f - base.transform.f) / py)
        window = Window(col_off, row_off, base.width, base.height)

        data = src.read(window=window, boundless=True, fill_value=255)

        profile = src.profile.copy()
        profile.update(
            transform=base.transform, width=base.width, height=base.height,
            crs=base.crs, compress="lzw",
        )
        for k in ("photometric", "interleave", "blockxsize", "blockysize", "tiled"):
            profile.pop(k, None)

        descriptions = src.descriptions
        scales = src.scales
        offsets = src.offsets
        units = src.units
        ds_tags = src.tags()
        band_tags = [src.tags(i) for i in range(1, src.count + 1)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rio.open(out_path, "w", **profile) as dst:
        dst.write(data)
        dst.scales = scales
        dst.offsets = offsets
        dst.units = units
        for i, desc in enumerate(descriptions, start=1):
            if desc:
                dst.set_band_description(i, desc)
        dst.update_tags(**ds_tags)
        dst.update_tags(SOURCE_MARGIN_OCM=str(margin_ocm_path),
                        CROPPED_TO=str(base_path))
        for i, tags in enumerate(band_tags, start=1):
            if tags:
                dst.update_tags(i, **tags)


@click.command(
    help="Crop margin-extent OCM masks to the base cutout grid.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option("--base-dir", type=click.Path(path_type=Path, file_okay=False),
              default=DEFAULT_BASE_DIR, show_default=True,
              help="Base 4-band cutout tree defining the target grids.")
@click.option("--ocm-dir", type=click.Path(path_type=Path, file_okay=False),
              default=DEFAULT_OCM_DIR, show_default=True,
              help="Margin-extent OCM mask tree.")
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False),
              default=DEFAULT_OUTPUT_DIR, show_default=True)
@click.option("--glob", "pattern", default=DEFAULT_GLOB, show_default=True)
@click.option("--force", is_flag=True, default=False,
              help="Overwrite existing cropped masks (default: skip).")
def main(base_dir: Path, ocm_dir: Path, output_dir: Path, pattern: str,
         force: bool) -> None:
    if not base_dir.exists():
        click.echo(f"base-dir does not exist: {base_dir}", err=True)
        sys.exit(1)
    if not ocm_dir.exists():
        click.echo(f"ocm-dir does not exist: {ocm_dir}", err=True)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenes = sorted(base_dir.rglob(pattern))
    if not scenes:
        click.echo(f"no files matched {pattern!r} under {base_dir}", err=True)
        sys.exit(1)
    click.echo(f"found {len(scenes)} base cutouts under {base_dir}")

    n_done = n_skipped = n_missing = n_errored = 0
    pbar = tqdm(scenes, desc="cropping", unit="scene")
    for scene in pbar:
        out_path = _output_for(scene, base_dir, output_dir)
        if out_path.exists() and not force:
            n_skipped += 1
            pbar.set_postfix(done=n_done, skipped=n_skipped,
                             missing=n_missing, errored=n_errored)
            continue
        margin_ocm = _margin_ocm_for(scene, base_dir, ocm_dir)
        if not margin_ocm.exists():
            n_missing += 1
            pbar.set_postfix(done=n_done, skipped=n_skipped,
                             missing=n_missing, errored=n_errored)
            continue
        try:
            crop_one(scene, margin_ocm, out_path)
            n_done += 1
        except Exception as exc:  # noqa: BLE001
            n_errored += 1
            click.echo(f"failed: {scene} -> {out_path}: {exc}", err=True)
        pbar.set_postfix(done=n_done, skipped=n_skipped,
                         missing=n_missing, errored=n_errored)

    click.echo(f"done: {n_done} written, {n_skipped} skipped, "
               f"{n_missing} missing margin-ocm, {n_errored} errored")
    if n_errored:
        sys.exit(2)


if __name__ == "__main__":
    main()
