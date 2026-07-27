#!/usr/bin/env python
"""Crop margin-extent AVA training chips to the plot extent, then pad to a
square SegFormer tile.

The AVA plot is too small to coregister directly (AROSICS ws=(200,200) needs
>=200 px), so `apply_drone_labels_coreg.py` is run on the larger *margin*
cutouts. The drone crown labels, however, only cover the plot itself, so the
300 m margin around it is unlabelled forest that must not become training
"background". The 4-band training pipeline
(`scripts/train_planet_image_segformer_4b.py`) also consumes chips at 0.75 m
(`--resize 4`) and, for AVA, whole 512x512 tiles. This script therefore:

1. Crops each margin chip (and its paired `.mask.png` / QA PNGs) to the fixed
   AVA plot extent defined by `config/clip_ava_plot.yml`, using an
   integer-offset window read (no resampling) against the chip's own transform.
   At 0.75 m the plot is ~324x467 px. The plot bounds lie on the 3 m Planet
   clip grid and hence on the 0.75 m grid, so the window is pixel-aligned.
2. Center-pads the cropped chip up to a square tile (512x512 px by default) so
   the training loader can consume it whole without a left/right split. The
   4-band image is padded with its nodata value and the mask with 0
   (background) -- the pad genuinely contains no labelled crowns, and constant
   (not reflect) padding avoids duplicating real crowns into the pad.

Typical usage:
    python scripts/crop_train_to_base.py
    python scripts/crop_train_to_base.py --plot-config config/clip_ava_plot.yml --size 512
    python scripts/crop_train_to_base.py --no-pad
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import imageio.v3 as iio
import rasterio as rio
import yaml
from rasterio.windows import Window
from tqdm import tqdm

DEFAULT_MARGIN_DIR = Path("/Volumes/Earth03/flower/ava/train_margin")
DEFAULT_PLOT_CONFIG = Path("config/clip_ava_plot.yml")
DEFAULT_OUTPUT_DIR = Path("/Volumes/Earth03/flower/ava/train")
DEFAULT_GLOB = "*_4band.tif"
DEFAULT_SIZE = 512

# QA PNGs written alongside the chip by apply_drone_labels_coreg.py that should
# be cropped (and padded) to match the chip.
PNG_SUFFIXES = (".mask.png", ".png", ".drone.png", ".ocm.png")


def _plot_bounds(plot_config: Path) -> tuple[float, float, float, float]:
    """Read (xmin, ymin, xmax, ymax) from a clip config's polygon ring."""
    with open(plot_config) as f:
        cfg = yaml.safe_load(f)
    ring = cfg["region"]["coordinates"][0]
    xs = [pt[0] for pt in ring]
    ys = [pt[1] for pt in ring]
    return min(xs), min(ys), max(xs), max(ys)


def _window_for_bounds(src, bounds: tuple[float, float, float, float]) -> Window:
    """Integer-offset window of the geographic `bounds` within the src grid."""
    xmin, ymin, xmax, ymax = bounds
    px = src.transform.a
    py = -src.transform.e
    col_off = round((xmin - src.transform.c) / px)
    row_off = round((src.transform.f - ymax) / py)
    width = round((xmax - xmin) / px)
    height = round((ymax - ymin) / py)
    return Window(col_off, row_off, width, height)


def _pad_to(arr: np.ndarray, target_hw: tuple[int, int], value) -> tuple[np.ndarray, int, int]:
    """Center-pad arr (H,W) or (H,W,C) to target (H,W). Returns (out, top, left)."""
    th, tw = target_hw
    h, w = arr.shape[:2]
    top = (th - h) // 2
    left = (tw - w) // 2
    pad = [(top, th - h - top), (left, tw - w - left)]
    if arr.ndim == 3:
        pad.append((0, 0))
    return np.pad(arr, pad, mode="constant", constant_values=value), top, left


def crop_one(margin_tif: Path, bounds: tuple[float, float, float, float],
             out_tif: Path, pad_hw: tuple[int, int] | None) -> None:
    with rio.open(margin_tif) as src:
        window = _window_for_bounds(src, bounds)
        nodata = src.nodata if src.nodata is not None else 0
        data = src.read(window=window, boundless=True, fill_value=nodata)  # (bands,H,W)

        profile = src.profile.copy()
        transform = rio.windows.transform(window, src.transform)
        width, height = int(window.width), int(window.height)
        if pad_hw is not None:
            padded = [_pad_to(data[b], pad_hw, nodata) for b in range(data.shape[0])]
            data = np.stack([p[0] for p in padded], axis=0)
            _, top, left = padded[0]
            height, width = pad_hw
            transform = rio.Affine(
                transform.a, transform.b, transform.c - left * transform.a,
                transform.d, transform.e, transform.f - top * transform.e,
            )

        profile.update(transform=transform, width=width, height=height,
                       nodata=nodata, compress="lzw")
        for k in ("photometric", "interleave", "blockxsize", "blockysize", "tiled"):
            profile.pop(k, None)
        descriptions = src.descriptions
        scales, offsets, units = src.scales, src.offsets, src.units

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rio.open(out_tif, "w", **profile) as dst:
        dst.write(data)
        dst.scales = scales
        dst.offsets = offsets
        dst.units = units
        for i, desc in enumerate(descriptions, start=1):
            if desc:
                dst.set_band_description(i, desc)
        dst.update_tags(SOURCE_MARGIN=str(margin_tif))

    # Crop (and pad) the paired PNGs using the same window/offsets. Masks and
    # RGB QA previews pad with 0.
    for suffix in PNG_SUFFIXES:
        src_png = margin_tif.with_name(margin_tif.stem + suffix)
        if not src_png.exists():
            continue
        arr = iio.imread(src_png)
        cropped = _crop_array(arr, int(window.col_off), int(window.row_off),
                              int(window.height), int(window.width))
        if pad_hw is not None:
            cropped, _, _ = _pad_to(cropped, pad_hw, 0)
        out_png = out_tif.with_name(out_tif.stem + suffix)
        iio.imwrite(out_png, cropped)


def _crop_array(arr: np.ndarray, col_off: int, row_off: int, h: int, w: int) -> np.ndarray:
    """Window a plain (H,W[,C]) array with boundless behaviour, zero-filled."""
    ah, aw = arr.shape[:2]
    shape = (h, w) + arr.shape[2:]
    out = np.zeros(shape, dtype=arr.dtype)
    r0, c0 = max(row_off, 0), max(col_off, 0)
    r1, c1 = min(row_off + h, ah), min(col_off + w, aw)
    if r1 <= r0 or c1 <= c0:
        return out
    out[r0 - row_off:r1 - row_off, c0 - col_off:c1 - col_off] = arr[r0:r1, c0:c1]
    return out


@click.command(
    help="Crop margin AVA training chips to the plot extent and pad to a square "
         "SegFormer tile.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option("--margin-dir", type=click.Path(path_type=Path, file_okay=False),
              default=DEFAULT_MARGIN_DIR, show_default=True,
              help="Directory of margin-extent training chips (flat).")
@click.option("--plot-config", type=click.Path(path_type=Path, dir_okay=False),
              default=DEFAULT_PLOT_CONFIG, show_default=True,
              help="Clip config whose polygon ring defines the plot extent.")
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False),
              default=DEFAULT_OUTPUT_DIR, show_default=True)
@click.option("--glob", "pattern", default=DEFAULT_GLOB, show_default=True)
@click.option("--size", default=DEFAULT_SIZE, type=int, show_default=True,
              help="Center-pad to SIZExSIZE px (the SegFormer tile size).")
@click.option("--no-pad", is_flag=True, default=False,
              help="Skip padding; emit plot-extent chips only.")
@click.option("--force", is_flag=True, default=False,
              help="Overwrite existing outputs (default: skip).")
def main(margin_dir: Path, plot_config: Path, output_dir: Path, pattern: str,
         size: int, no_pad: bool, force: bool) -> None:
    if not margin_dir.exists():
        click.echo(f"margin-dir does not exist: {margin_dir}", err=True)
        sys.exit(1)
    if not plot_config.exists():
        click.echo(f"plot-config does not exist: {plot_config}", err=True)
        sys.exit(1)

    bounds = _plot_bounds(plot_config)
    pad_hw: tuple[int, int] | None = None if no_pad else (size, size)

    output_dir.mkdir(parents=True, exist_ok=True)
    chips = sorted(margin_dir.glob(pattern))
    if not chips:
        click.echo(f"no files matched {pattern!r} under {margin_dir}", err=True)
        sys.exit(1)
    click.echo(f"found {len(chips)} margin chips under {margin_dir}")
    click.echo(f"plot bounds (xmin,ymin,xmax,ymax): {bounds}")

    n_done = n_skipped = n_errored = 0
    pbar = tqdm(chips, desc="cropping", unit="chip")
    for chip in pbar:
        out_tif = output_dir / chip.name
        if out_tif.exists() and not force:
            n_skipped += 1
        else:
            try:
                crop_one(chip, bounds, out_tif, pad_hw)
                n_done += 1
            except Exception as exc:  # noqa: BLE001
                n_errored += 1
                click.echo(f"failed: {chip} -> {out_tif}: {exc}", err=True)
        pbar.set_postfix(done=n_done, skipped=n_skipped, errored=n_errored)

    click.echo(f"done: {n_done} written, {n_skipped} skipped, {n_errored} errored")
    if n_errored:
        sys.exit(2)


if __name__ == "__main__":
    main()
