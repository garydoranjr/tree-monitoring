#!/usr/bin/env python
"""Crop margin-extent AVA training chips to the base plot grid, then pad to
the 50ha footprint.

The AVA plot is too small to coregister directly (AROSICS ws=(200,200) needs
>=200 px, but the base AVA cutout is only ~82x117 px), so
`apply_drone_labels_coreg.py` is run on the larger *margin* cutouts
(~282x317 px). The drone crown labels, however, only cover the plot itself, so
the 300 m margin around it is unlabelled forest that must not become training
"background". This script therefore:

1. Crops each margin chip (and its paired `.mask.png` / QA PNGs) back to the
   base AVA plot extent, using the paired base 4-band cutout as the target grid
   (a pure integer-offset window read, no resampling -- the margin and base
   cutouts are both `rasterio.mask` crops of the same source scene and share
   one pixel grid).
2. Center-pads the cropped chip up to the 50ha cutout footprint (367x205 px by
   default) so that, once the SegFormer dataloader resizes every chip to
   512x512, an AVA crown covers a comparable pixel count to a 50ha crown
   (both are 3 m/px; only the footprint differs). The 4-band image is padded
   with its nodata value and the mask with 0 (background) -- the pad region
   genuinely contains no labelled crowns, and constant (not reflect) padding
   avoids duplicating real crowns into the pad.

Typical usage:
    python scripts/crop_train_to_base.py
    python scripts/crop_train_to_base.py --pad-to 367x205 --force
    python scripts/crop_train_to_base.py --no-pad
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import imageio.v3 as iio
import rasterio as rio
from rasterio.windows import Window
from tqdm import tqdm

DEFAULT_MARGIN_DIR = Path("/Volumes/Earth03/flower/ava/train_margin")
DEFAULT_BASE_DIR = Path("/Volumes/Earth03/flower/ava/planet/4band")
DEFAULT_OUTPUT_DIR = Path("/Volumes/Earth03/flower/ava/train")
DEFAULT_GLOB = "*_4band.tif"
DEFAULT_PAD = "367x205"

# QA PNGs written alongside the chip by apply_drone_labels_coreg.py that should
# be cropped (and, for the RGB previews, padded) to match the chip.
PNG_SUFFIXES = (".mask.png", ".png", ".drone.png", ".ocm.png")


def _base_cutout_for(margin_tif: Path, base_dir: Path) -> Path:
    """Map <margin>/<prefix>_4band.tif -> <base>/<year>/<prefix>_4band.tif."""
    name = margin_tif.name
    year = name[:4]
    return base_dir / year / name


def _window_for(base, src) -> Window:
    """Integer-offset window of the base extent within the margin (src) grid."""
    px = src.transform.a
    py = -src.transform.e
    col_off = round((base.transform.c - src.transform.c) / px)
    row_off = round((src.transform.f - base.transform.f) / py)
    return Window(col_off, row_off, base.width, base.height)


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


def crop_one(margin_tif: Path, base_tif: Path, out_tif: Path,
             pad_hw: tuple[int, int] | None) -> None:
    with rio.open(base_tif) as base, rio.open(margin_tif) as src:
        window = _window_for(base, src)
        nodata = src.nodata if src.nodata is not None else 0
        data = src.read(window=window, boundless=True, fill_value=nodata)  # (bands,H,W)

        profile = src.profile.copy()
        transform = base.transform
        width, height = base.width, base.height
        top = left = 0
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
                       crs=base.crs, nodata=nodata, compress="lzw")
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
        dst.update_tags(SOURCE_MARGIN=str(margin_tif), CROPPED_TO=str(base_tif))

    # Crop (and pad) the paired PNGs using the same window/offsets. Masks pad
    # with 0 (background); RGB QA previews pad with 0 as well.
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
    help="Crop margin AVA training chips to the base plot grid and pad to the "
         "50ha footprint.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option("--margin-dir", type=click.Path(path_type=Path, file_okay=False),
              default=DEFAULT_MARGIN_DIR, show_default=True,
              help="Directory of margin-extent training chips (flat).")
@click.option("--base-dir", type=click.Path(path_type=Path, file_okay=False),
              default=DEFAULT_BASE_DIR, show_default=True,
              help="Base 4-band cutout tree (<year>/) defining target grids.")
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False),
              default=DEFAULT_OUTPUT_DIR, show_default=True)
@click.option("--glob", "pattern", default=DEFAULT_GLOB, show_default=True)
@click.option("--pad-to", default=DEFAULT_PAD, show_default=True,
              help='Center-pad to WIDTHxHEIGHT (px), matching the 50ha footprint.')
@click.option("--no-pad", is_flag=True, default=False,
              help="Skip padding; emit base-extent chips only.")
@click.option("--force", is_flag=True, default=False,
              help="Overwrite existing outputs (default: skip).")
def main(margin_dir: Path, base_dir: Path, output_dir: Path, pattern: str,
         pad_to: str, no_pad: bool, force: bool) -> None:
    if not margin_dir.exists():
        click.echo(f"margin-dir does not exist: {margin_dir}", err=True)
        sys.exit(1)
    if not base_dir.exists():
        click.echo(f"base-dir does not exist: {base_dir}", err=True)
        sys.exit(1)

    pad_hw: tuple[int, int] | None = None
    if not no_pad:
        w_str, _, h_str = pad_to.lower().partition("x")
        pad_hw = (int(h_str), int(w_str))  # (height, width)

    output_dir.mkdir(parents=True, exist_ok=True)
    chips = sorted(margin_dir.glob(pattern))
    if not chips:
        click.echo(f"no files matched {pattern!r} under {margin_dir}", err=True)
        sys.exit(1)
    click.echo(f"found {len(chips)} margin chips under {margin_dir}")

    n_done = n_skipped = n_missing = n_errored = 0
    pbar = tqdm(chips, desc="cropping", unit="chip")
    for chip in pbar:
        out_tif = output_dir / chip.name
        if out_tif.exists() and not force:
            n_skipped += 1
        else:
            base_tif = _base_cutout_for(chip, base_dir)
            if not base_tif.exists():
                n_missing += 1
            else:
                try:
                    crop_one(chip, base_tif, out_tif, pad_hw)
                    n_done += 1
                except Exception as exc:  # noqa: BLE001
                    n_errored += 1
                    click.echo(f"failed: {chip} -> {out_tif}: {exc}", err=True)
        pbar.set_postfix(done=n_done, skipped=n_skipped,
                         missing=n_missing, errored=n_errored)

    click.echo(f"done: {n_done} written, {n_skipped} skipped, "
               f"{n_missing} missing base cutout, {n_errored} errored")
    if n_errored:
        sys.exit(2)


if __name__ == "__main__":
    main()
