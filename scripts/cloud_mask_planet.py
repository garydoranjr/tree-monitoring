#!/usr/bin/env python
"""Generate OmniCloudMask cloud/cloud-shadow masks for clipped Planet 4-band scenes.

Walks --input-dir for `*_4band.tif` files and writes a parallel tree of 5-band
uint8 GeoTIFFs under --output-dir. Each output contains the argmax class label
in band 1 and the per-class softmax confidences (scaled to 0-255) in bands 2-5.
Existing outputs are skipped unless --force is given.

Class encoding (band 1):
    0=Clear, 1=Thick Cloud, 2=Thin Cloud, 3=Cloud Shadow, 255=NoData

Probability bands (2-5), values 0-255, scale 1/255:
    band 2 = P(Clear), band 3 = P(Thick Cloud),
    band 4 = P(Thin Cloud), band 5 = P(Cloud Shadow)

Inference uses OmniCloudMask (Wright et al. 2025, https://github.com/DPIRD-DMA/OmniCloudMask).
The model runs at 10 m and the output is nearest-neighbor upsampled to the native
~3 m grid; mask edge precision is therefore ~10 m even on the 3 m grid.

Required dependency (install into the `flower` conda env before running):
    pip install 'omnicloudmask>=1,<2'

Typical usage:
    python scripts/cloud_mask_planet.py
    python scripts/cloud_mask_planet.py --force
    python scripts/cloud_mask_planet.py \\
        --input-dir /Volumes/Earth03/flower/planet_clipped/4band \\
        --output-dir /Volumes/Earth03/flower/planet_clipped/ocm
"""
from __future__ import annotations

import logging
import math
import shutil
import sys
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import click
import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject
from tqdm import tqdm

import omnicloudmask as ocm

log = logging.getLogger("cloud_mask_planet")

# ---------------------------------------------------------------------------
# Constants

OCM_CLASSES = {
    0: "Clear",
    1: "Thick Cloud",
    2: "Thin Cloud",
    3: "Cloud Shadow",
    255: "NoData",
}
CLASS_ORDER = [0, 1, 2, 3]  # band order for probability bands 2..5
CLASS_NAMES_ORDERED = [OCM_CLASSES[c] for c in CLASS_ORDER]

# OmniCloudMask wants Red, Green, NIR. PlanetScope 4-band is BGRN, so [3, 2, 4].
# SuperDove 8-band is [Coastal, Blue, GreenI, Green, Yellow, Red, RedEdge, NIR].
PLANETSCOPE_4BAND_ORDER = [3, 2, 4]
PLANETSCOPE_8BAND_ORDER = [6, 4, 8]

DEFAULT_INPUT_DIR = Path("/Volumes/Earth03/flower/planet_clipped/4band")
DEFAULT_OUTPUT_DIR = Path("/Volumes/Earth03/flower/planet_clipped/ocm")
DEFAULT_GLOB = "*_4band.tif"
DEVICE_CHOICES = ["auto", "cpu", "cuda", "mps"]


# ---------------------------------------------------------------------------
# Device resolution


@dataclass(frozen=True)
class DeviceConfig:
    device: str  # "cuda" | "mps" | "cpu"
    dtype: str   # "bf16" | "fp32"
    batch_size: int


def _resolve_device(pref: str = "auto", batch_size: int = 4, dtype: str = "bf16") -> DeviceConfig:
    """Pick an inference device. CPU is forced to fp32+bs1; MPS to fp32."""
    pref = (pref or "auto").lower()
    if pref == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except Exception:
            device = "cpu"
    else:
        device = pref

    if device == "cpu":
        return DeviceConfig("cpu", "fp32", 1)
    if device == "mps":
        return DeviceConfig("mps", "fp32", batch_size)
    return DeviceConfig(device, dtype, batch_size)


# ---------------------------------------------------------------------------
# Input handling: band detection, UTM reprojection, saturation


def detect_band_order(path: Path) -> list[int]:
    with rio.open(path) as src:
        n = src.count
    if n < 4:
        raise ValueError(f"{path}: need at least 4 bands, got {n}")
    if n == 4:
        return list(PLANETSCOPE_4BAND_ORDER)
    if n == 8:
        log.warning("%s: 8-band SuperDove detected; using band_order=%s",
                    path, PLANETSCOPE_8BAND_ORDER)
        return list(PLANETSCOPE_8BAND_ORDER)
    raise ValueError(f"{path}: unsupported band count {n}; expected 4 or 8")


def _utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    return (32600 if lat >= 0 else 32700) + zone


def ensure_utm(path: Path, tmp_dir: Path) -> Path:
    """Reproject to local UTM if input is geographic; pass through otherwise."""
    with rio.open(path) as src:
        if src.crs is not None and not src.crs.is_geographic:
            return path
        if src.crs is None:
            raise ValueError(f"{path}: missing CRS")
        b = src.bounds
        lon = 0.5 * (b.left + b.right)
        lat = 0.5 * (b.bottom + b.top)
        dst_epsg = _utm_epsg_from_lonlat(lon, lat)
        dst_crs = rio.crs.CRS.from_epsg(dst_epsg)
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(crs=dst_crs, transform=transform, width=width, height=height)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        out_path = tmp_dir / f"{path.stem}_utm.tif"
        with rio.open(out_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
        log.warning("%s: reprojected from %s to EPSG:%d -> %s",
                    path, src.crs, dst_epsg, out_path)
        return out_path


def zero_saturated(arr: np.ndarray, dtype: np.dtype | str) -> np.ndarray:
    """Zero out pixels where any band is within 1% of the dtype max.

    Required preprocessing — without this the model misses clouds at sensor-saturation
    patches.
    """
    dt = np.dtype(dtype)
    if not np.issubdtype(dt, np.integer):
        return arr
    threshold = int(0.99 * np.iinfo(dt).max)
    if arr.ndim == 2:
        sat = arr >= threshold
        out = arr.copy()
        out[sat] = 0
        return out
    sat_any = (arr >= threshold).any(axis=0)
    out = arr.copy()
    out[:, sat_any] = 0
    return out


def _write_temp_like(src_path: Path, arr: np.ndarray, out_path: Path) -> Path:
    with rio.open(src_path) as src:
        profile = src.profile.copy()
    profile.update(compress="lzw")
    for k in ("photometric", "interleave", "blockxsize", "blockysize", "tiled"):
        profile.pop(k, None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rio.open(out_path, "w", **profile) as dst:
        dst.write(arr)
    return out_path


def _saturate_zero_to_temp(path: Path, tmp_dir: Path) -> Path:
    with rio.open(path) as src:
        arr = src.read()
        dtype = src.dtypes[0]
    cleaned = zero_saturated(arr, dtype)
    return _write_temp_like(path, cleaned, tmp_dir / f"{path.stem}_sat0.tif")


# ---------------------------------------------------------------------------
# Nodata overlay (apply input nodata to model output grid)


def _build_validity_overlay(source_path: Path, mask_profile: dict) -> np.ndarray:
    """Return uint8 (H, W) with 1=valid, 0=invalid in the model-output grid.

    Conservative: any source-band nodata at any contributing pixel forces
    invalid via Resampling.min.
    """
    with rio.open(source_path) as src:
        src_nodata = src.nodata if src.nodata is not None else 0
        bands = src.read()
        valid_native = (bands != src_nodata).all(axis=0).astype("uint8")
        out_valid = np.zeros((mask_profile["height"], mask_profile["width"]),
                             dtype="uint8")
        reproject(
            source=valid_native,
            destination=out_valid,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=mask_profile["transform"],
            dst_crs=mask_profile["crs"],
            resampling=Resampling.min,
        )
    return out_valid


# ---------------------------------------------------------------------------
# Inference (with OOM/MPS fallbacks)


def _predict_with_retry(arr: np.ndarray, cfg: DeviceConfig) -> tuple[np.ndarray, DeviceConfig]:
    """Run predict_from_array with export_confidence=True and a fallback chain.

    Returns (probs, used_cfg) where probs has shape (4, H, W) of softmax probs.
    """
    kwargs = dict(
        inference_device=cfg.device,
        inference_dtype=cfg.dtype,
        batch_size=cfg.batch_size,
        export_confidence=True,
        softmax_output=True,
    )
    try:
        probs = ocm.predict_from_array(arr, **kwargs)
        return probs, cfg
    except Exception as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or "oom" in msg:
            log.warning("OOM on %s; retrying with batch_size=1 + mosaic_device=cpu", cfg.device)
            try:
                probs = ocm.predict_from_array(
                    arr,
                    inference_device=cfg.device,
                    inference_dtype=cfg.dtype,
                    batch_size=1,
                    mosaic_device="cpu",
                    export_confidence=True,
                    softmax_output=True,
                )
                return probs, DeviceConfig(cfg.device, cfg.dtype, 1)
            except Exception as exc2:
                log.warning("Second OOM; falling back to CPU/fp32. orig=%s second=%s", exc, exc2)
        if cfg.device != "cpu":
            cpu_cfg = DeviceConfig("cpu", "fp32", 1)
            probs = ocm.predict_from_array(
                arr,
                inference_device=cpu_cfg.device,
                inference_dtype=cpu_cfg.dtype,
                batch_size=cpu_cfg.batch_size,
                export_confidence=True,
                softmax_output=True,
            )
            return probs, cpu_cfg
        raise


# ---------------------------------------------------------------------------
# 5-band assembly + nearest-neighbor upsample to native grid


def _assemble_stack(probs: np.ndarray, valid_overlay: np.ndarray) -> np.ndarray:
    """Build the 5-band uint8 stack at the model-output grid.

    Inputs:
        probs         (4, H, W) float in [0, 1] — softmax over CLASS_ORDER.
        valid_overlay (H, W)    uint8: 1=valid, 0=invalid.

    Output:
        stack (5, H, W) uint8.
            band 1: argmax class (0..3) or 255 at invalid pixels.
            bands 2..5: round(prob * 255), or 255 at invalid pixels (treated as nodata).
    """
    if probs.ndim != 3 or probs.shape[0] != 4:
        raise ValueError(f"expected probs shape (4, H, W); got {probs.shape}")
    H, W = probs.shape[1], probs.shape[2]

    label = np.argmax(probs, axis=0).astype(np.uint8)
    prob_u8 = np.clip(np.round(probs * 255.0), 0, 255).astype(np.uint8)

    invalid = valid_overlay == 0
    label[invalid] = 255
    prob_u8[:, invalid] = 255

    stack = np.empty((5, H, W), dtype=np.uint8)
    stack[0] = label
    stack[1:5] = prob_u8
    return stack


def _upsample_stack_to_reference(
    stack_path: Path, reference_path: Path, out_path: Path,
) -> tuple[np.ndarray, dict]:
    """Nearest-neighbor upsample a 5-band uint8 stack onto reference's grid.

    Returns (out_array, profile) describing the upsampled raster on disk.
    """
    with rio.open(stack_path) as src, rio.open(reference_path) as ref:
        n = src.count
        out = np.full((n, ref.height, ref.width), 255, dtype="uint8")
        for i in range(1, n + 1):
            reproject(
                source=rio.band(src, i),
                destination=out[i - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                resampling=Resampling.nearest,
                src_nodata=255,
                dst_nodata=255,
            )
        profile = ref.profile.copy()
    profile.update(count=n, dtype="uint8", nodata=255, compress="lzw")
    for k in ("photometric", "interleave", "blockxsize", "blockysize", "tiled"):
        profile.pop(k, None)
    return out, profile


# ---------------------------------------------------------------------------
# Output writing with full band metadata


def _write_output(out_path: Path, data: np.ndarray, profile: dict, *,
                  source_path: Path, band_order: list[int],
                  used_cfg: DeviceConfig, ocm_version: str) -> None:
    """Write the 5-band uint8 GeoTIFF with band descriptions, scales, and tags."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    descriptions = (
        "class",
        f"P({CLASS_NAMES_ORDERED[0]})",
        f"P({CLASS_NAMES_ORDERED[1]})",
        f"P({CLASS_NAMES_ORDERED[2]})",
        f"P({CLASS_NAMES_ORDERED[3]})",
    )
    scales = (1.0, 1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0)
    offsets = (0.0, 0.0, 0.0, 0.0, 0.0)
    units = ("class", "probability", "probability", "probability", "probability")

    with rio.open(out_path, "w", **profile) as dst:
        dst.write(data)
        for i, desc in enumerate(descriptions, start=1):
            dst.set_band_description(i, desc)
        dst.scales = scales
        dst.offsets = offsets
        dst.units = units

        dst.update_tags(
            OCM_VERSION=ocm_version,
            MODEL_RESOLUTION_M="10",
            OUTPUT_RESOLUTION_M=f"{abs(profile['transform'].a):.6f}",
            SOURCE_FILE=str(source_path),
            BAND_ORDER_USED=",".join(str(b) for b in band_order),
            DEVICE=used_cfg.device,
            DTYPE=used_cfg.dtype,
            CREATED_UTC=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            CLASS_ENCODING=("0=Clear,1=Thick Cloud,2=Thin Cloud,"
                            "3=Cloud Shadow,255=NoData"),
            NODATA_DESCRIPTION=("All 5 bands carry 255 at NoData pixels; "
                                "consumers should mask them out."),
            PROB_BAND_SCALE="1/255 (uint8 -> probability in [0,1])",
        )

        dst.update_tags(
            1,
            CLASS_VALUES="0,1,2,3,255",
            CLASS_NAMES="Clear,Thick Cloud,Thin Cloud,Cloud Shadow,NoData",
        )
        for i, cls in enumerate(CLASS_ORDER, start=2):
            dst.update_tags(
                i,
                CLASS_INDEX=str(cls),
                CLASS_NAME=OCM_CLASSES[cls],
                SCALE="1/255",
            )


# ---------------------------------------------------------------------------
# Per-scene pipeline


def mask_one(input_path: Path, out_path: Path, *, device: str = "auto",
             keep_tmp: bool = False) -> None:
    """Run the full pipeline on a single scene; write a 5-band uint8 GeoTIFF."""
    input_path = Path(input_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_parent = out_path.parent / "_tmp"
    tmp_parent.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix=f"ocm_{input_path.stem}_", dir=tmp_parent))

    try:
        band_order = detect_band_order(input_path)
        working = ensure_utm(input_path, tmp_root)
        sat0 = _saturate_zero_to_temp(working, tmp_root)

        with rio.open(working) as src:
            pixel_size = abs(src.transform.a)
        resample_res: float | None = 10.0
        if pixel_size >= 10.0:
            log.warning("%s: native pixel size %.2fm >= 10m; pass-through (resample_res=None)",
                        input_path, pixel_size)
            resample_res = None

        rgn, src_profile = ocm.load_multiband(
            input_path=sat0, resample_res=resample_res, band_order=band_order
        )

        cfg = _resolve_device(device)
        probs, used_cfg = _predict_with_retry(rgn, cfg)

        # OmniCloudMask returns confidence as either (4, H, W) or (1, 4, H, W);
        # normalize.
        if probs.ndim == 4 and probs.shape[0] == 1:
            probs = probs[0]
        if probs.ndim != 3 or probs.shape[0] != 4:
            raise RuntimeError(
                f"unexpected confidence shape {probs.shape}; expected (4, H, W)")

        # Build 10 m profile from the source profile (load_multiband's profile
        # keeps the source dimensions, not the resampled ones).
        src_transform = src_profile["transform"]
        if resample_res is None:
            mask_transform = src_transform
        else:
            mask_transform = Affine(
                resample_res, src_transform.b, src_transform.c,
                src_transform.d, -resample_res, src_transform.f,
            )
        mask_profile = src_profile.copy()
        mask_profile.update(
            count=5, dtype="uint8", nodata=255, compress="lzw",
            width=int(probs.shape[2]), height=int(probs.shape[1]),
            transform=mask_transform,
        )
        for k in ("photometric", "interleave", "blockxsize", "blockysize", "tiled"):
            mask_profile.pop(k, None)

        valid_overlay = _build_validity_overlay(sat0, mask_profile)
        stack10 = _assemble_stack(probs, valid_overlay)

        # Persist the 10 m stack to a temp file so we can reproject all 5 bands
        # in one pass via reproject(rio.band(...)) onto the native grid.
        stack10_path = tmp_root / f"{input_path.stem}_stack10.tif"
        with rio.open(stack10_path, "w", **mask_profile) as dst:
            dst.write(stack10)

        upsampled, out_profile = _upsample_stack_to_reference(
            stack10_path, input_path, out_path,
        )

        ocm_version = getattr(ocm, "__version__", "unknown")
        _write_output(
            out_path, upsampled, out_profile,
            source_path=input_path, band_order=band_order,
            used_cfg=used_cfg, ocm_version=ocm_version,
        )
    finally:
        if not keep_tmp:
            shutil.rmtree(tmp_root, ignore_errors=True)
            try:
                tmp_parent.rmdir()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Output path mapping


def _output_path_for(input_path: Path, input_dir: Path, output_dir: Path) -> Path:
    """Map <input_dir>/<year>/<prefix>_4band.tif -> <output_dir>/<year>/<prefix>_ocm.tif."""
    rel = input_path.relative_to(input_dir)
    stem = input_path.stem
    if stem.endswith("_4band"):
        stem = stem[: -len("_4band")]
    return output_dir / rel.parent / f"{stem}_ocm.tif"


# ---------------------------------------------------------------------------
# CLI


@click.command(
    help="Generate OmniCloudMask cloud/shadow masks for clipped Planet 4-band scenes.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option("--input-dir", type=click.Path(path_type=Path, file_okay=False),
              default=DEFAULT_INPUT_DIR, show_default=True,
              help="Directory containing <year>/*_4band.tif scenes.")
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False),
              default=DEFAULT_OUTPUT_DIR, show_default=True,
              help="Output directory; year subdirs are created to mirror input.")
@click.option("--glob", "pattern", default=DEFAULT_GLOB, show_default=True,
              help="Filename pattern to match under --input-dir (recursive).")
@click.option("--device", type=click.Choice(DEVICE_CHOICES), default="auto",
              show_default=True)
@click.option("--force", is_flag=True, default=False,
              help="Regenerate masks that already exist (default: skip).")
@click.option("-v", "--verbose", is_flag=True, default=False)
def main(input_dir: Path, output_dir: Path, pattern: str, device: str,
         force: bool, verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    warnings.filterwarnings("ignore", category=UserWarning, module="rasterio")

    if not input_dir.exists():
        click.echo(f"input-dir does not exist: {input_dir}", err=True)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenes = sorted(input_dir.rglob(pattern))
    if not scenes:
        click.echo(f"no files matched {pattern!r} under {input_dir}", err=True)
        sys.exit(1)

    click.echo(f"found {len(scenes)} scenes under {input_dir}")

    n_done = n_skipped = n_errored = 0
    pbar = tqdm(scenes, desc="masking", unit="scene")
    for scene in pbar:
        out_path = _output_path_for(scene, input_dir, output_dir)
        if out_path.exists() and not force:
            n_skipped += 1
            pbar.set_postfix(done=n_done, skipped=n_skipped, errored=n_errored)
            continue
        try:
            mask_one(scene, out_path, device=device)
            n_done += 1
        except Exception as exc:
            n_errored += 1
            log.exception("failed: %s -> %s: %s", scene, out_path, exc)
        pbar.set_postfix(done=n_done, skipped=n_skipped, errored=n_errored)

    click.echo(f"done: {n_done} written, {n_skipped} skipped, {n_errored} errored")
    if n_errored:
        sys.exit(2)


if __name__ == "__main__":
    main()
