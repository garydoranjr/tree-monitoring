"""
Filter and copy image/mask pairs from a label application output directory based on
configurable thresholds for image size, band correlation, and clear pixel fraction.

Usage:
    python scripts/filter_label_application.py <src_dir> <dst_dir> <config.yml>

Copies {stem}.png, {stem}.mask.png, and any {stem}.drone.png / {stem}.ocm.png that
exist. Writes a filtered coreg_log.json to the output directory.
"""

import argparse
import glob
import json
import os
import shutil

import yaml
from PIL import Image


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_scene_lookup(log_entries):
    """Return {stem: entry} mapping, handling both _rgb and _4band suffix conventions."""
    lookup = {}
    for entry in log_entries:
        scene = entry["scene"]
        lookup[scene] = entry
        for suffix in ("_rgb", "_4band"):
            if scene.endswith(suffix):
                stripped = scene[: -len(suffix)]
                if stripped not in lookup:
                    lookup[stripped] = entry
                break
    return lookup


def check_filters(entry, img_size, cfg):
    """Return (passes, reason_for_skip) for the entry + image size."""
    if cfg.get("require_coreg_ok") and not entry.get("coreg_ok"):
        return False, "coreg_ok"

    exact_sizes = cfg.get("exact_sizes")
    if exact_sizes:
        allowed = {tuple(s) for s in exact_sizes}
        if img_size not in allowed:
            return False, "size"
    else:
        min_w = cfg.get("min_width")
        min_h = cfg.get("min_height")
        if min_w is not None and img_size[0] < min_w:
            return False, "size"
        if min_h is not None and img_size[1] < min_h:
            return False, "size"

    min_cf = cfg.get("min_clear_fraction")
    if min_cf is not None:
        cf = entry.get("clear_fraction")
        if cf is None:
            if not cfg.get("allow_null_clear_fraction", False):
                return False, "clear_fraction_null"
        elif cf < min_cf:
            return False, "clear_fraction"

    min_bcm = cfg.get("min_band_corr_mean")
    if min_bcm is not None:
        bcm = entry.get("band_corr_mean")
        if bcm is None:
            if not cfg.get("allow_null_band_corr_mean", False):
                return False, "band_corr_mean_null"
        elif bcm < min_bcm:
            return False, "band_corr_mean"

    return True, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src_dir", help="Source label application directory")
    parser.add_argument("dst_dir", help="Destination directory for filtered output")
    parser.add_argument("config", help="Filter config YAML file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    log_path = os.path.join(args.src_dir, "coreg_log.json")
    with open(log_path) as f:
        log = json.load(f)
    lookup = build_scene_lookup(log)

    all_pngs = sorted(glob.glob(os.path.join(args.src_dir, "*.png")))
    image_files = [
        p for p in all_pngs
        if not any(p.endswith(s) for s in (".mask.png", ".drone.png", ".ocm.png"))
    ]

    os.makedirs(args.dst_dir, exist_ok=True)

    counts = {
        "copied": 0,
        "no_entry": 0,
        "coreg_ok": 0,
        "size": 0,
        "clear_fraction": 0,
        "clear_fraction_null": 0,
        "band_corr_mean": 0,
        "band_corr_mean_null": 0,
    }
    passed_entries = []

    for img_path in image_files:
        stem = os.path.basename(img_path)[: -len(".png")]

        entry = lookup.get(stem)
        if entry is None:
            counts["no_entry"] += 1
            continue

        with Image.open(img_path) as img:
            size = img.size  # (width, height)

        ok, reason = check_filters(entry, size, cfg)
        if not ok:
            counts[reason] += 1
            continue

        for ext in (".png", ".mask.png", ".drone.png", ".ocm.png"):
            src = os.path.join(args.src_dir, stem + ext)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(args.dst_dir, stem + ext))

        passed_entries.append(entry)
        counts["copied"] += 1
        cf = entry.get("clear_fraction")
        bcm = entry.get("band_corr_mean")
        cf_str = f"{cf:.2f}" if cf is not None else "null"
        bcm_str = f"{bcm:.4f}" if bcm is not None else "null"
        print(f"  [{counts['copied']:3d}] {stem}  clear={cf_str}  band_corr_mean={bcm_str}  size={size}")

    out_log = os.path.join(args.dst_dir, "coreg_log.json")
    with open(out_log, "w") as f:
        json.dump(passed_entries, f, indent=2)

    total = len(image_files)
    print(f"\nDone: {counts['copied']} / {total} scenes copied")
    if counts["no_entry"]:
        print(f"  Skipped (not in coreg_log):         {counts['no_entry']}")
    if counts["coreg_ok"]:
        print(f"  Skipped (coreg_ok=false):            {counts['coreg_ok']}")
    if counts["size"]:
        print(f"  Skipped (image size):                {counts['size']}")
    if counts["clear_fraction"]:
        print(f"  Skipped (clear_fraction below min):  {counts['clear_fraction']}")
    if counts["clear_fraction_null"]:
        print(f"  Skipped (clear_fraction null):       {counts['clear_fraction_null']}")
    if counts["band_corr_mean"]:
        print(f"  Skipped (band_corr_mean below min):  {counts['band_corr_mean']}")
    if counts["band_corr_mean_null"]:
        print(f"  Skipped (band_corr_mean null):       {counts['band_corr_mean_null']}")
    print(f"  Filtered coreg_log.json → {out_log}")


if __name__ == "__main__":
    main()
