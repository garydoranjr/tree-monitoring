#!/usr/bin/env python3
"""
Copy files for Planet vetting chips rated "Good" in the Labelbox export
`labels/20260706_planet_vetting.ndjson` from the 4-band stretch-stats
source directory into the curated destination directory.

For each Good chip (external_id ends in `.png`), all sibling files that
share the same stem are copied (e.g. `.png`, `.tif`, `.mask.png`,
`.drone.png`, `.ocm.png`). The source root's `coreg_log.json` is also
copied.
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


DEFAULT_NDJSON = Path("labels/20260706_planet_vetting.ndjson")
DEFAULT_SRC = Path(
    "/Volumes/Earth03/flower/20260608_full_label_application_x4_coreg_4band_stretch_stats"
)
DEFAULT_DST = Path(
    "/Volumes/Earth03/flower/20260706_full_label_application_x4_coreg_4band_stretch_stats_curated"
)


def collect_good_stems(ndjson_path: Path) -> list[str]:
    stems: list[str] = []
    with ndjson_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            external_id = rec["data_row"]["external_id"]
            quality = None
            for proj in rec.get("projects", {}).values():
                for label in proj.get("labels", []):
                    for cls in label.get("annotations", {}).get("classifications", []):
                        if cls.get("name") == "Quality":
                            quality = cls.get("radio_answer", {}).get("name")
            if quality == "Good":
                stems.append(Path(external_id).stem)
    return stems


def copy_good(
    ndjson_path: Path,
    src_dir: Path,
    dst_dir: Path,
    dry_run: bool = False,
) -> None:
    print(f"NDJSON        : {ndjson_path}")
    print(f"Source dir    : {src_dir}")
    print(f"Destination   : {dst_dir}")
    print(f"Dry run       : {dry_run}\n")

    stems = collect_good_stems(ndjson_path)
    print(f"Good chips    : {len(stems)}")

    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)

    src_files_by_stem: dict[str, list[Path]] = {}
    for stem in stems:
        matches = sorted(src_dir.glob(f"{stem}.*"))
        src_files_by_stem[stem] = matches

    missing = [s for s, m in src_files_by_stem.items() if not m]
    if missing:
        print(f"WARNING: {len(missing)} Good stem(s) had no matching files:")
        for s in missing:
            print(f"  {s}")

    ext_counts: Counter[str] = Counter()
    copied = 0
    for stem, matches in src_files_by_stem.items():
        for src_path in matches:
            suffix = src_path.name[len(stem):]
            ext_counts[suffix] += 1
            dst_path = dst_dir / src_path.name
            if dry_run:
                print(f"  [dry-run] {src_path.name}")
            else:
                shutil.copy2(src_path, dst_path)
            copied += 1

    coreg_src = src_dir / "coreg_log.json"
    if coreg_src.exists():
        if dry_run:
            print(f"  [dry-run] {coreg_src.name}")
        else:
            shutil.copy2(coreg_src, dst_dir / coreg_src.name)
        print(f"\nAlso copied   : {coreg_src.name}")
    else:
        print(f"\nWARNING: {coreg_src} not found; skipped")

    print(f"\nTotal chip files copied: {copied}")
    print("By suffix:")
    for suffix, n in sorted(ext_counts.items()):
        print(f"  {suffix:<15} {n}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ndjson", type=Path, default=DEFAULT_NDJSON)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    copy_good(args.ndjson, args.src, args.dst, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
