#!/usr/bin/env python3
"""
Copy files for Planet vetting chips that meet a minimum "Quality" rating in
a Labelbox export (e.g. `labels/20260706_planet_vetting.ndjson`) from the
4-band stretch-stats source directory into the curated destination directory.

The Quality radio has three levels, ranked Poor < Fair < Good. By default only
"Good" chips are copied (the original 50ha behaviour); pass `--min-quality Fair`
to also include Fair chips, etc.

For each accepted chip (external_id ends in `.png`), all sibling files that
share the same stem are copied (e.g. `.png`, `.tif`, `.mask.png`,
`.drone.png`, `.ocm.png`). The source root's `coreg_log.json` is also
copied, or *merged* into the destination's when one is already there — so a
set assembled from more than one source build (e.g. the 2020-2023 local
mosaics plus the 2024-2026 globus mosaics) keeps the provenance of both.
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

# Quality radio levels, worst to best.
QUALITY_RANK = {"Poor": 0, "Fair": 1, "Good": 2}


def collect_stems(ndjson_path: Path, min_quality: str) -> list[str]:
    min_rank = QUALITY_RANK[min_quality]
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
            if quality in QUALITY_RANK and QUALITY_RANK[quality] >= min_rank:
                stems.append(Path(external_id).stem)
    return stems


def merge_coreg_log(src_log: Path, dst_log: Path) -> tuple[list[dict], int, int, int]:
    """Union of the source and destination coreg logs, keyed on (scene, label).

    Destination records win on a collision, so re-running against the same
    source is idempotent and an existing curated log is never rewritten by a
    later build. Returns (merged, n_src, n_dst, n_new).
    """
    with src_log.open() as f:
        src_records = json.load(f)

    dst_records = []
    if dst_log.exists():
        with dst_log.open() as f:
            dst_records = json.load(f)

    def key(rec):
        return (rec.get("scene"), rec.get("label"))

    seen = {key(r) for r in dst_records}
    new_records = [r for r in src_records if key(r) not in seen]

    return dst_records + new_records, len(src_records), len(dst_records), len(new_records)


def copy_good(
    ndjson_path: Path,
    src_dir: Path,
    dst_dir: Path,
    min_quality: str = "Good",
    dry_run: bool = False,
) -> None:
    print(f"NDJSON        : {ndjson_path}")
    print(f"Source dir    : {src_dir}")
    print(f"Destination   : {dst_dir}")
    print(f"Min quality   : {min_quality}")
    print(f"Dry run       : {dry_run}\n")

    stems = collect_stems(ndjson_path, min_quality)
    print(f"Accepted chips: {len(stems)} (>= {min_quality})")

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
        merged, n_src, n_dst, n_new = merge_coreg_log(coreg_src, dst_dir / "coreg_log.json")
        if dry_run:
            print(f"  [dry-run] coreg_log.json ({len(merged)} records)")
        else:
            with (dst_dir / "coreg_log.json").open("w") as f:
                json.dump(merged, f, indent=2)
        if n_dst:
            print(f"\ncoreg_log.json: {n_dst} existing + {n_new} new "
                  f"(of {n_src} in source) = {len(merged)} records")
        else:
            print(f"\nAlso copied   : coreg_log.json ({len(merged)} records)")
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
    parser.add_argument(
        "--min-quality",
        choices=list(QUALITY_RANK),
        default="Good",
        help='Minimum Quality rating to copy (Poor < Fair < Good). Default: Good.',
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    copy_good(
        args.ndjson, args.src, args.dst,
        min_quality=args.min_quality, dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
