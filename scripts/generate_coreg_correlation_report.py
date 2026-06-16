"""Generate an HTML report of coreg scenes sorted by band_corr_mean (worst first).

Reads `coreg_log.json` from a directory of per-scene PNG visualizations
emitted by the coregistration pipeline and writes a static HTML page that
lists each scene with its main RGB PNG and band_corr_mean value, sorted
ascending so the most decorrelated scenes appear first.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing coreg_log.json and per-scene PNGs.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path (default: <input_dir>/coreg_correlation_report.html).",
    )
    return p.parse_args()


def render_html(rows: list[tuple[str, float, str]], title: str) -> str:
    head = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; background: #fafafa; }}
  h1 {{ font-size: 18px; }}
  .entry {{ display: flex; align-items: flex-start; gap: 16px; padding: 12px 0; border-top: 1px solid #ddd; }}
  .meta {{ min-width: 280px; font-family: ui-monospace, Menlo, monospace; font-size: 13px; }}
  .meta .scene {{ font-weight: 600; word-break: break-all; }}
  .meta .corr {{ margin-top: 4px; color: #444; }}
  img {{ max-width: 600px; height: auto; border: 1px solid #ccc; background: #fff; }}
</style>
</head>
<body>
<h1>{html.escape(title)} &mdash; {len(rows)} scenes (sorted ascending by band_corr_mean)</h1>
"""
    body_parts = [head]
    for scene, corr, img_rel in rows:
        body_parts.append(
            f'<div class="entry">'
            f'<div class="meta">'
            f'<div class="scene">{html.escape(scene)}</div>'
            f'<div class="corr">band_corr_mean = {corr:.4f}</div>'
            f'</div>'
            f'<img src="{html.escape(img_rel)}" alt="{html.escape(scene)}">'
            f'</div>\n'
        )
    body_parts.append("</body></html>\n")
    return "".join(body_parts)


def main() -> int:
    args = parse_args()
    input_dir: Path = args.input_dir.resolve()
    if not input_dir.is_dir():
        print(f"error: not a directory: {input_dir}", file=sys.stderr)
        return 2

    log_path = input_dir / "coreg_log.json"
    if not log_path.is_file():
        print(f"error: missing coreg_log.json in {input_dir}", file=sys.stderr)
        return 2

    output = (args.output or (input_dir / "coreg_correlation_report.html")).resolve()
    output_dir = output.parent

    with log_path.open() as f:
        records = json.load(f)

    if not isinstance(records, list):
        print("error: coreg_log.json is not a list of records", file=sys.stderr)
        return 2

    n_total = len(records)
    rows: list[tuple[str, float, str]] = []
    n_no_corr = 0
    n_no_image = 0

    for rec in records:
        scene = rec.get("scene")
        corr = rec.get("band_corr_mean")
        if scene is None or corr is None:
            n_no_corr += 1
            continue
        # Scene IDs in coreg_log.json may carry a band-suffix (e.g. "_4band")
        # that the per-scene PNGs do not include — try the literal name first,
        # then progressively shorter underscore-stripped variants.
        img_path = None
        candidate = scene
        while True:
            p = input_dir / f"{candidate}.png"
            if p.is_file():
                img_path = p
                break
            if "_" not in candidate:
                break
            candidate = candidate.rsplit("_", 1)[0]
        if img_path is None:
            n_no_image += 1
            continue
        img_rel = os.path.relpath(img_path, start=output_dir)
        rows.append((scene, float(corr), img_rel))

    rows.sort(key=lambda r: r[1])

    html_text = render_html(rows, title=f"Coreg correlation report: {input_dir.name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_text)

    print(
        f"loaded {n_total} records; rendered {len(rows)}; "
        f"skipped {n_no_corr} (no band_corr_mean), {n_no_image} (no PNG). "
        f"-> {output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
