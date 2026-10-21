#!/usr/bin/env python
"""Index PlanetScope SR box clips and stage them for cloud masking.

Walks one or more Planet order roots for `*_3B_AnalyticMS_SR_clip.tif` (the
`analytic_sr_udm2` bundle clipped to the BCNM box of config/planet_search.yml),
keeps one copy per scene ID (first root wins), and symlinks each scene as
`<out-dir>/4band/<YYYY>/<scene>_4band.tif`, the layout that
scripts/cloud_mask_planet.py walks. Writes `<out-dir>/index.csv` with the
per-scene metadata the MODIS calibration needs (instrument, satellite, strip,
sun/view geometry) from the order's `<scene>_metadata.json`.

Typical usage:
    python scripts/stage_planet_sr.py \\
        /Volumes/Earth03/flower/planet/planet_bci_orders_sr \\
        /Volumes/Earth03/flower/planet/BCNM_February_2024_psscene_analytic_sr_udm2
    python scripts/cloud_mask_planet.py \\
        --input-dir /Volumes/Earth03/flower/planet_bcnm_sr/4band \\
        --output-dir /Volumes/Earth03/flower/planet_bcnm_sr/ocm
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import click

SR_SUFFIX = '_3B_AnalyticMS_SR_clip.tif'
DEFAULT_OUT = Path('/Volumes/Earth03/flower/planet_bcnm_sr')
INDEX_FIELDS = [
    'scene_id', 'acquired', 'instrument', 'satellite_id', 'strip_id',
    'sun_elevation', 'sun_azimuth', 'view_angle', 'satellite_azimuth',
    'sr_path', 'udm2_path', 'staged_path',
]


def discover(roots: list[Path]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for root in roots:
        for p in sorted(root.rglob(f'*{SR_SUFFIX}')):
            sid = p.name[:-len(SR_SUFFIX)]
            found.setdefault(sid, p)
    return found


def scene_record(sid: str, sr_path: Path) -> dict:
    meta_path = sr_path.with_name(f'{sid}_metadata.json')
    props = json.loads(meta_path.read_text())['properties'] if meta_path.exists() else {}
    udm2 = sr_path.with_name(f'{sid}_3B_udm2_clip.tif')
    rec = {k: props.get(k, '') for k in INDEX_FIELDS[1:9]}
    rec.update(scene_id=sid, sr_path=str(sr_path),
               udm2_path=str(udm2) if udm2.exists() else '')
    return rec


@click.command()
@click.argument('roots', nargs=-1, required=True,
                type=click.Path(path_type=Path, exists=True, file_okay=False))
@click.option('--out-dir', type=click.Path(path_type=Path), default=DEFAULT_OUT,
              show_default=True)
def main(roots, out_dir):
    found = discover(list(roots))
    records = []
    for sid, sr_path in sorted(found.items()):
        rec = scene_record(sid, sr_path)
        link = out_dir / '4band' / sid[:4] / f'{sid}_4band.tif'
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() and link.resolve() != sr_path.resolve():
            link.unlink()
        if not link.exists():
            link.symlink_to(sr_path)
        rec['staged_path'] = str(link)
        records.append(rec)
    with open(out_dir / 'index.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows(records)
    click.echo(f'staged {len(records)} SR scenes under {out_dir}')


if __name__ == '__main__':
    main()
