#!/usr/bin/env python
"""Rebuild a coreg_log.json that was lost or truncated by a failed write.

The 20260915 globus build processed all 370 (label, scene) pairs and wrote
every chip, then died in the final `json.dump` on a numpy float32 shift
(fixed since: see `compute_coreg_shift`). That left a truncated log holding
19 complete records out of 370.

Everything in a record except the AROSICS shift can be recovered from disk:

  - `coreg_ok`      -> whether the chip files for the pair exist
  - `clear_fraction`-> recomputed from the OCM mask
  - the noise/quality stats -> recomputed from `load_planet(planetfile,
    resize)`, which is the same array the `.tif` was written from, using this
  - pipeline's own `compute_noise_metrics`, so the values are exact and not
    approximations
  - the remaining fields are derivable from the paths

`x_shift_m` / `y_shift_m` are *not* recoverable and are written as null on
reconstructed records. Any record this script rebuilds carries
`reconstructed: true` so a null shift is never mistaken for a measured zero.
Records salvaged from the truncated log keep their real shifts and are copied
through verbatim, without the flag.

Usage:
    python scripts/reconstruct_coreg_log.py <labelglob...> <dronedir> \
        <planetdir> <outputdir> [-t 2] [-r 4] [-b 4] [-k <maskdir>]
"""
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path

import click
import numpy as np
import pandas as pd
import rasterio

import apply_drone_labels_coreg as adlc

log = logging.getLogger(__name__)

CHIP_SUFFIXES = ('.tif', '.png', '.mask.png', '.drone.png', '.ocm.png')


def salvage_truncated(path):
    """Return the complete records from a truncated JSON array, or []."""
    if not Path(path).exists():
        return []
    text = Path(path).read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Trim back to the last complete record and close the array.
    cut = text.rfind('\n  },')
    if cut == -1:
        return []
    try:
        return json.loads(text[:cut + 4] + '\n]')
    except json.JSONDecodeError:
        return []


def chip_stem(planetfile, bands):
    basename = Path(planetfile).stem
    for suffix in ('_4band', '_rgb'):
        if basename.endswith(suffix):
            basename = basename[:-len(suffix)]
            break
    return f"{basename}_{'4band' if bands == 4 else 'rgb'}"


def chip_exists(outputdir, stem):
    return all((Path(outputdir) / f'{stem}{s}').exists() for s in CHIP_SUFFIXES)


@click.command()
@click.argument('labelfiles', nargs=-1)
@click.argument('dronedir')
@click.argument('planetdir')
@click.argument('outputdir')
@click.option('-t', '--timewindow', default=2, type=int)
@click.option('-r', '--resize', default=None, type=float)
@click.option('-m', '--mode', default='both')
@click.option('-k', '--maskdir', default=None, type=click.Path())
@click.option('-b', '--bands', default=3, type=click.Choice(['3', '4']),
              callback=lambda c, p, v: int(v))
def main(labelfiles, dronedir, planetdir, outputdir, timewindow, resize, mode, maskdir, bands):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    outputdir = Path(outputdir)
    out_path = outputdir / 'coreg_log.json'

    salvaged = {(r['scene'], r['label']): r for r in salvage_truncated(out_path)}
    log.info('Salvaged %d complete records (real shifts preserved)', len(salvaged))
    if salvaged:
        backup = outputdir / 'coreg_log.truncated.json'
        if not backup.exists():
            backup.write_text(out_path.read_text())
            log.info('Backed up the truncated log to %s', backup.name)

    # Enumerate pairs exactly the way apply_drone_labels_coreg.main() does.
    planet_glob = '*4band.tif' if bands == 4 else '*rgb.tif'
    planetfiles = adlc.filter_files(list(Path(planetdir).rglob(planet_glob)), None)
    planet_df = pd.DataFrame({'path': [str(p) for p in planetfiles]})
    planet_df['date'] = pd.to_datetime(
        planet_df['path'].apply(adlc.get_planet_date), format='%Y%m%d',
    )

    pairs = []
    for labelfile in labelfiles:
        label_date = pd.to_datetime(adlc.get_cls_date(labelfile), format='%Y_%m_%d')
        in_window = (planet_df['date'] - label_date).abs() <= pd.Timedelta(days=timewindow)
        for planetfile in planet_df.loc[in_window, 'path']:
            pairs.append((labelfile, planetfile))

    # A chip stem is keyed on the Planet scene alone, so a scene matched to two
    # label dates cannot be attributed to one of them from disk.
    stem_counts = Counter(chip_stem(pf, bands) for _, pf in pairs)
    ambiguous = {s for s, c in stem_counts.items() if c > 1}
    if ambiguous:
        log.warning(
            '%d scenes pair with more than one label date; coreg_ok for those '
            '%d pairs is attributed to the last-processed date and flagged',
            len(ambiguous), sum(stem_counts[s] for s in ambiguous),
        )

    records, n_recomputed = [], 0
    seen_ambiguous = set()
    for labelfile, planetfile in pairs:
        key = (Path(planetfile).stem, Path(labelfile).name)
        if key in salvaged:
            records.append(salvaged[key])
            continue

        stem = chip_stem(planetfile, bands)
        dronefile = adlc.find_drone(labelfile, dronedir)

        ocm_path = None
        clear_fraction = None
        if maskdir is not None:
            ocm_path = adlc.find_ocm_mask(planetfile, planetdir, maskdir)
            if ocm_path is not None:
                clear_fraction = adlc.compute_clear_fraction(ocm_path)

        with rasterio.open(planetfile) as src:
            planet_res_m = src.res[0]
        with rasterio.open(dronefile) as src:
            drone_res_m = src.res[0]

        exists = chip_exists(outputdir, stem)
        coreg_ok = exists
        if stem in ambiguous:
            # Files on disk came from whichever date ran last and won the write.
            coreg_ok = exists and stem not in seen_ambiguous
            seen_ambiguous.add(stem)

        record = {
            'scene': Path(planetfile).stem,
            'label': Path(labelfile).name,
            'coreg_ok': coreg_ok,
            'clear_fraction': clear_fraction,
            'drone_file': str(Path(dronefile).resolve()),
            'x_shift_m': None,
            'y_shift_m': None,
            'planet_res_m': planet_res_m,
            'drone_res_m': drone_res_m,
            'reconstructed': True,
        }
        if stem in ambiguous:
            record['ambiguous_coreg_ok'] = True

        # Recompute the stats the same way create_mask() does, from the same
        # array the chip was written from.
        if coreg_ok and bands == 4:
            pimg = adlc.load_planet(planetfile, resize)
            ocm_rep = None
            if maskdir is not None and ocm_path is not None:
                ocm_rep = adlc.reproject_ocm_to_grid(ocm_path, pimg)
            if ocm_rep is not None:
                clear_mask = (ocm_rep.values == 0)
            else:
                clear_mask = np.ones(pimg.values.shape[1:], dtype=bool)
            record.update(adlc.compute_noise_metrics(pimg, clear_mask))
            n_recomputed += 1

        records.append(record)

    adlc.write_coreg_log(records, out_path)
    n_ok = sum(1 for r in records if r['coreg_ok'])
    log.info(
        'Wrote %d records (%d coreg_ok, %d failed) to %s; %d salvaged with real '
        'shifts, %d reconstructed, %d with recomputed stats',
        len(records), n_ok, len(records) - n_ok, out_path,
        len(salvaged), len(records) - len(salvaged), n_recomputed,
    )


if __name__ == '__main__':
    main()
