#!/usr/bin/env python
"""Extract each crown's cloud-free Planet pixels on every scene, as input to
the crown temporal model (train_crown_pse_ltae.py).

`run` reads every clipped 4-band scene under PLANETDIR (native 3 m TOA) with
its OmniCloudMask raster under OCMDIR, and writes ``scenes/<scene>.npz`` to
OUTDIR with, for every crown of --crowns:

* ``n_px``: pixels of the crown polygon, shrunk inwards by --buffer meters
  against misregistration and translated by plus the scene's coregistration
  shift onto the Planet grid (as crown_rows in apply_planet_maskrcnn.py);
* ``n_valid``: of those, pixels with data;
* ``n_clear``: of those, pixels that are clear: OCM class 0, P(clear) at least
  --min-clear-prob, and more than --dilate pixels from any OCM cloud, thin
  cloud or cloud-shadow pixel (OCM runs at 10 m, so its edges are coarse);
* the B/G/R/N values of the clear pixels only.

A crown pixel that is not clear is never stored, so nothing downstream can
train or evaluate on it. Crowns are rasterized largest first, so where
buffered polygons still overlap the smaller crown wins.

Shifts come from the --coreg-json files (coreg_planet_to_nearest_drone.py
output; where several hold a scene, the first wins). A scene with no
successful shift uses the unshifted polygons and is flagged coreg_ok =
false, as in the apply step.

`concat` joins the per-scene files into one OUTDIR/crown_pixel_series.npz:
``scenes`` (scene, date, sat_id, coreg_ok, shift, scene clear fraction),
``crowns`` (crown index, tag, mnemonic, area), per crown-scene observation
arrays (scene index, crown index, n_px, n_valid, n_clear, offset into
``pixels``) and the flat uint16 ``pixels`` array (N x 4, B/G/R/N). The
tables are stored as JSON strings so the file loads without pickle.
See docs/crown_temporal_model.md.

Typical usage:
    python scripts/build_crown_pixel_series.py run \\
        /Volumes/Earth03/flower/planet_clipped/4band \\
        /Volumes/Earth03/flower/planet_clipped/ocm \\
        /Volumes/Earth03/flower/crown_series_20261029 \\
        --crowns .../BCI_50ha_2022_09_29_crownmap_improved.shp \\
        --coreg-json .../coreg_all_scenes.json --workers 8
    python scripts/build_crown_pixel_series.py concat \\
        /Volumes/Earth03/flower/crown_series_20261029
"""
import json
import multiprocessing as mp
import os
from glob import glob

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from scipy.ndimage import binary_dilation
from tqdm import tqdm

from apply_drone_labels_coreg import find_ocm_mask

OCM_CLEAR = 0
OCM_OBSCURED = (1, 2, 3)  # thick cloud, thin cloud, cloud shadow
OCM_NODATA = 255

_crowns = None  # buffered crown polygons, set per worker by _init


def scene_name(path):
    return os.path.basename(path)[:-len('.tif')]


def sat_id(scene):
    """Planet instrument id, the last token before _4band (e.g. 2448)."""
    return scene.split('_')[-2]


def load_coreg(paths):
    """{scene: (coreg_ok, x_shift_m, y_shift_m)}; the first file wins."""
    shifts = {}
    for path in paths:
        with open(path) as fh:
            for rec in json.load(fh):
                if rec['scene'] not in shifts or not shifts[rec['scene']][0]:
                    ok = bool(rec.get('coreg_ok'))
                    shifts[rec['scene']] = (ok, rec['x_shift_m'] if ok else 0.0,
                                            rec['y_shift_m'] if ok else 0.0)
    return shifts


def read_ocm(ocm_path, src):
    """OCM class and P(clear) (0-1) on the scene grid."""
    with rasterio.open(ocm_path) as ocm:
        if (ocm.shape == src.shape and ocm.transform == src.transform
                and ocm.crs == src.crs):
            return ocm.read(1), ocm.read(2) / 255.0
        out = np.full((2,) + src.shape, OCM_NODATA, dtype=np.uint8)
        reproject(ocm.read([1, 2]), out, src_transform=ocm.transform,
                  src_crs=ocm.crs, dst_transform=src.transform,
                  dst_crs=src.crs, resampling=Resampling.nearest,
                  src_nodata=OCM_NODATA, dst_nodata=OCM_NODATA)
        return out[0], out[1] / 255.0


def clear_mask(cls, p_clear, valid, min_clear_prob, dilate):
    obscured = np.isin(cls, OCM_OBSCURED)
    if dilate > 0 and obscured.any():
        obscured = binary_dilation(obscured, iterations=dilate)
    return valid & (cls == OCM_CLEAR) & (p_clear >= min_clear_prob) & ~obscured


def crown_index(geoms, shift, transform, shape):
    """Raster of crown index + 1 (0 = none), largest crown drawn first."""
    moved = geoms.translate(xoff=shift[0], yoff=shift[1])
    order = np.argsort(-geoms.area.values)
    return rasterize(((moved.iloc[i], i + 1) for i in order
                      if not moved.iloc[i].is_empty),
                     out_shape=shape, transform=transform, fill=0,
                     dtype='int32')


def extract(planetfile, ocm_path, shift, min_clear_prob, dilate):
    """Per-crown counts and clear-pixel values for one scene."""
    with rasterio.open(planetfile) as src:
        arr = src.read()
        nodata = src.nodata if src.nodata is not None else 0
        valid = (arr != nodata).all(axis=0)
        cls, p_clear = read_ocm(ocm_path, src)
        if _crowns.crs != src.crs:
            raise click.ClickException(f'{planetfile}: CRS {src.crs} != crowns')
        index = crown_index(_crowns.geometry, shift, src.transform, src.shape)
    clear = clear_mask(cls, p_clear, valid, min_clear_prob, dilate)

    n = len(_crowns) + 1
    flat = index.ravel()
    n_px = np.bincount(flat, minlength=n)[1:]
    n_valid = np.bincount(flat, weights=valid.ravel(), minlength=n)[1:]
    keep = clear.ravel() & (flat > 0)
    crown_of = flat[keep] - 1
    order = np.argsort(crown_of, kind='stable')
    pixels = arr.reshape(arr.shape[0], -1)[:, keep].T[order]
    n_clear = np.bincount(crown_of, minlength=n - 1)
    scene_valid = valid.sum()
    return {
        'n_px': n_px.astype(np.int32),
        'n_valid': n_valid.astype(np.int32),
        'n_clear': n_clear.astype(np.int32),
        'pixels': pixels.astype(np.uint16),
        'scene_clear_fraction': np.float32(clear.sum() / scene_valid
                                           if scene_valid else 0.0),
    }


def _init(crowns_path, crs, buffer):
    global _crowns
    crowns = gpd.read_file(crowns_path).to_crs(crs)
    crowns['geometry'] = crowns.geometry.buffer(-buffer)
    _crowns = crowns


def _work(job):
    planetfile, ocm_path, out_path, shift, min_clear_prob, dilate = job
    res = extract(planetfile, ocm_path, shift, min_clear_prob, dilate)
    np.savez(out_path + '.tmp.npz', **res)
    os.replace(out_path + '.tmp.npz', out_path)
    return out_path


@click.group()
def cli():
    pass


@cli.command()
@click.argument('planetdir', type=click.Path(exists=True, file_okay=False))
@click.argument('ocmdir', type=click.Path(exists=True, file_okay=False))
@click.argument('outdir', type=click.Path(file_okay=False))
@click.option('--crowns', 'crowns_path', required=True,
              type=click.Path(exists=True, dir_okay=False),
              help='Crown polygons with tag and mnemonic fields (the apply '
                   'step uses BCI_50ha_2022_09_29_crownmap_improved.shp).')
@click.option('--coreg-json', 'coreg_jsons', multiple=True,
              type=click.Path(exists=True, dir_okay=False),
              help='coreg_planet_to_nearest_drone.py output; repeatable, the '
                   'first file holding a successful shift for a scene wins.')
@click.option('--buffer', default=1.0, show_default=True,
              help='Inward buffer (m) applied to every crown polygon.')
@click.option('--dilate', default=1, show_default=True,
              help='Pixels by which OCM cloud and shadow are grown before '
                   'they are excluded.')
@click.option('--min-clear-prob', default=0.0, show_default=True,
              help='Minimum OCM P(clear) for a clear pixel, on top of the '
                   'argmax class being clear.')
@click.option('--start', default=None, help='Earliest scene date, YYYY-MM-DD.')
@click.option('--end', default=None, help='Latest scene date, YYYY-MM-DD.')
@click.option('--workers', default=1, show_default=True)
@click.option('--overwrite/--no-overwrite', default=False,
              help='Redo scenes whose outputs already exist.')
def run(planetdir, ocmdir, outdir, crowns_path, coreg_jsons, buffer, dilate,
        min_clear_prob, start, end, workers, overwrite):
    """Extract crown pixel sets from every scene in PLANETDIR."""
    os.makedirs(os.path.join(outdir, 'scenes'), exist_ok=True)
    files = sorted(glob(os.path.join(planetdir, '**', '*_4band.tif'),
                        recursive=True))
    dates = pd.to_datetime([os.path.basename(f)[:8] for f in files],
                           format='%Y%m%d')
    sel = np.ones(len(files), dtype=bool)
    if start:
        sel &= dates >= pd.Timestamp(start)
    if end:
        sel &= dates <= pd.Timestamp(end)
    files = [f for f, s in zip(files, sel) if s]
    shifts = load_coreg(coreg_jsons)
    with rasterio.open(files[0]) as src:
        crs = src.crs

    jobs, records, no_ocm = [], [], []
    for f in files:
        scene = scene_name(f)
        ocm_path = find_ocm_mask(f, planetdir, ocmdir)
        if ocm_path is None:
            no_ocm.append(scene)
            continue
        ok, dx, dy = shifts.get(scene, (False, 0.0, 0.0))
        records.append({'scene': scene,
                        'date': pd.Timestamp(scene[:8]).strftime('%Y-%m-%d'),
                        'sat_id': sat_id(scene),
                        'planet_rel': os.path.relpath(f, planetdir),
                        'coreg_ok': ok, 'x_shift_m': dx, 'y_shift_m': dy})
        out_path = os.path.join(outdir, 'scenes', f'{scene}.npz')
        if overwrite or not os.path.exists(out_path):
            jobs.append((f, str(ocm_path), out_path, (dx, dy),
                         min_clear_prob, dilate))
    click.echo(f'{len(records)} scenes ({sum(r["coreg_ok"] for r in records)} '
               f'coregistered), {len(no_ocm)} without OCM, {len(jobs)} to do')

    with open(os.path.join(outdir, 'build.json'), 'w') as fh:
        json.dump({'planetdir': planetdir, 'ocmdir': ocmdir,
                   'crowns': crowns_path, 'coreg_json': list(coreg_jsons),
                   'buffer_m': buffer, 'dilate_px': dilate,
                   'min_clear_prob': min_clear_prob,
                   'scenes': records, 'no_ocm': no_ocm}, fh, indent=1)

    init = (crowns_path, crs, buffer)
    if workers > 1:
        with mp.get_context('spawn').Pool(workers, _init, init) as pool:
            for _ in tqdm(pool.imap_unordered(_work, jobs), total=len(jobs)):
                pass
    else:
        _init(*init)
        for job in tqdm(jobs):
            _work(job)


@cli.command()
@click.argument('outdir', type=click.Path(exists=True, file_okay=False))
def concat(outdir):
    """Join the per-scene files into OUTDIR/crown_pixel_series.npz."""
    with open(os.path.join(outdir, 'build.json')) as fh:
        build = json.load(fh)
    crowns = gpd.read_file(build['crowns'])
    crown_tab = pd.DataFrame({'crown': np.arange(len(crowns)),
                              # integer, as in the nc4 tables and the apply
                              # CSVs; -9999 marks an untagged crown
                              'tag': pd.to_numeric(crowns['tag'], errors='coerce')
                                       .fillna(-9999).astype(int).values,
                              'mnemonic': crowns['mnemonic'].values,
                              'area_m2': crowns.geometry.area.values})

    scenes, obs, pixels, offset = [], [], [], 0
    for rec in tqdm(build['scenes']):
        path = os.path.join(outdir, 'scenes', f'{rec["scene"]}.npz')
        if not os.path.exists(path):
            continue
        d = np.load(path)
        s = len(scenes)
        scenes.append({**rec, 'scene_clear_fraction':
                       float(d['scene_clear_fraction'])})
        has = np.flatnonzero(d['n_px'] > 0)
        starts = np.concatenate([[0], np.cumsum(d['n_clear'])[:-1]])
        obs.append(np.stack([np.full(len(has), s), has, d['n_px'][has],
                             d['n_valid'][has], d['n_clear'][has],
                             offset + starts[has]], axis=1))
        pixels.append(d['pixels'])
        offset += len(d['pixels'])
    if not scenes:
        raise click.ClickException(f'no per-scene files in {outdir}/scenes')
    obs = np.concatenate(obs).astype(np.int64)
    out = os.path.join(outdir, 'crown_pixel_series.npz')
    np.savez(out,
             scenes=json.dumps(scenes), crowns=crown_tab.to_json(orient='records'),
             build=json.dumps({k: v for k, v in build.items()
                               if k not in ('scenes', 'no_ocm')}),
             obs_scene=obs[:, 0].astype(np.int32),
             obs_crown=obs[:, 1].astype(np.int32),
             obs_n_px=obs[:, 2].astype(np.int32),
             obs_n_valid=obs[:, 3].astype(np.int32),
             obs_n_clear=obs[:, 4].astype(np.int32),
             obs_offset=obs[:, 5],
             pixels=np.concatenate(pixels))
    click.echo(f'{len(scenes)} scenes, {len(obs)} crown-scenes, '
               f'{offset} clear pixels -> {out}')


if __name__ == '__main__':
    cli()
