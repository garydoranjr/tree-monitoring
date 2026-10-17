#!/usr/bin/env python
"""Apply a Planet Mask R-CNN model to Planet scenes away from drone dates.

`run` takes the scenes coregistered by coreg_planet_to_nearest_drone.py,
rebuilds each one as a training-style chip (4-band, 0.75 m, on the exact
grid of a training chip given by --grid-like), runs the model over
overlapping 512 x 512 tiles -- the input size it was trained at, since the
detector rescales whole-chip inputs -- and writes per scene:

* ``pred/<scene>.pred.tif``: float32 GeoTIFF on the raw Planet chip grid,
  band 1 the id of the highest-scoring instance covering each pixel (0 =
  none), band 2 the highest instance score covering it, band 3 the clear
  flag (OCM clear and not nodata). Every detection the model returns
  (score >= its own score_thresh, 0.05 by default) is kept, so band 2 can
  be re-thresholded later.
* ``pred/<scene>.pred_coreg.tif``: the same raster translated by minus the
  AROSICS shift into drone / crown-map geometry (only when coregistered).
* ``qa/<scene>.png``: RGB render with outlines of instances scoring at
  least --det-thresh, and cloud / nodata tinted.
* ``crowns/<scene>.csv``: one row per crown in --crowns, scored on the
  crown polygons translated by plus the shift onto the Planet grid.
* ``scenes/<scene>.json``: the scene's summary row.

`concat` joins the per-scene files into crown_timeseries.csv and
scene_summary.csv. See docs/planet_temporal_generalization.md.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must precede torch import
import json
from glob import glob

import click
import geopandas as gpd
import imageio.v3 as iio
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
import torch
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import rasterize
from skimage.segmentation import find_boundaries
from tqdm import tqdm

# Importing compare_maskrcnn_runs also re-exports the custom model classes
# into __main__, which unpickling the checkpoints needs.
from compare_maskrcnn_runs import load_checkpoint
from apply_drone_labels_coreg import (
    find_ocm_mask,
    render_rgb_from_4band,
    reproject_ocm_to_grid,
)
from train_planet_image_maskrcnn import build_input_channels, select_device


def tile_starts(length, tile, stride):
    """Tile origins covering [0, length), the last one flush to the end, and
    the core interval of each tile: overlaps are split at their midpoint so
    every pixel belongs to exactly one tile's core."""
    if length < tile:
        raise ValueError(f'image dimension {length} is smaller than tile {tile}')
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    bounds = [0] + [(s + tile + n) // 2 for s, n in zip(starts, starts[1:])] \
        + [length]
    return [(s, bounds[i], bounds[i + 1]) for i, s in enumerate(starts)]


def load_chip(planetfile, grid):
    """The scene reprojected (bilinear) onto the training-chip grid."""
    img = rxr.open_rasterio(planetfile)
    return img.rio.reproject(
        grid['crs'], transform=grid['transform'],
        shape=(grid['height'], grid['width']),
        resampling=Resampling.bilinear,
    )


@torch.no_grad()
def predict_tiled(model, img, clear, device, tile, stride, mask_thresh=0.5):
    """Instances over the whole chip as (masks (N, H, W) bool, scores (N,)).
    A detection is kept from the tile whose core holds its box centre, and
    dropped if that centre is cloudy, as in evaluate()."""
    H, W = img.shape[:2]
    masks, scores = [], []
    for r0, rc0, rc1 in tile_starts(H, tile, stride):
        for c0, cc0, cc1 in tile_starts(W, tile, stride):
            crop = img[r0:r0 + tile, c0:c0 + tile]
            x = torch.from_numpy(crop.transpose(2, 0, 1).copy()).float()
            out = model([x.to(device)])[0]
            if out['boxes'].shape[0] == 0:
                continue
            boxes = out['boxes'].cpu().numpy()
            cx = (boxes[:, 0] + boxes[:, 2]) / 2 + c0
            cy = (boxes[:, 1] + boxes[:, 3]) / 2 + r0
            keep = (cy >= rc0) & (cy < rc1) & (cx >= cc0) & (cx < cc1)
            iy = np.clip(cy.astype(int), 0, H - 1)
            ix = np.clip(cx.astype(int), 0, W - 1)
            keep &= clear[iy, ix]
            if not keep.any():
                continue
            m = (out['masks'][keep, 0] >= mask_thresh).cpu().numpy()
            full = np.zeros((m.shape[0], H, W), dtype=bool)
            full[:, r0:r0 + tile, c0:c0 + tile] = m
            masks.append(full)
            scores.append(out['scores'][keep].cpu().numpy())
    if not masks:
        return np.zeros((0, H, W), dtype=bool), np.zeros(0, dtype=np.float32)
    return np.concatenate(masks), np.concatenate(scores).astype(np.float32)


def instance_rasters(masks, scores, shape):
    """(id, max score) rasters; ids follow descending score, 1-based, and the
    highest-scoring instance wins where instances overlap."""
    ids = np.zeros(shape, dtype=np.float32)
    best = np.zeros(shape, dtype=np.float32)
    order = np.argsort(scores)  # ascending: higher scores paint last
    rank = {int(i): n + 1 for n, i in enumerate(np.argsort(-scores))}
    for i in order:
        m = masks[i] & (scores[i] >= best)
        ids[m] = rank[int(i)]
        best[m] = scores[i]
    return ids, best


def write_raster(path, bands, transform, crs, descriptions):
    with rasterio.open(
        path, 'w', driver='GTiff', height=bands.shape[1], width=bands.shape[2],
        count=bands.shape[0], dtype='float32', crs=crs, transform=transform,
        compress='deflate',
    ) as dst:
        dst.write(bands.astype(np.float32))
        for i, d in enumerate(descriptions, start=1):
            dst.set_band_description(i, d)


def qa_png(path, pimg, clear, valid, ids, best, det_thresh):
    rgb = render_rgb_from_4band(pimg, clear).astype(np.float32)
    rgb[~clear & valid] = 0.5 * rgb[~clear & valid] + 0.5 * np.array([90, 140, 255])
    # A pixel's max score is its id's score, so this keeps whole instances.
    shown = np.where(best >= det_thresh, ids, 0)
    edges = find_boundaries(shown.astype(np.int32), mode='thick')
    rgb[edges] = [255, 40, 40]
    iio.imwrite(path, rgb.astype(np.uint8))


def crown_rows(crowns, shift, transform, shape, clear, best, ids, det_thresh):
    """Per-crown stats with the crown map shifted by +shift onto Planet."""
    geoms = crowns.geometry.translate(xoff=shift[0], yoff=shift[1])
    # Largest crowns first, so where polygons overlap the smaller one wins.
    order = np.argsort(-crowns.geometry.area.values)
    index = rasterize(
        ((geoms.iloc[i], i + 1) for i in order),
        out_shape=shape, transform=transform, fill=0, dtype='int32',
    )
    flat = index.ravel()
    n = len(crowns) + 1
    px = np.bincount(flat, minlength=n)
    clear_px = np.bincount(flat, weights=clear.ravel(), minlength=n)
    pos = (best >= det_thresh) & clear
    pos_px = np.bincount(flat, weights=pos.ravel(), minlength=n)
    max_score = np.zeros(n, dtype=np.float32)
    np.maximum.at(max_score, flat, np.where(clear, best, 0).ravel())
    pos_ids = pd.DataFrame({'crown': flat[pos.ravel()],
                            'inst': ids.ravel()[pos.ravel()]})
    n_inst = pos_ids.groupby('crown')['inst'].nunique()

    k = np.arange(1, n)
    with np.errstate(invalid='ignore', divide='ignore'):
        return pd.DataFrame({
            'tag': crowns['tag'].values,
            'mnemonic': crowns['mnemonic'].values,
            'crown_px': px[k].astype(int),
            'crown_clear_frac': clear_px[k] / px[k],
            'pos_frac': pos_px[k] / clear_px[k],
            'max_score': np.where(clear_px[k] > 0, max_score[k], np.nan),
            'n_instances': n_inst.reindex(k, fill_value=0).values,
        })


@click.group()
def cli():
    pass


@cli.command()
@click.argument('model_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('planetdir', type=click.Path(exists=True, file_okay=False))
@click.argument('ocmdir', type=click.Path(exists=True, file_okay=False))
@click.argument('coreg_json', type=click.Path(exists=True, dir_okay=False))
@click.argument('outdir', type=click.Path(file_okay=False))
@click.option('--grid-like', required=True,
              type=click.Path(exists=True, dir_okay=False),
              help='A training chip .tif whose grid every scene is '
                   'reprojected onto (0.75 m, 1468 x 820 for the 50ha set).')
@click.option('--crowns', 'crowns_path', required=True,
              type=click.Path(exists=True, dir_okay=False),
              help='Crown polygons (e.g. BCI_50ha_2022_09_29_crownmap_'
                   'improved.shp) with tag and mnemonic fields.')
@click.option('--tile', default=512, show_default=True)
@click.option('--stride', default=384, show_default=True)
@click.option('--det-thresh', default=0.5, show_default=True,
              help='Instance score at or above which a pixel counts as '
                   'positive in pos_frac / n_instances and the QA outlines.')
@click.option('--envelope-days', default=30, show_default=True,
              help='in_envelope is true when the scene is at most this many '
                   'days from a training chip.')
@click.option('--shard', default='0/1', show_default=True,
              help='i/n: process every n-th scene starting at i.')
@click.option('--overwrite/--no-overwrite', default=False,
              help='Redo scenes whose outputs already exist.')
def run(model_path, planetdir, ocmdir, coreg_json, outdir, grid_like,
        crowns_path, tile, stride, det_thresh, envelope_days, shard,
        overwrite):
    """Predict on every coregistered scene in COREG_JSON."""
    i, n = (int(v) for v in shard.split('/'))
    for sub in ('pred', 'qa', 'crowns', 'scenes'):
        os.makedirs(os.path.join(outdir, sub), exist_ok=True)

    device = select_device()
    model, params, epoch = load_checkpoint(model_path, device)
    channel_kinds = params.get('channel_kinds', ('red', 'green', 'blue'))
    if not params.get('train_chips'):
        raise click.ClickException(
            f'{model_path} records no train_chips; train it with --folds '
            '... --fold all so the distance to training data is known'
        )
    train_dates = pd.to_datetime(
        [c[:8] for c in params['train_chips']], format='%Y%m%d'
    ).values

    with rasterio.open(grid_like) as src:
        grid = {'crs': src.crs, 'transform': src.transform,
                'height': src.height, 'width': src.width}
    crowns = gpd.read_file(crowns_path).to_crs(grid['crs'])

    with open(coreg_json) as fh:
        records = sorted(json.load(fh), key=lambda r: r['scene'])
    records = records[i::n]

    for rec in tqdm(records):
        scene = rec['scene']
        crown_csv = os.path.join(outdir, 'crowns', f'{scene}.csv')
        if os.path.exists(crown_csv) and not overwrite:
            continue
        planetfile = os.path.join(planetdir, rec['planet_rel'])
        pimg = load_chip(planetfile, grid)
        arr = pimg.values.transpose(1, 2, 0).astype(np.float32)
        nodata = pimg.rio.nodata
        valid = (np.ones(arr.shape[:2], dtype=bool) if nodata is None
                 else (arr != nodata).all(axis=2))
        ocm_path = find_ocm_mask(planetfile, planetdir, ocmdir)
        if ocm_path is None:
            raise click.ClickException(f'no OCM mask for {planetfile}')
        clear = (reproject_ocm_to_grid(ocm_path, pimg).values == 0) & valid

        img = build_input_channels(arr, channel_kinds)
        masks, scores = predict_tiled(model, img, clear, device, tile, stride)
        ids, best = instance_rasters(masks, scores, clear.shape)

        transform = pimg.rio.transform()
        bands = np.stack([ids, best, clear.astype(np.float32)])
        desc = ('instance_id', 'max_score', 'clear')
        write_raster(os.path.join(outdir, 'pred', f'{scene}.pred.tif'),
                     bands, transform, grid['crs'], desc)
        ok = bool(rec['coreg_ok'])
        shift = (rec['x_shift_m'], rec['y_shift_m']) if ok else (0.0, 0.0)
        if ok:
            write_raster(
                os.path.join(outdir, 'pred', f'{scene}.pred_coreg.tif'),
                bands, Affine.translation(-shift[0], -shift[1]) * transform,
                grid['crs'], desc,
            )
        qa_png(os.path.join(outdir, 'qa', f'{scene}.png'), pimg, clear,
               valid, ids, best, det_thresh)

        date = pd.Timestamp(rec['date'])
        days_train = int(np.abs((train_dates - date.to_datetime64())
                                / np.timedelta64(1, 'D')).min())
        meta = {
            'scene': scene, 'date': rec['date'],
            'days_to_drone': rec['days_to_drone'],
            'days_to_nearest_train': days_train,
            'in_envelope': days_train <= envelope_days,
            'coreg_ok': ok,
            'x_shift_m': rec['x_shift_m'], 'y_shift_m': rec['y_shift_m'],
        }
        rows = crown_rows(crowns, shift, transform, clear.shape, clear, best,
                          ids, det_thresh)
        for k, v in reversed(meta.items()):
            rows.insert(0, k, v)
        rows.to_csv(crown_csv + '.tmp', index=False)
        os.replace(crown_csv + '.tmp', crown_csv)

        summary = {
            **meta,
            'drone_date': rec['drone_date'],
            'clear_fraction': float(clear[valid].mean()) if valid.any() else 0.0,
            'valid_fraction': float(valid.mean()),
            'n_detections': int(len(scores)),
            'n_detections_above_thresh': int((scores >= det_thresh).sum()),
            'positive_px': int(((best >= det_thresh) & clear).sum()),
            'det_thresh': det_thresh,
            'model': model_path, 'model_epoch': epoch,
        }
        with open(os.path.join(outdir, 'scenes', f'{scene}.json'), 'w') as fh:
            json.dump(summary, fh, indent=2)


@cli.command()
@click.argument('outdir', type=click.Path(exists=True, file_okay=False))
def concat(outdir):
    """Join per-scene outputs into crown_timeseries.csv and scene_summary.csv."""
    crown_files = sorted(glob(os.path.join(outdir, 'crowns', '*.csv')))
    scene_files = sorted(glob(os.path.join(outdir, 'scenes', '*.json')))
    if not crown_files:
        raise click.ClickException(f'no per-scene crown CSVs in {outdir}/crowns')
    crowns = pd.concat((pd.read_csv(f) for f in crown_files), ignore_index=True)
    crowns.to_csv(os.path.join(outdir, 'crown_timeseries.csv'), index=False)
    scenes = pd.DataFrame([json.load(open(f)) for f in scene_files])
    scenes.sort_values('scene').to_csv(
        os.path.join(outdir, 'scene_summary.csv'), index=False)
    print(f'{len(scene_files)} scenes, {len(crowns)} crown rows -> '
          f'{outdir}/crown_timeseries.csv, scene_summary.csv')


if __name__ == '__main__':
    cli()
