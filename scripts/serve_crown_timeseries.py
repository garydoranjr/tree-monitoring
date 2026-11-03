#!/usr/bin/env python
"""Serve the per-crown timeseries viewer with drone and Planet image layers.

The page is the one make_crown_timeseries_html.py writes, plus image layers
under the crown map:

- the drone orthomosaic of the current flight: a cached 0.25 m overview of
  the whole plot (crown_viewer_imagery.py prerender fills the cache), and a
  near-native crop of the visible window when zoomed in;
- the Planet scene of the current date (RGB, 2/98 % stretch over clear
  pixels), with its detection scores (pred band 2) and cloud mask (pred
  band 3) as optional overlays. Coregistered scenes are shown in crown-map
  geometry, as pred_coreg.tif is.

    python scripts/serve_crown_timeseries.py            # http://127.0.0.1:8050
    python scripts/serve_crown_timeseries.py \\
        --apply-dir /Volumes/Earth03/flower/202610_temporal_apply_cal \\
        --planet-dir /Volumes/Earth03/flower/planet_clipped_calfill/4band \\
        --coreg-json /Volumes/Earth03/flower/202610_temporal_apply/coreg_nearest_drone.json
"""
import json
import os
import threading
from collections import OrderedDict

import click
import rasterio
from flask import Flask, Response, abort, request
from plotly.offline import get_plotlyjs

import crown_viewer_imagery as cv
from make_crown_timeseries_html import APPLY, CROWNMAP, NC4S, build_data, render_page

MAX_CROP_M = 400
MAX_CROP_PX = 2400


class _LRU:
    """Bounded cache of encoded image bytes."""

    def __init__(self, maxsize):
        self._lock = threading.Lock()
        self._data = OrderedDict()
        self._maxsize = maxsize

    def get(self, key):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            return self._data.get(key)

    def put(self, key, value):
        with self._lock:
            self._data[key] = value
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)


def make_app(data, apply_dir, planet_dir, cache_dir, alignment, coreg_json):
    bounds = cv.grid_bounds(CROWNMAP)
    ox, oy = data['origin']
    data['layers'] = {
        # Placement of every image layer in the page's map coordinates.
        'x': bounds[0] - ox, 'y': oy - bounds[3],
        'w': bounds[2] - bounds[0], 'h': bounds[3] - bounds[1],
        'overviewRes': cv.OVERVIEW_RES,
    }
    page = render_page(data, plotlyjs='plotly.js')
    plotlyjs = get_plotlyjs()
    flights = {d: cv.drone_file(d, alignment) for d in data['droneDates']}
    with open(coreg_json) as fh:
        coreg = {r['scene']: r for r in json.load(fh)}
    scenes = set(data['scenes']['name'])
    cache = _LRU(64)
    locks, locks_lock = {}, threading.Lock()

    def lock_for(key):
        with locks_lock:
            return locks.setdefault(key, threading.Lock())

    def image(body):
        return Response(body, mimetype='image/webp',
                        headers={'Cache-Control': 'max-age=86400'})

    def planet_inputs(scene):
        if scene not in scenes or scene not in coreg:
            abort(404)
        r = coreg[scene]
        if r['coreg_ok']:
            pred = os.path.join(apply_dir, 'pred', f'{scene}.pred_coreg.tif')
            shift = (r['x_shift_m'], r['y_shift_m'])
        else:
            pred = os.path.join(apply_dir, 'pred', f'{scene}.pred.tif')
            shift = (0.0, 0.0)
        return os.path.join(planet_dir, r['planet_rel']), pred, shift

    app = Flask(__name__)

    @app.route('/')
    def index():
        return Response(page, mimetype='text/html')

    @app.route('/plotly.js')
    def plotly():
        return Response(plotlyjs, mimetype='application/javascript',
                        headers={'Cache-Control': 'max-age=86400'})

    @app.route('/drone/<date>.webp')
    def drone_overview(date):
        if date not in flights:
            abort(404)
        path = cv.overview_path(cache_dir, date)
        if not path.exists():
            with lock_for(('drone', date)):
                cv.render_overview(date, cache_dir, bounds, cv.OVERVIEW_RES, alignment)
        return image(path.read_bytes())

    @app.route('/drone/<date>/crop')
    def drone_crop(date):
        if date not in flights:
            abort(404)
        try:
            box = tuple(float(request.args[k]) for k in ('x0', 'y0', 'x1', 'y1'))
            w = int(request.args['w'])
        except (KeyError, ValueError):
            abort(400)
        if not (0 < box[2] - box[0] <= MAX_CROP_M and 0 < box[3] - box[1] <= MAX_CROP_M):
            abort(400)
        key = ('crop', date, box, w)
        body = cache.get(key)
        if body is None:
            with rasterio.open(flights[date]) as src:
                native = src.res[0]
            width = int(min(max(w, 1), MAX_CROP_PX, round((box[2] - box[0]) / native)))
            height = max(1, int(round(width * (box[3] - box[1]) / (box[2] - box[0]))))
            rgba, _ = cv.render_drone(flights[date], box, width, height)
            body = cv.encode_webp(rgba)
            cache.put(key, body)
        return image(body)

    @app.route('/planet/<scene>/<layer>.webp')
    def planet(scene, layer):
        if layer not in ('rgb', 'score', 'cloud'):
            abort(404)
        key = ('planet', scene, layer)
        body = cache.get(key)
        if body is None:
            src, pred, shift = planet_inputs(scene)
            if layer == 'rgb':
                rgba = cv.render_planet_rgb(src, pred, (0.0, 0.0), bounds, shift)
            elif layer == 'score':
                rgba = cv.render_planet_score(pred, bounds)
            else:
                rgba = cv.render_planet_cloud(pred, bounds)
            body = cv.encode_webp(rgba, quality=95)
            cache.put(key, body)
        return image(body)

    return app


@click.command()
@click.option('--apply-dir', default=APPLY, show_default=True,
              type=click.Path(exists=True, file_okay=False))
@click.option('--planet-dir', default=str(cv.PLANET_DIR), show_default=True,
              type=click.Path(exists=True, file_okay=False),
              help='Planet 4-band scenes the apply run read (planet_rel paths).')
@click.option('--coreg-json', type=click.Path(exists=True, dir_okay=False),
              help='Coregistration log the apply run read '
                   '[default: APPLY_DIR/coreg_nearest_drone.json].')
@click.option('--cache-dir', default=str(cv.CACHE_DIR), show_default=True,
              type=click.Path(file_okay=False),
              help='Drone overview cache (crown_viewer_imagery.py prerender).')
@click.option('--alignment', default='local', show_default=True,
              type=click.Choice(['local', 'global']),
              help='Phantom mosaic alignment (crown_classifications.nc4 is local).')
@click.option('--nc4', 'nc4s', multiple=True, default=NC4S, show_default=True)
@click.option('--store-min-clear', default=0.1, show_default=True)
@click.option('--simplify-m', default=0.3, show_default=True)
@click.option('--host', default='127.0.0.1', show_default=True)
@click.option('--port', default=8050, show_default=True)
def main(apply_dir, planet_dir, coreg_json, cache_dir, alignment, nc4s,
         store_min_clear, simplify_m, host, port):
    coreg_json = coreg_json or os.path.join(apply_dir, 'coreg_nearest_drone.json')
    data = build_data(apply_dir, CROWNMAP, nc4s, store_min_clear, simplify_m)
    app = make_app(data, apply_dir, planet_dir, cache_dir, alignment, coreg_json)
    click.echo(f'Serving on http://{host}:{port}/')
    app.run(host=host, port=port, threaded=True)


if __name__ == '__main__':
    main()
