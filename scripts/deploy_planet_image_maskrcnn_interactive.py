#!/usr/bin/env python
"""Interactive Dash viewer for Planet Mask R-CNN predictions.

Navigate images in a directory, toggle ground truth and prediction
overlays, zoom into problem cases. A background worker pre-computes
predictions for the whole directory; navigating to an uncached image
bumps it to the front of the queue so the user waits at most a few
seconds.
"""
import atexit
import io
import itertools
import json
import os
import queue
import sys
import threading
import urllib.parse
from dataclasses import dataclass
from glob import glob
from typing import Optional

import click
import flask
import numpy as np
import plotly.graph_objects as go
import torch
from PIL import Image as _PILImage
from skimage import measure

import dash
from dash import Dash, Input, Output, Patch, State, ctx, dcc, html

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deploy_planet_image_maskrcnn import load_image_and_gt  # noqa: E402
from train_planet_image_maskrcnn import (  # noqa: E402
    _split_window,
    classify_instances,
)


@dataclass
class PredictionResult:
    masks: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    classifications: Optional[dict] = None


class PredictionCache:
    """In-memory prediction cache. Interface is small on purpose so a
    disk-backed subclass can be dropped in later."""

    def __init__(self):
        self._store = {}
        self._lock = threading.Lock()

    def get(self, path):
        with self._lock:
            return self._store.get(path)

    def put(self, path, result):
        with self._lock:
            self._store[path] = result

    def has(self, path):
        with self._lock:
            return path in self._store


class InferenceWorker(threading.Thread):
    """Single background thread that runs Mask R-CNN inference and
    populates a PredictionCache. User bumps jump to the front of the
    priority queue; the most recent bump wins."""

    _STOP = object()

    def __init__(self, model, device, cache, image_paths, split, size,
                 score_thresh, mask_thresh, iou_thresh, min_instance_size):
        super().__init__(daemon=True)
        self.model = model
        self.device = device
        self.cache = cache
        self.image_paths = list(image_paths)
        self.split = split
        self.size = size
        self.score_thresh = score_thresh
        self.mask_thresh = mask_thresh
        self.iou_thresh = iou_thresh
        self.min_instance_size = min_instance_size

        self._queue = queue.PriorityQueue()
        self._bump_counter = itertools.count(0, -1)
        self._fill_counter = itertools.count(1, 1)
        self._stop_event = threading.Event()

        for p in self.image_paths:
            self._queue.put((1, next(self._fill_counter), p))

    def bump(self, path):
        if self.cache.has(path):
            return
        self._queue.put((0, next(self._bump_counter), path))

    def stop(self):
        self._stop_event.set()
        self._queue.put((-1, 0, self._STOP))

    def run(self):
        self.model.to(self.device)
        self.model.eval()
        while not self._stop_event.is_set():
            _, _, path = self._queue.get()
            if path is self._STOP:
                break
            if self.cache.has(path):
                continue
            try:
                self.cache.put(path, self._run_inference(path))
            except Exception as e:
                print(f'[worker] error on {path}: {e}', file=sys.stderr)

    def _run_inference(self, path):
        _, gt_masks, img_tensor = load_image_and_gt(
            path, split=self.split, size=self.size,
            min_instance_size=self.min_instance_size,
        )
        with torch.no_grad():
            output = self.model([img_tensor.to(self.device)])[0]
        scores = output['scores'].cpu().numpy()
        keep = scores >= self.score_thresh
        soft_masks = output['masks'][keep, 0].cpu().numpy()
        boxes = output['boxes'][keep].cpu().numpy()
        scores = scores[keep]
        pred_masks = (soft_masks >= self.mask_thresh).astype(np.uint8)
        classifications = classify_instances(
            gt_masks, pred_masks, scores, iou_thresh=self.iou_thresh,
        )
        return PredictionResult(
            masks=pred_masks, boxes=boxes, scores=scores,
            classifications=classifications,
        )


def _mask_to_polygon_xy(mask):
    contours = measure.find_contours(mask.astype(np.float32), 0.5)
    if not contours:
        return [], []
    xs, ys = [], []
    for k, c in enumerate(contours):
        if k > 0:
            xs.append(None)
            ys.append(None)
        xs.extend(c[:, 1].tolist())
        ys.extend(c[:, 0].tolist())
    return xs, ys


def _hex_rgba(hex_color, alpha):
    s = hex_color.lstrip('#')
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


GT_COLOR = '#00E5FF'    # cyan
PRED_COLOR = '#FF00E5'  # magenta


def _trace_visible(kind, label, show_gt, show_pred, filter_val):
    show_kind = show_gt if kind == 'gt' else show_pred
    if not show_kind:
        return False
    if filter_val == 'all' or label is None:
        return True
    return label == filter_val


def build_figure(img, gt_masks, pred_result, show_gt, show_pred,
                 filter_val='all', drone_url=None, show_drone=False,
                 ocm_url=None, show_ocm=False, coreg_meta=None):
    h, w = img.shape[:2]
    fig = go.Figure()
    fig.add_trace(go.Image(z=img, hoverinfo='skip'))

    gt_labels = None
    pred_labels = None
    if pred_result is not None and pred_result.classifications is not None:
        gt_labels = pred_result.classifications['gt_labels']
        pred_labels = pred_result.classifications['pred_labels']

    for i in range(gt_masks.shape[0]):
        xs, ys = _mask_to_polygon_xy(gt_masks[i])
        if not xs:
            continue
        label = gt_labels[i] if gt_labels is not None else None
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode='lines', fill='toself',
            line=dict(color=GT_COLOR, width=1.5),
            fillcolor=_hex_rgba(GT_COLOR, 0.2),
            legendgroup='gt',
            meta=label,
            name='ground truth',
            showlegend=False,
            hoverinfo='skip',
            visible=_trace_visible('gt', label, show_gt, show_pred, filter_val),
        ))

    if pred_result is not None:
        for i in range(pred_result.masks.shape[0]):
            xs, ys = _mask_to_polygon_xy(pred_result.masks[i])
            score = float(pred_result.scores[i])
            label = pred_labels[i] if pred_labels is not None else None
            vis = _trace_visible('pred', label, show_gt, show_pred, filter_val)
            if xs:
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode='lines', fill='toself',
                    line=dict(color=PRED_COLOR, width=1.5),
                    fillcolor=_hex_rgba(PRED_COLOR, 0.2),
                    legendgroup='pred',
                    meta=label,
                    name=f'pred s={score:.2f}',
                    showlegend=False,
                    hoverinfo='name',
                    visible=vis,
                ))
            x1, y1, x2, y2 = pred_result.boxes[i]
            fig.add_trace(go.Scatter(
                x=[x1, x2, x2, x1, x1],
                y=[y1, y1, y2, y2, y1],
                mode='lines',
                line=dict(color=PRED_COLOR, width=1.2),
                legendgroup='pred',
                meta=label,
                name=f'pred bbox s={score:.2f}',
                showlegend=False,
                hoverinfo='skip',
                visible=vis,
            ))

    if coreg_meta is not None:
        x_m = coreg_meta.get('x_shift_m', 0.0)
        y_m = coreg_meta.get('y_shift_m', 0.0)
        ok = coreg_meta.get('coreg_ok', False)
        p_res = coreg_meta.get('planet_res_m')
        d_res = coreg_meta.get('drone_res_m')
        parts = [f'Δx={x_m:+.2f}m  Δy={y_m:+.2f}m  coreg={"OK" if ok else "FAIL"}']
        if p_res is not None and d_res is not None:
            parts.append(f'Planet {p_res:.1f}m/px  Drone {d_res:.3f}m/px')
        fig.add_annotation(
            text='<br>'.join(parts),
            xref='paper', yref='paper',
            x=0.01, y=0.99, xanchor='left', yanchor='top',
            showarrow=False,
            font=dict(size=11, color='white'),
            bgcolor='rgba(0,0,0,0.55)',
            borderpad=4,
        )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            range=[0, w], showgrid=False, zeroline=False, visible=False,
            constrain='domain',
        ),
        yaxis=dict(
            range=[h, 0], showgrid=False, zeroline=False, visible=False,
            scaleanchor='x', scaleratio=1,
        ),
        dragmode='pan',
        showlegend=False,
    )

    overlays = []
    if show_drone and drone_url is not None:
        overlays.append(dict(
            source=drone_url,
            xref='x', yref='y',
            x=0, y=0, sizex=w, sizey=h,
            xanchor='left', yanchor='top',
            sizing='stretch',
            opacity=0.5,
            layer='above',
        ))
    if show_ocm and ocm_url is not None:
        overlays.append(dict(
            source=ocm_url,
            xref='x', yref='y',
            x=0, y=0, sizex=w, sizey=h,
            xanchor='left', yanchor='top',
            sizing='stretch',
            layer='above',
        ))
    fig.update_layout(images=overlays)

    return fig


_INDEX_HTML = '''<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>Planet Mask R-CNN Viewer</title>
{%favicon%}
{%css%}
<style>
body { font-family: sans-serif; margin: 0; padding: 12px; }
.toolbar { display: flex; gap: 14px; align-items: center; margin-bottom: 8px; flex-wrap: wrap; }
.spinner { width: 16px; height: 16px; border: 2px solid #ddd; border-top-color: #3b82f6; border-radius: 50%; animation: spin 0.8s linear infinite; display: inline-block; vertical-align: middle; }
@keyframes spin { to { transform: rotate(360deg); } }
#header { font-family: monospace; font-size: 13px; color: #444; margin-left: auto; }
button { padding: 4px 10px; }
</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>
'''


def _serve_cropped(path, fracs):
    if fracs is None:
        return flask.send_file(path, mimetype='image/png',
                               conditional=True, max_age=86400)
    y0f, y1f, x0f, x1f = fracs
    with _PILImage.open(path) as im:
        w_im, h_im = im.size
        box = (
            int(round(x0f * w_im)),
            int(round(y0f * h_im)),
            int(round(x1f * w_im)),
            int(round(y1f * h_im)),
        )
        cropped = im.crop(box)
        buf = io.BytesIO()
        cropped.save(buf, format='PNG')
    buf.seek(0)
    return flask.send_file(buf, mimetype='image/png', max_age=86400)


def make_app(image_paths, cache, worker, split, size, min_instance_size,
             coreg_info=None, drone_paths=None, ocm_paths=None,
             crop_fracs=None):
    app = Dash(__name__)
    app.index_string = _INDEX_HTML

    drone_paths = drone_paths or {}
    ocm_paths = ocm_paths or {}
    crop_fracs = crop_fracs or {}

    @app.server.route('/drone/<scene>')
    def _serve_drone(scene):
        scene_key = urllib.parse.unquote(scene)
        path = drone_paths.get(scene_key)
        if path is None:
            flask.abort(404)
        return _serve_cropped(path, crop_fracs.get(scene_key))

    @app.server.route('/ocm/<scene>')
    def _serve_ocm(scene):
        scene_key = urllib.parse.unquote(scene)
        path = ocm_paths.get(scene_key)
        if path is None:
            flask.abort(404)
        return _serve_cropped(path, crop_fracs.get(scene_key))

    filenames = [os.path.basename(p) for p in image_paths]

    app.layout = html.Div([
        dcc.Store(id='store-current-idx', data=0),
        dcc.Store(id='store-last-rendered', data=None),
        dcc.Interval(id='poll-cache', interval=500),
        html.Div(className='toolbar', children=[
            html.Button('◀ Prev', id='btn-prev', n_clicks=0),
            html.Button('Next ▶', id='btn-next', n_clicks=0),
            dcc.Dropdown(
                id='dd-image',
                options=[{'label': fn, 'value': i}
                         for i, fn in enumerate(filenames)],
                value=0, clearable=False,
                style={'minWidth': '360px'},
            ),
            dcc.Checklist(
                id='toggle-gt',
                options=[{'label': ' Show ground truth', 'value': 'gt'}],
                value=['gt'], inline=True,
            ),
            dcc.Dropdown(
                id='dd-filter',
                options=[
                    {'label': 'All', 'value': 'all'},
                    {'label': 'True Positive', 'value': 'TP'},
                    {'label': 'False Positive', 'value': 'FP'},
                    {'label': 'False Negative', 'value': 'FN'},
                ],
                value='all', clearable=False,
                style={'minWidth': '180px'},
            ),
            html.Div(id='pred-toggle-slot', style={'display': 'inline-block'},
                     children=[
                dcc.Checklist(
                    id='toggle-pred',
                    options=[{'label': ' Show predictions', 'value': 'pred'}],
                    value=['pred'], inline=True,
                    style={'display': 'inline-block'},
                ),
                html.Span(id='pred-spinner', style={'display': 'none'},
                          children=[
                    html.Span(className='spinner'),
                    html.Span(' computing predictions…',
                              style={'marginLeft': '6px'}),
                ]),
            ]),
            html.Div(id='drone-toggle-slot', style={'display': 'none'},
                     children=[
                dcc.Checklist(
                    id='toggle-drone',
                    options=[{'label': ' Show drone overlay', 'value': 'drone'}],
                    value=[], inline=True,
                    style={'display': 'inline-block'},
                ),
            ]),
            html.Div(id='ocm-toggle-slot', style={'display': 'none'},
                     children=[
                dcc.Checklist(
                    id='toggle-ocm',
                    options=[{'label': ' Show cloud mask', 'value': 'ocm'}],
                    value=[], inline=True,
                    style={'display': 'inline-block'},
                ),
            ]),
            html.Div(id='header'),
        ]),
        dcc.Graph(
            id='viewer',
            config={'scrollZoom': True, 'displaylogo': False},
            style={'height': '82vh'},
        ),
    ])

    @app.callback(
        Output('store-current-idx', 'data'),
        Input('btn-prev', 'n_clicks'),
        Input('btn-next', 'n_clicks'),
        Input('dd-image', 'value'),
        State('store-current-idx', 'data'),
        prevent_initial_call=True,
    )
    def on_nav(_prev, _next, dd_val, cur):
        trigger = ctx.triggered_id
        cur = cur or 0
        if trigger == 'btn-prev':
            return max(0, cur - 1)
        if trigger == 'btn-next':
            return min(len(image_paths) - 1, cur + 1)
        if trigger == 'dd-image' and dd_val is not None and dd_val != cur:
            return dd_val
        raise dash.exceptions.PreventUpdate

    @app.callback(
        Output('dd-image', 'value'),
        Output('header', 'children'),
        Input('store-current-idx', 'data'),
    )
    def on_idx_sync(idx):
        idx = idx or 0
        return idx, f'{idx + 1} / {len(image_paths)}  —  {filenames[idx]}'

    @app.callback(
        Output('viewer', 'figure'),
        Output('pred-spinner', 'style'),
        Output('toggle-pred', 'style'),
        Output('drone-toggle-slot', 'style'),
        Output('ocm-toggle-slot', 'style'),
        Output('store-last-rendered', 'data'),
        Input('store-current-idx', 'data'),
        Input('poll-cache', 'n_intervals'),
        Input('toggle-drone', 'value'),
        Input('toggle-ocm', 'value'),
        State('store-last-rendered', 'data'),
        State('toggle-gt', 'value'),
        State('toggle-pred', 'value'),
        State('dd-filter', 'value'),
    )
    def render(idx, _tick, drone_val, ocm_val, last, gt_val, pred_val, filter_val):
        idx = idx or 0
        path = image_paths[idx]
        scene = os.path.splitext(os.path.basename(path))[0]
        if ctx.triggered_id == 'store-current-idx':
            worker.bump(path)

        pred = cache.get(path)
        pred_available = pred is not None

        drone_available = scene in drone_paths
        ocm_available = scene in ocm_paths

        show_drone = 'drone' in (drone_val or []) and drone_available
        show_ocm = 'ocm' in (ocm_val or []) and ocm_available

        if last is not None:
            if (last.get('path') == path
                    and last.get('pred_available') == pred_available
                    and last.get('drone_available') == drone_available
                    and last.get('ocm_available') == ocm_available
                    and last.get('show_drone') == show_drone
                    and last.get('show_ocm') == show_ocm):
                raise dash.exceptions.PreventUpdate

        img, gt_masks, _ = load_image_and_gt(
            path, split=split, size=size,
            min_instance_size=min_instance_size,
        )

        scene_q = urllib.parse.quote(scene, safe='')
        drone_url = f'/drone/{scene_q}' if drone_available else None
        ocm_url = f'/ocm/{scene_q}' if ocm_available else None

        show_gt = 'gt' in (gt_val or [])
        show_pred = 'pred' in (pred_val or []) and pred_available
        coreg_meta = (coreg_info or {}).get(scene)
        fig = build_figure(img, gt_masks, pred, show_gt, show_pred,
                           filter_val=filter_val or 'all',
                           drone_url=drone_url, show_drone=show_drone,
                           ocm_url=ocm_url, show_ocm=show_ocm,
                           coreg_meta=coreg_meta)

        if pred_available:
            spinner_style = {'display': 'none'}
            toggle_style = {'display': 'inline-block'}
        else:
            spinner_style = {'display': 'inline-block'}
            toggle_style = {'display': 'none'}

        drone_slot_style = {'display': 'inline-block'} if drone_available else {'display': 'none'}
        ocm_slot_style = {'display': 'inline-block'} if ocm_available else {'display': 'none'}

        return (fig, spinner_style, toggle_style, drone_slot_style, ocm_slot_style,
                {'path': path, 'pred_available': pred_available,
                 'drone_available': drone_available, 'ocm_available': ocm_available,
                 'show_drone': show_drone, 'show_ocm': show_ocm})

    @app.callback(
        Output('viewer', 'figure', allow_duplicate=True),
        Input('toggle-gt', 'value'),
        Input('toggle-pred', 'value'),
        Input('dd-filter', 'value'),
        State('viewer', 'figure'),
        prevent_initial_call=True,
    )
    def on_toggle(gt_val, pred_val, filter_val, fig):
        if not fig or 'data' not in fig:
            raise dash.exceptions.PreventUpdate
        show_gt = 'gt' in (gt_val or [])
        show_pred = 'pred' in (pred_val or [])
        filter_val = filter_val or 'all'
        patched = Patch()
        for i, trace in enumerate(fig['data']):
            lg = trace.get('legendgroup')
            if lg in ('gt', 'pred'):
                label = trace.get('meta')
                patched['data'][i]['visible'] = _trace_visible(
                    lg, label, show_gt, show_pred, filter_val,
                )
        return patched

    return app


@click.command()
@click.argument('modelfile')
@click.argument('imagedir')
@click.option('--score-thresh', default=0.5, type=float,
              help='Minimum score for a prediction to be kept.')
@click.option('--mask-thresh', default=0.5, type=float,
              help='Threshold applied to soft Mask R-CNN mask logits.')
@click.option('--split', default='right',
              type=click.Choice(['left', 'right']),
              help='Which half of each tile to visualize (test=right).')
@click.option('--size', default=512, type=int)
@click.option('--iou-thresh', default=0.25, type=float,
              help='IoU threshold for TP/FP/FN matching.')
@click.option('--host', default='127.0.0.1', type=str)
@click.option('--port', default=8050, type=int)
@click.option('--logfile', default=None, type=click.Path(exists=False),
              help='Path to coreg_log.json (default: imagedir/coreg_log.json).')
def main(modelfile, imagedir, score_thresh, mask_thresh, split,
         size, iou_thresh, host, port, logfile):

    image_paths = sorted(glob(os.path.join(imagedir, '*rgb.png')))
    image_paths = [p for p in image_paths if not p.endswith('.mask.png')]
    if not image_paths:
        raise click.ClickException(f'no *rgb.png files in {imagedir}')

    def _meets_size(p):
        from PIL import Image as _Image
        with _Image.open(p) as im:
            h, w = im.height, im.width
        return h >= size and w >= 2 * size

    skipped = [p for p in image_paths if not _meets_size(p)]
    image_paths = [p for p in image_paths if _meets_size(p)]
    if skipped:
        print(f'Skipping {len(skipped)} image(s) smaller than {size}x{2*size}: '
              + ', '.join(os.path.basename(p) for p in skipped))
    if not image_paths:
        raise click.ClickException(f'no images meet the minimum size ({size}x{2*size})')

    if logfile is None:
        logfile = os.path.join(imagedir, 'coreg_log.json')
    coreg_info = None
    if os.path.exists(logfile):
        with open(logfile) as f:
            records = json.load(f)
        coreg_info = {rec['scene']: rec for rec in records}

    drone_paths = {}
    ocm_paths = {}
    crop_fracs = {}
    for p in image_paths:
        scene = os.path.splitext(os.path.basename(p))[0]
        drone_png = os.path.join(imagedir, f'{scene}.drone.png')
        if os.path.exists(drone_png):
            drone_paths[scene] = drone_png
        ocm_png = os.path.join(imagedir, f'{scene}.ocm.png')
        if os.path.exists(ocm_png):
            ocm_paths[scene] = ocm_png
        if scene in drone_paths or scene in ocm_paths:
            with _PILImage.open(p) as im:
                w_p, h_p = im.size
            row_start, row_end, col_start, col_end = _split_window(
                h_p, w_p, split, size,
            )
            crop_fracs[scene] = (
                row_start / h_p, row_end / h_p,
                col_start / w_p, col_end / w_p,
            )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(modelfile, map_location='cpu', weights_only=False)
    model = ckpt['model']
    min_instance_size = ckpt['params']['min_instance_size']

    cache = PredictionCache()
    worker = InferenceWorker(
        model=model, device=device, cache=cache,
        image_paths=image_paths, split=split, size=size,
        score_thresh=score_thresh, mask_thresh=mask_thresh,
        iou_thresh=iou_thresh, min_instance_size=min_instance_size,
    )
    worker.start()
    atexit.register(worker.stop)

    app = make_app(image_paths, cache, worker, split=split, size=size,
                   min_instance_size=min_instance_size,
                   coreg_info=coreg_info, drone_paths=drone_paths,
                   ocm_paths=ocm_paths, crop_fracs=crop_fracs)
    print(f'Serving on http://{host}:{port} (device={device}, '
          f'{len(image_paths)} images, {len(drone_paths)} drone overlays, '
          f'{len(ocm_paths)} cloud masks)')
    app.run(host=host, port=port, debug=False)


if __name__ == '__main__':
    main()
