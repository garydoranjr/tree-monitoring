#!/usr/bin/env python
"""Interactive Dash viewer for Planet Mask R-CNN predictions.

Navigate images in a directory, toggle ground truth and prediction
overlays, zoom into problem cases. A background worker pre-computes
predictions for the whole directory; navigating to an uncached image
bumps it to the front of the queue so the user waits at most a few
seconds.
"""
import atexit
import itertools
import os
import queue
import sys
import threading
from dataclasses import dataclass
from glob import glob
from typing import Optional

import click
import numpy as np
import plotly.graph_objects as go
import torch
from skimage import measure

import dash
from dash import Dash, Input, Output, Patch, State, ctx, dcc, html

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deploy_planet_image_maskrcnn import load_image_and_gt  # noqa: E402


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


def classify_instances(gt_masks, pred_masks, pred_scores, iou_thresh=0.5):
    """Greedy IoU matching between predicted and GT instance masks.

    Returns per-instance TP/FP/FN labels plus the raw IoU matrix. The
    UI filter hook for TP/FP/FN lives off of this dict."""
    P = pred_masks.shape[0]
    G = gt_masks.shape[0]
    iou = np.zeros((P, G), dtype=np.float32)
    if P and G:
        p_flat = pred_masks.reshape(P, -1).astype(np.int32)
        g_flat = gt_masks.reshape(G, -1).astype(np.int32)
        inter = p_flat @ g_flat.T
        p_area = p_flat.sum(axis=1)[:, None]
        g_area = g_flat.sum(axis=1)[None, :]
        union = p_area + g_area - inter
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = np.where(union > 0, inter / union, 0.0).astype(np.float32)

    pred_labels = ['FP'] * P
    gt_labels = ['FN'] * G
    pred_to_gt = [-1] * P
    if P and G:
        order = np.argsort(-pred_scores)
        gt_used = np.zeros(G, dtype=bool)
        for pi in order:
            if gt_used.all():
                break
            avail = np.where(~gt_used)[0]
            best_local = int(np.argmax(iou[pi, avail]))
            gi = int(avail[best_local])
            if iou[pi, gi] >= iou_thresh:
                gt_used[gi] = True
                pred_labels[pi] = 'TP'
                gt_labels[gi] = 'TP'
                pred_to_gt[pi] = gi
    return {
        'pred_labels': pred_labels,
        'gt_labels': gt_labels,
        'pred_to_gt': pred_to_gt,
        'iou': iou,
    }


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


def build_figure(img, gt_masks, pred_result, show_gt, show_pred):
    fig = go.Figure()
    fig.add_trace(go.Image(z=img, hoverinfo='skip'))

    for i in range(gt_masks.shape[0]):
        xs, ys = _mask_to_polygon_xy(gt_masks[i])
        if not xs:
            continue
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode='lines', fill='toself',
            line=dict(color=GT_COLOR, width=1.5),
            fillcolor=_hex_rgba(GT_COLOR, 0.2),
            legendgroup='gt',
            name='ground truth',
            showlegend=False,
            hoverinfo='skip',
            visible=show_gt,
        ))

    if pred_result is not None:
        for i in range(pred_result.masks.shape[0]):
            xs, ys = _mask_to_polygon_xy(pred_result.masks[i])
            score = float(pred_result.scores[i])
            if xs:
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode='lines', fill='toself',
                    line=dict(color=PRED_COLOR, width=1.5),
                    fillcolor=_hex_rgba(PRED_COLOR, 0.2),
                    legendgroup='pred',
                    name=f'pred s={score:.2f}',
                    showlegend=False,
                    hoverinfo='name',
                    visible=show_pred,
                ))
            x1, y1, x2, y2 = pred_result.boxes[i]
            fig.add_trace(go.Scatter(
                x=[x1, x2, x2, x1, x1],
                y=[y1, y1, y2, y2, y1],
                mode='lines',
                line=dict(color=PRED_COLOR, width=1.2),
                legendgroup='pred',
                name=f'pred bbox s={score:.2f}',
                showlegend=False,
                hoverinfo='skip',
                visible=show_pred,
            ))

    h, w = img.shape[:2]
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


def make_app(image_paths, cache, worker, split, size):
    app = Dash(__name__)
    app.index_string = _INDEX_HTML

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
        Output('store-last-rendered', 'data'),
        Input('store-current-idx', 'data'),
        Input('poll-cache', 'n_intervals'),
        State('store-last-rendered', 'data'),
        State('toggle-gt', 'value'),
        State('toggle-pred', 'value'),
    )
    def render(idx, _tick, last, gt_val, pred_val):
        idx = idx or 0
        path = image_paths[idx]
        if ctx.triggered_id == 'store-current-idx':
            worker.bump(path)

        pred = cache.get(path)
        pred_available = pred is not None

        if last is not None:
            if (last.get('path') == path
                    and last.get('pred_available') == pred_available):
                raise dash.exceptions.PreventUpdate

        img, gt_masks, _ = load_image_and_gt(path, split=split, size=size)
        show_gt = 'gt' in (gt_val or [])
        show_pred = 'pred' in (pred_val or []) and pred_available
        fig = build_figure(img, gt_masks, pred, show_gt, show_pred)

        if pred_available:
            spinner_style = {'display': 'none'}
            toggle_style = {'display': 'inline-block'}
        else:
            spinner_style = {'display': 'inline-block'}
            toggle_style = {'display': 'none'}

        return (fig, spinner_style, toggle_style,
                {'path': path, 'pred_available': pred_available})

    @app.callback(
        Output('viewer', 'figure', allow_duplicate=True),
        Input('toggle-gt', 'value'),
        Input('toggle-pred', 'value'),
        State('viewer', 'figure'),
        prevent_initial_call=True,
    )
    def on_toggle(gt_val, pred_val, fig):
        if not fig or 'data' not in fig:
            raise dash.exceptions.PreventUpdate
        show_gt = 'gt' in (gt_val or [])
        show_pred = 'pred' in (pred_val or [])
        patched = Patch()
        for i, trace in enumerate(fig['data']):
            lg = trace.get('legendgroup')
            if lg == 'gt':
                patched['data'][i]['visible'] = show_gt
            elif lg == 'pred':
                patched['data'][i]['visible'] = show_pred
        return patched

    return app


@click.command()
@click.argument('modelfile')
@click.argument('imagedir')
@click.argument('outputdir')
@click.option('--score-thresh', default=0.5, type=float,
              help='Minimum score for a prediction to be kept.')
@click.option('--mask-thresh', default=0.5, type=float,
              help='Threshold applied to soft Mask R-CNN mask logits.')
@click.option('--split', default='right',
              type=click.Choice(['left', 'right']),
              help='Which half of each tile to visualize (test=right).')
@click.option('--size', default=512, type=int)
@click.option('--iou-thresh', default=0.5, type=float,
              help='IoU threshold for TP/FP/FN matching.')
@click.option('--host', default='127.0.0.1', type=str)
@click.option('--port', default=8050, type=int)
def main(modelfile, imagedir, outputdir, score_thresh, mask_thresh, split,
         size, iou_thresh, host, port):
    os.makedirs(outputdir, exist_ok=True)

    image_paths = sorted(glob(os.path.join(imagedir, '*rgb.png')))
    image_paths = [p for p in image_paths if not p.endswith('.mask.png')]
    if not image_paths:
        raise click.ClickException(f'no *rgb.png files in {imagedir}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(modelfile, map_location='cpu')
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

    app = make_app(image_paths, cache, worker, split=split, size=size)
    print(f'Serving on http://{host}:{port} (device={device}, '
          f'{len(image_paths)} images)')
    app.run(host=host, port=port, debug=False)


if __name__ == '__main__':
    main()
