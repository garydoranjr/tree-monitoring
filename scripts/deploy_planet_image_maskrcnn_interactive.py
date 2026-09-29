#!/usr/bin/env python
"""Interactive Dash viewer for Planet chips: drone ortho, ground truth and
(optionally) Mask R-CNN predictions.

Navigate images in a directory, toggle the layers, zoom into problem cases.
With `--model` a background worker pre-computes predictions for the whole
directory, and navigating to an uncached image bumps it to the front of the
queue so the user waits at most a few seconds.

Without `--model` the prediction layer is simply absent and the app becomes a
coregistration reviewer: blend or swipe the drone ortho against the Planet
chip, or hit `b` to blink it, to judge whether the two -- and the crown mask
applied from the drone -- line up. Chips with no `.mask.png` (every scene
whose coregistration failed) lose the ground-truth layer rather than erroring.
"""
import atexit
import io
import itertools
import json
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must precede torch import
import queue
import sys
import threading
import urllib.parse
from collections import OrderedDict
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
from train_planet_image_maskrcnn import (  # noqa: E402, F401
    OCMMaskRCNN,  # needed for torch.load to unpickle ocm-mask checkpoints
    OCMRPN,
    OCMRoIHeads,
    _lookup_clear_at_centers,
    _split_window,
    binary_mask_to_instances,
    classify_instances,
)
from util import select_device  # noqa: E402


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
                 score_thresh, mask_thresh, iou_thresh, min_instance_size,
                 use_ocm_masks=False, channel_kinds=None):
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
        self.use_ocm_masks = use_ocm_masks
        self.channel_kinds = channel_kinds

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
        if self.use_ocm_masks:
            _, gt_masks, img_tensor, clear_crop = load_image_and_gt(
                path, split=self.split, size=self.size,
                min_instance_size=self.min_instance_size,
                load_ocm_mask=True, channel_kinds=self.channel_kinds,
            )
        else:
            _, gt_masks, img_tensor = load_image_and_gt(
                path, split=self.split, size=self.size,
                min_instance_size=self.min_instance_size,
                channel_kinds=self.channel_kinds,
            )
            clear_crop = None

        with torch.no_grad():
            output = self.model([img_tensor.to(self.device)])[0]
        scores_t = output['scores']
        boxes_t = output['boxes']
        soft_masks_t = output['masks'][:, 0]

        keep_score = scores_t >= self.score_thresh
        scores_t = scores_t[keep_score]
        boxes_t = boxes_t[keep_score]
        soft_masks_t = soft_masks_t[keep_score]

        if clear_crop is not None and boxes_t.shape[0] > 0:
            cm = torch.from_numpy(np.ascontiguousarray(clear_crop).astype(bool))
            cm = cm.to(self.device)
            keep_clear = _lookup_clear_at_centers([cm], boxes_t, 0)
            scores_t = scores_t[keep_clear]
            boxes_t = boxes_t[keep_clear]
            soft_masks_t = soft_masks_t[keep_clear]

        scores = scores_t.cpu().numpy()
        boxes = boxes_t.cpu().numpy()
        pred_masks = (soft_masks_t.cpu().numpy() >= self.mask_thresh).astype(
            np.uint8
        )
        classifications = classify_instances(
            gt_masks, pred_masks, scores, iou_thresh=self.iou_thresh,
        )
        return PredictionResult(
            masks=pred_masks, boxes=boxes, scores=scores,
            classifications=classifications,
        )


def crop_window(path, split, size):
    """The (row_start, row_end, col_start, col_end) of the displayed region of
    a chip, read from the PNG header. `split == "whole"` takes the full tile."""
    with _PILImage.open(path) as im:
        w, h = im.size
    if split == 'whole':
        return 0, h, 0, w
    return _split_window(h, w, split, size)


def load_display_shape_and_gt(path, split, size, min_instance_size):
    """The `(h, w)` of the displayed crop and its ground-truth instances, read
    from the PNGs alone.

    The figure needs only the crop's dimensions -- the pixels on screen are
    served straight from the PNG on disk -- so going through the PNG here
    avoids re-reading and re-stretching the 4-band tif on every render. The
    crop and the instance extraction match `load_image_and_gt` exactly, so the
    ground truth is identical to what the model is scored against.

    Returns an empty instance stack when the chip has no `.mask.png`, which is
    the case for every scene whose coregistration failed.
    """
    r0, r1, c0, c1 = crop_window(path, split, size)
    shape = (r1 - r0, c1 - c0)

    maskfile = path[:-4] + '.mask.png' if path.endswith('.png') else path + '.mask.png'
    if not os.path.exists(maskfile):
        return shape, np.zeros((0, *shape), dtype=np.uint8)

    mask = np.array(_PILImage.open(maskfile))
    mask = (mask == 255).astype(np.uint8)[r0:r1, c0:c1]
    return shape, binary_mask_to_instances(
        mask, min_instance_size=min_instance_size,
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


def build_figure(img_shape, gt_masks, pred_result, show_gt, show_pred,
                 chip_url, filter_val='all',
                 ocm_array=None, show_ocm=False, coreg_meta=None,
                 ocm_opacity=0.5, swipe_frac=1.0, show_divider=False):
    h, w = img_shape
    fig = go.Figure()

    # OCM is a go.Image trace so its alpha composites correctly above the
    # layer='below' Planet/Drone images, while polygon traces added after
    # still render on top of it.
    if show_ocm and ocm_array is not None:
        oh, ow = ocm_array.shape[:2]
        fig.add_trace(go.Image(
            z=ocm_array,
            colormodel='rgba256',
            x0=0, y0=0, dx=w / ow, dy=h / oh,
            opacity=ocm_opacity,
            hoverinfo='skip',
            name='ocm',
        ))

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
        x_m = coreg_meta.get('x_shift_m')
        y_m = coreg_meta.get('y_shift_m')
        ok = coreg_meta.get('coreg_ok', False)
        p_res = coreg_meta.get('planet_res_m')
        d_res = coreg_meta.get('drone_res_m')
        # A log rebuilt by reconstruct_coreg_log.py carries no recoverable
        # shift, so render the null as "n/a" and say where the record came
        # from -- a missing offset must never read as a measured zero.
        x_txt = 'n/a' if x_m is None else f'{x_m:+.2f}m'
        y_txt = 'n/a' if y_m is None else f'{y_m:+.2f}m'
        line = f'Δx={x_txt}  Δy={y_txt}  coreg={"OK" if ok else "FAIL"}'
        flags = []
        if coreg_meta.get('reconstructed'):
            flags.append('reconstructed')
        if coreg_meta.get('ambiguous_coreg_ok'):
            flags.append('ambiguous')
        if flags:
            line += '  (' + ', '.join(flags) + ')'
        parts = [line]
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

    # Always present so the swipe slider can move it with a Patch; parked
    # off the right edge (frac 1.0) when swiping is off.
    x_div = swipe_frac * w
    fig.add_trace(go.Scatter(
        x=[x_div, x_div], y=[0, h],
        mode='lines',
        line=dict(color='white', width=1.5, dash='dot'),
        name='swipe-divider',
        showlegend=False,
        hoverinfo='skip',
        visible=bool(show_divider),
    ))

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

    # A single image: the drone ortho is composited onto the Planet chip
    # server-side (see `composite_chip`). Stacking them as two `layer='below'`
    # entries does not work -- Plotly keys those on their footprint, so two
    # images sharing a subplot and geometry collapse into one and the drone
    # simply replaces the chip. Compositing also keeps the crown outlines
    # drawing above both layers, which `layer='above'` would not.
    fig.update_layout(images=[dict(
        source=chip_url,
        xref='x', yref='y',
        x=0, y=0, sizex=w, sizey=h,
        xanchor='left', yanchor='top',
        sizing='stretch',
        layer='below',
    )])

    return fig


_INDEX_HTML = '''<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>Planet Chip Viewer</title>
{%favicon%}
{%css%}
<style>
body { font-family: sans-serif; margin: 0; padding: 12px; }
.toolbar { display: flex; gap: 14px; align-items: center; margin-bottom: 8px; flex-wrap: wrap; }
.spinner { width: 16px; height: 16px; border: 2px solid #ddd; border-top-color: #3b82f6; border-radius: 50%; animation: spin 0.8s linear infinite; display: inline-block; vertical-align: middle; }
@keyframes spin { to { transform: rotate(360deg); } }
#header { font-family: monospace; font-size: 13px; color: #444; margin-left: auto; }
button { padding: 4px 10px; }
.hotkeys { font-size: 12px; color: #888; }
</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
<script>
// Keyboard shortcuts. Dash has no keydown input, and dash_extensions is not
// in environment.yml, so route keys to hidden buttons the callbacks listen on.
(function () {
    var KEYS = {
        'b': 'btn-blink', ' ': 'btn-blink',
        'ArrowLeft': 'btn-prev', 'ArrowRight': 'btn-next'
    };
    document.addEventListener('keydown', function (ev) {
        if (ev.metaKey || ev.ctrlKey || ev.altKey) { return; }
        var tag = (ev.target.tagName || '').toLowerCase();
        if (tag === 'input' || tag === 'textarea') { return; }
        var id = KEYS[ev.key];
        if (!id) { return; }
        var el = document.getElementById(id);
        if (!el) { return; }
        ev.preventDefault();
        el.click();
    });
})();
</script>
</body>
</html>
'''


def _load_crop(path, fracs):
    """Open a chip PNG, cropped to the fractional box (`None` = the whole
    tile). Fractions rather than pixels so the Planet chip, the 2x drone
    ortho and the OCM all crop to the same ground footprint."""
    with _PILImage.open(path) as im:
        im.load()
        if fracs is None:
            return im.copy()
        y0f, y1f, x0f, x1f = fracs
        w_im, h_im = im.size
        return im.crop((
            int(round(x0f * w_im)),
            int(round(y0f * h_im)),
            int(round(x1f * w_im)),
            int(round(y1f * h_im)),
        ))


def _cropped_rgba_array(path, fracs):
    return np.array(_load_crop(path, fracs).convert('RGBA'))


def composite_chip(planet_path, drone_path, fracs, drone_opacity, swipe):
    """The Planet chip with the drone ortho composited over it.

    Blending here rather than stacking two Plotly images is not a
    micro-optimization: Plotly keys its `layer='below'` images on their
    footprint, so two entries sharing a subplot and geometry collapse into
    one and the drone simply replaces the chip. It also keeps the wire
    payload to a single PNG and leaves the crown outlines drawing on top of
    both layers.

    `drone_opacity` blends the whole frame; `swipe < 1` instead wipes the
    drone in only left of that fraction of the width, for a hard edge to
    judge alignment against.
    """
    base = _load_crop(planet_path, fracs).convert('RGB')
    if drone_path is None or drone_opacity <= 0:
        return base

    over = _load_crop(drone_path, fracs).convert('RGB')
    if over.size != base.size:
        over = over.resize(base.size, _PILImage.BILINEAR)

    if swipe >= 1.0:
        return _PILImage.blend(base, over, drone_opacity)

    cut = int(round(max(swipe, 0.0) * base.size[0]))
    if cut <= 0:
        return base
    box = (0, 0, cut, base.size[1])
    out = base.copy()
    out.paste(_PILImage.blend(base.crop(box), over.crop(box), drone_opacity),
              box)
    return out


class _ImageCache:
    """Bounded FIFO cache of encoded PNG bytes, keyed by request parameters.

    The browser caches each distinct URL, so this only has to absorb the
    first hit per (scene, opacity, swipe) -- enough for blinking and for
    stepping back and forth through the set to feel immediate.
    """

    def __init__(self, maxsize=16):
        self._lock = threading.Lock()
        self._data = OrderedDict()
        self._maxsize = maxsize

    def get(self, key):
        with self._lock:
            return self._data.get(key)

    def put(self, key, value):
        with self._lock:
            self._data[key] = value
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)


def make_app(image_paths, cache, worker, split, size, min_instance_size,
             coreg_info=None, drone_paths=None, ocm_paths=None,
             crop_fracs=None):
    app = Dash(__name__)
    app.index_string = _INDEX_HTML

    drone_paths = drone_paths or {}
    ocm_paths = ocm_paths or {}
    crop_fracs = crop_fracs or {}
    planet_paths = {
        os.path.splitext(os.path.basename(p))[0]: p for p in image_paths
    }
    mask_paths = {
        scene: path[:-4] + '.mask.png'
        for scene, path in planet_paths.items()
        if os.path.exists(path[:-4] + '.mask.png')
    }
    has_model = worker is not None

    image_cache = _ImageCache()

    def _float_arg(name, default):
        try:
            return min(max(float(flask.request.args.get(name, default)), 0.0), 1.0)
        except (TypeError, ValueError):
            return default

    @app.server.route('/chip/<scene>')
    def _serve_chip(scene):
        scene_key = urllib.parse.unquote(scene)
        path = planet_paths.get(scene_key)
        if path is None:
            flask.abort(404)
        drone_op = _float_arg('drone', 0.0)
        swipe = _float_arg('swipe', 1.0)
        fracs = crop_fracs.get(scene_key)

        if drone_op <= 0 and fracs is None:
            return flask.send_file(path, mimetype='image/png',
                                   conditional=True, max_age=86400)

        key = (scene_key, round(drone_op, 3), round(swipe, 3))
        png = image_cache.get(key)
        if png is None:
            im = composite_chip(path, drone_paths.get(scene_key), fracs,
                                drone_op, swipe)
            buf = io.BytesIO()
            im.save(buf, format='PNG')
            png = buf.getvalue()
            image_cache.put(key, png)
        return flask.send_file(io.BytesIO(png), mimetype='image/png',
                               max_age=86400)

    filenames = [os.path.basename(p) for p in image_paths]

    def drone_params(show_drone, mode, opacity_val, swipe_val):
        """(drone opacity, swipe fraction) for the composite route. Swiping
        wipes a hard edge, so the covered part is opaque; blending honours
        the opacity slider across the whole frame."""
        if not show_drone:
            return 0.0, 1.0
        if mode == 'swipe':
            return 1.0, (0.5 if swipe_val is None else float(swipe_val))
        return (0.5 if opacity_val is None else float(opacity_val)), 1.0

    def chip_url(scene, drone_op, swipe):
        scene_q = urllib.parse.quote(scene, safe='')
        mtimes = [os.path.getmtime(planet_paths[scene])]
        if scene in drone_paths:
            mtimes.append(os.path.getmtime(drone_paths[scene]))
        v = int(max(mtimes))
        return (f'/chip/{scene_q}?v={v}'
                f'&drone={drone_op:.3f}&swipe={swipe:.3f}')

    def _respell_chip_url(src, drone_op, swipe):
        """Re-point an existing chip URL at new overlay parameters, keeping
        its scene and cache-busting stamp."""
        return (f'{src.split("&drone=")[0]}'
                f'&drone={drone_op:.3f}&swipe={swipe:.3f}')

    SHOW = {'display': 'inline-block'}
    HIDE = {'display': 'none'}
    SHOW_FLEX = {'display': 'inline-flex', 'alignItems': 'center'}

    app.layout = html.Div([
        dcc.Store(id='store-current-idx', data=0),
        dcc.Store(id='store-last-rendered', data=None),
        dcc.Interval(id='poll-cache', interval=500, disabled=not has_model),
        # Clicked by the keydown handler in _INDEX_HTML; never shown.
        html.Button(id='btn-blink', n_clicks=0, style=HIDE),
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
            html.Div(id='gt-toggle-slot', style=SHOW, children=[
                dcc.Checklist(
                    id='toggle-gt',
                    options=[{'label': ' Show ground truth', 'value': 'gt'}],
                    value=['gt'], inline=True,
                ),
            ]),
            html.Div(id='filter-slot',
                     style=SHOW if has_model else HIDE, children=[
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
            ]),
            html.Div(id='pred-toggle-slot',
                     style=SHOW if has_model else HIDE, children=[
                dcc.Checklist(
                    id='toggle-pred',
                    options=[{'label': ' Show predictions', 'value': 'pred'}],
                    value=['pred'], inline=True,
                    style=SHOW,
                ),
                html.Span(id='pred-spinner', style=HIDE, children=[
                    html.Span(className='spinner'),
                    html.Span(' computing predictions…',
                              style={'marginLeft': '6px'}),
                ]),
            ]),
            html.Div(id='drone-toggle-slot', style=HIDE, children=[
                dcc.Checklist(
                    id='toggle-drone',
                    options=[{'label': ' Show drone overlay', 'value': 'drone'}],
                    value=[], inline=True,
                    style=SHOW,
                ),
            ]),
            html.Div(id='ocm-toggle-slot', style=HIDE, children=[
                dcc.Checklist(
                    id='toggle-ocm',
                    options=[{'label': ' Show cloud mask', 'value': 'ocm'}],
                    value=[], inline=True,
                    style=SHOW,
                ),
            ]),
            html.Div(id='drone-mode-slot', style=HIDE, children=[
                dcc.RadioItems(
                    id='drone-mode',
                    options=[{'label': ' Blend', 'value': 'blend'},
                             {'label': ' Swipe', 'value': 'swipe'}],
                    value='blend', inline=True,
                ),
            ]),
            html.Div(id='overlay-opacity-slot', style=HIDE, children=[
                html.Span('Overlay opacity:',
                          style={'fontSize': '13px', 'marginRight': '8px'}),
                html.Div(
                    dcc.Slider(
                        id='slider-overlay-opacity',
                        min=0, max=1, step=0.05, value=0.5,
                        marks=None,
                        tooltip={'always_visible': False,
                                 'placement': 'bottom'},
                    ),
                    style={'width': '160px', 'display': 'inline-block',
                           'verticalAlign': 'middle'},
                ),
            ]),
            html.Div(id='swipe-slot', style=HIDE, children=[
                html.Span('Swipe:',
                          style={'fontSize': '13px', 'marginRight': '8px'}),
                html.Div(
                    dcc.Slider(
                        id='slider-swipe',
                        min=0, max=1, step=0.01, value=0.5,
                        marks=None, updatemode='mouseup',
                        tooltip={'always_visible': False,
                                 'placement': 'bottom'},
                    ),
                    style={'width': '200px', 'display': 'inline-block',
                           'verticalAlign': 'middle'},
                ),
            ]),
            html.Span('← → navigate  ·  b blinks the drone layer',
                      className='hotkeys'),
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
        Output('gt-toggle-slot', 'style'),
        Output('drone-toggle-slot', 'style'),
        Output('ocm-toggle-slot', 'style'),
        Output('drone-mode-slot', 'style'),
        Output('overlay-opacity-slot', 'style'),
        Output('swipe-slot', 'style'),
        Output('store-last-rendered', 'data'),
        Input('store-current-idx', 'data'),
        Input('poll-cache', 'n_intervals'),
        Input('toggle-ocm', 'value'),
        Input('drone-mode', 'value'),
        State('store-last-rendered', 'data'),
        State('toggle-gt', 'value'),
        State('toggle-pred', 'value'),
        State('toggle-drone', 'value'),
        State('dd-filter', 'value'),
        State('slider-overlay-opacity', 'value'),
        State('slider-swipe', 'value'),
    )
    def render(idx, _tick, ocm_val, mode, last, gt_val, pred_val, drone_val,
               filter_val, opacity_val, swipe_val):
        idx = idx or 0
        path = image_paths[idx]
        scene = os.path.splitext(os.path.basename(path))[0]
        if has_model and ctx.triggered_id == 'store-current-idx':
            worker.bump(path)

        pred = cache.get(path) if has_model else None
        pred_available = pred is not None

        drone_available = scene in drone_paths
        ocm_available = scene in ocm_paths
        mask_available = scene in mask_paths

        show_drone = 'drone' in (drone_val or []) and drone_available
        show_ocm = 'ocm' in (ocm_val or []) and ocm_available
        mode = mode or 'blend'
        drone_op, swipe = drone_params(show_drone, mode, opacity_val, swipe_val)

        # Everything that changes the figure's *structure*. Layer visibility
        # and opacity are patched instead, so they are deliberately absent:
        # this is what keeps the 500 ms cache poll from thrashing the figure.
        state = {
            'path': path,
            'pred_available': pred_available,
            'drone_available': drone_available,
            'ocm_available': ocm_available,
            'mask_available': mask_available,
            'show_ocm': show_ocm,
            'mode': mode,
        }
        if last == state:
            raise dash.exceptions.PreventUpdate

        img_shape, gt_masks = load_display_shape_and_gt(
            path, split=split, size=size,
            min_instance_size=min_instance_size,
        )

        ocm_array = None
        if show_ocm:
            ocm_array = _cropped_rgba_array(
                ocm_paths[scene], crop_fracs.get(scene),
            )

        show_gt = 'gt' in (gt_val or []) and mask_available
        show_pred = 'pred' in (pred_val or []) and pred_available
        coreg_meta = (coreg_info or {}).get(scene)
        ocm_opacity = 0.5 if opacity_val is None else float(opacity_val)
        fig = build_figure(img_shape, gt_masks, pred, show_gt, show_pred,
                           chip_url=chip_url(scene, drone_op, swipe),
                           filter_val=filter_val or 'all',
                           ocm_array=ocm_array, show_ocm=show_ocm,
                           coreg_meta=coreg_meta,
                           ocm_opacity=ocm_opacity,
                           swipe_frac=swipe,
                           show_divider=show_drone and mode == 'swipe')
        fig.update_layout(uirevision=scene)

        if not has_model:
            spinner_style, toggle_style = HIDE, HIDE
        elif pred_available:
            spinner_style, toggle_style = HIDE, SHOW
        else:
            spinner_style, toggle_style = SHOW, HIDE

        any_overlay = drone_available or ocm_available
        return (
            fig,
            spinner_style,
            toggle_style,
            SHOW if mask_available else HIDE,
            SHOW if drone_available else HIDE,
            SHOW if ocm_available else HIDE,
            SHOW if drone_available else HIDE,
            SHOW_FLEX if (any_overlay and mode == 'blend') else HIDE,
            SHOW_FLEX if (drone_available and mode == 'swipe') else HIDE,
            state,
        )

    # `b` / spacebar blinks the drone layer by flipping its checkbox; the
    # opacity callback below turns that into a one-number Patch.
    app.clientside_callback(
        """
        function (n, value) {
            if (!n) { return window.dash_clientside.no_update; }
            var on = (value || []).indexOf('drone') !== -1;
            return on ? [] : ['drone'];
        }
        """,
        Output('toggle-drone', 'value'),
        Input('btn-blink', 'n_clicks'),
        State('toggle-drone', 'value'),
        prevent_initial_call=True,
    )

    @app.callback(
        Output('viewer', 'figure', allow_duplicate=True),
        Input('toggle-drone', 'value'),
        Input('slider-overlay-opacity', 'value'),
        Input('slider-swipe', 'value'),
        State('drone-mode', 'value'),
        State('store-last-rendered', 'data'),
        State('viewer', 'figure'),
        prevent_initial_call=True,
    )
    def on_overlay(drone_val, opacity_val, swipe_val, mode, last, fig):
        """Re-point the composite at new overlay parameters and move the wipe
        edge. The figure is patched, never rebuilt, so blinking the drone
        layer costs one cached image fetch."""
        if not fig:
            raise dash.exceptions.PreventUpdate
        images = (fig.get('layout') or {}).get('images') or []
        if not images:
            raise dash.exceptions.PreventUpdate

        mode = mode or 'blend'
        show_drone = ('drone' in (drone_val or [])
                      and bool((last or {}).get('drone_available')))
        drone_op, swipe = drone_params(show_drone, mode, opacity_val, swipe_val)
        ocm_opacity = 0.5 if opacity_val is None else float(opacity_val)

        patched = Patch()
        patched['layout']['images'][0]['source'] = _respell_chip_url(
            images[0].get('source') or '', drone_op, swipe,
        )

        x_range = ((fig.get('layout') or {}).get('xaxis') or {}).get('range')
        w = x_range[1] if x_range else 1.0
        for i, trace in enumerate(fig.get('data') or []):
            if trace.get('type') == 'image':
                patched['data'][i]['opacity'] = ocm_opacity
            elif trace.get('name') == 'swipe-divider':
                patched['data'][i]['x'] = [swipe * w, swipe * w]
                patched['data'][i]['visible'] = bool(
                    show_drone and mode == 'swipe')
        return patched

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
@click.argument('imagedir')
@click.option('--model', 'modelfile', default=None,
              type=click.Path(exists=True),
              help='Mask R-CNN checkpoint. Without it the prediction layer is '
                   'omitted and the app is a drone/Planet coregistration '
                   'reviewer.')
@click.option('--score-thresh', default=0.5, type=float,
              help='Minimum score for a prediction to be kept.')
@click.option('--mask-thresh', default=0.5, type=float,
              help='Threshold applied to soft Mask R-CNN mask logits.')
@click.option('--split', default='right',
              type=click.Choice(['left', 'right', 'whole']),
              help='Which part of each tile to visualize (test=right, '
                   'whole=the entire chip).')
@click.option('--size', default=512, type=int)
@click.option('--min-instance-size', default=4, type=int,
              help='Minimum ground-truth instance size in pixels. Ignored '
                   'with --model, which takes it from the checkpoint.')
@click.option('--iou-thresh', default=0.25, type=float,
              help='IoU threshold for TP/FP/FN matching.')
@click.option('--host', default='127.0.0.1', type=str)
@click.option('--port', default=8050, type=int)
@click.option('--logfile', default=None, type=click.Path(exists=False),
              help='Path to coreg_log.json (default: imagedir/coreg_log.json).')
def main(imagedir, modelfile, score_thresh, mask_thresh, split,
         size, min_instance_size, iou_thresh, host, port, logfile):

    SIDECAR_SUFFIXES = ('.mask.png', '.drone.png', '.ocm.png')
    image_paths = sorted(glob(os.path.join(imagedir, '*.png')))
    image_paths = [p for p in image_paths if not p.endswith(SIDECAR_SUFFIXES)]
    if not image_paths:
        raise click.ClickException(f'no Planet *.png tiles in {imagedir}')

    # 'whole' consumes the tile as-is, so the left/right window constraint
    # does not apply and nothing needs to be dropped for being too small.
    if split != 'whole':
        def _meets_size(p):
            with _PILImage.open(p) as im:
                h, w = im.height, im.width
            return h >= size and w >= 2 * size

        skipped = [p for p in image_paths if not _meets_size(p)]
        image_paths = [p for p in image_paths if _meets_size(p)]
        if skipped:
            print(f'Skipping {len(skipped)} image(s) smaller than '
                  f'{size}x{2*size}: '
                  + ', '.join(os.path.basename(p) for p in skipped))
        if not image_paths:
            raise click.ClickException(
                f'no images meet the minimum size ({size}x{2*size})')

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
        # Every served layer is cropped by the same *fractional* box, so the
        # Planet chip, the 2x drone ortho and the OCM line up despite their
        # different resolutions. The Planet chip needs it too, whether or not
        # it has sidecars, since the axes are sized to the crop.
        if split != 'whole':
            with _PILImage.open(p) as im:
                w_p, h_p = im.size
            row_start, row_end, col_start, col_end = _split_window(
                h_p, w_p, split, size,
            )
            crop_fracs[scene] = (
                row_start / h_p, row_end / h_p,
                col_start / w_p, col_end / w_p,
            )

    cache = None
    worker = None
    device = None
    use_ocm_masks = False
    if modelfile is not None:
        device = select_device()
        ckpt = torch.load(modelfile, map_location='cpu', weights_only=False)
        model = ckpt['model']
        min_instance_size = ckpt['params']['min_instance_size']
        use_ocm_masks = ckpt['params'].get('use_ocm_masks', False)
        channel_kinds = ckpt['params'].get('channel_kinds')

        cache = PredictionCache()
        worker = InferenceWorker(
            model=model, device=device, cache=cache,
            image_paths=image_paths, split=split, size=size,
            score_thresh=score_thresh, mask_thresh=mask_thresh,
            iou_thresh=iou_thresh, min_instance_size=min_instance_size,
            use_ocm_masks=use_ocm_masks, channel_kinds=channel_kinds,
        )
        worker.start()
        atexit.register(worker.stop)

    app = make_app(image_paths, cache, worker, split=split, size=size,
                   min_instance_size=min_instance_size,
                   coreg_info=coreg_info, drone_paths=drone_paths,
                   ocm_paths=ocm_paths, crop_fracs=crop_fracs)
    if modelfile is None:
        model_desc = 'no model'
    else:
        model_desc = (f'device={device}, '
                      f"ocm_filtering={'on' if use_ocm_masks else 'off'}")
    print(f'Serving on http://{host}:{port} ({model_desc}, '
          f'{len(image_paths)} images, {len(drone_paths)} drone overlays, '
          f'{len(ocm_paths)} cloud masks)')
    app.run(host=host, port=port, debug=False)


if __name__ == '__main__':
    main()
