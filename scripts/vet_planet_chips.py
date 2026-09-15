#!/usr/bin/env python
"""Rate Planet training chips Poor / Fair / Good, to decide which belong in
the training set at all.

A local replacement for the Labelbox "Quality" radio that produced
`labels/20260706_planet_vetting.ndjson`. It reads nothing but the Planet RGB
previews and their cloud-mask sidecars -- no tif, no crown mask, no drone
ortho, no torch -- so it starts in a second and stays responsive over a set of
several hundred chips. Use `deploy_planet_image_maskrcnn_interactive.py`
instead when the question is whether the drone, the chip and the crown mask
line up.

A contact sheet shows every chip at thumbnail size, bordered by its rating;
clicking one opens it full-resolution with pan and zoom. Ratings are written
to `vetting.json` after every click, so the session is resumable and survives
a crash. `copy_good_planet_vetting.py` reads that file directly.

Usage:
    python scripts/vet_planet_chips.py <imagedir> [-o vetting.json]
"""
import io
import json
import os
import threading
import urllib.parse
from collections import OrderedDict
from datetime import datetime
from glob import glob

import click
import flask
import plotly.graph_objects as go
from PIL import Image

import dash
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html

# Sidecars of a chip, not chips in their own right.
SIDECAR_SUFFIXES = ('.mask.png', '.drone.png', '.ocm.png')

# Worst to best, matching the Labelbox radio and QUALITY_RANK in
# copy_good_planet_vetting.py, so --min-quality keeps its meaning and old
# NDJSON exports stay comparable with the files written here.
QUALITIES = ('Poor', 'Fair', 'Good')

RATING_COLOR = {
    'Poor': '#dc2626',
    'Fair': '#d97706',
    'Good': '#16a34a',
    None: '#d4d4d8',
}

FILTERS = ('all', 'unrated') + QUALITIES

KEY_QUALITY = {'key-poor': 'Poor', 'key-fair': 'Fair', 'key-good': 'Good'}
DETAIL_BUTTONS = {'det-poor': 'Poor', 'det-fair': 'Fair', 'det-good': 'Good'}


class Ratings:
    """The vetting decisions, persisted after every change.

    Keyed by chip *filename* rather than stem, matching Labelbox's
    `data_row.external_id`, so a file written here drops into the same place
    in the pipeline as an export did.
    """

    def __init__(self, path, imagedir):
        self.path = path
        self.imagedir = imagedir
        self._lock = threading.Lock()
        self._data = {}
        if os.path.exists(path):
            with open(path) as f:
                self._data = json.load(f).get('ratings', {})

    def get(self, filename):
        return self._data.get(filename, {})

    def quality(self, filename):
        return self._data.get(filename, {}).get('quality')

    def note(self, filename):
        return self._data.get(filename, {}).get('note', '')

    def counts(self):
        tally = {q: 0 for q in QUALITIES}
        for rec in self._data.values():
            if rec.get('quality') in tally:
                tally[rec['quality']] += 1
        return tally

    def set(self, filename, quality=None, note=None):
        """Record a rating and/or a note. `quality=None` leaves the rating
        alone; passing the rating a chip already has clears it, so a mis-click
        is undone by repeating it."""
        with self._lock:
            rec = dict(self._data.get(filename, {}))
            if quality is not None:
                rec['quality'] = (None if rec.get('quality') == quality
                                  else quality)
            if note is not None:
                rec['note'] = note
            if not rec.get('quality') and not rec.get('note'):
                self._data.pop(filename, None)
            else:
                rec['updated'] = datetime.now().isoformat(timespec='seconds')
                self._data[filename] = rec
            self._write()

    def _write(self):
        """Atomic replace, as apply_drone_labels_coreg.write_coreg_log does:
        the file is rewritten on every click and must never be left half
        written if the process dies mid-session."""
        payload = {
            'version': 1,
            'imagedir': os.path.abspath(self.imagedir),
            'updated': datetime.now().isoformat(timespec='seconds'),
            'ratings': self._data,
        }
        tmp = f'{self.path}.tmp'
        with open(tmp, 'w') as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp, self.path)


class ThumbnailCache:
    """Bounded FIFO cache of encoded thumbnail PNGs."""

    def __init__(self, maxsize=512):
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


_INDEX_HTML = '''<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>Planet Chip Vetting</title>
{%favicon%}
{%css%}
<style>
body { font-family: sans-serif; margin: 0; padding: 12px; }
.toolbar { display: flex; gap: 14px; align-items: center; margin-bottom: 10px; flex-wrap: wrap; }
.grid { display: grid; gap: 10px; }
.cell { border: 3px solid #d4d4d8; border-radius: 4px; padding: 4px; background: #fafafa; }
.cell img { width: 100%; display: block; cursor: zoom-in; border-radius: 2px; }
.cap { font-family: monospace; font-size: 11px; color: #555; margin-top: 3px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.rate { display: flex; gap: 4px; margin-top: 4px; }
.rate button { flex: 1; padding: 2px 0; font-size: 12px; cursor: pointer; }
.hotkeys { font-size: 12px; color: #888; }
#progress { font-family: monospace; font-size: 13px; color: #444; margin-left: auto; }
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
<script>
// Keyboard shortcuts. Dash has no keydown input and dash_extensions is not in
// environment.yml, so route keys to hidden buttons the callbacks listen on.
(function () {
    var KEYS = {
        '1': 'key-poor', '2': 'key-fair', '3': 'key-good',
        'ArrowLeft': 'btn-prev', 'ArrowRight': 'btn-next',
        'Escape': 'btn-back'
    };
    document.addEventListener('keydown', function (ev) {
        if (ev.metaKey || ev.ctrlKey || ev.altKey) { return; }
        var tag = (ev.target.tagName || '').toLowerCase();
        if (tag === 'input' || tag === 'textarea') { return; }
        var el = document.getElementById(KEYS[ev.key]);
        if (!el) { return; }
        ev.preventDefault();
        el.click();
    });
})();
</script>
</body>
</html>
'''


def rate_style(quality, active):
    return {
        'background': RATING_COLOR[quality] if active else '#fff',
        'color': '#fff' if active else '#333',
        'border': f'1px solid {RATING_COLOR[quality]}',
        'borderRadius': '3px',
    }


def make_app(image_paths, ratings, cols, thumb_width):
    app = Dash(__name__)
    app.index_string = _INDEX_HTML

    filenames = [os.path.basename(p) for p in image_paths]
    paths_by_name = dict(zip(filenames, image_paths))
    ocm_by_name = {}
    for name, path in paths_by_name.items():
        ocm = path[:-4] + '.ocm.png'
        if os.path.exists(ocm):
            ocm_by_name[name] = ocm

    thumbs = ThumbnailCache()
    grid_template = {'gridTemplateColumns': f'repeat({cols}, 1fr)'}

    @app.server.route('/thumb/<name>')
    def _serve_thumb(name):
        key = urllib.parse.unquote(name)
        path = paths_by_name.get(key)
        if path is None:
            flask.abort(404)
        png = thumbs.get(key)
        if png is None:
            with Image.open(path) as im:
                im.draft('RGB', (thumb_width, thumb_width))
                im = im.convert('RGB')
                im.thumbnail((thumb_width, thumb_width * 4), Image.BILINEAR)
                buf = io.BytesIO()
                im.save(buf, format='PNG')
            png = buf.getvalue()
            thumbs.put(key, png)
        return flask.send_file(io.BytesIO(png), mimetype='image/png',
                               max_age=86400)

    @app.server.route('/chip/<name>')
    def _serve_chip(name):
        key = urllib.parse.unquote(name)
        path = paths_by_name.get(key)
        if path is None:
            flask.abort(404)
        ocm_path = ocm_by_name.get(key)
        if not flask.request.args.get('ocm') or ocm_path is None:
            return flask.send_file(path, mimetype='image/png',
                                   conditional=True, max_age=86400)
        # Composited here rather than stacked as a second Plotly image:
        # Plotly keys below-layer images on their footprint, so two sharing a
        # subplot collapse into one and only the first is drawn.
        with Image.open(path) as base, Image.open(ocm_path) as over:
            out = Image.alpha_composite(base.convert('RGBA'),
                                        over.convert('RGBA')).convert('RGB')
        buf = io.BytesIO()
        out.save(buf, format='PNG')
        buf.seek(0)
        return flask.send_file(buf, mimetype='image/png', max_age=86400)

    def visible_names(filter_val):
        if filter_val == 'unrated':
            return [n for n in filenames if not ratings.quality(n)]
        if filter_val in QUALITIES:
            return [n for n in filenames if ratings.quality(n) == filter_val]
        return list(filenames)

    def progress_text():
        tally = ratings.counts()
        rated = sum(tally.values())
        parts = ', '.join(f'{tally[q]} {q}' for q in reversed(QUALITIES))
        return f'{rated}/{len(filenames)} rated — {parts}'

    def cell_style(name, filter_val):
        style = {'borderColor': RATING_COLOR[ratings.quality(name)]}
        if name not in visible_names(filter_val):
            style['display'] = 'none'
        return style

    def detail_figure(name, show_ocm):
        with Image.open(paths_by_name[name]) as im:
            w, h = im.size
        url = f'/chip/{urllib.parse.quote(name, safe="")}'
        if show_ocm:
            url += '?ocm=1'
        fig = go.Figure()
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(range=[0, w], showgrid=False, zeroline=False,
                       visible=False, constrain='domain'),
            yaxis=dict(range=[h, 0], showgrid=False, zeroline=False,
                       visible=False, scaleanchor='x', scaleratio=1),
            dragmode='pan',
            showlegend=False,
            images=[dict(
                source=url,
                xref='x', yref='y', x=0, y=0, sizex=w, sizey=h,
                xanchor='left', yanchor='top', sizing='stretch',
                layer='below',
            )],
            uirevision=name,
        )
        return fig

    # The grid is built once, here, and never rebuilt: a callback that
    # replaced its children would re-fire on its own pattern-matching inputs.
    # Rating a chip and changing the filter only restyle existing cells.
    cells = []
    for name in filenames:
        quality = ratings.quality(name)
        cells.append(html.Div(
            id={'type': 'cell', 'name': name},
            className='cell',
            style={'borderColor': RATING_COLOR[quality]},
            children=[
                html.Img(src=f'/thumb/{urllib.parse.quote(name, safe="")}',
                         id={'type': 'open', 'name': name}, n_clicks=0),
                html.Div(name, className='cap', title=name),
                html.Div(className='rate', children=[
                    html.Button(
                        q[0], id={'type': 'rate', 'name': name, 'q': q},
                        n_clicks=0, title=f'{q} ({QUALITIES.index(q) + 1})',
                        style=rate_style(q, quality == q),
                    )
                    for q in QUALITIES
                ]),
            ],
        ))

    app.layout = html.Div([
        dcc.Store(id='store-view', data={'mode': 'grid', 'name': None}),
        # Clicked by the keydown handler in _INDEX_HTML; never shown.
        html.Div(style={'display': 'none'}, children=[
            html.Button(id=key, n_clicks=0) for key in KEY_QUALITY
        ]),
        html.Div(className='toolbar', children=[
            html.Div(id='nav-slot', style={'display': 'none'}, children=[
                html.Button('◀ Prev', id='btn-prev', n_clicks=0),
                html.Button('Next ▶', id='btn-next', n_clicks=0),
                html.Button('↑ Back to grid', id='btn-back', n_clicks=0),
            ]),
            dcc.Dropdown(
                id='dd-filter',
                options=[{'label': f'Show: {f}', 'value': f} for f in FILTERS],
                value='all', clearable=False,
                style={'minWidth': '170px'},
            ),
            html.Div(id='ocm-slot', style={'display': 'none'}, children=[
                dcc.Checklist(
                    id='toggle-ocm',
                    options=[{'label': ' Cloud mask', 'value': 'ocm'}],
                    value=[], inline=True,
                ),
            ]),
            html.Span('1 Poor · 2 Fair · 3 Good · '
                      '← → navigate · esc back',
                      className='hotkeys'),
            html.Div(id='progress', children=progress_text()),
        ]),
        html.Div(id='grid', className='grid', style=grid_template,
                 children=cells),
        html.Div(id='detail', style={'display': 'none'}, children=[
            html.Div(id='detail-header', style={
                'fontFamily': 'monospace', 'fontSize': '13px',
                'marginBottom': '6px',
            }),
            # responsive: the graph is first laid out inside a
            # display:none panel, so without it Plotly keeps the zero
            # width it measured there.
            dcc.Graph(id='detail-fig', style={'height': '72vh'},
                      responsive=True,
                      config={'scrollZoom': True, 'displaylogo': False}),
            html.Div(style={'display': 'flex', 'gap': '10px',
                            'alignItems': 'center', 'marginTop': '8px'},
                     children=[
                html.Div(className='rate', style={'width': '280px'}, children=[
                    html.Button(q, id=bid, n_clicks=0,
                                style=rate_style(q, False))
                    for bid, q in DETAIL_BUTTONS.items()
                ]),
                dcc.Input(id='detail-note', type='text', debounce=True,
                          placeholder='note (optional)', value='',
                          style={'flex': '1', 'padding': '5px'}),
            ]),
        ]),
    ])

    @app.callback(
        Output('store-view', 'data'),
        Input({'type': 'open', 'name': ALL}, 'n_clicks'),
        Input('btn-back', 'n_clicks'),
        Input('btn-prev', 'n_clicks'),
        Input('btn-next', 'n_clicks'),
        State('store-view', 'data'),
        State('dd-filter', 'value'),
        prevent_initial_call=True,
    )
    def on_nav(_open_clicks, _back, _prev, _next, view, filter_val):
        trigger = ctx.triggered_id
        view = view or {'mode': 'grid', 'name': None}

        if isinstance(trigger, dict) and trigger.get('type') == 'open':
            if not ctx.triggered[0]['value']:
                raise dash.exceptions.PreventUpdate
            return {'mode': 'detail', 'name': trigger['name']}

        if trigger == 'btn-back':
            return {'mode': 'grid', 'name': view.get('name')}

        # Step through whatever the filter is showing, so a pass over the
        # unrated chips stays a pass over the unrated chips.
        names = visible_names(filter_val or 'all') or list(filenames)
        if view.get('mode') != 'detail':
            raise dash.exceptions.PreventUpdate
        try:
            idx = names.index(view.get('name'))
        except ValueError:
            idx = 0
        idx = (max(0, idx - 1) if trigger == 'btn-prev'
               else min(len(names) - 1, idx + 1))
        return {'mode': 'detail', 'name': names[idx]}

    @app.callback(
        Output('progress', 'children'),
        Output({'type': 'cell', 'name': ALL}, 'style'),
        Output({'type': 'rate', 'name': ALL, 'q': ALL}, 'style'),
        Output('det-poor', 'style'),
        Output('det-fair', 'style'),
        Output('det-good', 'style'),
        Output('detail-note', 'value'),
        Input({'type': 'rate', 'name': ALL, 'q': ALL}, 'n_clicks'),
        Input('key-poor', 'n_clicks'),
        Input('key-fair', 'n_clicks'),
        Input('key-good', 'n_clicks'),
        Input('dd-filter', 'value'),
        Input('store-view', 'data'),
    )
    def on_rate(_rate_clicks, _poor, _fair, _good, filter_val, view):
        trigger = ctx.triggered_id
        view = view or {'mode': 'grid', 'name': None}
        current = view.get('name')

        if isinstance(trigger, dict) and trigger.get('type') == 'rate':
            if ctx.triggered[0]['value']:
                ratings.set(trigger['name'], quality=trigger['q'])
        elif trigger in KEY_QUALITY and current and ctx.triggered[0]['value']:
            ratings.set(current, quality=KEY_QUALITY[trigger])

        filter_val = filter_val or 'all'
        # Drive the outputs off ctx.outputs_list rather than assuming the
        # order Dash matched the wildcards in.
        cell_ids = [o['id']['name'] for o in ctx.outputs_list[1]]
        rate_ids = [(o['id']['name'], o['id']['q'])
                    for o in ctx.outputs_list[2]]
        current_q = ratings.quality(current) if current else None
        return (
            progress_text(),
            [cell_style(n, filter_val) for n in cell_ids],
            [rate_style(q, ratings.quality(n) == q) for n, q in rate_ids],
            rate_style('Poor', current_q == 'Poor'),
            rate_style('Fair', current_q == 'Fair'),
            rate_style('Good', current_q == 'Good'),
            ratings.note(current) if current else '',
        )

    @app.callback(
        Output('store-view', 'data', allow_duplicate=True),
        Input('det-poor', 'n_clicks'),
        Input('det-fair', 'n_clicks'),
        Input('det-good', 'n_clicks'),
        State('store-view', 'data'),
        prevent_initial_call=True,
    )
    def on_detail_rate(_p, _f, _g, view):
        """Rate from the detail view. Writing the rating and then touching the
        view store lets on_rate restyle everything through its single set of
        outputs, instead of two callbacks fighting over the same styles."""
        view = view or {}
        name = view.get('name')
        if not name or not ctx.triggered[0]['value']:
            raise dash.exceptions.PreventUpdate
        ratings.set(name, quality=DETAIL_BUTTONS[ctx.triggered_id])
        return dict(view)

    @app.callback(
        Output('progress', 'children', allow_duplicate=True),
        Input('detail-note', 'value'),
        State('store-view', 'data'),
        prevent_initial_call=True,
    )
    def on_note(note, view):
        name = (view or {}).get('name')
        if not name or (note or '') == ratings.note(name):
            raise dash.exceptions.PreventUpdate
        ratings.set(name, note=note or '')
        return progress_text()

    @app.callback(
        Output('grid', 'style'),
        Output('detail', 'style'),
        Output('nav-slot', 'style'),
        Output('ocm-slot', 'style'),
        Output('detail-fig', 'figure'),
        Output('detail-header', 'children'),
        Input('store-view', 'data'),
        Input('toggle-ocm', 'value'),
    )
    def on_view(view, ocm_val):
        view = view or {'mode': 'grid', 'name': None}
        hidden = {'display': 'none'}
        if view.get('mode') != 'detail' or not view.get('name'):
            return (grid_template, hidden, hidden, hidden,
                    dash.no_update, dash.no_update)

        name = view['name']
        idx = filenames.index(name)
        header = f'{idx + 1} / {len(filenames)}  —  {name}'
        ocm_style = ({'display': 'inline-block'} if name in ocm_by_name
                     else hidden)
        return (hidden, {'display': 'block'},
                {'display': 'flex', 'gap': '8px'}, ocm_style,
                detail_figure(name, 'ocm' in (ocm_val or [])), header)

    return app


@click.command()
@click.argument('imagedir', type=click.Path(exists=True, file_okay=False))
@click.option('-o', '--out', default=None, type=click.Path(),
              help='Where to write the ratings '
                   '(default: imagedir/vetting.json).')
@click.option('--cols', default=5, type=int, help='Contact-sheet columns.')
@click.option('--thumb', default=320, type=int,
              help='Thumbnail width in pixels.')
@click.option('--host', default='127.0.0.1', type=str)
@click.option('--port', default=8051, type=int)
def main(imagedir, out, cols, thumb, host, port):
    image_paths = sorted(glob(os.path.join(imagedir, '*.png')))
    image_paths = [p for p in image_paths if not p.endswith(SIDECAR_SUFFIXES)]
    if not image_paths:
        raise click.ClickException(f'no Planet *.png chips in {imagedir}')

    if out is None:
        out = os.path.join(imagedir, 'vetting.json')
    ratings = Ratings(out, imagedir)

    app = make_app(image_paths, ratings, cols=cols, thumb_width=thumb)
    rated = sum(ratings.counts().values())
    print(f'Serving on http://{host}:{port} ({len(image_paths)} chips, '
          f'{rated} already rated)')
    print(f'Ratings: {out}')
    app.run(host=host, port=port, debug=False)


if __name__ == '__main__':
    main()
