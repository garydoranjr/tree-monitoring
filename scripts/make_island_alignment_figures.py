#!/usr/bin/env python
"""Slide figures explaining the whole-island Planet alignment correction
(island_coreg.py, island_coreg_diagnose.py; docs/island_coregistration.md).

Subcommands (outputs in OUTDIR; all sized for a 12.5 x 6 in slide):

  problem    one example scene: its tile displacement field over the island, the
             terrain, and displacement against elevation.
  diagnosis  how the cause was found: triangle closure (not false matches), our RPC
             ortho against Planet's (not the orthorectification), and the elevation
             term against the drone mosaic (real geometry).
  steps      the procedure on the example scene: match one tile, the field of all
             tiles, the fitted model, the tie to the drone mosaic, the per-pixel
             correction.
  method     the example scene's field split into affine part, elevation part and
             what is left.
  result     residual by model on plateau and lower tiles, and the offset from the
             drone mosaic before and after the correction.
  compare    short MP4/GIF over the 50ha plot: drone mosaic, Planet as delivered and
             Planet corrected, for the scenes that were furthest off, with fixed
             crown outlines as reference marks.
  geometry   the observation geometry of the example scene: the reported and the
             RPC line of sight from orbit, at the ground, and in map view.
  tilt       all scenes with an RPC: reported against RPC line of sight, the
             elevation term against the RPC tilt, and the size of the tilt.
  rpcmap     the example scene's elevation-related displacement over the island as
             measured from imagery and as predicted from its RPC and metadata.
  model3d    a self-contained interactive 3-D page of the same geometry at three
             scales (orbit, island, one tree crown) for a handful of scenes.
  all        everything.

Typical usage:
    python scripts/make_island_alignment_figures.py all
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import click  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from rasterio.windows import from_bounds  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import island_coreg as ic  # noqa: E402
import island_coreg_diagnose as dg  # noqa: E402

DEFAULT_OUT = ic.FLOWER / 'figs' / '20261118_updates'
FINAL = 'Ra'  # the correction that is applied: model R (affine or shift + RPC tilt) anchored to the drone mosaic
CP = 'r'      # prefix of its coefficient columns in scene_models.csv
W = 12.5  # slide width, inches
RES = ic.RES
CROWNS = ic.FLOWER / 'crowns_20240803' / 'BCI_50ha_crownmap_timeseries.shp'
COMPARE_WINDOW = (626200, 1011950, 626500, 1012250)  # 300 m inside the 50ha plot, on the plateau
C_HIGH, C_LOW = '#8c564b', '#7f9f7f'
C_BEFORE, C_AFTER = '#d95f02', '#1b9e77'
C_OUTLINE = '#ffe11a'
FS, FS_T, FS_S = 9.5, 11, 8


@click.group()
def cli():
    pass


out_option = click.option('--outdir', type=click.Path(file_okay=False, path_type=Path),
                          default=DEFAULT_OUT, show_default=True)


def save(fig, outdir, name, caption):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'{name}.{ext}', dpi=300)
    (outdir / f'{name}_caption.md').write_text(' '.join(caption.split()) + '\n')
    plt.close(fig)
    click.echo(f'wrote {outdir / name}.png')


def models_table():
    sm = pd.read_csv(ic.RESULTS / 'scene_models.csv', parse_dates=['acquired'])
    return sm[(sm.band == 'both') & sm.med_H.notna() & sm.med_R.notna()].set_index('scene_id')


def master_k():
    a = json.loads((ic.RESULTS / 'master_abs.json').read_text())
    return -a['median']['h_ch_y'] * RES * 100, a['median']['h_ch_x'] * RES * 100, a['n']


def example_scene():
    """A fully clear video scene with a large elevation term (relative to the ground)."""
    sm = models_table()
    kn, ke, _ = master_k()
    sc = ic.load_scenes(ic.RESULTS).set_index('scene_id')
    sm = sm[sm.index.isin(sc[sc.use_video].index) & (sm.fraction_clear >= 0.97)].copy()
    sm['k'] = np.hypot(sm.k_north_m100 + kn, sm.k_east_m100 + ke)
    sm = sm[sm.n_tiles >= sm.n_tiles.quantile(0.5)]
    return sm.sort_values('k', ascending=False).index[0]


def scene_field(scene_id):
    """Tile lattice table of one scene (red band) with the parts of its R model, all
    in meters east/north and with the scene's median displacement removed."""
    t = pd.read_csv(ic.RESULTS / 'tiles' / f'{scene_id}.csv')
    t = dg.add_lattice(t[t.band == 'red'])
    bad = ic.neighbor_outliers(t.ri.values, t.ci.values, t.dy_px.values, t.dx_px.values, ic.grid_shape())
    t = t[~bad].copy()
    row = models_table().loc[scene_id]
    t['h'] = ic.tile_elevation()[t.ri, t.ci]
    parts = {}
    for comp, sgn, d in (('n', -1, 'y'), ('e', 1, 'x')):
        meas = sgn * RES * t[f'd{d}_px']
        aff = sgn * RES * (row[f'{CP}_c0_{d}'] + row[f'{CP}_cr_{d}'] * t.row + row[f'{CP}_cc_{d}'] * t.col)
        elev = sgn * RES * row[f'{CP}_ch_{d}'] * (t.h - ic.HREF)
        t[f'meas_{comp}'] = meas - meas.median()
        t[f'aff_{comp}'] = aff - aff.median()
        t[f'elev_{comp}'] = elev - elev.median()
        t[f'res_{comp}'] = (meas - aff - elev) - (meas - aff - elev).median()
        parts[comp] = sgn * RES * row[f'{CP}_ch_{d}'] * 100
    l, b, r, top = ic.ISLAND_BOUNDS
    t['x'], t['y'] = l + RES * t.col, top - RES * t.row
    return t, parts, row


def island_axes(ax, shade=None, cmap='gray', alpha=1.0, **kw):
    ext = ic.extent()
    if shade is not None:
        ax.imshow(shade, extent=ext, cmap=cmap, alpha=alpha, interpolation='bilinear', **kw)
    ic.island_outline().boundary.plot(ax=ax, color='k', lw=0.6)
    ax.set_xlim(ext[0] + 200, ext[1] - 100)
    ax.set_ylim(ext[2] + 150, ext[3] - 150)
    ic.bare(ax)


def terrain():
    elev = ic.dem_arrays()[0]
    land = ic.land_mask()
    return elev, land, np.where(land, ic.hillshade(elev), np.nan)


def quiver(ax, t, key, scale, step=1):
    q = t.iloc[::step]
    mag = np.hypot(q[f'{key}_e'], q[f'{key}_n'])
    return ax.quiver(q.x, q.y, q[f'{key}_e'], q[f'{key}_n'], mag, cmap='plasma', clim=(0, 12),
                     scale=scale, scale_units='xy', angles='xy', width=0.004, headwidth=3.5)


def scene_title(scene_id, row):
    return f"{pd.Timestamp(row.acquired):%d %b %Y}, {ic.INSTRUMENTS[row.instrument][0]} {row.satellite_id}"


# --------------------------------------------------------------------------- problem

@cli.command()
@out_option
@click.option('--scene', default=None)
def problem(outdir, scene):
    """What was wrong: one scene's displacement follows the terrain."""
    scene = scene or example_scene()
    t, k, row = scene_field(scene)
    elev, land, hs = terrain()
    fig = plt.figure(figsize=(W, 4.6))
    ax1 = fig.add_axes([0.01, 0.04, 0.33, 0.84])
    island_axes(ax1, hs, vmin=-0.3, vmax=1.3)
    qv = quiver(ax1, t, 'meas', 0.035)
    ax1.quiverkey(qv, 0.14, 0.07, 10, '10 m', labelpos='E', fontproperties=dict(size=FS_S))
    ax1.set_title('Displacement of one scene', fontsize=FS_T)
    ax2 = fig.add_axes([0.345, 0.04, 0.33, 0.84])
    island_axes(ax2, hs, vmin=-0.3, vmax=1.3)
    im = ax2.imshow(np.where(land, elev, np.nan), extent=ic.extent(), cmap='terrain', vmin=0, vmax=190, alpha=0.75)
    ic.add_scalebar(ax2, 1000, '1 km', loc=(0.74, 0.05), fontsize=FS_S)
    cax = fig.add_axes([0.357, 0.09, 0.012, 0.22])
    cb = fig.colorbar(im, cax=cax)
    cax.set_title('m', fontsize=FS_S, pad=3)
    cb.ax.tick_params(labelsize=FS_S - 1)
    ax2.set_title('Terrain elevation', fontsize=FS_T)
    ax3 = fig.add_axes([0.745, 0.15, 0.24, 0.73])
    for comp, col, name in (('n', '#d62728', 'north'), ('e', '#1f77b4', 'east')):
        ax3.scatter(t.h, t[f'meas_{comp}'], s=3, color=col, alpha=0.35, lw=0)
        xs = np.array([25, 185])
        b = np.polyfit(t.h, t[f'meas_{comp}'], 1)
        ax3.plot(xs, np.polyval(b, xs), color=col, lw=1.8, label=f'{name}: {b[0] * 100:+.1f} m per 100 m')
    ax3.axhline(0, color='0.6', lw=0.6)
    ax3.set_xlabel('elevation (m)', fontsize=FS)
    ax3.set_ylabel('displacement (m)', fontsize=FS)
    ax3.tick_params(labelsize=FS_S)
    ax3.legend(fontsize=FS_S, loc='lower left', frameon=False)
    ax3.set_title('Displacement against elevation', fontsize=FS_T)
    fig.suptitle(f'PlanetScope {scene_title(scene, row)}', fontsize=FS, y=0.985, color='0.3')
    save(fig, outdir, 'island_align_problem', f"""
        What is wrong with the orthorectified PlanetScope scenes, shown on one scene ({scene}, {row.fraction_clear:.0%}
        of Barro Colorado Island clear, view angle {row.view_angle:.1f} degrees). Left: displacement of each 192 m
        matching tile of the scene (arrows; color and length give the magnitude, key 10 m) relative to the master
        scene {ic.MASTER}, measured by normalized cross-correlation in the red band with the scene's median shift
        removed, over a hillshade of the Copernicus GLO-30 elevation model. Middle: the same elevation model (color,
        meters above sea level). Right: north (red) and east (blue) components of each tile's displacement against
        the mean elevation of the tile, with least-squares lines; the slopes are the scene's elevation term.
        The displacement grows with terrain height, so the central plateau moves against the low shore by
        {np.hypot(k['n'], k['e']) * 1.4:.0f} m in this scene.""")


# --------------------------------------------------------------------------- diagnosis

@cli.command()
@out_option
def diagnosis(outdir):
    """How it was found: not false matches, not the orthorectification, real geometry."""
    fig, axes = plt.subplots(1, 3, figsize=(W, 4.4))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.14, top=0.88, wspace=0.27)
    # closure
    ax = axes[0]
    c = pd.read_csv(dg.DIAG / 'closure_planet.csv.gz', usecols=['band', 'closure_m', 'in_band', 'scene_id', 'mk'])
    c = c[c.band == 'red']
    bins = np.linspace(0, 6, 61)
    for flag, col, name in ((True, C_HIGH, 'central plateau'), (False, C_LOW, 'rest of the island')):
        v = c[c.in_band == flag].closure_m
        ax.hist(v.clip(upper=6), bins=bins, density=True, histtype='step', lw=1.8, color=col,
                label=f'{name} (median {v.median():.1f} m)')
    ax.axvline(3, color='0.4', ls=':', lw=1)
    ax.text(3.08, ax.get_ylim()[1] * 0.62, '1 px', fontsize=FS_S, color='0.35')
    ax.set_xlabel('triangle closure error (m)', fontsize=FS)
    ax.set_ylabel('density of tiles', fontsize=FS)
    ax.set_yticks([])
    ax.legend(fontsize=FS_S, frameon=False, loc='center right')
    ax.set_title('Matches agree across reference scenes', fontsize=FS_T)
    # ortho
    ax = axes[1]
    o = pd.read_csv(dg.DIAG / 'orthocmp_scenes.csv')
    bins = np.logspace(-2, 1, 46)
    ax.hist(RES * o.med_T, bins=bins, color=C_AFTER, alpha=0.85, label='our reprojection vs Planet ortho\n(same image)')
    ax.hist(RES * o.island_med_T, bins=bins, color=C_BEFORE, alpha=0.85, label='scene vs master scene')
    ax.set_xscale('log')
    ax.set_xticks([0.01, 0.1, 1, 10])
    ax.set_xticklabels(['0.01', '0.1', '1', '10'])
    ax.set_xlabel('median tile residual per scene (m)', fontsize=FS)
    ax.set_ylabel('scenes', fontsize=FS)
    ax.legend(fontsize=FS_S, frameon=False, loc='upper center')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    ax.set_title('Orthorectification adds nothing', fontsize=FS_T)
    # drone
    ax = axes[2]
    sm = models_table()
    kn, ke, n = master_k()
    f = pd.read_csv(dg.DIAG / 'drone_fits.csv').set_index('scene_id')
    j = f.join(sm[['k_north_m100', 'k_east_m100']], rsuffix='_rel', how='inner')
    j = j[j.index != ic.MASTER]
    ax.scatter(j.k_north_m100_rel + kn, j.k_north_m100, s=16, color='#d62728', label='north', zorder=3)
    ax.scatter(j.k_east_m100_rel + ke, j.k_east_m100, s=16, color='#1f77b4', label='east', zorder=3)
    lim = [-14, 10]
    ax.plot(lim, lim, color='0.5', lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.set_xlabel('measured against the Planet master', fontsize=FS)
    ax.set_ylabel('measured against the drone mosaic', fontsize=FS)
    ax.legend(fontsize=FS_S, frameon=False, loc='upper left', title='elevation term\n(m per 100 m)', title_fontsize=FS_S)
    ax.set_title('The drone mosaic sees the same term', fontsize=FS_T)
    for a in axes:
        a.tick_params(labelsize=FS_S)
    tri = c[['scene_id', 'mk']].drop_duplicates()
    save(fig, outdir, 'island_align_diagnosis', f"""
        Three tests that locate the cause of the misalignment. Left: triangle closure error of the red-band tile
        displacements, |d(scene to master) - d(alternate master to master) - d(scene to alternate master)|, for
        {len(c):,} tile triangles from {tri.scene_id.nunique()} scenes and {tri.mk.nunique()} alternate master
        scenes, on the central plateau (brown, the {int(dg.band_mask().sum())} tiles with the largest mean
        residual) and elsewhere (green); values above 6 m are counted in the last bin and the dotted line marks one
        3 m pixel. Displacements that are real close to zero, false matches would not. Middle: per scene, the
        median tile residual between our own reprojection of the non-orthorectified (basic) scene with its
        rational polynomial coefficients and the Copernicus elevation model and Planet's orthorectified product
        of the same image (green, {len(o)} scenes, median {RES * o.med_T.median():.2f} m), against the residual of
        the same scenes relative to the master scene after a translation (orange, median
        {RES * o.island_med_T.median():.1f} m). Right: north (red) and east (blue) components of the elevation term
        of {len(j)} scenes, in meters of displacement per 100 m of elevation, measured against the 2024-10-14
        whole-island drone orthomosaic versus measured against the Planet master scene and referred to the ground
        by adding the master's own term; the line is equality
        (r = {j.k_north_m100.corr(j.k_north_m100_rel):.2f} north, {j.k_east_m100.corr(j.k_east_m100_rel):.2f} east).""")


# --------------------------------------------------------------------------- method

@cli.command()
@out_option
@click.option('--scene', default=None)
def method(outdir, scene):
    """The correction: measured field = affine part + elevation part + remainder."""
    scene = scene or example_scene()
    t, k, row = scene_field(scene)
    _, _, hs = terrain()
    fig = plt.figure(figsize=(W, 3.35))
    names = [('meas', 'Measured'), ('aff', 'Affine part'), ('elev', 'Elevation part'), ('res', 'Remainder')]
    w, gap, x0 = 0.225, 0.028, 0.008
    for i, (key, title) in enumerate(names):
        ax = fig.add_axes([x0 + i * (w + gap), 0.02, w, 0.82])
        island_axes(ax, hs, vmin=-0.3, vmax=1.3)
        qv = quiver(ax, t, key, 0.035)
        med = np.hypot(t[f'{key}_e'], t[f'{key}_n']).median()
        ax.set_title(f'{title} (median {med:.1f} m)', fontsize=FS_T)
        if i == 0:
            ax.quiverkey(qv, 0.16, 0.07, 10, '10 m', labelpos='E', fontproperties=dict(size=FS_S))
        if i < 3:
            fig.text(x0 + i * (w + gap) + w + gap / 2, 0.43, '=' if i == 0 else '+', fontsize=20, ha='center',
                     va='center')
    fig.suptitle(f'PlanetScope {scene_title(scene, row)}', fontsize=FS, y=0.99, color='0.3')
    src = ('not fitted: it is the tilt of the scene\'s RPC, its line of sight minus the view direction reported in '
           'the metadata, relative to the same quantity for the master' if row.elev_source == 'rpc' else
           'fitted with the affine, because this scene has no RPC on disk')
    save(fig, outdir, 'island_align_method', f"""
        The correction model applied to one scene ({scene}). Each panel shows displacement vectors of the scene's
        192 m matching tiles relative to the master scene {ic.MASTER} (arrows; color and length give the magnitude,
        key 10 m), with the scene's median shift removed, over a hillshade of the Copernicus GLO-30 elevation model.
        From left to right: the measured displacement; the part explained by an affine function of map position
        (scale, rotation and shear); the part proportional to terrain elevation
        ({k['n']:+.1f} m north and {k['e']:+.1f} m east per 100 m of elevation in this scene, relative to the
        master); and the remainder after both are removed, which is at the level of the matching noise
        (about 1.1 m). The elevation part is {src}; the six affine coefficients are fitted per scene by least
        squares over the tiles after it is removed, and the correction is applied to every 3 m pixel using the
        elevation model.""")


# --------------------------------------------------------------------------- steps

def tile_example(scene_id, t, search=15):
    """Master tile, scene tile (as delivered, same map position) and the NCC surface
    of the tile with the largest measured displacement on the plateau."""
    from skimage.feature import match_template
    md = ic.master_data()
    bands, clear, data = ic.read_scene(scene_id, bands=('red',))
    valid = clear & data & ic.land_mask()
    c = t[t.h >= 150].assign(m=lambda d: np.hypot(d.meas_e, d.meas_n)).sort_values('m').iloc[-1]
    r0, c0 = int(c.row - ic.TILE / 2), int(c.col - ic.TILE / 2)
    sl = (slice(r0, r0 + ic.TILE), slice(c0, c0 + ic.TILE))
    big = (slice(r0 - search, r0 + ic.TILE + search), slice(c0 - search, c0 + ic.TILE + search))
    mz = ic.zscore(md['bands']['red'], md['valid'])
    sz = ic.zscore(bands['red'], valid)
    surf = match_template(sz[big], mz[sl])
    i, j = np.unravel_index(np.argmax(surf), surf.shape)
    return dict(master=md['bands']['red'][sl], scene=bands['red'][sl], surface=surf, search=search,
                east=RES * (j - search), north=-RES * (i - search), peak=float(surf[i, j]), x=c.x, y=c.y)


@cli.command()
@out_option
@click.option('--scene', default=None)
def steps(outdir, scene):
    """The procedure, step by step, on one scene."""
    scene = scene or example_scene()
    t, k, row = scene_field(scene)
    elev, land, hs = terrain()
    ex = tile_example(scene, t)
    H = 3.9
    fig = plt.figure(figsize=(W, H))
    w, gap, x0, y0, h = 0.178, 0.0245, 0.006, 0.03, 0.74
    xs = [x0 + i * (w + gap) for i in range(5)]
    titles = ['1  Match each tile', '2  Displacement of all tiles', '3  Affine fit + RPC tilt',
              '4  Tie to the drone mosaic', '5  Move every pixel']
    for x, title in zip(xs, titles):
        fig.text(x + w / 2, y0 + h + 0.035, title, ha='center', va='bottom', fontsize=FS_T)
    for x in xs[1:]:
        fig.text(x - gap / 2, y0 + h / 2, '\u2192', ha='center', va='center', fontsize=17, color='0.3')

    # 1: master tile, scene tile, correlation surface
    hw = (w - 0.006) / 2
    hh = hw * W / H
    lo, hi = np.percentile(ex['master'], [2, 98])
    lo2, hi2 = np.percentile(ex['scene'], [2, 98])
    for xi, img, name, a, b in ((xs[0], ex['master'], 'master', lo, hi), (xs[0] + hw + 0.006, ex['scene'], 'scene', lo2, hi2)):
        ax = fig.add_axes([xi, y0 + h - hh, hw, hh])
        ax.imshow(img, cmap='gray', vmin=a, vmax=b, interpolation='nearest')
        ax.axhline(ic.TILE / 2 - 0.5, color=C_OUTLINE, lw=0.7)
        ax.axvline(ic.TILE / 2 - 0.5, color=C_OUTLINE, lw=0.7)
        ic.bare(ax)
        ax.text(0.04, 0.95, name, transform=ax.transAxes, fontsize=FS_S, color='white', va='top',
                path_effects=[__import__('matplotlib.patheffects', fromlist=['x']).withStroke(linewidth=2, foreground='black')])
    m = ex['search'] * RES
    sh = y0 + h - hh - 0.075 - y0
    sw = sh * H / W
    ax = fig.add_axes([xs[0] + (w - sw) / 2 + 0.012, y0 + 0.055, sw, sh - 0.055])
    ax.imshow(ex['surface'], extent=(-m - 1.5, m + 1.5, -m - 1.5, m + 1.5), cmap='viridis', interpolation='nearest')
    ax.plot(ex['east'], ex['north'], marker='+', color='white', ms=9, mew=1.6)
    ax.axhline(0, color='white', lw=0.4, alpha=0.6)
    ax.axvline(0, color='white', lw=0.4, alpha=0.6)
    ax.set_xlabel('east shift (m)', fontsize=FS_S, labelpad=1)
    ax.set_ylabel('north shift (m)', fontsize=FS_S, labelpad=0)
    ax.tick_params(labelsize=FS_S - 1.5, length=2, pad=1)
    ax.set_xticks([-30, 0, 30])
    ax.set_yticks([-30, 0, 30])

    # 2: all tiles
    ax = fig.add_axes([xs[1], y0, w, h])
    island_axes(ax, hs, vmin=-0.3, vmax=1.3)
    ax.set_anchor('N')
    qv = quiver(ax, t, 'meas', 0.035)
    ax.plot(ex['x'], ex['y'], marker='s', mfc='none', mec='k', ms=7, mew=1.2)
    ax.quiverkey(qv, 0.18, 0.06, 10, '10 m', labelpos='E', fontproperties=dict(size=FS_S))

    # 3: fitted model (affine + elevation)
    ax = fig.add_axes([xs[2], y0, w, h])
    island_axes(ax, hs, vmin=-0.3, vmax=1.3)
    ax.set_anchor('N')
    t['fit_e'], t['fit_n'] = t.aff_e + t.elev_e, t.aff_n + t.elev_n
    quiver(ax, t, 'fit', 0.035)

    # 4: the master's own model relative to the drone mosaic
    ax = fig.add_axes([xs[3], y0, w, h])
    island_axes(ax, hs, vmin=-0.3, vmax=1.3)
    ax.set_anchor('N')
    c = ic.master_anchor(kind=FINAL[0])
    X = np.stack([np.ones(len(t)), t.row, t.col, t.h - ic.HREF], axis=1)
    t['anc_e'], t['anc_n'] = RES * (X @ c[1]), -RES * (X @ c[0])
    quiver(ax, t, 'anc', 0.035, step=2)

    # 5: per-pixel correction (scene to master plus master to drone)
    ax = fig.add_axes([xs[4], y0, w, h])
    island_axes(ax)
    dy, dx = ic.displacement_maps(ic.load_model(ic.model_path(ic.RESULTS, scene)), FINAL)
    mag = np.where(land, RES * np.hypot(dy, dx), np.nan)
    im = ax.imshow(mag, extent=ic.extent(), cmap='plasma', vmin=np.nanpercentile(mag, 1), vmax=np.nanpercentile(mag, 99))
    ic.island_outline().boundary.plot(ax=ax, color='k', lw=0.6)
    ic.bare(ax)
    ax.set_anchor('N')
    cax = fig.add_axes([xs[4] + 0.03, y0 + 0.13, w - 0.06, 0.028])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label('correction (m)', fontsize=FS_S, labelpad=2)
    cb.ax.tick_params(labelsize=FS_S - 1, length=2)
    fig.suptitle(f'PlanetScope {scene_title(scene, row)}', fontsize=FS, y=0.985, color='0.3')
    nanchor = json.loads((ic.RESULTS / 'master_abs.json').read_text()).get('R_n', 0)
    save(fig, outdir, 'island_align_steps', f"""
        The alignment procedure, step by step, on one PlanetScope scene ({scene}). (1) The island is cut into
        192 m tiles every 96 m; each tile of the scene is compared with the same tile of the master scene
        {ic.MASTER} by normalized cross-correlation of the red band (and separately the near-infrared band) over
        cloud-free land pixels. Top: one tile on the central plateau in the master and, at the same map position,
        in the scene as delivered (yellow cross at the tile center). Bottom: the correlation of that tile as a
        function of the trial shift; the white cross marks the best match ({ex['east']:+.0f} m east,
        {ex['north']:+.0f} m north, correlation {ex['peak']:.2f}), which the pipeline finds in two stages (one
        whole-island shift searched over 90 m, then a search of 18 m per tile) and refines to a fraction of a
        pixel by Lucas-Kanade. (2) The resulting displacement of every tile (arrows; color and length give the
        magnitude, key 10 m) with the scene's median shift removed, over a hillshade of the Copernicus GLO-30
        elevation model; the square marks the tile of panel 1, and tiles with a correlation peak below 0.35 or
        more than 2 pixels from the median of their neighbors are dropped. (3) The model of those vectors,
        drawn on the same scale: a displacement proportional to terrain elevation that is not fitted but read
        from the scene's RPC (the tilt of its line of sight against the view direction in the metadata, relative
        to the master's), plus a displacement linear in map position (affine, six coefficients) fitted by robust
        least squares to what is left. (4) The master scene's own displacement relative to the 2024-10-14
        whole-island drone orthomosaic: its elevation part from its own RPC tilt, and its affine part estimated
        once as the median over {nanchor} scenes of the difference between a scene's fit against the drone mosaic
        and its fit against the master (same arrow scale, every second tile, median shift included); it is added
        to every scene's model. (5) The correction applied to this scene: magnitude of the total displacement at every 3 m
        pixel (color, meters), computed from the summed model and the elevation model, by which the scene is
        resampled.""")


# --------------------------------------------------------------------------- result

def drone_offsets():
    """Per scene matched to the drone mosaic: median tile distance from the drone as
    delivered and after the drone-anchored correction (model FINAL)."""
    t = dg.add_lattice(pd.read_csv(dg.DIAG / 'drone_tiles.csv.gz'))
    t = t[(t.band == 'red') & t.global_ok]
    sm = models_table()
    anchor = ic.master_anchor(kind=FINAL[0])
    h = ic.tile_elevation()
    rows = []
    for s, d in t.groupby('scene_id'):
        if s == ic.MASTER:
            c = anchor
        elif s in sm.index:
            r = sm.loc[s]
            c = anchor + np.array([[r[f'{CP}_{k}_{a}'] for k in ('c0', 'cr', 'cc', 'ch')] for a in ('y', 'x')])
        else:
            continue
        X = np.stack([np.ones(len(d)), d.row, d.col, h[d.ri, d.ci] - ic.HREF], axis=1)
        after = np.hypot(d.dy_px - X @ c[0], d.dx_px - X @ c[1])
        rows.append(dict(scene_id=s, before_m=RES * np.hypot(d.dy_px, d.dx_px).median(), after_m=RES * after.median(),
                         after_high_m=RES * after[h[d.ri, d.ci] >= 120].median()))
    return pd.DataFrame(rows)


@cli.command()
@out_option
def result(outdir):
    """Residual by model, and the offset from the drone mosaic before and after."""
    fig, axes = plt.subplots(1, 2, figsize=(W, 4.4), gridspec_kw=dict(width_ratios=[1.15, 1]))
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.15, top=0.88, wspace=0.2)
    ax = axes[0]
    rm = pd.read_csv(ic.RESULTS / 'remeasure.csv')
    kinds = ['T', 'A', 'L', 'H', 'R']
    labels = ['translation', 'affine', 'local warp', 'affine\n+ fitted\nelevation', 'affine\n+ RPC tilt']
    med = rm.groupby('model')[['med_high_px', 'med_low_px']].median().reindex(kinds) * RES
    x = np.arange(len(kinds))
    ax.bar(x - 0.2, med.med_high_px, 0.4, color=C_HIGH, label='central plateau')
    ax.bar(x + 0.2, med.med_low_px, 0.4, color=C_LOW, label='lower ground')
    for xi, (a, b) in zip(x, med.values):
        ax.text(xi - 0.2, a - 0.05, f'{a:.1f}', ha='center', va='top', fontsize=FS_S, color='white')
        ax.text(xi + 0.2, b - 0.05, f'{b:.1f}', ha='center', va='top', fontsize=FS_S, color='white')
    noise = RES * models_table().noise_px.median()
    ax.axhline(noise, color='k', ls='--', lw=0.9)
    ax.set_xlim(-0.55, 4.6)
    ax.text(4.57, noise + 0.05, 'matching noise', fontsize=FS_S, ha='right')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS)
    ax.set_ylabel('residual misalignment (m)', fontsize=FS)
    ax.legend(fontsize=FS_S, frameon=False, loc='upper right')
    ax.set_title('Scene to scene, after each correction', fontsize=FS_T)
    ax = axes[1]
    d = drone_offsets()
    d.to_csv(dg.DIAG / 'drone_offsets_before_after.csv', index=False)
    bins = np.arange(0, 21, 1)
    ax.hist(d.before_m.clip(upper=20), bins=bins, color=C_BEFORE, alpha=0.85,
            label=f'as delivered (median {d.before_m.median():.1f} m)')
    ax.hist(d.after_m, bins=bins, color=C_AFTER, alpha=0.85, label=f'corrected (median {d.after_m.median():.1f} m)')
    ax.axvline(3, color='0.4', ls=':', lw=1)
    ax.text(3.2, ax.get_ylim()[1] * 0.55, '1 px', fontsize=FS_S, color='0.35')
    ax.set_xlabel('distance from the drone mosaic (m)', fontsize=FS)
    ax.set_ylabel('scenes', fontsize=FS)
    ax.legend(fontsize=FS_S, frameon=False, loc='upper right')
    ax.set_title('Against the drone mosaic', fontsize=FS_T)
    for a in axes:
        a.tick_params(labelsize=FS_S)
        a.spines[['top', 'right']].set_visible(False)
    save(fig, outdir, 'island_align_result', f"""
        Alignment of PlanetScope scenes over Barro Colorado Island before and after the correction. Left: residual
        misalignment between each scene and the master scene {ic.MASTER} after warping the scene with each model,
        as the median over {rm.scene_id.nunique()} scenes (2020-2026, at least 80 % of the island clear) of the
        median displacement of 192 m matching tiles re-measured in the red band, for tiles on the central plateau
        (brown, mean elevation of at least 120 m) and on lower ground (green). The models are one translation per
        scene, an affine, a local warp (affine plus a 144 m Gaussian-smoothed residual field), an affine plus a
        fitted term proportional to elevation, and an affine plus the elevation term given by the tilt of the
        scene's RPC against its reported view direction (fitted where no RPC is on disk); the dashed line is the matching noise floor ({noise:.1f} m, the median
        red-versus-near-infrared disagreement of a tile's displacement divided by the square root of 2). Right:
        median distance of each scene's tiles from the 2024-10-14 whole-island drone orthomosaic (20 cm, averaged
        to 3 m), for the {len(d)} scenes matched to it, as delivered by Planet (orange; values above 20 m in the
        last bin) and after the affine + RPC tilt correction anchored to the drone mosaic (green, computed by
        subtracting each scene's model from its measured displacements); the dotted line marks one 3 m pixel. The
        corrected value includes the cross-sensor matching noise of about 1 m.""")
    click.echo(d[['before_m', 'after_m', 'after_high_m']].describe().round(2).to_string())


# --------------------------------------------------------------------------- compare

def pick_frames(n, b, min_clear=0.995):
    """The n video scenes furthest from the ground at the window as delivered (by their
    drone-anchored model), fully clear in the window, at most one per day."""
    sc = ic.load_scenes(ic.RESULTS)
    sc = sc[sc.use_video & sc.scene_id.map(lambda s: ic.model_path(ic.RESULTS, s).exists())]
    win = map_window(b)
    rows = []
    for r in sc.itertuples():
        m = ic.load_model(ic.model_path(ic.RESULTS, r.scene_id))
        c = (win[0] + win[1]) // 2, (win[2] + win[3]) // 2
        dy, dx = ic.displacement_maps(m, FINAL, window=(c[0], c[0] + 1, c[1], c[1] + 1))
        rows.append(dict(scene_id=r.scene_id, off_m=RES * float(np.hypot(dy, dx)[0, 0]),
                         east=RES * float(dx[0, 0]), north=-RES * float(dy[0, 0])))
    d = pd.DataFrame(rows).sort_values('off_m', ascending=False)
    keep = []
    for r in d.itertuples():
        if len(keep) >= n:
            break
        if any(k[:8] == r.scene_id[:8] for k in keep):
            continue
        ocm = ic.read_island(ic.scene_paths(r.scene_id)[1], [1], fill=ic.OCM_NODATA)[0]
        pad = 8
        if (ocm[win[0] - pad:win[1] + pad, win[2] - pad:win[3] + pad] == ic.OCM_CLEAR).mean() >= min_clear:
            keep.append(r.scene_id)
    return d[d.scene_id.isin(keep)].sort_values('scene_id'), d


def map_window(b):
    """(r0, r1, c0, c1) of map bounds b on the island grid."""
    l, _, _, top = ic.ISLAND_BOUNDS
    return (int(round((top - b[3]) / RES)), int(round((top - b[1]) / RES)),
            int(round((b[0] - l) / RES)), int(round((b[2] - l) / RES)))


def stretch(rgb, lo, hi, gamma=0.8):
    return np.clip((rgb - lo) / (hi - lo), 0, 1) ** gamma


@cli.command()
@out_option
@click.option('--n', default=12, show_default=True, help='Number of scenes.')
@click.option('--hold', default=1.0, show_default=True, help='Seconds per scene.')
@click.option('--dpi', default=150, show_default=True)
def compare(outdir, n, hold, dpi):
    """Drone | Planet as delivered | Planet corrected, for the worst scenes."""
    import geopandas as gpd
    outdir.mkdir(parents=True, exist_ok=True)
    b = COMPARE_WINDOW
    win = map_window(b)
    ext = (b[0], b[2], b[1], b[3])
    crowns = gpd.read_file(CROWNS, bbox=b)
    crowns = crowns[crowns.date == sorted(crowns.date.unique())[-1]]
    crowns = crowns[crowns.geometry.area >= 250]
    with rasterio.open(dg.DRONE_SRC) as src:
        w = from_bounds(*b, transform=src.transform)
        drone = src.read([1, 2, 3], window=w, out_shape=(3, 750, 750)).transpose(1, 2, 0)
    frames, ranked = pick_frames(n, b)
    sc = ic.load_scenes(ic.RESULTS).set_index('scene_id')

    H = 4.75
    fig = plt.figure(figsize=(W, H), dpi=dpi)
    s = 3.9
    gap = (W - 3 * s) / 4
    axes, ims = [], []
    titles = ['Drone mosaic (14 Oct 2024)', 'PlanetScope as delivered', 'PlanetScope corrected']
    for i, title in enumerate(titles):
        ax = fig.add_axes([(gap + i * (s + gap)) / W, 0.42 / H, s / W, s / H])
        ims.append(ax.imshow(drone if i == 0 else np.zeros((100, 100, 3)), extent=ext,
                             interpolation='bilinear' if i else 'antialiased'))
        crowns.boundary.plot(ax=ax, color=C_OUTLINE, lw=1.1)
        ax.set_xlim(b[0], b[2])
        ax.set_ylim(b[1], b[3])
        ic.bare(ax)
        ax.set_title(title, fontsize=FS_T, pad=4)
        axes.append(ax)
    ic.add_scalebar(axes[0], 50, '50 m', loc=(0.06, 0.05), fontsize=FS_S, lw=2.5)
    label = fig.text(0.5, 0.1 / H, '', ha='center', va='bottom', fontsize=FS + 0.5)
    fig.canvas.draw()
    wpx, hpx = fig.canvas.get_width_height()
    mp4 = outdir / 'island_align_compare.mp4'
    fps = 10
    ff = subprocess.Popen(
        ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{wpx}x{hpx}',
         '-r', str(fps), '-i', '-', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264',
         '-pix_fmt', 'yuv420p', '-crf', '18', '-movflags', '+faststart', str(mp4)], stdin=subprocess.PIPE)
    r0, r1, c0, c1 = win
    for k, fr in enumerate(frames.itertuples()):
        sr = ic.read_island(ic.scene_paths(fr.scene_id)[0], [3, 2, 1]).astype(np.float32)
        raw = sr[:, r0:r1, c0:c1]
        model = ic.load_model(ic.model_path(ic.RESULTS, fr.scene_id))
        dy, dx = ic.displacement_maps(model, FINAL, window=win)
        cor = np.stack(ic.warp_arrays(list(sr), dy, dx, [3] * 3, [0] * 3, window=win))
        lo = np.percentile(raw, 1, axis=(1, 2))
        hi = np.percentile(raw, 99.7, axis=(1, 2))
        ims[1].set_data(stretch(raw.transpose(1, 2, 0), lo, hi))
        ims[2].set_data(stretch(cor.transpose(1, 2, 0), lo, hi))
        row = sc.loc[fr.scene_id]
        label.set_text(f"{pd.Timestamp(row.acquired):%d %b %Y}   {ic.INSTRUMENTS[row.instrument][0]} "
                       f"{row.satellite_id}   as delivered: {fr.off_m:.0f} m from the ground position")
        fig.canvas.draw()
        buf = bytes(fig.canvas.buffer_rgba())
        for _ in range(int(round(hold * fps))):
            ff.stdin.write(buf)
        if k == 0:
            plt.imsave(outdir / 'island_align_compare.png', np.frombuffer(buf, np.uint8).reshape(hpx, wpx, 4))
    ff.stdin.close()
    if ff.wait():
        raise click.ClickException('ffmpeg failed')
    plt.close(fig)
    gif = outdir / 'island_align_compare.gif'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(mp4), '-vf',
                    f'fps={max(1, round(1 / hold))},split[a][b];[a]palettegen[p];[b][p]paletteuse', '-loop', '0',
                    str(gif)], check=True)
    frames.to_csv(outdir / 'island_align_compare_scenes.csv', index=False)
    (outdir / 'island_align_compare_caption.md').write_text(' '.join(f"""
        Alignment of PlanetScope scenes before and after the correction, over a 300 m window on the central plateau
        of Barro Colorado Island inside the 50ha plot. Left: the whole-island drone orthomosaic of 2024-10-14
        (20 cm), fixed in every frame. Middle: each PlanetScope scene as delivered (Planet's orthorectified surface
        reflectance, 3 m, true color with a per-scene 1-99.7 percentile stretch). Right: the same scene after the
        affine + RPC tilt correction anchored to the drone mosaic (bicubic resampling). Yellow outlines are the
        {len(crowns)} crowns of at least 250 m2 in the 2024 drone crown map, drawn at the same position in every
        panel and frame as reference marks. The sequence shows the {len(frames)} scenes that were furthest from
        their ground position at this window as delivered ({frames.off_m.min():.0f}-{frames.off_m.max():.0f} m; the
        median over all {len(ranked)} scenes is {ranked.off_m.median():.1f} m), among scenes at least 80 % clear
        over the island and fully clear in the window, at most one per day, in date order, {hold:g} s each.
        """.split()) + '\n')
    click.echo(f'wrote {mp4}, {gif}: {len(frames)} scenes, offsets {frames.off_m.round(1).tolist()}')


# --------------------------------------------------------------------------- RPC line of sight
C_REP, C_RPC, C_MEAS = '0.2', C_BEFORE, C_AFTER
LAKE = dg.RPC_H0


def los_table():
    """Fitted scenes with a basic RPC (island_coreg_diagnose.py metadata); lines of sight
    and terms in m east/north per 100 m of height."""
    d = pd.read_csv(dg.DIAG / 'metadata_checks.csv').set_index('scene_id')
    return d[d.rpc_los_east.notna()].copy()


def frame_center(rpc):
    return dg.rpc_ground(rpc, rpc['LINE_OFF'], rpc['SAMP_OFF'], LAKE, (rpc['LONG_OFF'], rpc['LAT_OFF']))


def master_los():
    """The master is the reference of every fit and has no row in the table."""
    idx = pd.read_csv(ic.SR / 'index.csv').set_index('scene_id').loc[ic.MASTER]
    meta = dg.scene_metadata(ic.MASTER, idx.sr_path)
    t, az = 100 * np.tan(np.radians(meta['incidence_angle'])), np.radians(idx.satellite_azimuth)
    rpc = dg.load_rpc(dg.rpc_txt_path(ic.MASTER))
    c = frame_center(rpc)
    e, n, _ = dg.rpc_los(rpc, c[0], c[1])
    kn, ke, _ = master_k()
    return pd.Series(dict(meta_los_east=t * np.sin(az), meta_los_north=t * np.cos(az), rpc_los_east=e,
                          rpc_los_north=n, los_gap_east=e - t * np.sin(az), los_gap_north=n - t * np.cos(az),
                          k_abs_east=ke, k_abs_north=kn, incidence_angle=meta['incidence_angle'],
                          satellite_azimuth=idx.satellite_azimuth, scan_azimuth=meta['scan_azimuth'],
                          instrument=idx.instrument, acquired=idx.acquired, satellite_id=idx.satellite_id))


def camera_height_km(scene_id):
    """Height above the ground at which the RPC's rays from pixels across the frame meet."""
    from pyproj import Transformer
    rpc = dg.load_rpc(dg.rpc_txt_path(scene_id))
    tr = Transformer.from_crs('EPSG:4326', ic.CRS, always_xy=True)
    c = frame_center(rpc)

    def g(samp, h):
        return np.array(tr.transform(*dg.rpc_ground(rpc, rpc['LINE_OFF'], samp, h, c)))
    s0, s1 = rpc['SAMP_OFF'], rpc['SAMP_OFF'] + 0.6 * rpc['SAMP_SCALE']
    lo, hi = np.linalg.norm(g(s1, LAKE) - g(s0, LAKE)), np.linalg.norm(g(s1, LAKE + 1000) - g(s0, LAKE + 1000))
    return lo / (lo - hi)


def deg(v):
    return np.degrees(np.arctan(np.asarray(v) / 100))


def rings(ax, radii_deg, lim, north=-1):
    th = np.linspace(0, 2 * np.pi, 200)
    for r in radii_deg:
        m = 100 * np.tan(np.radians(r))
        if m < lim * 1.35:
            ax.plot(m * np.sin(th), m * np.cos(th), color='0.8', lw=0.6, zorder=0)
            ax.text(m * 0.71, north * m * 0.71, f'{r}°', fontsize=FS_S - 1, color='0.55', ha='left',
                    va='top' if north < 0 else 'bottom', clip_on=True)
    ax.axhline(0, color='0.85', lw=0.6, zorder=0)
    ax.axvline(0, color='0.85', lw=0.6, zorder=0)


@cli.command()
@out_option
@click.option('--scene', default=None)
def geometry(outdir, scene):
    """Observation geometry of one scene: reported and RPC line of sight."""
    scene = scene or example_scene()
    r = los_table().loc[scene]
    H = camera_height_km(scene)
    gap = np.array([r.los_gap_east, r.los_gap_north])
    u = gap / np.hypot(*gap)
    m_rep = (r.meta_los_east * u[0] + r.meta_los_north * u[1]) / 100   # tan of each line of sight in the
    m_rpc = (r.rpc_los_east * u[0] + r.rpc_los_north * u[1]) / 100     # vertical plane of the tilt
    tilt_deg = float(deg(np.hypot(*gap)))
    plot_h = float(ic.dem_arrays()[0][dg.PLOT_CENTER])
    rise = plot_h - LAKE
    shift = rise * (m_rpc - m_rep)
    az_u = np.degrees(np.arctan2(u[0], u[1])) % 360

    fig = plt.figure(figsize=(W, 4.7))
    # (a) from orbit, to scale
    ax = fig.add_axes([0.035, 0.1, 0.25, 0.78])
    xr, xp = H * m_rep, H * m_rpc
    mid = (xr + xp) / 2
    ax.plot([mid - 230, mid + 230], [0, 0], color='0.4', lw=1.2)
    ax.plot([-16.25, 16.25], [0, 0], color='0.4', lw=4, solid_capstyle='butt')
    ax.plot([0, xr], [0, H], color=C_REP, lw=1.3)
    ax.plot([0, xp], [0, H], color=C_RPC, lw=1.3)
    ax.plot([0, 0], [0, H], color='0.75', lw=0.7, ls=':')
    ax.plot([xr], [H], marker='s', ms=7, color=C_REP)
    ax.plot([xp], [H], marker='s', ms=7, color=C_RPC)
    ax.annotate('', (xp, H + 22), (xr, H + 22), arrowprops=dict(arrowstyle='<->', color='0.3', lw=0.8))
    ax.text((xr + xp) / 2, H + 30, f'{abs(xp - xr):.0f} km', ha='center', fontsize=FS)
    side = 1 if xp > xr else -1
    ax.text(xr - side * 14, H, 'camera where\nthe metadata\nputs it', ha='right' if side > 0 else 'left',
            va='center', fontsize=FS_S, color=C_REP)
    ax.text(xp + side * 14, H, 'camera where\nthe RPC\nputs it', ha='left' if side > 0 else 'right', va='center',
            fontsize=FS_S, color=C_RPC)
    ax.text(0, -14, 'island (inside the 32 km frame)', ha='center', va='top', fontsize=FS_S, color='0.3')
    ax.text(xp * 0.42 + side * 12, H * 0.42, f'{tilt_deg:.1f}° apart', ha='left' if side > 0 else 'right',
            va='center', fontsize=FS)
    ax.set_aspect('equal')
    ax.set_xlim(mid - 235, mid + 235)
    ax.set_ylim(-45, H + 60)
    ax.set_ylabel('Height above the ground (km)', fontsize=FS)
    ax.set_xlabel(f'Distance toward azimuth {az_u:.0f}° (km)', fontsize=FS)
    ax.tick_params(labelsize=FS_S)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.set_title('From orbit (to scale)', fontsize=FS_T)

    # (b) at the ground, to scale
    ax = fig.add_axes([0.365, 0.1, 0.27, 0.78])
    x0 = -rise * m_rep                       # where both rays cross lake level
    z = np.array([LAKE - 14, plot_h + 26])
    lim = (shift / 2 - 82, shift / 2 + 82)
    side = 1 if shift > 0 else -1
    ax.fill_between(lim, LAKE, plot_h, color='#c9b79c', alpha=0.35, lw=0)
    ax.plot(lim, [plot_h, plot_h], color=C_HIGH, lw=1.2)
    ax.plot(lim, [LAKE, LAKE], color='#3b7dd8', lw=1.2)
    ax.plot(x0 + (z - LAKE) * m_rep, z, color=C_REP, lw=1.3)
    ax.plot(x0 + (z - LAKE) * m_rpc, z, color=C_RPC, lw=1.3)
    ax.plot([x0], [LAKE], 'o', ms=4, color='k', zorder=5)
    from matplotlib.patches import Ellipse
    ax.add_patch(Ellipse((0, plot_h + 5), 12, 10, fc='#3a8f3a', ec='none', alpha=0.9))
    ax.add_patch(Ellipse((shift, plot_h + 5), 12, 10, fc='none', ec=C_RPC, lw=1.3, ls='--'))
    ax.annotate('', (shift, plot_h - 9), (0, plot_h - 9), arrowprops=dict(arrowstyle='->', color=C_RPC, lw=1.4))
    ax.text(shift / 2, plot_h - 12, f'{abs(shift):.0f} m', ha='center', va='top', fontsize=FS, color=C_RPC)
    ax.text(lim[0] + 3, plot_h - 3, f'plateau, {plot_h:.0f} m', fontsize=FS_S, color=C_HIGH, va='top')
    ax.text(lim[0] + 3, LAKE + 3, f'lake level, {LAKE:.0f} m:\nwhere the tie points are', fontsize=FS_S,
            color='#3b7dd8', va='bottom')
    ax.text(-side * 9, plot_h + 7, 'tree crown', ha='right' if side > 0 else 'left', va='center', fontsize=FS_S,
            color='#2b6e2b')
    ax.text(shift + side * 9, plot_h + 3, 'where the\northo puts it', ha='left' if side > 0 else 'right',
            va='bottom', fontsize=FS_S, color=C_RPC)
    ax.text(x0 + side * 5, LAKE - 8, 'the two lines of\nsight agree here', fontsize=FS_S, va='center',
            ha='left' if side > 0 else 'right')
    ax.set_aspect('equal')
    ax.set_xlim(*lim)
    ax.set_ylim(z[0], plot_h + 34)
    ax.set_xlabel(f'Distance toward azimuth {az_u:.0f}° (m)', fontsize=FS)
    ax.set_ylabel('Elevation (m)', fontsize=FS)
    ax.tick_params(labelsize=FS_S)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.set_title('At the 50ha plot (to scale)', fontsize=FS_T)

    # (c) map view of the lines of sight
    ax = fig.add_axes([0.715, 0.1, 0.275, 0.78])
    vecs = [((r.meta_los_east, r.meta_los_north), C_REP, 'reported by the satellite'),
            ((r.rpc_los_east, r.rpc_los_north), C_RPC, 'RPC')]
    lim = 1.25 * max(np.hypot(*v) for v, _, _ in vecs)
    rings(ax, (2, 4, 6, 8, 10, 12), lim, north=1)
    a = np.radians(r.scan_azimuth)
    ax.plot([-lim * np.sin(a), lim * np.sin(a)], [-lim * np.cos(a), lim * np.cos(a)], color='0.7', lw=0.7, ls='--')
    ax.text(0.93 * lim * np.sin(a), 0.93 * lim * np.cos(a), ' flight line', fontsize=FS_S - 1, color='0.45',
            rotation=90 - r.scan_azimuth, rotation_mode='anchor', ha='right', va='bottom')
    for v, c, lab in vecs:
        ax.annotate('', v, (0, 0), arrowprops=dict(arrowstyle='-|>', color=c, lw=1.8, mutation_scale=14))
        ax.plot([], [], color=c, lw=1.8, label=f'{lab}: {deg(np.hypot(*v)):.1f}° off vertical')
    b0 = np.array([r.meta_los_east, r.meta_los_north])
    ax.annotate('', b0 + gap, b0, arrowprops=dict(arrowstyle='-|>', color='#c0392b', lw=1.3, ls='--',
                                                   mutation_scale=11))
    ax.plot([], [], color='#c0392b', lw=1.3, ls='--', label=f'tilt (RPC − reported): {tilt_deg:.1f}°')
    k = np.array([r.k_abs_east, r.k_abs_north])
    ax.plot(*(b0 + k), marker='o', ms=7, mfc='none', mec=C_MEAS, mew=1.8, ls='none',
            label='reported + elevation term\nmeasured from imagery')
    ax.set_aspect('equal')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim * 0.75, lim * 1.15)
    ax.set_xlabel('East (m per 100 m of height)', fontsize=FS)
    ax.set_ylabel('North (m per 100 m of height)', fontsize=FS)
    ax.tick_params(labelsize=FS_S)
    ax.legend(fontsize=FS_S, loc='lower center', frameon=True, framealpha=0.92, edgecolor='none',
              bbox_to_anchor=(0.5, 0.0))
    ax.set_title('Direction to the satellite, seen from above', fontsize=FS_T)
    fig.suptitle(f'PlanetScope {scene_title(scene, models_table().loc[scene])}', fontsize=FS, y=0.995,
                 color='0.3')
    save(fig, outdir, 'island_rpc_geometry', f"""
        Observation geometry of one PlanetScope scene ({scene}) as reported by the satellite and as modeled by the
        scene's rational polynomial coefficients (RPC), the geometric model Planet uses to orthorectify it.
        Left: vertical section from orbit, to scale, along the direction in which the two disagree (azimuth
        {az_u:.0f}°); the line of sight to the island from the camera position implied by the metadata incidence
        angle and satellite azimuth (dark gray) and from the position implied by the RPC (orange), with the rays of
        the RPC converging {H:.0f} km above the ground; the two positions are {abs(xp - xr):.0f} km apart, an
        angle of {tilt_deg:.1f}° at the ground. Center: the same two lines of sight for one image pixel at the
        50ha plot, to scale; they coincide at lake level ({LAKE:.0f} m), the height of the tie points that
        constrain the RPC around the island, and separate with height, so a crown on the plateau
        ({plot_h:.0f} m; green) is placed {abs(shift):.0f} m away by the orthorectification (dashed orange
        outline), which is {rise:.0f} m of height times the tangent of the tilt. Right: horizontal displacement
        of a pixel's ground position per 100 m of height, east and north, which is the direction to the satellite
        seen from above (rings mark the angle off vertical; dashed line, the flight line), for the reported line of
        sight (dark gray arrow), the RPC at the frame center (orange arrow) and their difference, the tilt of the
        RPC (dashed red arrow); the green circle is the reported line of sight plus the elevation term measured
        independently by image matching ({r.k_abs_east:+.1f} m east, {r.k_abs_north:+.1f} m north per 100 m,
        against {gap[0]:+.1f} and {gap[1]:+.1f} for the tilt), referred to the ground with the drone orthomosaic
        of 2024-10-14.""")


@cli.command()
@out_option
def tilt(outdir):
    """All scenes with an RPC: reported against RPC line of sight, term against tilt."""
    d = los_table()
    m = master_los()
    fig = plt.figure(figsize=(W, 4.35))
    # (a) lines of sight in map view
    ax = fig.add_axes([0.045, 0.125, 0.285, 0.77])
    lim = 20
    rings(ax, (2, 4, 6, 8, 10), lim)
    ax.scatter(d.rpc_los_east, d.rpc_los_north, s=7, c=[ic.INSTRUMENTS[i][1] for i in d.instrument], alpha=0.55,
               lw=0)
    ax.scatter(d.meta_los_east, d.meta_los_north, s=7, c='k', alpha=0.6, lw=0)
    ax.scatter([], [], s=14, c='k', label='reported by the satellite')
    for inst, (name, col) in ic.INSTRUMENTS.items():
        ax.scatter([], [], s=14, c=col, label=f'RPC, {name} ({(d.instrument == inst).sum()})')
    ax.set_aspect('equal')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('East (m per 100 m of height)', fontsize=FS)
    ax.set_ylabel('North (m per 100 m of height)', fontsize=FS)
    ax.tick_params(labelsize=FS_S)
    ax.legend(fontsize=FS_S - 0.5, loc='upper left', frameon=True, framealpha=0.9, edgecolor='none',
              handletextpad=0.2, borderpad=0.2)
    ax.set_title('Line of sight at the frame center', fontsize=FS_T)
    out = (np.hypot(d.rpc_los_east, d.rpc_los_north) > lim).sum()

    # (b) elevation term against tilt
    ax = fig.add_axes([0.395, 0.125, 0.285, 0.77])
    lim2 = 22
    ax.plot([-lim2, lim2], [-lim2, lim2], color='0.6', lw=0.8)
    cols = [ic.INSTRUMENTS[i][1] for i in d.instrument]
    ax.scatter(d.los_gap_east, d.k_abs_east, s=9, c=cols, alpha=0.55, lw=0, marker='o')
    ax.scatter(d.los_gap_north, d.k_abs_north, s=11, c=cols, alpha=0.55, lw=0, marker='^')
    ax.scatter([m.los_gap_east], [m.k_abs_east], s=70, marker='o', fc='none', ec='k', lw=1.2)
    ax.scatter([m.los_gap_north], [m.k_abs_north], s=80, marker='^', fc='none', ec='k', lw=1.2)
    ax.scatter([], [], s=14, c='0.4', marker='o', label='east component')
    ax.scatter([], [], s=16, c='0.4', marker='^', label='north component')
    ax.scatter([], [], s=50, marker='o', fc='none', ec='k', label='master scene (drone)')
    c, r2, n = dg._vector_fit(d.k_abs_east.values, d.k_abs_north.values, d.los_gap_east.values,
                              d.los_gap_north.values)
    ax.text(0.97, 0.05, f'slope {c:.2f}\n{r2:.1%} of variance\n{n} scenes', transform=ax.transAxes, ha='right',
            fontsize=FS_S)
    ax.set_aspect('equal')
    ax.set_xlim(-lim2, lim2)
    ax.set_ylim(-lim2, lim2)
    ax.set_xlabel('Tilt of the RPC (m per 100 m of height)', fontsize=FS)
    ax.set_ylabel('Elevation term from image matching\n(m per 100 m of height)', fontsize=FS)
    ax.tick_params(labelsize=FS_S)
    ax.legend(fontsize=FS_S - 0.5, loc='upper left', frameon=False, handletextpad=0.2)
    ax.set_title('What the imagery shows against the tilt', fontsize=FS_T)
    out2 = ((d[['los_gap_east', 'los_gap_north', 'k_abs_east', 'k_abs_north']].abs() > lim2).any(axis=1)).sum()

    # (c) size of the tilt
    ax = fig.add_axes([0.745, 0.125, 0.24, 0.77])
    td = deg(d.los_gap_m100)
    bins = np.arange(0, 12.01, 0.5)
    ax.hist([np.clip(td[d.instrument == i], 0, 11.99) for i in ic.INSTRUMENTS], bins=bins, stacked=True,
            color=[v[1] for v in ic.INSTRUMENTS.values()], label=[v[0] for v in ic.INSTRUMENTS.values()])
    med, p90 = float(np.median(td)), float(np.quantile(td, 0.9))
    for v, lab in ((med, f'median {med:.1f}°'), (p90, f'90th percentile {p90:.1f}°')):
        ax.axvline(v, color='k', lw=0.8, ls='--')
        ax.text(v + 0.15, ax.get_ylim()[1] * 0.97, lab, rotation=90, va='top', fontsize=FS_S)
    ax.set_xlim(0, 12)
    ax.set_xlabel('Tilt of the RPC (degrees)', fontsize=FS)
    ax.set_ylabel('Scenes', fontsize=FS)
    ax.tick_params(labelsize=FS_S)
    ax.legend(fontsize=FS_S, frameon=False, loc='center right')
    top = ax.secondary_xaxis('top', functions=(lambda x: 475 * np.tan(np.radians(x)),
                                                lambda km: np.degrees(np.arctan(km / 475))))
    top.set_xlabel('Camera displacement at 475 km (km)', fontsize=FS_S)
    top.tick_params(labelsize=FS_S)
    for sp in ('right',):
        ax.spines[sp].set_visible(False)
    save(fig, outdir, 'island_rpc_tilt', f"""
        The line of sight of each PlanetScope scene's rational polynomial coefficients (RPC) against the view
        direction reported in its metadata, for the {len(d)} scenes over Barro Colorado Island (2020-2026) that
        were aligned to the master scene and have a basic (level 1B) RPC. Left: horizontal displacement of the
        ground position of the frame-center pixel per 100 m of height, east and north, which is the direction to
        the satellite seen from above (rings mark the angle off vertical), from the metadata incidence angle and
        satellite azimuth (black, along the across-track direction because only an across-track angle is
        reported) and from the RPC evaluated between {LAKE:.0f} and {LAKE + 100:.0f} m (colored by satellite
        generation; {out} scenes fall outside the axes). Center: the elevation term of each scene, the
        displacement per 100 m of terrain height measured by matching image tiles to the master scene and
        referred to the ground with the drone orthomosaic of 2024-10-14, against the tilt of its RPC, the RPC line
        of sight minus the reported one (circles, east component; triangles, north component; gray line, equality;
        {out2} scenes outside the axes); the black outlines are the master scene, whose term was measured against
        the drone orthomosaic. Right: magnitude of the tilt in degrees, stacked by generation (values above 12°
        in the last bin), with the horizontal displacement of the camera it corresponds to at 475 km altitude on
        the top axis.""")


@cli.command()
@out_option
@click.option('--scene', default=None)
def rpcmap(outdir, scene):
    """The example scene's elevation-related displacement: measured and predicted from the RPC."""
    scene = scene or example_scene()
    t, k, row = scene_field(scene)
    r, m = los_table().loc[scene], master_los()
    _, _, hs = terrain()
    rel = np.array([r.los_gap_east - m.los_gap_east, r.los_gap_north - m.los_gap_north])
    t['m_e'], t['m_n'] = t.elev_e + t.res_e, t.elev_n + t.res_n         # measured minus the affine part
    for j, comp in enumerate('en'):
        pred = rel[j] * (t.h - ic.HREF) / 100
        t[f'p_{comp}'] = pred - pred.median()
        t[f'd_{comp}'] = t[f'm_{comp}'] - t[f'p_{comp}']
        t[f'd_{comp}'] -= t[f'd_{comp}'].median()
    fig = plt.figure(figsize=(W, 4.0))
    names = [('m', 'Measured by image matching'), ('p', 'Predicted from RPC and metadata'), ('d', 'Difference')]
    w, gap, x0 = 0.3, 0.035, 0.012
    for i, (key, title) in enumerate(names):
        ax = fig.add_axes([x0 + i * (w + gap), 0.02, w, 0.84])
        island_axes(ax, hs, vmin=-0.3, vmax=1.3)
        qv = quiver(ax, t, key, 0.035)
        med = np.hypot(t[f'{key}_e'], t[f'{key}_n']).median()
        ax.set_title(f'{title} (median {med:.1f} m)', fontsize=FS_T)
        if i == 0:
            ax.quiverkey(qv, 0.16, 0.07, 10, '10 m', labelpos='E', fontproperties=dict(size=FS_S))
        if i < 2:
            fig.text(x0 + i * (w + gap) + w + gap / 2, 0.43, '−' if i == 0 else '=', fontsize=20, ha='center',
                     va='center')
    fig.suptitle(f'PlanetScope {scene_title(scene, row)}', fontsize=FS, y=0.99, color='0.3')
    save(fig, outdir, 'island_rpc_map', f"""
        Elevation-related displacement of one PlanetScope scene ({scene}) over Barro Colorado Island, measured from
        imagery and predicted from metadata alone. Each panel shows displacement vectors at the scene's 192 m
        matching tiles (arrows; color and length give the magnitude, key 10 m) with the median removed, over a
        hillshade of the Copernicus GLO-30 elevation model. Left: the displacement measured by matching the scene
        to the master scene {ic.MASTER}, after removing the part that is an affine function of map position.
        Center: the displacement predicted with no image matching, as tile elevation times the tilt of the scene's
        rational polynomial coefficients (RPC) relative to the master's, where the tilt is the RPC line of sight at
        the frame center minus the line of sight given by the metadata incidence angle and satellite azimuth
        ({rel[0]:+.1f} m east and {rel[1]:+.1f} m north per 100 m of elevation; the fit to the imagery gives
        {k['e']:+.1f} and {k['n']:+.1f}). Right: measured minus predicted.""")


@cli.command()
@out_option
def model3d(outdir):
    """Interactive 3-D page: reported and RPC lines of sight at orbit, island and crown scale."""
    d, m = los_table(), master_los()
    kn, ke, _ = master_k()
    ex = example_scene()
    mag = d.los_gap_m100
    picks = [ex]
    for inst in ic.INSTRUMENTS:                       # the scene nearest each generation's median tilt
        g = mag[d.instrument == inst]
        picks.append((g - g.median()).abs().idxmin())
    picks.append((mag - mag.quantile(0.9)).abs().idxmin())
    scenes = []
    for sid, r, note in [(ic.MASTER, m, 'master scene')] + [
            (s, d.loc[s], n) for s, n in zip(picks, ['largest clear example', 'median Dove Classic', 'median Dove-R',
                                                     'median SuperDove', '90th percentile of all scenes'])]:
        scenes.append(dict(
            id=sid, note=note, date=f'{pd.Timestamp(r.acquired):%d %b %Y}', inst=ic.INSTRUMENTS[r.instrument][0],
            H=round(float(camera_height_km(sid)), 1), rep=[round(float(r.meta_los_east), 3),
                                                           round(float(r.meta_los_north), 3)],
            rpc=[round(float(r.rpc_los_east), 3), round(float(r.rpc_los_north), 3)],
            term=[round(float(r.k_abs_east), 3), round(float(r.k_abs_north), 3)],
            scan=round(float(r.scan_azimuth), 1)))
    elev, land, _ = terrain()
    step = 30                                           # 90 m nodes
    z = elev[::step, ::step]
    lm = land[::step, ::step]
    pr, pc = dg.PLOT_CENTER
    data = dict(lake=LAKE, dx=step * RES, nx=int(z.shape[1]), ny=int(z.shape[0]),
                z=[round(float(v), 1) for v in np.where(lm, z, LAKE).ravel()], land=[int(v) for v in lm.ravel()],
                plot=dict(col=pc / step, row=pr / step, h=round(float(elev[pr, pc]), 1)), scenes=scenes)
    html = (Path(__file__).resolve().parent / 'island_rpc_model.html').read_text()
    out = outdir / 'island_rpc_model.html'
    outdir.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace('__DATA__', json.dumps(data, separators=(',', ':'))))
    click.echo(f'wrote {out} ({out.stat().st_size / 1e3:.0f} kB, {len(scenes)} scenes)')


@cli.command(name='all')
@out_option
@click.pass_context
def all_(ctx, outdir):
    for cmd in (problem, diagnosis, steps, method, result, compare, geometry, tilt, rpcmap, model3d):
        ctx.invoke(cmd, outdir=outdir)


if __name__ == '__main__':
    cli()
