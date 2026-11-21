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

DEFAULT_OUT = ic.FLOWER / 'figs' / '20261117_updates'
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
    return sm[(sm.band == 'both') & sm.med_H.notna()].set_index('scene_id')


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
    """Tile lattice table of one scene (red band) with the parts of its H model, all
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
        aff = sgn * RES * (row[f'h_c0_{d}'] + row[f'h_cr_{d}'] * t.row + row[f'h_cc_{d}'] * t.col)
        elev = sgn * RES * row[f'h_ch_{d}'] * (t.h - ic.HREF)
        t[f'meas_{comp}'] = meas - meas.median()
        t[f'aff_{comp}'] = aff - aff.median()
        t[f'elev_{comp}'] = elev - elev.median()
        t[f'res_{comp}'] = (meas - aff - elev) - (meas - aff - elev).median()
        parts[comp] = sgn * RES * row[f'h_ch_{d}'] * 100
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
    cax = fig.add_axes([0.637, 0.615, 0.012, 0.22])
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
    save(fig, outdir, 'island_align_method', f"""
        The correction model applied to one scene ({scene}). Each panel shows displacement vectors of the scene's
        192 m matching tiles relative to the master scene {ic.MASTER} (arrows; color and length give the magnitude,
        key 10 m), with the scene's median shift removed, over a hillshade of the Copernicus GLO-30 elevation model.
        From left to right: the measured displacement; the part explained by an affine function of map position
        (scale, rotation and shear); the part proportional to terrain elevation
        ({k['n']:+.1f} m north and {k['e']:+.1f} m east per 100 m of elevation in this scene, relative to the
        master); and the remainder after both are removed, which is at the level of the matching noise
        (about 1.1 m). The eight coefficients are fitted per scene by least squares over the tiles, and the
        correction is applied to every 3 m pixel using the elevation model.""")


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
    titles = ['1  Match each tile', '2  Displacement of all tiles', '3  Fit affine + elevation',
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
    c = ic.master_anchor()
    X = np.stack([np.ones(len(t)), t.row, t.col, t.h - ic.HREF], axis=1)
    t['anc_e'], t['anc_n'] = RES * (X @ c[1]), -RES * (X @ c[0])
    quiver(ax, t, 'anc', 0.035, step=2)

    # 5: per-pixel correction (scene to master plus master to drone)
    ax = fig.add_axes([xs[4], y0, w, h])
    island_axes(ax)
    dy, dx = ic.displacement_maps(ic.load_model(ic.model_path(ic.RESULTS, scene)), 'Ha')
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
        more than 2 pixels from the median of their neighbors are dropped. (3) The model fitted to those
        vectors in one robust least-squares fit: a displacement linear in map position (affine) and in terrain
        elevation, eight coefficients per scene, drawn on the same scale. (4) The master scene's own displacement
        relative to the 2024-10-14 whole-island drone orthomosaic, modeled the same way and estimated once as the
        median over 53 scenes of the difference between a scene's fit against the drone mosaic and its fit
        against the master (same arrow scale, every second tile, median shift included); it is added to every
        scene's model. (5) The correction applied to this scene: magnitude of the total displacement at every 3 m
        pixel (color, meters), computed from the summed model and the elevation model, by which the scene is
        resampled.""")


# --------------------------------------------------------------------------- result

def drone_offsets():
    """Per scene matched to the drone mosaic: median tile distance from the drone as
    delivered and after the drone-anchored affine + elevation correction."""
    t = dg.add_lattice(pd.read_csv(dg.DIAG / 'drone_tiles.csv.gz'))
    t = t[(t.band == 'red') & t.global_ok]
    sm = models_table()
    anchor = ic.master_anchor()
    h = ic.tile_elevation()
    rows = []
    for s, d in t.groupby('scene_id'):
        if s == ic.MASTER:
            c = anchor
        elif s in sm.index:
            r = sm.loc[s]
            c = anchor + np.array([[r[f'h_{k}_{a}'] for k in ('c0', 'cr', 'cc', 'ch')] for a in ('y', 'x')])
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
    kinds = ['T', 'A', 'L', 'H']
    labels = ['translation', 'affine', 'local warp', 'affine\n+ elevation']
    med = rm.groupby('model')[['med_high_px', 'med_low_px']].median().reindex(kinds) * RES
    x = np.arange(len(kinds))
    ax.bar(x - 0.2, med.med_high_px, 0.4, color=C_HIGH, label='central plateau')
    ax.bar(x + 0.2, med.med_low_px, 0.4, color=C_LOW, label='lower ground')
    for xi, (a, b) in zip(x, med.values):
        ax.text(xi - 0.2, a - 0.05, f'{a:.1f}', ha='center', va='top', fontsize=FS_S, color='white')
        ax.text(xi + 0.2, b - 0.05, f'{b:.1f}', ha='center', va='top', fontsize=FS_S, color='white')
    noise = RES * models_table().noise_px.median()
    ax.axhline(noise, color='k', ls='--', lw=0.9)
    ax.set_xlim(-0.55, 4.45)
    ax.text(4.42, noise + 0.05, 'matching noise', fontsize=FS_S, ha='right')
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
        scene, an affine, a local warp (affine plus a 144 m Gaussian-smoothed residual field) and an affine plus a
        term proportional to elevation; the dashed line is the matching noise floor ({noise:.1f} m, the median
        red-versus-near-infrared disagreement of a tile's displacement divided by the square root of 2). Right:
        median distance of each scene's tiles from the 2024-10-14 whole-island drone orthomosaic (20 cm, averaged
        to 3 m), for the {len(d)} scenes matched to it, as delivered by Planet (orange; values above 20 m in the
        last bin) and after the affine + elevation correction anchored to the drone mosaic (green, computed by
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
        dy, dx = ic.displacement_maps(m, 'Ha', window=(c[0], c[0] + 1, c[1], c[1] + 1))
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
        dy, dx = ic.displacement_maps(model, 'Ha', window=win)
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
        affine + elevation correction anchored to the drone mosaic (bicubic resampling). Yellow outlines are the
        {len(crowns)} crowns of at least 250 m2 in the 2024 drone crown map, drawn at the same position in every
        panel and frame as reference marks. The sequence shows the {len(frames)} scenes that were furthest from
        their ground position at this window as delivered ({frames.off_m.min():.0f}-{frames.off_m.max():.0f} m; the
        median over all {len(ranked)} scenes is {ranked.off_m.median():.1f} m), among scenes at least 80 % clear
        over the island and fully clear in the window, at most one per day, in date order, {hold:g} s each.
        """.split()) + '\n')
    click.echo(f'wrote {mp4}, {gif}: {len(frames)} scenes, offsets {frames.off_m.round(1).tolist()}')


@cli.command(name='all')
@out_option
@click.pass_context
def all_(ctx, outdir):
    for cmd in (problem, diagnosis, steps, method, result, compare):
        ctx.invoke(cmd, outdir=outdir)


if __name__ == '__main__':
    cli()
