#!/usr/bin/env python
"""Diagnose the central-island band of the whole-island Planet coregistration
(island_coreg.py): is it match error on ambiguous canopy texture, the master
scene's own distortion, or Planet's orthorectification?

Subcommands (in order):

  spread     per-tile maps over all fitted scenes from the existing tiles table:
             mean translation residual, across-scene spread, red-NIR
             disagreement, NCC peak, survival rate; defines the "band" tiles.
  sample     the scene sample used by every test below (stratified by
             instrument and season, with a basic 1B version on disk) and the
             alternate masters.
  multi      measure the sample against the master and each alternate master,
             and each alternate master against the master, keeping per-tile
             NCC-surface diagnostics (secondary local maximum, peak curvature).
  closure    triangle closure d(S->M0) - d(Mk->M0) - d(S->Mk) per tile, and the
             mean residual field under each master.
  ortho1b    orthorectify basic (1B) scenes with their RPCs and the Copernicus
             DEM onto the island lattice (red, NIR and the udm2 clear band).
  orthocmp   tile field between our RPC ortho and Planet's ortho of the same
             image: what Planet's orthorectification did beyond RPC + DEM.
  island1b   the standard measurement and T/A/L fit on the RPC orthos against
             the RPC ortho of the master, next to the Planet-ortho results.
  metadata   Planet's quality flags and ortho XML fields, the reported view
             direction and the line of sight of each basic scene's RPC, against
             the fitted offset and elevation term of every scene.
  figures    figures and captions.

Displacement convention as in island_coreg.py: d = (dy, dx) px means the content
at reference pixel p appears at p + d in the scene.

Outputs: results/island_coreg/diagnose/ (tables), planet_bci_1b_ortho/ (rasters),
figs/20261116_updates/island_diag_* (figures). See docs/island_coregistration.md and,
for the metadata checks, docs/planet_geolocation_literature.md.
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import click  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import island_coreg as ic  # noqa: E402

RESULTS = ic.RESULTS
DIAG = RESULTS / 'diagnose'
FIGS = ic.FLOWER / 'figs' / '20261116_updates'
BASIC = ic.BASIC
ORTHO1B = ic.FLOWER / 'planet_bci_1b_ortho'
DEM_SRC = ic.DEM_DIR / f'{ic.DEM_NAME}.tif'
RES = ic.RES
MARGIN = 40  # px of margin around the island grid in the RPC orthos (search range + slack)
DRONE = 'drone_20241014'
DRONE_SRC = ic.FLOWER / 'whole_island' / '2024-10-14_orthoWholeBCI_res20.tif'
DRONE_3M = DIAG / 'drone_20241014_3m.tif'
BAND_MIN_M = 2.0  # archive-mean translation residual that defines the central band


def basic_path(scene_id):
    return BASIC / f'PSScene-{scene_id}' / 'assets' / 'basic_analytic_4b' / f'{scene_id}_1B_AnalyticMS.tif'


def basic_udm_path(scene_id):
    return BASIC / f'PSScene-{scene_id}' / 'assets' / 'basic_udm2' / f'{scene_id}_1A_udm2.tif'


def mad(v):
    v = np.asarray(v, float)
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def add_lattice(df):
    df = df.copy()
    df['ri'], df['ci'] = ic.lattice_index(df)
    return df


def grid(df, col):
    return ic.to_grid(df.ri.values, df.ci.values, df[col].values, ic.grid_shape())


def band_mask():
    """Boolean tile-lattice grid of the central band (from `spread`)."""
    s = pd.read_csv(DIAG / 'spread_tiles.csv')
    g = np.zeros(ic.grid_shape(), bool)
    b = s[s.in_band]
    g[b.ri, b.ci] = True
    return g


# --------------------------------------------------------------------------- spread

@click.group()
def cli():
    """Diagnostics of the whole-island coregistration."""


@cli.command()
def spread():
    """Per-tile statistics over all fitted scenes from results/island_coreg/tiles.csv.gz."""
    DIAG.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(RESULTS / 'tiles.csv.gz')
    g = pd.read_csv(RESULTS / 'global_shifts.csv')
    t = add_lattice(t[t.scene_id.isin(set(g[g.global_ok].scene_id))])
    n_scenes = t.scene_id.nunique()
    med = t.groupby(['scene_id', 'band'])[['dy_px', 'dx_px']].transform('median')
    t['ry'], t['rx'] = t.dy_px - med.dy_px, t.dx_px - med.dx_px
    a = t.groupby(['band', 'ri', 'ci']).agg(
        n=('ry', 'size'), mean_dy=('ry', 'mean'), mean_dx=('rx', 'mean'), mad_dy=('ry', mad),
        mad_dx=('rx', mad), peak=('peak', 'mean')).reset_index()
    a['mean_m'] = RES * np.hypot(a.mean_dy, a.mean_dx)
    a['spread_m'] = RES * np.hypot(a.mad_dy, a.mad_dx)
    a['survival'] = a.n / n_scenes
    p = t.pivot_table(index=['scene_id', 'ri', 'ci'], columns='band', values=['ry', 'rx']).dropna()
    p['dis'] = RES * np.hypot(p[('ry', 'red')] - p[('ry', 'nir')], p[('rx', 'red')] - p[('rx', 'nir')])
    dis = p.groupby(['ri', 'ci']).dis.median().rename('rednir_m').reset_index()
    r = a[a.band == 'red'].merge(a[a.band == 'nir'][['ri', 'ci', 'mean_dy', 'mean_dx', 'mean_m', 'spread_m', 'peak']],
                                 on=['ri', 'ci'], how='left', suffixes=('', '_nir')).merge(dis, on=['ri', 'ci'], how='left')
    r = r[r.n >= 100].drop(columns='band')
    # the band: the largest connected set of tiles with an archive-mean residual >= BAND_MIN_M
    gm = grid(r, 'mean_m')
    lab, n = ndimage.label(np.nan_to_num(gm) >= BAND_MIN_M, structure=np.ones((3, 3)))
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    big = lab == (1 + int(np.argmax(sizes)))
    r['in_band'] = big[r.ri, r.ci]
    r.to_csv(DIAG / 'spread_tiles.csv', index=False)
    cols = ['mean_m', 'spread_m', 'rednir_m', 'peak', 'survival']
    click.echo(f'{n_scenes} scenes, {len(r)} tiles with >= 100 scenes; band = {int(r.in_band.sum())} tiles')
    click.echo(r.groupby('in_band')[cols].median().round(2).to_string())
    click.echo('Spearman:\n' + r[cols].corr(method='spearman').round(2).to_string())


# --------------------------------------------------------------------------- sample

@cli.command()
@click.option('--n', default=60, show_default=True)
@click.option('--n-masters', default=3, show_default=True)
@click.option('--seed', default=0, show_default=True)
def sample(n, n_masters, seed):
    """Scene sample (with basic scenes on disk) and alternate masters."""
    DIAG.mkdir(parents=True, exist_ok=True)
    sm = pd.read_csv(RESULTS / 'scene_models.csv', parse_dates=['acquired'])
    sm = sm[(sm.band == 'both') & sm.med_T.notna()].copy()
    sm['has_basic'] = sm.scene_id.map(lambda s: basic_path(s).exists())
    sm['season'] = np.where(sm.acquired.dt.month.isin([1, 2, 3, 4]), 'dry', 'wet')
    pool = sm[sm.has_basic & (sm.scene_id != ic.MASTER)]
    # alternate masters: fully clear, one per (instrument, season) class away from the master's
    cand = pool[(pool.fraction_clear >= 0.995) & (pool.n_tiles >= pool.n_tiles.quantile(0.5))]
    masters = []
    for key in [('PSB.SD', 'wet'), ('PS2', 'dry'), ('PSB.SD', 'dry'), ('PS2.SD', 'dry'), ('PS2', 'wet')]:
        c = cand[(cand.instrument == key[0]) & (cand.season == key[1])
                 & ~cand.satellite_id.isin([m['satellite_id'] for m in masters] + ['24be'])]
        if len(c) and len(masters) < n_masters:
            row = c.sort_values('med_Lh').iloc[0]
            masters.append(dict(scene_id=row.scene_id, instrument=row.instrument, season=row.season,
                                satellite_id=row.satellite_id, fraction_clear=float(row.fraction_clear),
                                view_angle=float(row.view_angle), med_T_px=float(row.med_T),
                                affine_amp_m=float(row.affine_amp_m)))
    (DIAG / 'masters.json').write_text(json.dumps(dict(master=ic.MASTER, alternates=masters), indent=1))
    # sample: the three worst cases of the doc, then stratified at random among clear scenes
    forced = ['20201010_155324_99_2412', '20230328_144849_31_241a', '20240117_155649_55_24d1']
    forced = [s for s in forced if s in set(pool.scene_id)]
    rest = pool[(pool.fraction_clear >= 0.8) & ~pool.scene_id.isin(forced + [m['scene_id'] for m in masters])]
    rng = np.random.default_rng(seed)
    groups = list(rest.groupby(['instrument', 'season']))
    per = max(1, (n - len(forced)) // len(groups))
    chosen = []
    for _, gdf in groups:
        chosen += list(rng.choice(gdf.scene_id.values, min(per, len(gdf)), replace=False))
    left = rest[~rest.scene_id.isin(chosen)]
    extra = n - len(forced) - len(chosen)
    if extra > 0:
        chosen += list(rng.choice(left.scene_id.values, min(extra, len(left)), replace=False))
    out = sm[sm.scene_id.isin(forced + chosen)][
        ['scene_id', 'acquired', 'instrument', 'satellite_id', 'season', 'fraction_clear', 'view_angle',
         'med_T', 'med_A', 'med_Lh', 'affine_amp_m', 'coherence']].sort_values('acquired')
    out.to_csv(DIAG / 'sample.csv', index=False)
    click.echo(f'{len(out)} sample scenes: {out.groupby(["instrument", "season"]).size().to_dict()}')
    click.echo('alternate masters:\n' + pd.DataFrame(masters).to_string())


def load_sample():
    return pd.read_csv(DIAG / 'sample.csv', parse_dates=['acquired'])


def load_masters():
    j = json.loads((DIAG / 'masters.json').read_text())
    return [j['master']] + [m['scene_id'] for m in j['alternates']]


# --------------------------------------------------------------------------- multi-master measurement

_REF = {}


def ref_data(scene_id, source='planet'):
    """Reference dict (bands, valid) of any scene, cached per process."""
    key = (scene_id, source)
    if key not in _REF:
        if scene_id == DRONE:
            with rasterio.open(DRONE_3M) as src:
                d = src.read().astype(np.float32)
            # red, and green in the slot the matcher calls 'nir'
            _REF[key] = dict(bands=dict(red=d[0], nir=d[1]), valid=(d[3] > 250) & ic.land_mask())
        else:
            bands, clear, data = read_any(scene_id, source)
            _REF[key] = dict(bands=bands, valid=clear & data & ic.land_mask())
    return _REF[key]


def read_any(scene_id, source):
    """(bands dict, clear, data) on the island grid from Planet's ortho ('planet')
    or our RPC ortho of the basic scene ('rpc')."""
    if source == 'planet':
        return ic.read_scene(scene_id)
    if source == 'drone':  # Planet red and green, to match the drone's red and green
        p4, pocm = ic.scene_paths(scene_id)
        a = ic.read_island(p4, [3, 2]).astype(np.float32)
        ocm = ic.read_island(pocm, [1], fill=ic.OCM_NODATA)[0]
        return dict(red=a[0], nir=a[1]), ocm == ic.OCM_CLEAR, a[0] > 0
    # cloud mask: OmniCloudMask of Planet's ortho of the same image (they agree to a
    # fraction of a pixel, see `orthocmp`), and the udm2 clear band of the basic scene
    arr, udm_clear = read_rpc(scene_id)
    _, clear, _ = ic.read_scene(scene_id, bands=('red',))
    return dict(red=arr[0], nir=arr[1]), clear & udm_clear, arr[0] > 0


def surface_stats(r):
    """Ambiguity diagnostics of an NCC surface: the best secondary local maximum
    (value and distance from the peak, NaN if none), the peak curvature (peak minus
    the mean of its 4 neighbors) and the number of cells within 10% of the peak."""
    i, j = np.unravel_index(np.argmax(r), r.shape)
    peak = r[i, j]
    locmax = (r == ndimage.maximum_filter(r, size=3, mode='nearest'))
    locmax[i, j] = False
    yy, xx = np.nonzero(locmax)
    if len(yy):
        k = np.argmax(r[yy, xx])
        sec, dist = float(r[yy[k], xx[k]]), float(np.hypot(yy[k] - i, xx[k] - j))
    else:
        sec, dist = np.nan, np.nan
    if 0 < i < r.shape[0] - 1 and 0 < j < r.shape[1] - 1:
        curv = float(peak - (r[i - 1, j] + r[i + 1, j] + r[i, j - 1] + r[i, j + 1]) / 4)
    else:
        curv = np.nan
    return dict(sec_local=sec, sec_dist=dist, curv=curv, n_near=int((r >= 0.9 * peak).sum()))


def measure_pair(args):
    scene_id, ref_id, source, outdir = args
    out = outdir / f'{scene_id}__{ref_id}.csv'
    if out.exists():
        return scene_id, ref_id, 'cached'
    try:
        bands, clear, data = read_any(scene_id, source)
        rec, tiles = ic.measure_arrays(bands, clear, data, surfaces=True, md=ref_data(ref_id, source))
    except Exception as e:  # noqa: BLE001
        return scene_id, ref_id, f'error: {e}'
    rows = []
    for t in tiles:
        s = surface_stats(t.pop('surface'))
        rows.append({**t, **s})
    df = pd.DataFrame(rows, columns=['band', 'row', 'col', 'dy_px', 'dx_px', 'peak', 'valid_frac', 'method',
                                     'second', 'sec_local', 'sec_dist', 'curv', 'n_near'])
    df = df.assign(scene_id=scene_id, ref_id=ref_id, global_dy_px=rec['global_dy_px'],
                   global_dx_px=rec['global_dx_px'], global_peak=rec['global_peak'], global_ok=rec['global_ok'])
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return scene_id, ref_id, f"ok {len(df)} tiles, global peak {rec['global_peak']:.2f}"


def run_pairs(pairs, source, outdir, jobs):
    t0 = time.time()
    with Pool(jobs) as pool:
        for k, (s, r, msg) in enumerate(pool.imap_unordered(
                measure_pair, [(s, r, source, outdir) for s, r in pairs]), 1):
            if msg.startswith('error') or k % 20 == 0 or k == len(pairs):
                click.echo(f'[{k}/{len(pairs)} {time.time() - t0:5.0f}s] {s} vs {r}: {msg}')
    dfs = [pd.read_csv(outdir / f'{s}__{r}.csv') for s, r in pairs if (outdir / f'{s}__{r}.csv').exists()]
    return pd.concat([d for d in dfs if len(d)], ignore_index=True)  # failed global matches have no tiles


@cli.command()
@click.option('--jobs', default=5, show_default=True)
@click.option('--source', type=click.Choice(['planet', 'rpc']), default='planet', show_default=True)
def multi(jobs, source):
    """Sample scenes against the master and each alternate master; alternates against
    each other and the master."""
    masters = load_masters()
    ids = load_sample().scene_id.tolist()
    if source == 'rpc':
        ids = [s for s in ids if rpc_path(s).exists()]
        masters = [m for m in masters if rpc_path(m).exists()]
    pairs = [(s, m) for m in masters for s in ids]
    pairs += [(a, b) for a in masters for b in masters if a != b]
    df = run_pairs(pairs, source, DIAG / f'pairs_{source}', jobs)
    df.to_csv(DIAG / f'multi_{source}.csv.gz', index=False)
    click.echo(f'{len(df)} tile vectors over {len(pairs)} pairs -> {DIAG / f"multi_{source}.csv.gz"}')


def residual_T(df):
    """Translation residual per (scene, ref, band): tile vector minus the pair's median."""
    med = df.groupby(['scene_id', 'ref_id', 'band'])[['dy_px', 'dx_px']].transform('median')
    return df.assign(ry=df.dy_px - med.dy_px, rx=df.dx_px - med.dx_px)


@cli.command()
@click.option('--source', type=click.Choice(['planet', 'rpc']), default='planet', show_default=True)
def closure(source):
    """Triangle closure per tile and the mean residual field under each master."""
    masters = load_masters()
    m0 = masters[0]
    df = add_lattice(pd.read_csv(DIAG / f'multi_{source}.csv.gz'))
    df = residual_T(df[df.global_ok])
    key = ['band', 'ri', 'ci']
    inband = band_mask()
    df['in_band'] = inband[df.ri, df.ci]
    is_master = df.scene_id.isin(masters)
    s0 = df[(df.ref_id == m0) & ~is_master]
    rows, summ = [], []
    for mk in masters[1:]:
        if mk not in set(df.ref_id):
            continue
        sk = df[(df.ref_id == mk) & ~is_master]
        k0 = df[(df.scene_id == mk) & (df.ref_id == m0)]
        j = s0.merge(sk, on=['scene_id'] + key, suffixes=('_0', '_k')).merge(
            k0[key + ['dy_px', 'dx_px']].rename(columns={'dy_px': 'dy_k0', 'dx_px': 'dx_k0'}), on=key)
        j['cy'] = j.dy_px_0 - j.dy_k0 - j.dy_px_k
        j['cx'] = j.dx_px_0 - j.dx_k0 - j.dx_px_k
        j['closure_m'] = RES * np.hypot(j.cy, j.cx)
        j['mk'] = mk
        rows.append(j[['scene_id', 'mk'] + key + ['cy', 'cx', 'closure_m', 'peak_0', 'peak_k', 'sec_local_0',
                                                   'curv_0', 'in_band_0']].rename(columns={'in_band_0': 'in_band'}))
    c = pd.concat(rows, ignore_index=True)
    c.to_csv(DIAG / f'closure_{source}.csv.gz', index=False)
    click.echo(f'closure over {len(c)} tile triangles ({c.scene_id.nunique()} scenes x {c.mk.nunique()} alternates)')
    click.echo('median |closure| (m), share > 3 m, share > 6 m, by band and region:')
    g = c.groupby(['band', 'in_band']).closure_m
    click.echo(pd.DataFrame(dict(n=g.size(), median=g.median(), p90=g.quantile(0.9), gt3=g.apply(lambda v: (v > 3).mean()),
                                 gt6=g.apply(lambda v: (v > 6).mean()))).round(3).to_string())
    # per-tile closure map (all alternates, red)
    ct = c.groupby(key).agg(closure_med_m=('closure_m', 'median'), closure_gt3=('closure_m', lambda v: (v > 3).mean()),
                            n=('closure_m', 'size')).reset_index()
    ct.to_csv(DIAG / f'closure_tiles_{source}.csv', index=False)
    # mean residual field under each master (sample scenes only) and the master-to-master fields
    nm = df[~is_master]
    mf = nm.groupby(['ref_id'] + key).agg(mean_dy=('ry', 'mean'), mean_dx=('rx', 'mean'), n=('ry', 'size'),
                                          mad_dy=('ry', mad), mad_dx=('rx', mad), peak=('peak', 'mean'),
                                          sec_local=('sec_local', 'mean'), has_sec=('sec_local', lambda v: v.notna().mean()),
                                          curv=('curv', 'mean'), n_near=('n_near', 'mean')).reset_index()
    mf['mean_m'] = RES * np.hypot(mf.mean_dy, mf.mean_dx)
    mf['spread_m'] = RES * np.hypot(mf.mad_dy, mf.mad_dx)
    mf['in_band'] = inband[mf.ri, mf.ci]
    mf.to_csv(DIAG / f'master_fields_{source}.csv', index=False)
    mm = df[is_master][['scene_id', 'ref_id'] + key + ['ry', 'rx', 'peak', 'in_band']]
    mm.to_csv(DIAG / f'master_pairs_{source}.csv', index=False)
    ok = mf[(mf.n >= 0.4 * nm.scene_id.nunique()) & (mf.band == 'red')]
    click.echo('mean residual field (red), median over tiles of |mean| (m) and spread (m), by master and region:')
    click.echo(ok.groupby(['ref_id', 'in_band'])[['mean_m', 'spread_m', 'peak', 'curv', 'has_sec', 'n_near']]
               .median().round(3).to_string())
    # the band direction: mean vector over band tiles, per master
    for ref, d in ok[ok.in_band].groupby('ref_id'):
        click.echo(f'  {ref}: band-mean vector dy {RES * d.mean_dy.mean():+.2f} m, dx {RES * d.mean_dx.mean():+.2f} m')
    for (s, r), d in mm[mm.band == 'red'].groupby(['scene_id', 'ref_id']):
        b = d[d.in_band]
        click.echo(f'  master pair {s} vs {r}: band-mean residual dy {RES * b.ry.mean():+.2f} m, '
                   f'dx {RES * b.rx.mean():+.2f} m; outside |mean| '
                   f'{RES * np.hypot(d[~d.in_band].ry.mean(), d[~d.in_band].rx.mean()):.2f} m')


# --------------------------------------------------------------------------- RPC orthorectification of basic scenes

def rpc_path(scene_id):
    return ORTHO1B / scene_id[:4] / f'{scene_id}_1b_ortho.tif'


def rpc_bounds():
    l, b, r, t = ic.ISLAND_BOUNDS
    m = MARGIN * RES
    return l - m, b - m, r + m, t + m


def ortho_one(args):
    scene_id, dem = args
    out = rpc_path(scene_id)
    if out.exists():
        return scene_id, 'cached'
    src, udm = basic_path(scene_id), basic_udm_path(scene_id)
    if not src.exists():
        return scene_id, 'no basic scene'
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    tmp.mkdir(exist_ok=True)
    l, b, r, t = rpc_bounds()
    common = ['-rpc', '-to', f'RPC_DEM={dem}', '-to', 'RPC_DEM_MISSING_VALUE=0', '-t_srs', ic.CRS,
              '-te', str(l), str(b), str(r), str(t), '-tr', str(RES), str(RES), '-overwrite', '-q',
              '-et', '0', '-wo', 'NUM_THREADS=1']
    try:
        a = tmp / 'a.tif'
        subprocess.run(['gdalwarp', *common, '-r', 'cubic', '-b', '3', '-b', '4', '-srcnodata', '0',
                        '-dstnodata', '0', str(src), str(a)], check=True, capture_output=True)
        layers = [rasterio.open(a).read()]
        if udm.exists():
            # the 1A udm2 has the basic scene's geometry but no RPC of its own: borrow it via a VRT
            u = tmp / 'u.tif'
            v = tmp / 'u.vrt'
            subprocess.run(['gdal_translate', '-q', '-of', 'VRT', '-b', '1', str(udm), str(v)], check=True,
                           capture_output=True)
            rpc = rasterio.open(src).tags(ns='RPC')
            txt = v.read_text().replace('</VRTDataset>', '  <Metadata domain="RPC">\n' + ''.join(
                f'    <MDI key="{k}">{val}</MDI>\n' for k, val in rpc.items()) + '  </Metadata>\n</VRTDataset>')
            v.write_text(txt)
            subprocess.run(['gdalwarp', *common, '-r', 'near', str(v), str(u)], check=True, capture_output=True)
            layers.append(rasterio.open(u).read())
        else:
            layers.append((layers[0][:1] > 0).astype(layers[0].dtype))
        arr = np.concatenate(layers).astype(np.uint16)
        ic.write_tif(out, arr, 'uint16', 0, transform=rasterio.open(a).transform)
    except subprocess.CalledProcessError as e:
        return scene_id, f'error: {e.stderr.decode()[-300:]}'
    finally:
        for f in tmp.glob('*'):
            f.unlink()
        tmp.rmdir()
    cover = float((arr[0] > 0).mean())
    return scene_id, f'ok, {cover:.0%} of the window has data'


@cli.command()
@click.option('--jobs', default=5, show_default=True)
@click.option('--scenes', default=None, help='Comma-separated ids (default: sample + masters).')
@click.option('--all-fitted', is_flag=True, help='Every fitted scene with a basic version.')
def ortho1b(jobs, scenes, all_fitted):
    """gdalwarp -rpc (cubic, RPC_DEM = Copernicus GLO-30) of the red and NIR bands of
    basic scenes onto the island lattice (+ margin); band 3 is the udm2 clear mask."""
    if scenes:
        ids = scenes.split(',')
    elif all_fitted:
        sm = pd.read_csv(RESULTS / 'scene_models.csv')
        ids = sorted(set(sm.scene_id))
    else:
        ids = load_masters() + load_sample().scene_id.tolist()
    ids = [s for s in ids if basic_path(s).exists()]
    t0 = time.time()
    with Pool(jobs) as pool:
        for k, (s, msg) in enumerate(pool.imap_unordered(ortho_one, [(s, DEM_SRC) for s in ids]), 1):
            if not msg.startswith('ok') or k % 10 == 0 or k == len(ids):
                click.echo(f'[{k}/{len(ids)} {time.time() - t0:5.0f}s] {s}: {msg}')


def read_rpc(scene_id):
    """Red, NIR (float32) of our RPC ortho on the island grid, and its clear mask
    (udm2 clear where available, combined with Planet-ortho OCM by the caller if wanted)."""
    with rasterio.open(rpc_path(scene_id)) as src:
        arr = src.read()[:, MARGIN:-MARGIN, MARGIN:-MARGIN]
    assert arr.shape[-2:] == ic.ISLAND_SHAPE, arr.shape
    return arr[:2].astype(np.float32), arr[2] == 1


def orthocmp_one(scene_id):
    try:
        arr, _ = read_rpc(scene_id)
        bands, clear, data = ic.read_scene(scene_id)
        md = dict(bands=bands, valid=clear & data & ic.land_mask())
        rec, tiles = ic.measure_arrays(dict(red=arr[0], nir=arr[1]), clear, arr[0] > 0, md=md)
    except Exception as e:  # noqa: BLE001
        return scene_id, None, None, f'error: {e}'
    t = pd.DataFrame(tiles).drop(columns='second').assign(scene_id=scene_id)
    if not rec['global_ok'] or len(t) < 60:
        return scene_id, None, None, f"global peak {rec['global_peak']:.2f}, {len(t)} tiles"
    out, _, _ = ic.fit_scene(t[t.band == 'red'], rec)
    out.update(scene_id=scene_id, global_dy_px=rec['global_dy_px'], global_dx_px=rec['global_dx_px'],
               global_peak=rec['global_peak'], tile_peak=float(t.peak.median()))
    return scene_id, out, t, f"ok T residual {RES * out['med_T']:.2f} m, affine {out['affine_amp_m']:.2f} m"


@cli.command()
@click.option('--jobs', default=5, show_default=True)
def orthocmp(jobs):
    """Our RPC ortho against Planet's ortho of the same image, for every RPC ortho on disk
    that is also a fitted scene (or a master)."""
    sm = pd.read_csv(RESULTS / 'scene_models.csv')
    ids = [s for s in sorted(set(sm.scene_id) | set(load_masters())) if rpc_path(s).exists()]
    rows, tiles = [], []
    t0 = time.time()
    with Pool(jobs) as pool:
        for k, (s, out, t, msg) in enumerate(pool.imap_unordered(orthocmp_one, ids), 1):
            if out is not None:
                rows.append(out)
                tiles.append(t)
            if out is None or k % 20 == 0 or k == len(ids):
                click.echo(f'[{k}/{len(ids)} {time.time() - t0:5.0f}s] {s}: {msg}')
    r = pd.DataFrame(rows)
    ref = sm[sm.band == 'both'].set_index('scene_id')
    for c in ('med_T', 'med_A', 'affine_amp_m', 'instrument', 'view_angle'):
        r[f'island_{c}'] = r.scene_id.map(ref[c])
    r.to_csv(DIAG / 'orthocmp_scenes.csv', index=False)
    t = add_lattice(pd.concat(tiles, ignore_index=True))
    t['in_band'] = band_mask()[t.ri, t.ci]
    t.to_csv(DIAG / 'orthocmp_tiles.csv.gz', index=False)
    r['offset_m'] = RES * np.hypot(r.t_dy_px, r.t_dx_px)
    r['med_T_m'], r['p90_T_m'] = RES * r.med_T, RES * r.p90_T
    click.echo(f'{len(r)} scenes. RPC ortho vs Planet ortho of the same image:')
    click.echo(r[['offset_m', 'med_T_m', 'p90_T_m', 'affine_amp_m', 'tile_peak']].describe().round(3).to_string())
    click.echo('against the scene-to-master distortion of the same scenes: island med_T '
               f'{RES * r.island_med_T.median():.2f} m, affine amplitude {r.island_affine_amp_m.median():.2f} m')
    click.echo(r.groupby('island_instrument')[['offset_m', 'med_T_m', 'affine_amp_m']].median().round(3).to_string())


def fit_pairs(df, tilts=None):
    """T/A/L/H fit (red, NIR referred to red) per (scene, ref) of a multi table. With
    `tilts` (scene_id -> elevation coefficients in px per m, or None) also model R."""
    rows = []
    for (s, r), d in df[df.global_ok].groupby(['scene_id', 'ref_id']):
        red, nir = d[d.band == 'red'], d[d.band == 'nir'].copy()
        if len(red) < 60:
            continue
        if len(nir):
            nir['dy_px'] -= nir.dy_px.median() - red.dy_px.median()
            nir['dx_px'] -= nir.dx_px.median() - red.dx_px.median()
        both = pd.concat([red, nir], ignore_index=True)[['band', 'row', 'col', 'dy_px', 'dx_px', 'peak',
                                                         'valid_frac', 'method']]
        out, tdf, _ = ic.fit_scene(both, None, tilt='skip' if tilts is None else tilts.get(s))
        if tdf is None:
            continue
        inb = band_mask()[tdf.ri, tdf.ci]
        out.update(scene_id=s, ref_id=r, peak=float(d.peak.median()),
                   med_T_band=float(np.median(np.hypot(tdf.rT_dy, tdf.rT_dx)[inb])) if inb.any() else np.nan,
                   med_T_out=float(np.median(np.hypot(tdf.rT_dy, tdf.rT_dx)[~inb])),
                   med_Lh_band=float(np.nanmedian(np.hypot(tdf.rLh_dy, tdf.rLh_dx)[inb])) if inb.any() else np.nan,
                   med_Lh_out=float(np.nanmedian(np.hypot(tdf.rLh_dy, tdf.rLh_dx)[~inb])))
        rows.append(out)
    return pd.DataFrame(rows)


@cli.command()
def island1b():
    """T/A/L fits of the sample against each master from the RPC orthos and from Planet's
    orthos, side by side (needs `multi --source rpc` and `multi --source planet`)."""
    res = {}
    for source in ('planet', 'rpc'):
        f = fit_pairs(pd.read_csv(DIAG / f'multi_{source}.csv.gz'))
        f.to_csv(DIAG / f'fits_{source}.csv', index=False)
        res[source] = f
    masters = load_masters()
    j = res['planet'].merge(res['rpc'], on=['scene_id', 'ref_id'], suffixes=('_planet', '_rpc'))
    j = j[~j.scene_id.isin(masters)]
    j.to_csv(DIAG / 'island1b_compare.csv', index=False)
    cols = ['med_T', 'med_A', 'med_Lh', 'med_T_band', 'med_T_out', 'affine_amp_m']
    click.echo(f'{len(j)} scene-master pairs measured in both geometries; medians (px, affine in m):')
    tab = pd.DataFrame({src: [j[f'{c}_{src}'].median() for c in cols] for src in ('planet', 'rpc')}, index=cols)
    tab['spearman'] = [j[f'{c}_planet'].corr(j[f'{c}_rpc'], method='spearman') for c in cols]
    click.echo(tab.round(3).to_string())
    for c in ('jac_yy_m', 'jac_xx_m', 'jac_xy_m', 'jac_yx_m', 't_dy_px', 't_dx_px'):
        click.echo(f'  {c}: r = {j[f"{c}_planet"].corr(j[f"{c}_rpc"]):.3f}, '
                   f'median |planet - rpc| = {(j[f"{c}_planet"] - j[f"{c}_rpc"]).abs().median():.3f}')


H_PARAMS = ['h_c0_y', 'h_cr_y', 'h_cc_y', 'h_ch_y', 'h_c0_x', 'h_cr_x', 'h_cc_x', 'h_ch_x']


@cli.command()
@click.option('--jobs', default=4, show_default=True)
@click.option('--n', default=120, show_default=True, help='Scenes nearest in date to the drone flight.')
@click.option('--min-clear', default=0.9, show_default=True)
def drone(jobs, n, min_clear):
    """Planet scenes against the whole-island drone orthomosaic of 2024-10-14 (20 cm,
    averaged to the 3 m island grid): absolute affine + elevation terms per scene, and
    the master's own terms as the median of (scene vs drone) - (scene vs master)."""
    if not DRONE_3M.exists():
        l, b, r, t = ic.ISLAND_BOUNDS
        subprocess.run(['gdalwarp', '-q', '-overwrite', '-t_srs', ic.CRS, '-te', str(l), str(b), str(r), str(t),
                        '-tr', str(RES), str(RES), '-r', 'average', str(DRONE_SRC), str(DRONE_3M)], check=True)
    sm = pd.read_csv(RESULTS / 'scene_models.csv', parse_dates=['acquired'])
    sm = sm[sm.band == 'both'].set_index('scene_id')
    c = sm[sm.fraction_clear >= min_clear].copy()
    c['dt'] = (c.acquired.dt.tz_localize(None) - pd.Timestamp('2024-10-14')).abs()
    ids = [ic.MASTER] + [s for s in c.sort_values('dt').index[:n] if s != ic.MASTER]
    df = run_pairs([(s, DRONE) for s in ids], 'drone', DIAG / 'pairs_drone', jobs)
    df.to_csv(DIAG / 'drone_tiles.csv.gz', index=False)
    # against the drone the elevation term of model R is the scene's whole tilt, not the tilt relative to the master
    tl = ic.load_tilts()
    abs_tilt = {s: np.array([-tl[s][1], tl[s][0]]) / (RES * 100) for s in ids if s in tl}
    f = fit_pairs(df, abs_tilt).set_index('scene_id')
    f.to_csv(DIAG / 'drone_fits.csv')
    click.echo(f'{len(f)} of {len(ids)} scenes matched to the drone mosaic; median tile peak {f.peak.median():.2f}; '
               f'median residual T {RES * f.med_T.median():.2f} m, A {RES * f.med_A.median():.2f} m, '
               f'H {RES * f.med_H.median():.2f} m, HL hold-out {RES * f.med_HLh.median():.2f} m')
    # the master's own model: direct, and through every other scene
    j = f.join(sm[H_PARAMS + ['k_north_m100', 'k_east_m100']], rsuffix='_rel', how='inner')
    j = j[j.index != ic.MASTER]
    est = pd.DataFrame({p: j[p] - j[f'{p}_rel'] for p in H_PARAMS})
    out = dict(n=len(est), median={p: float(est[p].median()) for p in H_PARAMS},
               mad={p: float(mad(est[p])) for p in H_PARAMS},
               direct={p: float(f.loc[ic.MASTER, p]) for p in H_PARAMS} if ic.MASTER in f.index else None)
    # the same for model R: planar terms as the median of (scene vs drone) - (scene vs master), each fitted
    # with the scene's elevation term fixed at its RPC tilt; elevation term = the master's own RPC tilt
    R_PARAMS = ['r' + p[1:] for p in H_PARAMS]
    if 'r_c0_y' in sm and 'r_c0_y' in f:
        jr = f[R_PARAMS].join(sm[R_PARAMS], rsuffix='_rel', how='inner')
        jr = jr[jr.index != ic.MASTER].dropna()
        est_r = pd.DataFrame({p: jr[p] - jr[f'{p}_rel'] for p in R_PARAMS})
        r = {'h' + p[1:]: float(est_r[p].median()) for p in R_PARAMS}
        if ic.MASTER in tl:
            r['h_ch_y'], r['h_ch_x'] = -tl[ic.MASTER][1] / (RES * 100), tl[ic.MASTER][0] / (RES * 100)
        out['R'] = r
        out['R_n'] = len(est_r)
        out['R_mad'] = {'h' + p[1:]: float(mad(est_r[p])) for p in R_PARAMS}
        click.echo(f'model R anchor from {len(est_r)} scenes: offset at the island origin '
                   f'{RES * r["h_c0_y"]:+.2f}, {RES * r["h_c0_x"]:+.2f} px->m (dy, dx); elevation term from the '
                   f'master RPC {-r["h_ch_y"] * RES * 100:+.2f} N, {r["h_ch_x"] * RES * 100:+.2f} E m per 100 m '
                   f'(drone tie: {-out["median"]["h_ch_y"] * RES * 100:+.2f} N, {out["median"]["h_ch_x"] * RES * 100:+.2f} E)')
    (DIAG / 'master_abs.json').write_text(json.dumps(out, indent=1))
    (RESULTS / 'master_abs.json').write_text(json.dumps(out, indent=1))  # read by island_coreg.py (models Ha, Ra)
    kn, ke = -est.h_ch_y * RES * 100, est.h_ch_x * RES * 100
    click.echo(f'master elevation term from {len(est)} scenes: north {kn.median():+.2f} (MAD {mad(kn):.2f}), '
               f'east {ke.median():+.2f} (MAD {mad(ke):.2f}) m per 100 m; direct: '
               + (f"{f.loc[ic.MASTER, 'k_north_m100']:+.2f}, {f.loc[ic.MASTER, 'k_east_m100']:+.2f}"
                  if ic.MASTER in f.index else 'n/a'))
    click.echo(f'archive consensus (minus the median over all scenes of the term relative to the master): '
               f'north {-sm.k_north_m100.median():+.2f}, east {-sm.k_east_m100.median():+.2f}')
    click.echo(f'scene terms, drone vs master-relative: r = {j.k_north_m100.corr(j.k_north_m100_rel):.2f} (north), '
               f'{j.k_east_m100.corr(j.k_east_m100_rel):.2f} (east)')
    h, w = ic.ISLAND_SHAPE
    click.echo('master affine terms across the island (m), median (MAD) over scenes: '
               + ', '.join(f'{nm} {RES * est[p].median() * L:+.2f} ({RES * mad(est[p]) * L:.2f})' for nm, p, L in
                           [('dy/row', 'h_cr_y', h), ('dy/col', 'h_cc_y', w), ('dx/row', 'h_cr_x', h), ('dx/col', 'h_cc_x', w)]))


PLOT_CENTER = (960, 736)  # row, col of the 50ha plot center on the island grid


@cli.command()
@click.option('--shifts', type=click.Path(path_type=Path),
              default=ic.FLOWER / 'crown_series_20261101' / 'coreg_all_scenes.json', show_default=True)
def crosscheck(shifts):
    """Island models evaluated at the 50ha plot center against the independent
    Planet-to-drone shifts (one AROSICS window per scene over the plot)."""
    d = pd.DataFrame(json.loads(Path(shifts).read_text()))
    d = d[d.coreg_ok].assign(scene_id=d.scene.str.replace('_4band', ''))[['scene_id', 'x_shift_m', 'y_shift_m']]
    r0, c0 = PLOT_CENTER
    win = (r0, r0 + 1, c0, c0 + 1)
    rows = []
    for s in d.scene_id:
        p = ic.model_path(RESULTS, s)
        if not p.exists():
            continue
        m = ic.load_model(p)
        row = dict(scene_id=s)
        for kind in ic.MODELS:
            try:
                dy, dx = ic.displacement_maps(m, kind, window=win)
            except KeyError:
                continue
            row[f'{kind}_east'], row[f'{kind}_north'] = RES * float(dx[0, 0]), -RES * float(dy[0, 0])
        rows.append(row)
    j = pd.DataFrame(rows).merge(d, on='scene_id')
    j.to_csv(DIAG / 'plot_crosscheck.csv', index=False)
    click.echo(f'{len(j)} scenes with an island model and a Planet-to-drone shift')
    for kind in ic.MODELS:
        if f'{kind}_east' not in j:
            continue
        # the sign of the drone shift relative to the island convention is fixed by the data
        best = None
        for sg in (1, -1):
            e = j[f'{kind}_east'] - sg * j.x_shift_m
            n = j[f'{kind}_north'] - sg * j.y_shift_m
            dis = np.hypot(e - e.median(), n - n.median())
            if best is None or dis.median() < best[0]:
                best = (dis.median(), (dis <= 3).mean(), sg)
        click.echo(f'  {ic.MODEL_NAMES[kind]:36s} median disagreement {best[0]:.2f} m, within 3 m {best[1]:.0%} '
                   f'(sign {best[2]:+d})')


# --------------------------------------------------------------------------- Planet metadata and RPC line of sight

# the helpers live in island_coreg.py (model R uses them at fit time)
META_KEYS, XML_KEYS, RPC_H0 = ic.META_KEYS, ic.XML_KEYS, ic.RPC_H0
rpc_txt_path, scene_metadata, load_rpc = ic.rpc_txt_path, ic.scene_metadata, ic.load_rpc
rpc_project, rpc_ground, rpc_los = ic.rpc_project, ic.rpc_ground, ic.rpc_los


def _vector_fit(y_e, y_n, x_e, x_n):
    """Least-squares scalar c in (y_e, y_n) = c (x_e, x_n) + const, and the share of
    the variance of y it explains."""
    ok = np.isfinite(y_e) & np.isfinite(y_n) & np.isfinite(x_e) & np.isfinite(x_n)
    y = np.concatenate([y_e[ok] - y_e[ok].mean(), y_n[ok] - y_n[ok].mean()])
    x = np.concatenate([x_e[ok] - x_e[ok].mean(), x_n[ok] - x_n[ok].mean()])
    c = float(x @ y / (x @ x))
    return c, float(1 - ((y - c * x) ** 2).sum() / (y ** 2).sum()), int(ok.sum())


@cli.command()
def metadata():
    """Planet's own metadata and the RPC line of sight against the fitted terms: (1) the
    ground_control / quality flags and cloud fields, (2) the elevation model recorded in
    the ortho XML, (3) the metadata view direction, and (4) the line of sight of each
    basic scene's RPC, each against the scene's offset and elevation term."""
    from pyproj import Transformer
    from scipy.stats import spearmanr
    sm = pd.read_csv(RESULTS / 'scene_models.csv')
    sm = sm[sm.band == 'both'].set_index('scene_id')
    idx = pd.read_csv(ic.SR / 'index.csv').set_index('scene_id')
    absm = json.loads((RESULTS / 'master_abs.json').read_text())['median']
    k_master = np.array([absm['h_ch_x'], -absm['h_ch_y']]) * RES * 100  # east, north m per 100 m
    cx, cy = (ic.ISLAND_BOUNDS[0] + ic.ISLAND_BOUNDS[2]) / 2, (ic.ISLAND_BOUNDS[1] + ic.ISLAND_BOUNDS[3]) / 2
    lon, lat = Transformer.from_crs(ic.CRS, 'EPSG:4326', always_xy=True).transform(cx, cy)

    def describe(s, view_angle, satellite_azimuth):
        return ic.scene_los(s, idx.loc[s, 'sr_path'], view_angle, satellite_azimuth, (lon, lat))

    rows = []
    for s, r in sm.iterrows():
        row = describe(s, r.view_angle, r.satellite_azimuth)
        row.update(instrument=r.instrument, acquired=r.acquired, fraction_clear=r.fraction_clear,
                   offset_m=RES * np.hypot(r.t_dy_px, r.t_dx_px), affine_amp_m=r.affine_amp_m,
                   res_T_m=RES * r.med_T, res_H_m=RES * r.med_H,
                   k_east=r.k_east_m100, k_north=r.k_north_m100, k_m100=np.hypot(r.k_east_m100, r.k_north_m100))
        rows.append(row)
    d = pd.DataFrame(rows).set_index('scene_id')
    d['k_abs_east'], d['k_abs_north'] = d.k_east + k_master[0], d.k_north + k_master[1]
    # the master is the reference of every fit, so it has no row of its own
    m = pd.Series(describe(ic.MASTER, idx.loc[ic.MASTER, 'view_angle'], idx.loc[ic.MASTER, 'satellite_azimuth']))
    m['los_gap_east'], m['los_gap_north'] = m.rpc_los_east - m.meta_los_east, m.rpc_los_north - m.meta_los_north
    for src in ('meta', 'rpc'):
        for ax in ('east', 'north'):
            d[f'd{src}_los_{ax}'] = d[f'{src}_los_{ax}'] - m[f'{src}_los_{ax}']
    d['los_gap_east'], d['los_gap_north'] = d.rpc_los_east - d.meta_los_east, d.rpc_los_north - d.meta_los_north
    d['los_gap_m100'] = np.hypot(d.los_gap_east, d.los_gap_north)
    DIAG.mkdir(parents=True, exist_ok=True)
    d.to_csv(DIAG / 'metadata_checks.csv')
    click.echo(f'{len(d)} fitted scenes; elevation term of the master against the drone: '
               f'{k_master[0]:+.2f} E, {k_master[1]:+.2f} N m per 100 m')

    targets = ['offset_m', 'k_m100', 'affine_amp_m', 'res_T_m']
    click.echo('\n(1) Planet quality flags (median of each quantity by flag value)')
    for col in ('ground_control', 'quality_category', 'publishing_stage'):
        click.echo(d.groupby(col, dropna=False)[targets].agg(['size', 'median']).round(2).to_string())
    click.echo('Spearman rho against the cloud and resolution fields:')
    for col in ('cloud_cover', 'clear_percent', 'visible_percent', 'clear_confidence_percent', 'fraction_clear',
                'gsd', 'view_angle'):
        v = pd.to_numeric(d[col], errors='coerce')
        click.echo(f'  {col:26s} ' + '  '.join(
            f'{t} {spearmanr(v, d[t], nan_policy="omit").statistic:+.2f}' for t in targets))

    click.echo('\n(2) geometric processing recorded in the ortho XML')
    for col in ('elevation_model', 'geo_level'):
        click.echo(d[col].value_counts(dropna=False).to_string())
    pv = d.assign(year=d.acquired.astype(str).str[:4]).groupby(['year', 'processor_version']).size().unstack(0)
    click.echo(pv.fillna(0).astype(int).to_string())

    click.echo('\n(3) elevation term (relative to the master) against the metadata view direction, m per 100 m')
    click.echo(f'  line of sight |tan(incidence)| x 100: median {np.hypot(d.meta_los_east, d.meta_los_north).median():.2f}'
               f' (master {np.hypot(m.meta_los_east, m.meta_los_north):.2f})')
    c, r2, n = _vector_fit(d.k_east.values, d.k_north.values, d.dmeta_los_east.values, d.dmeta_los_north.values)
    click.echo(f'  k = c x (line of sight - master line of sight): c = {c:+.3f}, variance explained {r2:.1%}, n = {n}')
    for ax in ('east', 'north'):
        click.echo(f'  Pearson r, {ax}: {d[f"k_{ax}"].corr(d[f"dmeta_los_{ax}"]):+.2f}')

    q = d[d.rpc_los_east.notna()]
    inside = (q.line_frac.abs() <= 1) & (q.samp_frac.abs() <= 1)
    click.echo(f'\n(4) RPC line of sight at the frame center against the metadata view direction, {len(q)} scenes '
               f'with a basic RPC ({int(inside.sum())} with the island center inside the frame), m per 100 m')
    click.echo(f'  master: RPC ({m.rpc_los_east:+.2f} E, {m.rpc_los_north:+.2f} N) - metadata '
               f'({m.meta_los_east:+.2f} E, {m.meta_los_north:+.2f} N) = ({m.los_gap_east:+.2f} E, '
               f'{m.los_gap_north:+.2f} N); elevation term measured against the drone '
               f'({k_master[0]:+.2f} E, {k_master[1]:+.2f} N)')
    g = q.los_gap_m100
    click.echo(f'  |RPC - metadata|: median {g.median():.2f}, 90th percentile {g.quantile(0.9):.2f} '
               f'({np.degrees(np.arctan(g.median() / 100)):.2f} and '
               f'{np.degrees(np.arctan(g.quantile(0.9) / 100)):.2f} deg)')
    ka = np.hypot(q.k_abs_east, q.k_abs_north)
    click.echo(f'  |absolute elevation term|: median {ka.median():.2f}, 90th percentile {ka.quantile(0.9):.2f}')
    c, r2, n = _vector_fit(q.k_abs_east.values, q.k_abs_north.values, q.los_gap_east.values, q.los_gap_north.values)
    click.echo(f'  elevation term = c x (RPC - metadata): c = {c:+.3f}, variance explained {r2:.1%}, n = {n}')
    for inst, gi in q.groupby('instrument'):
        c, r2, n = _vector_fit(gi.k_abs_east.values, gi.k_abs_north.values, gi.los_gap_east.values,
                               gi.los_gap_north.values)
        click.echo(f'    {inst:7s} c = {c:+.3f}, variance explained {r2:.1%}, n = {n}')
    res = np.hypot(q.k_abs_east - q.los_gap_east, q.k_abs_north - q.los_gap_north)
    click.echo(f'  |elevation term - (RPC - metadata)|: median {res.median():.2f}, 90th percentile '
               f'{res.quantile(0.9):.2f} (mean difference {(q.k_abs_east - q.los_gap_east).mean():+.2f} E, '
               f'{(q.k_abs_north - q.los_gap_north).mean():+.2f} N)')
    ve, vn = q.rpc_island_los_east - q.rpc_los_east, q.rpc_island_los_north - q.rpc_los_north
    c, r2, n = _vector_fit(q.k_abs_east.values, q.k_abs_north.values, ve.values, vn.values)
    click.echo(f'  look direction at the island minus at the frame center: median {np.hypot(ve, vn).median():.2f}; '
               f'elevation term on it: c = {c:+.3f}, variance explained {r2:.1%}')
    click.echo('  |RPC - metadata| by ground_control: ' + ', '.join(
        f'{k}: {v:.2f} (n = {int(cnt)})' for k, v, cnt in zip(*[
            q.groupby('ground_control').los_gap_m100.median().index,
            q.groupby('ground_control').los_gap_m100.median().values,
            q.groupby('ground_control').los_gap_m100.size().values])))
    for col in ('cloud_cover', 'clear_percent', 'fraction_clear'):
        click.echo(f'  Spearman rho of |RPC - metadata| with {col}: '
                   f'{spearmanr(pd.to_numeric(q[col], errors="coerce"), g, nan_policy="omit").statistic:+.2f}')
    click.echo(f'wrote {DIAG / "metadata_checks.csv"}')


def save_fig(fig, name, caption):
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(FIGS / f'{name}.{ext}', dpi=200, bbox_inches='tight')
    (FIGS / f'{name}_caption.md').write_text(caption.strip() + '\n')
    click.echo(f'wrote {FIGS / name}.png')


@cli.command()
@click.option('--outdir', type=click.Path(file_okay=False, path_type=Path), default=FIGS, show_default=True)
def figures(outdir):
    """Figures and captions (island_diag_*)."""
    import matplotlib.pyplot as plt
    global FIGS
    FIGS = outdir
    sp = pd.read_csv(DIAG / 'spread_tiles.csv')
    cl = pd.read_csv(DIAG / 'closure_tiles_planet.csv')
    cl = cl[cl.band == 'red']
    elev = ic.tile_elevation()
    land = np.zeros(ic.grid_shape(), bool)
    land[sp.ri, sp.ci] = True
    inband = band_mask()
    n_scenes = pd.read_csv(RESULTS / 'scene_models.csv').scene_id.nunique()
    n_tri = pd.read_csv(DIAG / 'closure_planet.csv.gz', usecols=['scene_id', 'mk']).drop_duplicates()

    # 1. maps on the tile lattice
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.4))
    panels = [(grid(sp, 'mean_m'), 'mean residual after translation (m)', 'viridis', 0, 4),
              (np.where(land, elev, np.nan), 'elevation (m a.s.l.)', 'terrain', 20, 190),
              (grid(sp, 'rednir_m'), 'red-NIR disagreement (m)', 'magma', 0, 4),
              (grid(cl, 'closure_med_m'), 'triangle closure error (m)', 'magma', 0, 4)]
    for ax, (g, title, cmap, vmin, vmax), lab in zip(axes.ravel(), panels, 'abcd'):
        im = ax.imshow(g, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.contour(inband.astype(float), levels=[0.5], colors='white' if lab != 'b' else 'black', linewidths=1.0)
        ax.set_title(f'({lab}) {title}', fontsize=10, loc='left')
        ic.bare(ax)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    q = sp.dropna(subset=['mean_dy'])
    axes[0, 0].quiver(q.ci[::2], q.ri[::2], q.mean_dx[::2], -q.mean_dy[::2], color='white', scale=18, width=0.003)
    fig.tight_layout()
    b, o = sp[sp.in_band], sp[~sp.in_band]
    cb = cl.assign(in_band=inband[cl.ri, cl.ci])
    save_fig(fig, 'island_diag_band', f"""
Where the central band of whole-island misalignment lies and whether it is match error. Each cell is one
64-pixel (192 m) matching tile on a 96 m lattice over Barro Colorado Island; the outline encloses the band,
defined as the largest connected set of tiles whose mean residual is at least {BAND_MIN_M:.0f} m ({len(b)} of
{len(sp)} tiles). (a) Mean residual displacement of each tile (color, meters; arrows give its direction) after
removing each scene's median shift, averaged over the {n_scenes} scenes matched in the red band to the master
scene {ic.MASTER}, 2020-2026. (b) Mean Copernicus GLO-30 elevation of each tile: the band is the island's
central plateau (mean {elev[inband].mean():.0f} m against {np.nanmean(elev[land & ~inband]):.0f} m elsewhere).
(c) Median over those scenes of the distance between the red-band and the near-infrared-band displacement of
the same tile and scene, a measure of matching noise ({b.rednir_m.median():.2f} m in the band,
{o.rednir_m.median():.2f} m outside). (d) Median triangle closure error of the red-band displacement,
|d(scene to master) - d(alternate master to master) - d(scene to alternate master)|, over {len(n_tri)}
triangles formed by {n_tri.scene_id.nunique()} scenes and {n_tri.mk.nunique()} alternate master scenes
({cb[cb.in_band].closure_med_m.median():.2f} m in the band, {cb[~cb.in_band].closure_med_m.median():.2f} m
outside); displacements that are real close to zero, false matches do not.
""")
    plt.close(fig)

    # 2. the elevation term
    sm = pd.read_csv(RESULTS / 'scene_models.csv')
    sm = sm[(sm.band == 'both') & sm.med_H.notna()].set_index('scene_id')
    anchor = json.loads((DIAG / 'master_abs.json').read_text())
    kM = (-anchor['median']['h_ch_y'] * RES * 100, anchor['median']['h_ch_x'] * RES * 100)
    df = pd.read_csv(DIAG / 'drone_fits.csv').set_index('scene_id')
    j = df.join(sm[['k_north_m100', 'k_east_m100']], rsuffix='_rel', how='inner')
    j = j[j.index != ic.MASTER]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    ax = axes[0]
    for inst, (nm, col) in ic.INSTRUMENTS.items():
        d = sm[sm.instrument == inst]
        ax.scatter(d.k_east_m100 + kM[1], d.k_north_m100 + kM[0], s=7, color=col, alpha=0.6, label=f'{nm} ({len(d)})')
    ax.scatter([kM[1]], [kM[0]], marker='*', s=220, color='k', zorder=5, label='master scene')
    ax.axhline(0, color='0.6', lw=0.6)
    ax.axvline(0, color='0.6', lw=0.6)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-20, 15)
    ax.set_xlabel('eastward displacement per 100 m of elevation (m)')
    ax.set_ylabel('northward displacement per 100 m of elevation (m)')
    ax.set_title('(a) elevation term of every scene', fontsize=10, loc='left')
    ax.legend(fontsize=8, loc='lower right')
    ax = axes[1]
    ax.scatter(j.k_north_m100_rel, j.k_north_m100, s=14, color='#d62728', label=f'north (r = {j.k_north_m100.corr(j.k_north_m100_rel):.2f})')
    ax.scatter(j.k_east_m100_rel, j.k_east_m100, s=14, color='#1f77b4', label=f'east (r = {j.k_east_m100.corr(j.k_east_m100_rel):.2f})')
    lim = np.array([-12, 8])
    ax.plot(lim, lim + kM[0], color='#d62728', lw=0.8)
    ax.plot(lim, lim + kM[1], color='#1f77b4', lw=0.8)
    ax.set_xlabel('term measured against the Planet master (m per 100 m)')
    ax.set_ylabel('term measured against the drone orthomosaic (m per 100 m)')
    ax.set_title('(b) the same term from two references', fontsize=10, loc='left')
    ax.legend(fontsize=8)
    ax = axes[2]
    rm = pd.read_csv(RESULTS / 'remeasure.csv')
    kinds = [k for k in ('T', 'A', 'L', 'H', 'HL', 'R') if k in set(rm.model)]
    x = np.arange(len(kinds))
    med = rm.groupby('model')[['med_high_px', 'med_low_px']].median().reindex(kinds) * RES
    ax.bar(x - 0.2, med.med_high_px, 0.4, color='#8c564b', label='plateau tiles (>= 120 m a.s.l.)')
    ax.bar(x + 0.2, med.med_low_px, 0.4, color='#7f9f7f', label='lower tiles')
    noise = RES * sm.noise_px.median()
    ax.axhline(noise, color='k', ls='--', lw=0.8, label=f'red-NIR noise floor ({noise:.2f} m)')
    ax.set_xticks(x)
    ax.set_xticklabels([ic.MODEL_NAMES[k].replace(' + ', '\n+ ') for k in kinds], fontsize=8)
    ax.set_ylabel('re-measured median tile residual (m)')
    ax.set_title('(c) residual after warping with each model', fontsize=10, loc='left')
    ax.legend(fontsize=8)
    fig.tight_layout()
    kmag = np.hypot(sm.k_north_m100 + kM[0], sm.k_east_m100 + kM[1])
    save_fig(fig, 'island_diag_elevation', f"""
The elevation-proportional displacement of PlanetScope scenes over Barro Colorado Island and what correcting
it achieves. (a) Elevation term of each of {len(sm)} scenes (2020-2026; color, Dove generation): the
displacement per 100 m of terrain elevation, fitted per scene together with an affine to the tile
displacements against the master scene {ic.MASTER} and referred to the ground by adding the master's own term
(star: {kM[0]:+.1f} m north, {kM[1]:+.1f} m east per 100 m, the median over {anchor['n']} scenes of the
difference between each scene's term against the 2024-10-14 drone orthomosaic and against the master). The
median magnitude is {kmag.median():.1f} m per 100 m and the 90th percentile {kmag.quantile(0.9):.1f} m; the
island spans 25-190 m. (b) For the {len(j)} scenes matched to both references, the term measured against the
drone orthomosaic (20 cm, averaged to 3 m) versus the term measured against the Planet master, north (red) and
east (blue) components, with lines of unit slope offset by the master's term. (c) Median residual tile
displacement re-measured against the master after warping each of {rm.scene_id.nunique()} scenes (at least 80 %
of the island clear) with each model, as the median over scenes, for tiles on the central plateau (brown, mean
elevation of at least 120 m) and for lower tiles (green); the dashed line is the matching noise floor (median
over scenes of the red-versus-near-infrared disagreement of a tile's displacement divided by the square root
of 2).
""")
    plt.close(fig)


if __name__ == '__main__':
    cli()
