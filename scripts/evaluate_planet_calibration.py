#!/usr/bin/env python
"""Evaluate the MODIS cross-calibration of PlanetScope over the 50ha plot.

Compares three versions of every 50ha clip that has all three:
  TOA  existing planet_clipped/4band DN x reflectanceCoefficient (metadata XML),
  SR   raw Planet surface reflectance (planet_clipped_sr),
  CAL  SR cross-calibrated to BRDF-adjusted MODIS (planet_clipped_cal),
using the existing OCM masks (planet_clipped/ocm). Each version is read on the
OCM mask's grid (a few masks sit 7 px inside the usual 205 x 367 clip). For each
clip with at least --min-clear OCM-clear plot coverage, the plot means of blue,
green, red, NIR, NDVI and EVI are computed from clear pixels, then averaged per
date and satellite so adjacent scenes of one strip count as one observation.

Metrics (results/planet_calibration/planet_calibration_eval.csv), per version and variable:
  jitter     median |difference| between consecutive observations <= 3 days apart
  resid_sd   SD of residuals from a centred 31-day rolling median
  rel_*      the above divided by the series median
  inst_*     median residual per instrument, 2020-2022 (sensor-generation offset)

Figures (figs/planet_calibration/):
  timeseries_plot_mean.png  plot-mean NIR and NDVI per version, coloured by
                            instrument, with the MODIS reference over the plot
  residual_sd_by_month.png  residual SD by calendar month per version
  gains_offsets.png         sigma-ratio gain (not applied by default) and applied
                            offset per band over time
  calibration_success.png   share of date-satellite groups calibrated by month
  matched_pixels.png        upscaled Planet vs MODIS at 500 m, before and after
  example_histograms.png    HKU Fig. S10-style histogram pair for one group

Typical usage:
    python scripts/evaluate_planet_calibration.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio as rio  # noqa: E402
import xarray as xr  # noqa: E402
from rasterio.features import geometry_mask  # noqa: E402
from rasterio.warp import transform_geom  # noqa: E402
from rasterio.windows import from_bounds  # noqa: E402
from shapely.geometry import mapping, shape  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calculate_ndvi import get_coeff  # noqa: E402
from calibrate_planet_modis import MODIS_SINU, modis_grid  # noqa: E402
from util import calibration_results, load_config  # noqa: E402

FLOWER = Path('/Volumes/Earth03/flower')
REPO = Path(__file__).resolve().parent.parent
CAL = calibration_results()
BANDS = ['blue', 'green', 'red', 'nir']
VARS = BANDS + ['ndvi', 'evi']
VERSIONS = ['TOA', 'SR', 'CAL']
# dataviz reference categorical slots 1-3 (validated, light surface)
VERSION_COLOR = dict(zip(VERSIONS, ['#2a78d6', '#eb6834', '#1baf7a']))
INSTRUMENT_COLOR = {'PS2': '#2a78d6', 'PS2.SD': '#eb6834', 'PSB.SD': '#1baf7a'}
INK, MUTED = '#2b2b2b', '#8a8a85'


def toa_xml(four_band_src: str) -> Path:
    p = Path(four_band_src)
    if p.name.endswith('_3B_AnalyticMS_clip.tif'):
        return p.with_name(p.name.replace('_3B_AnalyticMS_clip.tif',
                                          '_3B_AnalyticMS_metadata_clip.xml'))
    return (p.parent.parent / 'ortho_analytic_4b_xml'
            / p.name.replace('_3B_AnalyticMS.tif', '_3B_AnalyticMS_metadata.xml'))


def add_indices(d: dict) -> dict:
    b, r, n = d['blue'], d['red'], d['nir']
    d['ndvi'] = (n - r) / (n + r)
    d['evi'] = 2.5 * (n - r) / (n + 6 * r - 7.5 * b + 1)
    return d


def plot_means(path: Path, cls: np.ndarray, bounds, scale) -> dict:
    with rio.open(path) as src:
        dn = src.read(window=from_bounds(*bounds, src.transform), out_shape=(4,) + cls.shape,
                      boundless=True, fill_value=0).astype(np.float64)
    clear = (cls == 0) & np.all(dn > 0, axis=0)
    refl = dn * np.asarray(scale, float)[:, None, None]
    out = {'clear_frac': clear.mean()}
    if clear.any():
        out.update({b: refl[i][clear].mean() for i, b in enumerate(BANDS)})
        px = add_indices({b: refl[i][clear] for i, b in enumerate(BANDS)})
        out['ndvi'], out['evi'] = px['ndvi'].mean(), px['evi'].mean()
    return out


def collect(inventory, index, ocm_root, sr_root, cal_root, min_clear) -> pd.DataFrame:
    inst = {r['scene_id']: r['instrument'] for r in csv.DictReader(open(index))}
    rows = []
    for r in tqdm(list(csv.DictReader(open(inventory))), desc='clips'):
        sid = r['scene_id']
        y = sid[:4]
        paths = {'TOA': ocm_root.parent / '4band' / y / f'{sid}_4band.tif',
                 'SR': sr_root / '4band' / y / f'{sid}_4band.tif',
                 'CAL': cal_root / '4band' / y / f'{sid}_4band.tif'}
        ocm = ocm_root / y / f'{sid}_ocm.tif'
        if sid not in inst or not ocm.exists() or not all(p.exists() for p in paths.values()):
            continue
        xml = toa_xml(r['four_band_src'])
        if not xml.exists():
            continue
        coeff = get_coeff(str(xml))
        scales = {'TOA': [coeff[i] for i in (1, 2, 3, 4)], 'SR': [1e-4] * 4,
                  'CAL': [1e-4] * 4}
        with rio.open(ocm) as src:
            cls, bounds = src.read(1), src.bounds
        for v, p in paths.items():
            m = plot_means(p, cls, bounds, scales[v])
            if m['clear_frac'] < min_clear:
                break
            rows.append(dict(m, scene_id=sid, version=v, instrument=inst[sid],
                             satellite_id=sid.rsplit('_', 1)[-1], date=sid[:8],
                             time=pd.Timestamp(f'{sid[:4]}-{sid[4:6]}-{sid[6:8]}T'
                                               f'{sid[9:11]}:{sid[11:13]}:{sid[13:15]}')))
    df = pd.DataFrame(rows)
    complete = df.groupby('scene_id')['version'].nunique() == 3
    return df[df['scene_id'].isin(complete[complete].index)].sort_values('time')


def per_acquisition(df: pd.DataFrame) -> pd.DataFrame:
    """Mean plot values per version, date and satellite (one strip pass)."""
    agg = {v: 'mean' for v in VARS + ['clear_frac']}
    agg.update(time='mean', instrument='first', scene_id='first')
    return (df.groupby(['version', 'date', 'satellite_id'], as_index=False)
              .agg(agg).sort_values('time'))


def series_metrics(s: pd.DataFrame, var: str) -> tuple[dict, pd.Series]:
    s = s.set_index('time')[var].dropna()
    s = s[~s.index.duplicated()]
    dt = s.index.to_series().diff().dt.total_seconds() / 86400
    jit = s.diff().abs()[(dt > 0) & (dt <= 3)]
    resid = s - s.rolling('31D', center=True, min_periods=3).median()
    med = s.median()
    return {'n': len(s), 'jitter': jit.median(), 'resid_sd': resid.std(),
            'rel_jitter': jit.median() / med, 'rel_resid_sd': resid.std() / med}, resid


def modis_plot_series(reference: Path, region) -> pd.DataFrame:
    ref = xr.open_dataset(reference)
    tf, w, h, _ = modis_grid(ref)
    geom = transform_geom('EPSG:32617', MODIS_SINU.to_wkt(), mapping(region))
    inside = ~geometry_mask([geom], (h, w), tf, all_touched=True)
    vals = ref['reflectance'].values[:, :, inside].mean(axis=-1)       # (band, time)
    d = add_indices({b: vals[i] for i, b in enumerate(BANDS)})
    return pd.DataFrame(d, index=pd.DatetimeIndex(ref['time'].values))


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e6e6e3', lw=0.6)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)


@click.command()
@click.option('--inventory', type=click.Path(path_type=Path, exists=True),
              default=FLOWER / 'planet_clipped/inventory.csv', show_default=True)
@click.option('--index', type=click.Path(path_type=Path, exists=True),
              default=FLOWER / 'planet_bcnm_sr/index.csv', show_default=True)
@click.option('--ocm-root', type=click.Path(path_type=Path, exists=True),
              default=FLOWER / 'planet_clipped/ocm', show_default=True)
@click.option('--sr-root', type=click.Path(path_type=Path),
              default=FLOWER / 'planet_clipped_sr', show_default=True)
@click.option('--cal-root', type=click.Path(path_type=Path),
              default=FLOWER / 'planet_clipped_cal', show_default=True)
@click.option('--calibration', type=click.Path(path_type=Path, exists=True),
              default=CAL / 'planet_modis_calibration.csv', show_default=True)
@click.option('--pixels-dir', type=click.Path(path_type=Path),
              default=CAL / 'planet_modis_calibration_pixels', show_default=True)
@click.option('--reference', type=click.Path(path_type=Path, exists=True),
              default=FLOWER / 'modis/mcd43a1_bcnm/reference_nadir_sza45.nc',
              show_default=True)
@click.option('--clip-config', type=click.Path(path_type=Path, exists=True),
              default=REPO / 'config/clip_50ha_plot.yml', show_default=True)
@click.option('--fig-dir', type=click.Path(path_type=Path),
              default=REPO / 'figs/planet_calibration', show_default=True)
@click.option('--output', type=click.Path(path_type=Path),
              default=CAL / 'planet_calibration_eval.csv', show_default=True)
@click.option('--min-clear', type=float, default=0.8, show_default=True)
def main(inventory, index, ocm_root, sr_root, cal_root, calibration, pixels_dir,
         reference, clip_config, fig_dir, output, min_clear):
    fig_dir.mkdir(parents=True, exist_ok=True)
    clips = collect(inventory, index, ocm_root, sr_root, cal_root, min_clear)
    clips.to_csv(output.with_name('planet_calibration_plot_means.csv'), index=False)
    df = per_acquisition(clips)
    click.echo(f'{clips.scene_id.nunique()} clips ({len(df) // 3} date-satellite passes) '
               f'with all three versions and >= {min_clear:.0%} clear')

    # --- metrics -----------------------------------------------------------
    rows, resid = [], {}
    for v in VERSIONS:
        s = df[df.version == v]
        for var in VARS:
            m, r = series_metrics(s, var)
            resid[(v, var)] = r
            row = dict(version=v, variable=var, **m)
            ins = s.drop_duplicates('time').set_index('time')['instrument'].reindex(r.index)
            early = r.index.year <= 2022
            for name, grp in r[early].groupby(ins[early].values):
                row[f'inst_{name}'] = grp.median()
            rows.append(row)
    met = pd.DataFrame(rows)
    met.to_csv(output, index=False)
    with pd.option_context('display.float_format', '{:.4f}'.format, 'display.width', 160):
        click.echo(met[met.variable.isin(['red', 'nir', 'ndvi', 'evi'])].to_string(index=False))

    cal = pd.read_csv(calibration, parse_dates=['date'])
    ok = cal[cal.calibrated]
    oos = ok.groupby('band')[['rmse_oos_before', 'rmse_oos_after', 'rmse_before',
                              'rmse_after', 'r']].median()
    click.echo('median 500 m RMSE vs MODIS (out-of-sample checkerboard and in-sample):')
    click.echo(oos.reindex(BANDS).to_string(float_format='{:.4f}'.format))
    oos.to_csv(output.with_name('planet_calibration_modis_rmse.csv'))

    # --- time series -------------------------------------------------------
    region = shape(load_config(clip_config)['region'])
    mod = modis_plot_series(reference, region)
    t0, t1 = df.time.min() - pd.Timedelta('15D'), df.time.max() + pd.Timedelta('15D')
    mod = mod[(mod.index >= t0) & (mod.index <= t1)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey='row')
    for j, v in enumerate(VERSIONS):
        s = df[df.version == v]
        for i, var in enumerate(['nir', 'ndvi']):
            ax = axes[i, j]
            for ins, g in s.groupby('instrument'):
                ax.scatter(g.time, g[var], s=8, color=INSTRUMENT_COLOR.get(ins, MUTED),
                           label=ins, linewidths=0, alpha=0.8)
            ax.plot(mod.index, mod[var], color=INK, lw=1.2, label='MODIS ref.')
            style(ax)
            if i == 0:
                ax.set_title(v, color=INK, fontsize=10)
            if j == 0:
                ax.set_ylabel({'nir': 'NIR reflectance', 'ndvi': 'NDVI'}[var], color=INK)
    axes[0, 0].legend(frameon=False, fontsize=8, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(fig_dir / 'timeseries_plot_mean.png', dpi=150)
    plt.close(fig)

    # --- residual SD by month ---------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, var in zip(axes, ['nir', 'ndvi']):
        for v in VERSIONS:
            r = resid[(v, var)]
            sd = r.groupby(r.index.month).std().reindex(range(1, 13))
            ax.plot(sd.index, sd.values, marker='o', ms=4, lw=2,
                    color=VERSION_COLOR[v], label=v)
            ax.annotate(v, (12, sd.values[-1]), xytext=(4, 0), textcoords='offset points',
                        color=INK, fontsize=8, va='center')
        ax.set_xticks(range(1, 13))
        ax.set_xlabel('Month', color=INK)
        ax.set_ylabel(f'{var.upper()} residual SD', color=INK)
        style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / 'residual_sd_by_month.png', dpi=150)
    plt.close(fig)

    # --- gains and offsets -------------------------------------------------
    fig, axes = plt.subplots(4, 2, figsize=(11, 8), sharex=True)
    for i, b in enumerate(BANDS):
        g = ok[ok.band == b]
        # the applied gain is 1 by default, so show the sigma-ratio gain it stands in for
        for j, (col, lab) in enumerate([('gain_sd_ratio', 'σ-ratio gain'), ('offset', 'offset')]):
            ax = axes[i, j]
            for ins, gg in g.groupby('instrument'):
                ax.scatter(gg.date, gg[col], s=8, linewidths=0,
                           color=INSTRUMENT_COLOR.get(ins, MUTED), label=ins)
            ax.axhline(1 if j == 0 else 0, color=MUTED, lw=0.8)
            ax.set_ylabel(f'{b} {lab}', color=INK)
            style(ax)
    axes[0, 0].legend(frameon=False, fontsize=8, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(fig_dir / 'gains_offsets.png', dpi=150)
    plt.close(fig)

    # --- calibration success by month -------------------------------------
    grp = cal[cal.band == 'nir']
    rate = grp.groupby(grp.date.dt.month)['calibrated'].mean().reindex(range(1, 13))
    nmon = grp.groupby(grp.date.dt.month).size().reindex(range(1, 13)).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(rate.index, rate.values, color=VERSION_COLOR['CAL'], width=0.7)
    for m, (rv, n) in enumerate(zip(rate.values, nmon.values), start=1):
        if np.isfinite(rv):
            ax.annotate(f'{int(n)}', (m, rv), xytext=(0, 2), textcoords='offset points',
                        ha='center', fontsize=7, color=MUTED)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel('Month', color=INK)
    ax.set_ylabel('Share of groups calibrated', color=INK)
    ax.set_ylim(0, 1.08)
    style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / 'calibration_success.png', dpi=150)
    plt.close(fig)

    # --- matched pixels and an example histogram pair -----------------------
    npz = sorted(pixels_dir.glob('*.npz'))
    if npz:
        fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), sharex='col', sharey='col')
        for j, b in enumerate(BANDS):
            p_all, c_all, m_all = [], [], []
            for f in npz:
                z = np.load(f)
                p_all.append(z[f'planet_{b}'])
                c_all.append(z['gain'][j] * z[f'planet_{b}'] + z['offset'][j])
                m_all.append(z[f'modis_{b}'])
            m_all = np.concatenate(m_all)
            for i, (lab, pv) in enumerate([('raw SR', np.concatenate(p_all)),
                                           ('calibrated', np.concatenate(c_all))]):
                ax = axes[i, j]
                ax.hexbin(m_all, pv, gridsize=60, bins='log', cmap='Blues', mincnt=1)
                lo, hi = np.percentile(np.r_[m_all, pv], [0.5, 99.5])
                ax.plot([lo, hi], [lo, hi], color=INK, lw=0.8)
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                rm = np.sqrt(np.mean((pv - m_all) ** 2))
                ax.set_title(f'{b}, {lab}: RMSE {rm:.4f}', color=INK, fontsize=9)
                ax.set_xlabel('MODIS reflectance', color=INK)
                if j == 0:
                    ax.set_ylabel('Planet reflectance (500 m)', color=INK)
                style(ax)
        fig.tight_layout()
        fig.savefig(fig_dir / 'matched_pixels.png', dpi=150)
        plt.close(fig)

        best = ok[ok.band == 'nir'].sort_values('n_pixels').iloc[-1]
        f = pixels_dir / f'{best.date:%Y-%m-%d}_{best.satellite_id}.npz'
        z = np.load(f)
        fig, axes = plt.subplots(1, 4, figsize=(13, 3))
        for j, b in enumerate(BANDS):
            ax = axes[j]
            p, m = z[f'planet_{b}'], z[f'modis_{b}']
            c = z['gain'][j] * p + z['offset'][j]
            bins = np.linspace(min(p.min(), m.min(), c.min()),
                               max(p.max(), m.max(), c.max()), 30)
            ax.hist(m, bins, histtype='step', lw=2, color=INK, label='MODIS')
            ax.hist(p, bins, histtype='step', lw=2, color=VERSION_COLOR['SR'], label='raw SR')
            ax.hist(c, bins, histtype='step', lw=2, color=VERSION_COLOR['CAL'],
                    label='calibrated')
            ax.set_title(b, color=INK, fontsize=9)
            ax.set_xlabel('Reflectance', color=INK)
            style(ax)
        axes[0].legend(frameon=False, fontsize=8)
        axes[0].set_ylabel('500 m pixels', color=INK)
        fig.suptitle(f'{best.date:%Y-%m-%d}, satellite {best.satellite_id} '
                     f'({best.instrument}), {best.n_pixels} pixels', fontsize=9, color=INK)
        fig.tight_layout()
        fig.savefig(fig_dir / 'example_histograms.png', dpi=150)
        plt.close(fig)
    click.echo(f'figures in {fig_dir}')


if __name__ == '__main__':
    main()
