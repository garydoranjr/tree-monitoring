#!/usr/bin/env python
"""Species-specific Planet event detection and expected event observability.

Detection comes from the temporal cross-validation of the PlanetScope Mask
R-CNN (plot_temporal_eval.py: each fold's epoch-12 model on its held-out
chips). Its crown table holds one row per clear held-out crown-date (>= 90%
inside the window, >= 90% OCM-clear, >= 64 px of 0.75 m); a crown is an event
when >= 50% of it is in the label mask, and it counts as detected when >= 50%
of it is covered by predicted masks (score >= 0.5). The model has one event
class, so each event crown-date is typed flowering or leafless by the larger of
the drone classifier's crown probabilities (crown_classifications{,_mavic}.nc4)
for that flight.

Subcommands (outputs in OUTDIR, default the November 2026 presentation):
  species-eval  detection rate of event crowns by species and event type,
                false-alarm rate on the same species' no-event crowns, and
                detection rate by crown size and season
  summary       expected observability of each species' events: the chance of
                a clear Planet view during the event (OCM model of
                plot_ocm_observability.py, from event timing and length) times
                the chance of detecting the event in a clear view (logistic
                model of the temporal-CV detections on species x event type,
                event type x season and event type x log crown area)

Typical usage:
    python scripts/plot_planet_event_observability.py species-eval
    python scripts/plot_planet_event_observability.py summary
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_ocm_observability as po  # noqa: E402
import plot_spatial_eval as pse  # noqa: E402

FLOWER = pse.FLOWER
DEFAULT_OUT = pse.DEFAULT_OUT
CHIP_RES = 0.75  # m
DRY_MONTHS = (12, 1, 2, 3, 4)
C_FLOWER, C_DECID = po.C_FLOWER, po.C_DECID
KINDS = (('flowering', C_FLOWER, 'flowering'), ('deciduous', C_DECID, 'leafless'))
# crown-map names -> the Condit names used by the trap and leaf-cover events
SYNONYMS = {'Quararibea stenophylla': 'Quararibea asterolepis',
            'Platymiscium dimorphandrum': 'Platymiscium pinnatum',
            'Virola surinamensis': 'Virola nobilis',
            'Hybanthus prunifolius': 'Pombalia prunifolia'}


def wet(month):
    return ~np.isin(month, DRY_MONTHS)


def crown_table(outdir, rebuild=False):
    """Temporal-CV crown table with species, event type, size, season, detection."""
    import xarray as xr
    import plot_temporal_eval as pte
    cache = outdir / 'materials/temporal_eval_crowns_species.pkl'
    if cache.exists() and not rebuild:
        return pd.read_pickle(cache)
    z = np.load(outdir / 'materials/temporal_eval_cache.npz', allow_pickle=False)
    cr = pse.crown_table(z['chips'], z['windows'], z['score'], z['gt'], z['clear'],
                         chip_dir=pte.CHIPS)
    dc = pse.load_dated_crowns()
    latin = dc.drop_duplicates(['tag', 'date']).set_index(['tag', 'date']).latin
    cr['latin'] = [latin.get((t, d)) for t, d in zip(cr.tag, cr.date)]
    probs = []
    for f in ('crown_classifications.nc4', 'crown_classifications_mavic.nc4'):
        ds = xr.open_dataset(FLOWER / 'results' / f)
        if f == 'crown_classifications.nc4':
            cr['latin'] = cr.latin.fillna(cr.tag.map(ds.species.to_series()))
        p = ds[['flowering_probability', 'deciduous_probability']].to_dataframe().reset_index()
        p['date'] = pd.to_datetime(p.date).dt.strftime('%Y_%m_%d')
        probs.append(p)
    probs = pd.concat(probs).drop_duplicates(['tag', 'date']).set_index(['tag', 'date'])
    cr = cr.join(probs, on=['tag', 'date'])
    cr['latin'] = cr.latin.replace(SYNONYMS)
    cr['kind'] = np.where(~cr.event, 'none',
                          np.where(cr.flowering_probability >= cr.deciduous_probability,
                                   'flowering', 'deciduous'))
    cr['det'] = cr.pred_frac >= 0.5
    cr['area_m2'] = cr.px * CHIP_RES ** 2
    cr['month'] = pd.to_datetime(cr.date, format='%Y_%m_%d').dt.month
    cr['wet'] = wet(cr.month)
    cr.to_pickle(cache)
    return cr


def save(fig, outdir, name):
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'{name}.{ext}', dpi=300)
    plt.close(fig)
    click.echo(f'Wrote {outdir / name}.png/.pdf')


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=9)


@click.group()
def cli():
    pass


out_option = click.option('--outdir', type=click.Path(file_okay=False, path_type=Path),
                          default=DEFAULT_OUT, show_default=True)
rebuild_option = click.option('--rebuild', is_flag=True,
                              help='Rebuild the species crown table from the inference cache.')


@cli.command('species-eval')
@out_option
@rebuild_option
@click.option('--min-crowns', default=4, show_default=True,
              help='Event crowns of a type needed to show that type for a species.')
@click.option('--k', default=18, show_default=True, help='Species shown.')
def species_eval(outdir, rebuild, min_crowns, k):
    """Detection rate by species, event type, crown size and season."""
    cr = crown_table(outdir, rebuild)
    ev = cr[cr.event]
    rows = []
    for sp, c in cr.groupby('latin'):
        r = dict(latin=sp, crowns=c.tag.nunique(), crown_dates=len(c),
                 false_alarm=c.det[~c.event].mean(), n_none=int((~c.event).sum()),
                 area_m2=c.drop_duplicates('tag').area_m2.median())
        for kind, _, _ in KINDS:
            e = c[c.kind == kind]
            r[f'{kind}_crowns'] = e.tag.nunique()
            r[f'{kind}_dates'] = len(e)
            r[f'{kind}_det'] = e.det.mean() if len(e) else np.nan
        rows.append(r)
    sp = pd.DataFrame(rows)
    sp['event_crowns'] = sp.flowering_crowns + sp.deciduous_crowns
    show = sp[(sp.flowering_crowns >= min_crowns) | (sp.deciduous_crowns >= min_crowns)]
    show = show.sort_values('event_crowns', ascending=False).head(k)
    show = show.sort_values('deciduous_det', na_position='first').reset_index(drop=True)

    fig = plt.figure(figsize=(12.5, 6))
    ax = fig.add_axes([0.2, 0.1, 0.36, 0.84])
    y = np.arange(len(show))
    for kind, color, label in KINDS:
        ok = show[f'{kind}_crowns'] >= min_crowns
        off = 0.17 if kind == 'flowering' else -0.17
        ax.barh(y[ok] + off, show[f'{kind}_det'][ok], height=0.32, color=color,
                label=f'{label} crowns')
        for yi, r in zip(y[ok], show[ok].itertuples()):
            ax.text(getattr(r, f'{kind}_det') + 0.01, yi + off,
                    f'{getattr(r, f"{kind}_crowns")}', va='center', fontsize=7, color='0.35')
    ax.plot(show.false_alarm, y, 'k|', ms=11, mew=2, label='false alarms (no-event crowns)')
    ax.set_yticks(y, [po.italic(s) for s in show.latin], fontsize=9)
    ax.set_ylim(-0.6, len(show) - 0.4)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('share of clear crown-dates detected', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.07), ncol=3, fontsize=8.5,
              frameon=False, handlelength=1.4, columnspacing=1.0)
    style(ax)

    # by crown size
    bins = [36, 50, 100, 200, 400, 800, 1600]
    ax2 = fig.add_axes([0.66, 0.57, 0.32, 0.31])
    mids = np.sqrt(np.array(bins[:-1]) * np.array(bins[1:]))
    for kind, color, label in KINDS:
        e = ev[ev.kind == kind]
        g = e.groupby(pd.cut(e.area_m2, bins), observed=False).det.agg(['size', 'mean'])
        ok = g['size'].values >= 20
        ax2.plot(mids[ok], g['mean'].values[ok], 'o-', color=color, lw=2, ms=5, label=label)
    g = cr[~cr.event].groupby(pd.cut(cr.area_m2[~cr.event], bins), observed=False).det.mean()
    ax2.plot(mids, g.values, 's--', color='0.3', lw=1.2, ms=4, label='no event (false alarm)')
    ax2.set_xscale('log')
    ax2.set_xticks([50, 100, 200, 400, 800], ['50', '100', '200', '400', '800'])
    ax2.minorticks_off()
    ax2.set_xlabel('crown area in the chip (m²)', fontsize=9.5)
    ax2.set_ylabel('detected', fontsize=9.5)
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)
    sec = ax2.secondary_xaxis('top', functions=(lambda a: a / 9, lambda p: p * 9))
    sec.set_xticks([5, 10, 25, 50, 100], ['5', '10', '25', '50', '100'])
    sec.minorticks_off()
    sec.tick_params(labelsize=8)
    sec.set_xlabel('Planet pixels (3 m)', fontsize=8.5)
    ax2.legend(fontsize=8, frameon=False, loc='upper left')
    style(ax2)

    # by season
    ax3 = fig.add_axes([0.66, 0.1, 0.32, 0.34])
    xs = np.arange(2)
    for j, (kind, color, label) in enumerate(KINDS):
        e = ev[ev.kind == kind]
        g = e.groupby('wet').det.agg(['size', 'mean'])
        b = ax3.bar(xs + (j - 0.5) * 0.36, g['mean'].values, width=0.34, color=color)
        for rect, n in zip(b, g['size']):
            ax3.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.02, f'{n:,}',
                     ha='center', fontsize=7.5, color='0.35')
    ax3.set_xticks(xs, ['dry season (Dec-Apr)', 'wet season (May-Nov)'], fontsize=9)
    ax3.set_ylabel('detected', fontsize=9.5)
    ax3.set_ylim(0, 1)
    ax3.grid(axis='y', alpha=0.3)
    style(ax3)
    save(fig, outdir, 'planet_species_detection')

    by_kind = ev.groupby('kind').agg(dates=('det', 'size'), crowns=('tag', 'nunique'),
                                     detected=('det', 'mean'))
    out = dict(by_kind=by_kind.reset_index().to_dict('records'),
               by_kind_season=ev.groupby(['kind', 'wet']).det.agg(['size', 'mean'])
               .reset_index().to_dict('records'),
               false_alarm=float(cr.det[~cr.event].mean()),
               precision=float(cr.event[cr.det].mean()),
               species=sp.sort_values('event_crowns', ascending=False).to_dict('records'))
    (outdir / 'materials/planet_species_detection.json').write_text(
        json.dumps(out, indent=1, default=float))
    click.echo(by_kind.round(3).to_string())
    click.echo(show.round(2).to_string())


def detection_model(cr):
    """Logistic model of detection over event crown-dates, one unit weight per crown."""
    from sklearn.linear_model import LogisticRegression
    ev = cr[cr.event].copy()
    ev['w'] = 1 / ev.groupby(['tag', 'kind']).tag.transform('size')
    la_mu, la_sd = np.log(ev.area_m2).mean(), np.log(ev.area_m2).std()
    cells = sorted(set(zip(ev.latin, ev.kind)))

    def design(latin, kind, is_wet, area):
        X = np.zeros((len(latin), 6 + len(cells)))
        la = (np.log(area) - la_mu) / la_sd
        for j, (kd, _, _) in enumerate(KINDS):
            m = np.asarray(kind) == kd
            X[:, 3 * j] = m * 1.0
            X[:, 3 * j + 1] = m * np.asarray(is_wet)
            X[:, 3 * j + 2] = m * la
        idx = {c: i for i, c in enumerate(cells)}
        for r, c in enumerate(zip(latin, kind)):
            if c in idx:
                X[r, 6 + idx[c]] = 1
        return X

    X = design(ev.latin.values, ev.kind.values, ev.wet.values, ev.area_m2.values)
    # unpenalised fixed effects would need statsmodels; a mild L2 on all terms
    # shrinks sparse species x type cells toward their event type
    m = LogisticRegression(C=1.0, fit_intercept=False, max_iter=2000)
    m.fit(X, ev.det.values, sample_weight=ev.w.values * len(ev) / ev.w.sum())

    def predict(latin, kind, is_wet, area):
        return m.predict_proba(design(latin, kind, is_wet, area))[:, 1]
    coef = dict(zip([f'{k}:{t}' for k, _, _ in KINDS for t in ('base', 'wet', 'log_area_z')],
                    m.coef_[0][:6]))
    return predict, coef


def episodes(cr):
    """Temporal-CV event episodes: one crown's event crown-dates of one type no more
    than 21 days apart. Looks are the distinct Planet chips; a look detects if
    any of its windows does."""
    e = cr[cr.event].copy()
    e['d'] = pd.to_datetime(e.date, format='%Y_%m_%d')
    e = e.sort_values(['tag', 'kind', 'd'])
    e['ep'] = (e.groupby(['tag', 'kind']).d.diff().dt.days.fillna(np.inf) > 21).cumsum()
    looks = e.groupby(['ep', 'chip']).agg(kind=('kind', 'first'), det=('det', 'any'))
    return looks.groupby('ep').agg(kind=('kind', 'first'), n=('det', 'size'), k=('det', 'sum'))


def look_correlation(cr):
    """Beta-binomial fit of detections over the looks of each episode, by event type.

    Returns {kind: rho} (intra-episode correlation of looks) and a check table of
    observed and fitted chance of at least one detection by number of looks.
    """
    from scipy.optimize import minimize
    from scipy.stats import betabinom
    ep = episodes(cr)
    rho, check = {}, []
    for kind, _, _ in KINDS:
        g = ep[ep.kind == kind]
        r = minimize(lambda t: -betabinom.logpmf(g.k, g.n, *np.exp(t)).sum(), [0.0, 0.0],
                     method='Nelder-Mead')
        a, b = np.exp(r.x)
        rho[kind] = 1 / (a + b + 1)
        for lo, hi in ((1, 1), (2, 2), (3, 4), (5, 99)):
            s = g[(g.n >= lo) & (g.n <= hi)]
            check.append(dict(kind=kind, looks=f'{lo}-{hi}', episodes=len(s),
                              per_look=s.k.sum() / s.n.sum(), any_obs=(s.k > 0).mean(),
                              any_fit=float(np.mean(1 - betabinom.pmf(0, s.n, a, b))),
                              any_indep=float(np.mean(1 - (1 - a / (a + b)) ** s.n))))
    return rho, pd.DataFrame(check)


def p_any_detection(p, rho, n):
    """Chance of at least one detection in n looks; per-look rate p, look correlation rho."""
    from scipy.stats import betabinom
    a, b = p * (1 - rho) / rho, (1 - p) * (1 - rho) / rho
    return 1 - betabinom.pmf(0, n, a, b)


def clear_view_counts(vis, days, doy, length):
    """Days with a clear view of each crown (rows of vis) in each YEARS year, within
    the length-day window centred on day of year doy, as in po.capture_model."""
    c = np.concatenate([np.zeros((vis.shape[0], 1), int), np.cumsum(vis, axis=1)], axis=1)
    starts = np.array([(pd.Timestamp(f'{y}-01-01') - days[0]).days for y in po.YEARS])
    s = starts + int(doy) - 1
    lo = np.clip(s - length // 2, 0, vis.shape[1])
    hi = np.clip(s + length - length // 2, 0, vis.shape[1])
    return (c[:, hi] - c[:, lo]).ravel()


@cli.command()
@out_option
@rebuild_option
@click.option('--min-events', default=10, show_default=True,
              help='Trap or leaf-cover events needed for a species x type row.')
@click.option('--min-crowns', default=4, show_default=True,
              help='Temporal-CV event crowns needed for a species x type row.')
@click.option('--min-clear', default=0.9, show_default=True)
def summary(outdir, rebuild, min_events, min_crowns, min_clear):
    """Expected observability of each species' events from Planet."""
    cr = crown_table(outdir, rebuild)
    predict, coef = detection_model(cr)
    rho, check = look_correlation(cr)
    click.echo(f'look correlation {rho}')
    click.echo(check.round(2).to_string())
    ev = po.load_events(outdir / 'materials/sp_flower_counts_annual_stats.csv',
                        outdir / 'materials/decid_summary.csv')
    ev['wet'] = wet(pd.to_datetime(ev.event_peak, format='mixed').dt.month.values)
    crowns, vis, days = po.visibility(min_clear)
    size = cr.drop_duplicates('tag').groupby('latin').area_m2.median()
    n_cv = cr[cr.event].groupby(['latin', 'kind']).tag.nunique()
    rows = []
    for (sp, kind), e in ev.groupby(['latin', 'kind']):
        if len(e) < min_events or n_cv.get((sp, kind), 0) < min_crowns or sp not in size:
            continue
        sp_vis = vis[crowns.index[crowns.latin == sp]]
        if len(sp_vis) < 5:
            continue
        p_det = predict([sp] * len(e), [kind] * len(e), e.wet.values,
                        np.full(len(e), size[sp]))
        p_clear, p_event, n_views = [], [], []
        for r, p in zip(e.itertuples(), p_det):
            n = clear_view_counts(sp_vis, days, r.doy, int(r.event_length))
            p_clear.append((n > 0).mean())
            p_event.append(p_any_detection(p, rho[kind], n).mean())
            n_views.append(np.median(n))
        rows.append(dict(latin=sp, kind=kind, events=len(e), cv_crowns=int(n_cv[(sp, kind)]),
                         sp_crowns=len(sp_vis), area_m2=size[sp], days=e.event_length.median(),
                         wet_share=e.wet.mean(), clear_views=np.median(n_views),
                         p_clear=np.mean(p_clear), p_det=p_det.mean(),
                         single_look=np.mean(np.array(p_clear) * p_det),
                         expected=np.mean(p_event)))
    df = pd.DataFrame(rows)

    fig = plt.figure(figsize=(12.5, 6))
    panels = [('flowering', 0.155), ('deciduous', 0.655)]
    for kind, x0 in panels:
        d = df[df.kind == kind].sort_values('expected').reset_index(drop=True)
        color = C_FLOWER if kind == 'flowering' else C_DECID
        ax = fig.add_axes([x0, 0.12, 0.225, 0.8])
        y = np.arange(len(d))
        ax.barh(y, d.expected, height=0.62, color=color, alpha=0.9)
        ax.plot(d.p_det, y, 'D', color='k', ms=4.5, clip_on=False)
        ax.plot(d.p_clear, y, 'o', mfc='white', mec='k', ms=5.5, mew=1.2, clip_on=False)
        for yi, r in zip(y, d.itertuples()):
            for xx, txt in ((1.16, f'{r.area_m2:.0f}'), (1.31, f'{r.days:.0f}'),
                            (1.46, f'{r.clear_views:.0f}')):
                ax.text(xx, yi, txt, va='center', ha='right', fontsize=7.5, color='0.35',
                        transform=ax.get_yaxis_transform())
        for xx, lab in ((1.16, 'crown\nm²'), (1.31, 'event\ndays'), (1.46, 'clear\nviews')):
            ax.text(xx, len(d) - 0.4, lab, transform=ax.get_yaxis_transform(), ha='right',
                    va='bottom', fontsize=7.5, color='0.35')
        ax.set_yticks(y, [po.italic(s) for s in d.latin], fontsize=8.5)
        ax.set_ylim(-0.6, len(d) - 0.4)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.grid(axis='x', alpha=0.3)
        ax.set_title('flowering' if kind == 'flowering' else 'leafless', fontsize=11,
                     color=color, loc='left', fontweight='bold')
        ax.set_xlabel('probability', fontsize=9.5)
        style(ax)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker='o', mfc='white', mec='k', mew=1.2, ms=5.5, ls=''),
         Line2D([], [], marker='D', color='k', ms=4.5, ls=''),
         plt.Rectangle((0, 0), 1, 1, color='0.55')]
    l = ['at least one clear view during the event', 'detected in one clear view',
         'event detected over all its clear views']
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.005))
    save(fig, outdir, 'planet_event_observability')
    out = dict(detection_model_coef=coef, look_correlation=rho,
               look_check=check.to_dict('records'), rows=df.to_dict('records'))
    (outdir / 'materials/planet_event_observability.json').write_text(
        json.dumps(out, indent=1, default=float))
    click.echo(json.dumps(coef, indent=1))
    click.echo(df.sort_values(['kind', 'expected']).round(2).to_string())


if __name__ == '__main__':
    cli()
