#!/usr/bin/env python
"""Phenological event observability from OCM crown visibility (Planet, 50ha plot).

The empirical observation model of the observability pipeline
(fit_empirical_count_models.py) is rebuilt with OmniCloudMask in place of the
DINOv2 visibility classifier: a crown is visible in a Planet scene when at
least --min-clear of its pixels are OCM-clear (crown_pixel_series.npz). For
every day of year and window length the model gives the share of crown-years
with at least one visible scene in the window, over the full calendar years in
the record (2020-2025). Each species gets its own model from its own crowns,
and events are scored with event_summary_stats.get_obs_probabilities.

Events come from the pipeline's inputs:
  flowering      individual_trap_analysis.py output (litter traps, 2020-2024)
  deciduousness  individual_decid_analysis.py output (leaf-cover series,
                 2014-2015)

Subcommands (outputs in OUTDIR, default the November 2026 presentation):
  species  capture-probability heat map for one species with its events
           overlaid, and the distribution of their capture probabilities
  summary  calendar of event timing for the top species, how often crowns
           are visible through the year, and mean capture probability

Typical usage:
    python scripts/plot_ocm_observability.py species "Jacaranda copaia"
    python scripts/plot_ocm_observability.py summary --k 10
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import functools
import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_empirical_count_models import EmpiricalCountModel  # noqa: E402
from event_summary_stats import get_obs_probabilities  # noqa: E402

FLOWER = Path('/Volumes/Earth03/flower')
DEFAULT_OUT = FLOWER / 'figs' / '20261103_presentation'
CROWN_SERIES = FLOWER / 'crown_series_20261029/crown_pixel_series.npz'
TAXA = FLOWER / 'data/Condit_ValidBCITaxa.tab'
YEARS = range(2020, 2026)  # full calendar years of OCM crown visibility
MAX_WINDOW = 120  # days
C_FLOWER, C_DECID = '#d6604d', '#8c510a'
MONTHS = 'JFMAMJJASOND'


@functools.cache
def taxa():
    """Upper-case 4-letter trap codes and 6-letter mnemonics -> Latin name."""
    tx = pd.read_csv(TAXA, sep='\t')
    out = {}
    for r in tx.itertuples():
        for c in (r.past_code, r.Mnemonic):
            if isinstance(c, str):
                for cc in c.replace(';', ',').split(','):
                    out.setdefault(cc.strip().upper(), r.Latin)
    return out


@functools.cache
def visibility(min_clear):
    """(crowns DataFrame with latin, bool crowns x days, day index)."""
    z = np.load(CROWN_SERIES, allow_pickle=True)
    crowns = pd.DataFrame(json.loads(str(z['crowns'])))
    crowns['latin'] = crowns.mnemonic.str.upper().map(taxa())
    sdate = pd.to_datetime(pd.DataFrame(json.loads(str(z['scenes']))).date).values
    days = pd.date_range(f'{YEARS[0]}-01-01', f'{YEARS[-1]}-12-31')
    ok = (z['obs_n_clear'] >= min_clear * z['obs_n_px']) & (z['obs_n_px'] > 0)
    d = (sdate[z['obs_scene'][ok]] - days[0].to_datetime64()).astype('timedelta64[D]').astype(int)
    keep = (d >= 0) & (d < len(days))
    vis = np.zeros((len(crowns), len(days)), bool)
    vis[z['obs_crown'][ok][keep], d[keep]] = True
    return crowns, vis, days


def capture_model(vis, days):
    """EmpiricalCountModel over the crowns in `vis` (rows) and YEARS.

    P[w, s]: share of crown-years with a visible scene within the w-day window
    centred on day-of-year s (s = 0 .. 364), as in fit_empirical_count_models.
    """
    c = np.concatenate([np.zeros((vis.shape[0], 1), int), np.cumsum(vis, axis=1)], axis=1)
    windows = np.arange(1, MAX_WINDOW + 1)
    P = np.zeros((len(windows), 365))
    starts = np.array([(pd.Timestamp(f'{y}-01-01') - days[0]).days for y in YEARS])
    for i, w in enumerate(windows):
        lo_off, hi_off = -(w // 2), w - w // 2  # [s - w//2, s + w - w//2)
        for s in range(365):
            lo = np.clip(starts + s + lo_off, 0, vis.shape[1])
            hi = np.clip(starts + s + hi_off, 0, vis.shape[1])
            P[i, s] = ((c[:, hi] - c[:, lo]) > 0).mean()
    samples = pd.date_range('2021-01-01', '2021-12-31').values
    return EmpiricalCountModel(samples, windows, P)


@functools.cache
def species_model(latin, min_clear):
    crowns, vis, days = visibility(min_clear)
    rows = crowns.index[crowns.latin == latin]
    if len(rows) == 0:
        raise click.ClickException(f'no crowns of {latin} in the crown series')
    return capture_model(vis[rows], days), len(rows)


def load_events(flower_csv, decid_csv):
    f = pd.read_csv(flower_csv)
    f['latin'] = f.species.str.upper().map(taxa())
    f['kind'] = 'flowering'
    d = pd.read_csv(decid_csv)
    d['latin'] = d.species.str.upper().map(taxa())
    d['kind'] = 'deciduous'
    ev = pd.concat([f, d], ignore_index=True).dropna(subset=['latin'])
    ev = ev[ev.event_length > 0].copy()
    # the two sources write dates differently (ISO vs M/D/YYYY)
    ev['doy'] = pd.to_datetime(ev.event_peak, format='mixed').dt.dayofyear.astype(float)
    return ev


def score(ev, model):
    length = np.clip(ev.event_length.values, 1, MAX_WINDOW)
    return get_obs_probabilities(model, ev.doy.values, length)


def italic(latin):
    return ' '.join(f'$\\it{{{w}}}$' for w in latin.split())


def month_axis(ax, labels=True):
    ticks = [pd.Timestamp(f'2021-{m:02d}-01').dayofyear - 1 for m in range(1, 13)]
    ax.set_xticks(ticks, [pd.Timestamp(f'2021-{m:02d}-01').strftime('%b') for m in range(1, 13)]
                  if labels else [''] * 12)
    ax.set_xlim(0, 365)


def save(fig, outdir, name, dpi):
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'{name}.{ext}', dpi=dpi)
    plt.close(fig)
    click.echo(f'Wrote {outdir / name}.png/.pdf')


@click.group()
def cli():
    pass


common = [
    click.option('--outdir', type=click.Path(file_okay=False, path_type=Path),
                 default=DEFAULT_OUT, show_default=True),
    click.option('--flower-events', type=click.Path(exists=True, path_type=Path),
                 default=DEFAULT_OUT / 'materials/sp_flower_counts_annual_stats.csv',
                 show_default=True),
    click.option('--decid-events', type=click.Path(exists=True, path_type=Path),
                 default=DEFAULT_OUT / 'materials/decid_summary.csv', show_default=True),
    click.option('--min-clear', default=0.9, show_default=True,
                 help='Share of crown pixels OCM-clear for the crown to count as visible.'),
    click.option('--dpi', default=300, show_default=True),
]


def with_common(f):
    for opt in reversed(common):
        f = opt(f)
    return f


@cli.command()
@click.argument('latin')
@with_common
@click.option('--max-weeks', default=10, show_default=True)
def species(latin, outdir, flower_events, decid_events, min_clear, dpi, max_weeks):
    """Capture probability by timing and length, with one species' events."""
    model, n_crowns = species_model(latin, min_clear)
    ev = load_events(flower_events, decid_events)
    ev = ev[ev.latin == latin].copy()
    ev['p'] = score(ev, model)

    fig = plt.figure(figsize=(12.5, 5.6))
    ax = fig.add_axes([0.075, 0.13, 0.6, 0.8])
    cax = fig.add_axes([0.685, 0.13, 0.012, 0.8])
    hax = fig.add_axes([0.8, 0.13, 0.18, 0.74])
    wmax = 7 * max_weeks
    im = ax.imshow(model.probablities[:wmax], origin='lower', aspect='auto', cmap='magma',
                   vmin=0, vmax=1, extent=[0, 365, 0.5 / 7, (wmax + 0.5) / 7])
    for kind, color, marker in (('flowering', C_FLOWER, 'o'), ('deciduous', C_DECID, 's')):
        e = ev[ev.kind == kind]
        # trap events sit on a weekly grid, so identical events stack: size by count
        g = (e.assign(x=e.doy - 1, y=np.minimum(e.event_length, wmax) / 7)
             .groupby(['x', 'y']).size().reset_index(name='n'))
        ax.scatter(g.x, g.y, s=22 * np.sqrt(g.n), marker=marker, facecolor=color,
                   edgecolor='white', lw=0.7, zorder=3, label=f'{kind} ({len(e)} events)')
    month_axis(ax)
    ax.set_ylim(0.5 / 7, (wmax + 0.5) / 7)
    ax.set_ylabel('event length (weeks)', fontsize=11)
    ax.tick_params(labelsize=10)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.85, markerscale=1.2)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('chance of at least one clear Planet view', fontsize=10)
    cb.ax.tick_params(labelsize=9)
    # split histogram: flowering to the left of zero, deciduous to the right
    bins = np.linspace(0, 1, 11)
    width = bins[1] - bins[0]
    peak = 1
    for kind, color, sign in (('flowering', C_FLOWER, -1), ('deciduous', C_DECID, 1)):
        p = ev.p[ev.kind == kind]
        n, _ = np.histogram(p, bins=bins)
        peak = max(peak, n.max())
        hax.barh(bins[:-1] + width / 2, sign * n, height=width * 0.92, color=color, alpha=0.8)
        if len(p):
            hax.plot([0, sign * 1e6], [p.mean()] * 2, color=color, lw=1.8, ls='--')
            hax.text(0.5 + sign * 0.25, 1.01, f'{kind}\nmean {p.mean():.2f}',
                     transform=hax.transAxes, ha='center', va='bottom', fontsize=8.5,
                     color=color)
    lim = peak * 1.1
    hax.set_xlim(-lim, lim)
    ticks = hax.get_xticks()
    hax.set_xticks(ticks, [f'{abs(t):.0f}' for t in ticks])
    hax.set_xlim(-lim, lim)
    hax.axvline(0, color='0.3', lw=0.8)
    hax.set_ylim(0, 1)
    hax.set_xlabel('events', fontsize=10)
    hax.set_ylabel('chance of observing the event', fontsize=10)
    hax.tick_params(labelsize=9)
    hax.spines[['top', 'right']].set_visible(False)
    fig.text(0.075, 0.955, f'{italic(latin)} ({n_crowns} crowns)', fontsize=12)
    save(fig, outdir, f'ocm_observability_{latin.split()[0].lower()}', dpi)
    click.echo(ev.groupby('kind').p.agg(['size', 'mean']).round(2))


@cli.command()
@with_common
@click.option('--k', default=10, show_default=True, help='Number of species.')
@click.option('--min-events', default=10, show_default=True)
def summary(outdir, flower_events, decid_events, min_clear, dpi, k, min_events):
    """Event calendar of the top species against Planet crown visibility."""
    crowns, vis, days = visibility(min_clear)
    ev = load_events(flower_events, decid_events)
    n_ev = ev.groupby('latin').size()
    n_crowns = crowns.latin.value_counts()
    cand = [s for s in n_crowns.index if n_ev.get(s, 0) >= min_events][:k]
    all_model = capture_model(vis, days)

    fig = plt.figure(figsize=(12.5, 6.2))
    left, cal_w = 0.2, 0.55
    top_ax = fig.add_axes([left, 0.84, cal_w, 0.12])
    cal = fig.add_axes([left, 0.15, cal_w, 0.67])
    bar = fig.add_axes([left + cal_w + 0.04, 0.15, 0.17, 0.67])

    # how often a crown is seen: chance of a clear view within 7 days, all crowns
    w7 = all_model.probablities[6]
    top_ax.fill_between(np.arange(365), 100 * w7, color='#4c72b0', alpha=0.85, lw=0)
    top_ax.set_ylim(0, 100)
    top_ax.set_yticks([0, 50, 100])
    top_ax.tick_params(labelsize=8)
    month_axis(top_ax, labels=False)
    top_ax.set_ylabel('clear view\nwithin 7 days (%)', fontsize=8.5)
    top_ax.spines[['top', 'right']].set_visible(False)

    rows = []
    for i, sp in enumerate(cand):
        model, _ = species_model(sp, min_clear)
        e = ev[ev.latin == sp].copy()
        e['p'] = score(e, model)
        y = len(cand) - 1 - i
        for kind, color, off in (('flowering', C_FLOWER, 0.18), ('deciduous', C_DECID, -0.18)):
            ek = e[e.kind == kind]
            alpha = min(0.6, 6 / max(len(ek), 1))
            for r in ek.itertuples():
                a, b = r.doy - 1 - r.event_length / 2, r.doy - 1 + r.event_length / 2
                for lo, hi in ((a, b), (a + 365, b + 365), (a - 365, b - 365)):  # wrap the year
                    if hi > 0 and lo < 365:
                        cal.plot([max(lo, 0), min(hi, 365)], [y + off] * 2, color=color,
                                 lw=5, alpha=alpha, solid_capstyle='butt')
            rows.append({'species': sp, 'kind': kind, 'n': len(ek),
                         'p': ek.p.mean() if len(ek) else np.nan, 'y': y})
    cal.set_yticks(range(len(cand)), [f'{italic(s)}  ({n_crowns[s]})' for s in cand[::-1]],
                   fontsize=9)
    cal.set_ylim(-0.6, len(cand) - 0.4)
    month_axis(cal)
    cal.tick_params(axis='x', labelsize=9)
    cal.grid(axis='x', alpha=0.3)
    cal.spines[['top', 'right']].set_visible(False)
    cal.plot([], [], color=C_FLOWER, lw=5, label='flowering (litter traps)')
    cal.plot([], [], color=C_DECID, lw=5, label='leafless (leaf-cover series)')
    cal.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=8.5,
               frameon=False)

    df = pd.DataFrame(rows)
    for kind, color, off in (('flowering', C_FLOWER, 0.18), ('deciduous', C_DECID, -0.18)):
        d = df[df.kind == kind]
        bar.barh(d.y + off, d.p, height=0.34, color=color)
        for r in d.itertuples():
            if r.n:
                bar.text(r.p + 0.01, r.y + off, f'{r.p:.2f} (n={r.n})', va='center',
                         fontsize=7, color='0.3')
    bar.set_xlim(0, 1.25)
    bar.set_xticks([0, 0.5, 1])
    bar.set_ylim(-0.6, len(cand) - 0.4)
    bar.set_yticks([])
    bar.set_xlabel('mean chance of observing\nan event (clear view)', fontsize=9)
    bar.tick_params(labelsize=8.5)
    bar.spines[['top', 'right']].set_visible(False)
    save(fig, outdir, 'ocm_observability_summary', dpi)
    df.to_csv(outdir / 'materials/ocm_observability_summary.csv', index=False)
    click.echo(df.round(2).to_string())


if __name__ == '__main__':
    cli()
