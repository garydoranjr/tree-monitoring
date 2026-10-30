#!/usr/bin/env python
"""Train a crown-level temporal classifier (pixel-set encoder + temporal
transformer, after Garnot et al. 2020) on the clear-pixel series of
build_crown_pixel_series.py, and predict the flowering-or-deciduous event
for every crown on every Planet date.

Each crown's series is its sequence of **valid observations**: scenes on
which at least --min-pixels of its pixels are clear and at least
--min-clear-frac of its valid pixels are clear. Cloudy, shadowed and nodata
pixels were already left out by the builder; observations that fail these
tests are left out here, so they are neither inputs nor targets.

Model, per observation:
1. Pixel-set encoder: up to --pixels clear pixels (B, G, R, N, NDVI,
   green-red index; standardised) go through a shared MLP. They are pooled
   by masked mean and standard deviation, padding excluded, and joined with
   log crown area, clear fraction, pixel count, coreg_ok and a SuperDove
   flag.
2. Temporal transformer over the observations of a --window-day window,
   positioned by days from the window centre (sinusoidal) and day of year,
   with padding masked out of the attention.
3. One event logit per observation.

Labels are the drone crown events (load_drone in
plot_crown_species_comparison.py: 1-(1-p_flower)(1-p_decid) > 0.5), put on
every valid observation of the crown within --label-days of the flight. The
loss is a masked BCE over labelled observations only.

Folds: with --folds and --fold K, the labels of the flights of fold K's
chips are withheld (imagery on every date is still input, as it would be
when interpolating). By default only the fold CSV's flights give labels, as
for the Mask R-CNN CV; --all-flights adds every other drone flight more than
--exclude-days from a held-out flight. --fold all trains on every fold.
--crown-folds N --crown-fold J also withholds the labels of the crowns in
the J-th of N east-west strips of the plot, to test crowns the model never
saw labels for. --split-x X --test-side east instead withholds every crown
whose centroid lies east of x = X (or west of it): with X = 626281.5, the
line at which the Mask R-CNN spatial experiments split their chips into
training (left) and test (right) windows.

Writes OUTDIR/model.pt, OUTDIR/params.json (including the training flights)
and OUTDIR/predictions.csv: one row per valid crown observation with
p_event. See docs/crown_temporal_model.md.

Typical usage:
    python scripts/train_crown_pse_ltae.py SERIES OUTDIR/fold0 \\
        --folds config/temporal_folds_20261012.csv --fold 0
"""
import json
import math
import os
import time

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from plot_crown_species_comparison import NC4S, load_drone

EPOCH0 = pd.Timestamp('2018-01-01')
N_EXTRA = 5  # per-observation extras, see CrownSeries.extras


def select_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def days(dates):
    return ((pd.to_datetime(dates) - EPOCH0) / pd.Timedelta(days=1)).astype(int)


def flight_folds(folds_csv):
    """{flight date: fold} from a fold CSV, using each chip's first drone
    date as persistence_baseline.py does."""
    df = pd.read_csv(folds_csv)
    flights = pd.to_datetime(df['drone_dates'].astype(str).str.split(';').str[0])
    out = {}
    for fl, k in zip(flights, df['fold']):
        if out.setdefault(fl, k) != k:
            raise click.ClickException(f'flight {fl.date()} is in two folds')
    return out


def crown_centroid_x(series_path):
    """Centroid x of every crown of a crown_pixel_series.npz, in the crown
    map's CRS."""
    build = json.loads(str(np.load(series_path)['build']))
    return gpd.read_file(build['crowns']).geometry.centroid.x.values


class CrownSeries:
    """The valid observations of every crown, and their pixels."""

    def __init__(self, path, min_pixels, min_clear_frac):
        d = np.load(path)
        self.scenes = pd.DataFrame(json.loads(str(d['scenes'])))
        self.crowns = pd.DataFrame(json.loads(str(d['crowns'])))
        self.build = json.loads(str(d['build']))
        self.pixels = d['pixels']
        obs = pd.DataFrame({k[4:]: d[k] for k in d.files if k.startswith('obs_')})
        obs['clear_frac'] = obs['n_clear'] / np.maximum(obs['n_valid'], 1)
        obs = obs[(obs['n_clear'] >= min_pixels)
                  & (obs['clear_frac'] >= min_clear_frac)]
        sc = self.scenes
        obs['day'] = days(sc['date']).values[obs['scene']]
        obs['coreg_ok'] = sc['coreg_ok'].values[obs['scene']].astype(np.float32)
        superdove = sc['sat_id'].str.len().eq(4) & sc['sat_id'].str.startswith('2')
        obs['superdove'] = superdove.values[obs['scene']].astype(np.float32)
        obs = obs.sort_values(['crown', 'day', 'scene']).reset_index(drop=True)
        self.obs = obs
        area = self.crowns['area_m2'].values[obs['crown']]
        self.extras = np.stack([
            np.log(area) - 4.5,
            obs['clear_frac'].values,
            np.log(obs['n_clear'].values) - 2.0,
            obs['coreg_ok'].values,
            obs['superdove'].values,
        ], axis=1).astype(np.float32)
        # obs rows of crown c are start[c]:start[c + 1]
        self.start = np.searchsorted(obs['crown'].values,
                                     np.arange(len(self.crowns) + 1))

    def pixel_features(self, rows, n_pix, rng):
        """(T, n_pix, 6) features and (T, n_pix) validity for obs rows."""
        n = self.obs['n_clear'].values[rows]
        off = self.obs['offset'].values[rows]
        j = np.arange(n_pix)[None, :]
        draw = (rng.random((len(rows), n_pix)) * n[:, None]).astype(np.int64)
        # all pixels when they fit, else a random draw with replacement
        idx = np.where(n[:, None] <= n_pix, np.minimum(j, n[:, None] - 1), draw)
        valid = (j < n[:, None]) | (n[:, None] > n_pix)
        x = self.pixels[off[:, None] + idx].astype(np.float32) / 10000.0
        b, g, r, nir = (x[..., i] for i in range(4))
        ndvi = (nir - r) / np.maximum(nir + r, 1e-6)
        gri = (g - r) / np.maximum(g + r, 1e-6)
        return np.concatenate([x, ndvi[..., None], gri[..., None]], -1), valid


def attach_labels(series, drone, label_days):
    """obs-row -> (event, flight) for valid observations near a flight."""
    obs = series.obs
    tags = series.crowns['tag']
    ok = (tags != -9999) & ~tags.duplicated(keep=False)
    crown_of_tag = pd.Series(series.crowns.index[ok], index=tags[ok])
    drone = drone[drone['tag'].isin(crown_of_tag.index)]
    lab = pd.DataFrame({'crown': crown_of_tag[drone['tag']].values,
                        'flight': drone['date'].values,
                        'event': drone['event'].values})
    fday = days(lab['flight']).values
    # obs are sorted by (crown, day): find each label's rows by key range
    key = obs['crown'].values.astype(np.int64) * 100000 + obs['day'].values
    lkey = lab['crown'].values.astype(np.int64) * 100000 + fday
    lo = np.searchsorted(key, lkey - label_days)
    hi = np.searchsorted(key, lkey + label_days, side='right')
    which = np.repeat(np.arange(len(lab)), hi - lo)
    rows = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)] or [[]])
    m = pd.DataFrame({'row': rows.astype(np.int64),
                      'event': lab['event'].values[which],
                      'flight': lab['flight'].values[which],
                      'gap': np.abs(obs['day'].values[rows.astype(np.int64)]
                                    - fday[which])})
    # an observation near two flights takes the nearer
    m = m.sort_values(['row', 'gap']).drop_duplicates('row')
    return m.set_index('row')[['event', 'flight', 'gap']]


class Sinusoid(nn.Module):
    def __init__(self, d, tau=1000.0):
        super().__init__()
        self.register_buffer('freq', tau ** (-torch.arange(0, d, 2) / d))

    def forward(self, t):
        a = t[..., None] * self.freq
        return torch.cat([a.sin(), a.cos()], -1)


class PSELTAE(nn.Module):
    def __init__(self, n_in=6, d=128, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.mlp1 = nn.Sequential(nn.Linear(n_in, 32), nn.LayerNorm(32), nn.ReLU(),
                                  nn.Linear(32, 64), nn.LayerNorm(64), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Linear(128 + N_EXTRA, d), nn.LayerNorm(d),
                                  nn.ReLU(), nn.Linear(d, d))
        self.pos = Sinusoid(d)
        self.doy = nn.Linear(2, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout,
                                           batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, layers,
                                              enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x, pix_mask, extras, rel_day, doy, pad):
        """x (B, T, S, C), pix_mask (B, T, S), extras (B, T, E), rel_day and
        doy (B, T), pad (B, T) true where there is no observation."""
        h = self.mlp1(x)
        w = pix_mask[..., None].float()
        cnt = w.sum(2).clamp(min=1)
        mean = (h * w).sum(2) / cnt
        std = (((h - mean[:, :, None]) ** 2 * w).sum(2) / cnt).clamp(min=1e-6).sqrt()
        e = self.mlp2(torch.cat([mean, std, extras], -1))
        ang = 2 * math.pi * doy / 365.25
        e = e + self.pos(rel_day) + self.doy(torch.stack([ang.sin(), ang.cos()], -1))
        return self.head(self.temporal(e, src_key_padding_mask=pad)).squeeze(-1)


class Batcher:
    """Windows of one crown's observations, as model tensors."""

    def __init__(self, series, norm, window, max_obs, n_pix):
        self.s, self.norm = series, norm
        self.window, self.max_obs, self.n_pix = window, max_obs, n_pix

    def window_rows(self, crown, center, rng, keep=None):
        lo, hi = self.s.start[crown], self.s.start[crown + 1]
        d = self.s.obs['day'].values[lo:hi]
        a = lo + np.searchsorted(d, center - self.window / 2)
        b = lo + np.searchsorted(d, center + self.window / 2, side='right')
        rows = np.arange(a, b)
        if len(rows) > self.max_obs:
            # the rows in keep first, then the ones nearest the centre
            dist = np.abs(self.s.obs['day'].values[rows] - center).astype(float)
            if keep is not None:
                dist[np.isin(rows, keep)] = -1
            rows = np.sort(rows[np.argsort(dist, kind='stable')[:self.max_obs]])
        return rows

    def tensors(self, windows, rng, drop=0.0):
        """windows: list of (rows, center). Returns model inputs and the
        (B, T) obs-row index, -1 for padding."""
        if drop > 0:
            # never drop a whole window: an all-padding sequence makes the
            # attention return NaN
            kept = [r[rng.random(len(r)) >= drop] for r, _ in windows]
            windows = [(k if len(k) else r, c)
                       for k, (r, c) in zip(kept, windows)]
        T = max(1, max(len(r) for r, _ in windows))
        B, S = len(windows), self.n_pix
        x = np.zeros((B, T, S, 6), np.float32)
        pm = np.zeros((B, T, S), bool)
        ex = np.zeros((B, T, N_EXTRA), np.float32)
        rel = np.zeros((B, T), np.float32)
        doy = np.zeros((B, T), np.float32)
        idx = np.full((B, T), -1, np.int64)
        obs_day = self.s.obs['day'].values
        for i, (rows, center) in enumerate(windows):
            n = len(rows)
            if n == 0:
                continue
            f, v = self.s.pixel_features(rows, S, rng)
            x[i, :n], pm[i, :n] = (f - self.norm[0]) / self.norm[1], v
            ex[i, :n] = self.s.extras[rows]
            rel[i, :n] = obs_day[rows] - center
            doy[i, :n] = (EPOCH0 + pd.to_timedelta(obs_day[rows], 'D')).dayofyear
            idx[i, :n] = rows
        t = lambda a: torch.from_numpy(a)
        return (t(x), t(pm), t(ex), t(rel), t(doy), t(idx < 0)), idx


def pixel_norm(series, rng, n=200000):
    rows = rng.integers(0, len(series.obs), n // 8)
    f, v = series.pixel_features(rows, 8, rng)
    f = f[v]
    return f.mean(0), f.std(0) + 1e-6


def predict(model, batcher, crowns, device, rng, batch=64):
    """p_event for every valid observation of the given crowns. Windows are
    centred every window/4 days; each observation takes its prediction from
    the window whose centre is nearest."""
    s = batcher.s
    out = np.full(len(s.obs), np.nan, np.float32)
    step = batcher.window / 4
    jobs = []
    for c in crowns:
        lo, hi = s.start[c], s.start[c + 1]
        if hi == lo:
            continue
        d = s.obs['day'].values[lo:hi]
        for center in np.arange(d[0], d[-1] + step, step):
            own = lo + np.flatnonzero(np.abs(d - center) <= step / 2)
            if len(own):
                jobs.append((batcher.window_rows(c, center, rng, own),
                             center, own))
    model.eval()
    with torch.no_grad():
        for i in range(0, len(jobs), batch):
            chunk = jobs[i:i + batch]
            inp, idx = batcher.tensors([(r, c) for r, c, _ in chunk], rng)
            p = torch.sigmoid(model(*(a.to(device) for a in inp))).cpu().numpy()
            for b, (_, _, own) in enumerate(chunk):
                sel = np.isin(idx[b], own)
                out[idx[b][sel]] = p[b][sel]
    return out


@click.command()
@click.argument('series_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('outdir', type=click.Path(file_okay=False))
@click.option('--folds', 'folds_csv', required=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option('--fold', required=True, help="Fold K to withhold, or 'all'.")
@click.option('--all-flights', is_flag=True,
              help='Also take labels from drone flights not in the fold CSV.')
@click.option('--exclude-days', default=7, show_default=True,
              help='With --all-flights: skip extra flights this close to a '
                   'held-out flight.')
@click.option('--crown-folds', default=0, show_default=True,
              help='Number of east-west crown strips (0: no crown hold-out).')
@click.option('--crown-fold', default=0, show_default=True)
@click.option('--split-x', default=None, type=float,
              help='Withhold the crowns on --test-side of this x (m, crown '
                   'map CRS); 626281.5 is the Mask R-CNN east/west line.')
@click.option('--test-side', default='east', show_default=True,
              type=click.Choice(['east', 'west']))
@click.option('--nc4', 'nc4s', multiple=True, default=NC4S, show_default=True)
@click.option('--min-pixels', default=2, show_default=True, type=click.IntRange(1),
              help='Minimum clear pixels for a valid crown observation.')
@click.option('--min-clear-frac', default=0.5, show_default=True,
              help='Minimum clear share of the crown for a valid observation.')
@click.option('--min-crown-area', default=36.0, show_default=True,
              help='Crowns smaller than this (m2) give no labels; 36 m2 is '
                   'the 64 px at 0.75 m of persistence_baseline.py.')
@click.option('--label-days', default=2, show_default=True)
@click.option('--window', default=120, show_default=True, help='Days.')
@click.option('--max-obs', default=96, show_default=True)
@click.option('--pixels', 'n_pix', default=16, show_default=True)
@click.option('--d-model', default=128, show_default=True)
@click.option('--layers', default=2, show_default=True)
@click.option('--epochs', default=30, show_default=True)
@click.option('--batch-size', default=64, show_default=True)
@click.option('--lr', default=1e-3, show_default=True)
@click.option('--weight-decay', default=1e-2, show_default=True)
@click.option('--pos-weight', default=1.0, show_default=True,
              help='BCE weight on event labels (events are ~4%).')
@click.option('--date-dropout', default=0.2, show_default=True,
              help='Share of observations dropped from each training window, '
                   'as extra simulated cloud.')
@click.option('--max-crowns', default=0, show_default=True,
              help='Use only this many labelled crowns (smoke tests).')
@click.option('--seed', default=0, show_default=True)
def main(series_path, outdir, folds_csv, fold, all_flights, exclude_days,
         crown_folds, crown_fold, split_x, test_side, nc4s, min_pixels,
         min_clear_frac, min_crown_area, label_days, window, max_obs, n_pix,
         d_model, layers, epochs, batch_size, lr, weight_decay, pos_weight,
         date_dropout, max_crowns, seed):
    """Train on SERIES_PATH (crown_pixel_series.npz) and write OUTDIR."""
    if crown_folds and split_x is not None:
        raise click.ClickException('--crown-folds and --split-x are exclusive')
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = select_device()

    series = CrownSeries(series_path, min_pixels, min_clear_frac)
    labels = attach_labels(series, load_drone(nc4s), label_days)
    obs_crown = series.obs['crown'].values
    big = series.crowns['area_m2'].values >= min_crown_area
    labels = labels[big[obs_crown[labels.index]]]

    fmap = flight_folds(folds_csv)
    csv_flights = pd.Series(fmap)
    if fold == 'all':
        test_flights = []
    else:
        test_flights = list(csv_flights.index[csv_flights == int(fold)])
        if not test_flights:
            raise click.ClickException(f'fold {fold} not in {folds_csv}')
    train_flights = [f for f in csv_flights.index if f not in test_flights]
    if all_flights:
        held = np.array(days(test_flights)) if test_flights else np.array([])
        for f in labels['flight'].unique():
            f = pd.Timestamp(f)
            if f in fmap:
                continue
            if len(held) == 0 or np.abs(held - days([f])[0]).min() > exclude_days:
                train_flights.append(f)
    train_lab = labels[labels['flight'].isin(train_flights)]

    test_crowns = np.zeros(len(series.crowns), bool)
    if crown_folds:
        cx = crown_centroid_x(series_path)
        edges = np.quantile(cx, np.linspace(0, 1, crown_folds + 1))
        strip = np.clip(np.searchsorted(edges, cx, side='right') - 1, 0,
                        crown_folds - 1)
        test_crowns = strip == crown_fold
    elif split_x is not None:
        east = crown_centroid_x(series_path) >= split_x
        test_crowns = east if test_side == 'east' else ~east
    if test_crowns.any():
        train_lab = train_lab[~test_crowns[obs_crown[train_lab.index]]]
        assert not test_crowns[obs_crown[train_lab.index]].any()
        click.echo(f'crown hold-out: {(~test_crowns).sum()} training crowns, '
                   f'{test_crowns.sum()} withheld')

    # one training sample per labelled (crown, flight), centred on the flight
    samples = (train_lab.assign(crown=obs_crown[train_lab.index])
               .groupby(['crown', 'flight']).size().reset_index()[['crown', 'flight']])
    if max_crowns:
        keep = rng.permutation(samples['crown'].unique())[:max_crowns]
        samples = samples[samples['crown'].isin(keep)]
    samples['center'] = days(samples['flight'])
    target = np.full(len(series.obs), np.nan, np.float32)
    target[train_lab.index] = train_lab['event'].astype(np.float32).values
    click.echo(f'{len(series.obs)} valid crown observations of '
               f'{(np.diff(series.start) > 0).sum()} crowns; '
               f'{len(train_flights)} training flights '
               f'({len(test_flights)} withheld), {len(train_lab)} labelled '
               f'observations ({train_lab["event"].mean():.3f} events), '
               f'{len(samples)} crown-flight samples; device {device}')

    norm = pixel_norm(series, rng)
    batcher = Batcher(series, norm, window, max_obs, n_pix)
    model = PSELTAE(d=d_model, layers=layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * math.ceil(len(samples) / batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    pw = torch.tensor(pos_weight, device=device)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        order = rng.permutation(len(samples))
        for i in range(0, len(order), batch_size):
            sub = samples.iloc[order[i:i + batch_size]]
            wins = []
            for c, center in zip(sub['crown'], sub['center']):
                center = center + rng.integers(-window // 4, window // 4 + 1)
                wins.append((batcher.window_rows(c, center, rng), center))
            inp, idx = batcher.tensors(wins, rng, drop=date_dropout)
            y = np.where(idx >= 0, target[np.maximum(idx, 0)], np.nan)
            m = torch.from_numpy(~np.isnan(y)).to(device)
            if not m.any():
                continue
            logit = model(*(a.to(device) for a in inp))
            yt = torch.from_numpy(np.nan_to_num(y)).to(device)
            loss = F.binary_cross_entropy_with_logits(logit[m], yt[m], pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * int(m.sum())
            n += int(m.sum())
        history.append({'epoch': epoch, 'loss': tot / max(n, 1)})
        click.echo(f'epoch {epoch}: loss {tot / max(n, 1):.4f} '
                   f'({time.time() - t0:.0f} s)')

    torch.save({'model': model.state_dict(), 'norm': norm}, f'{outdir}/model.pt')
    crowns = np.flatnonzero(np.diff(series.start) > 0)
    p = predict(model, batcher, crowns, device, rng)
    obs = series.obs
    sc = series.scenes
    pred = pd.DataFrame({
        'scene': sc['scene'].values[obs['scene']],
        'date': sc['date'].values[obs['scene']],
        'crown': obs['crown'].values,
        'tag': series.crowns['tag'].values[obs['crown']],
        'mnemonic': series.crowns['mnemonic'].values[obs['crown']],
        'area_m2': series.crowns['area_m2'].values[obs['crown']],
        'n_px': obs['n_px'].values, 'n_clear': obs['n_clear'].values,
        'clear_frac': obs['clear_frac'].values,
        'coreg_ok': obs['coreg_ok'].values.astype(bool),
        'test_crown': test_crowns[obs['crown'].values],
        'p_event': p,
    })
    pred.to_csv(f'{outdir}/predictions.csv', index=False, float_format='%.5g')
    with open(f'{outdir}/params.json', 'w') as fh:
        json.dump({
            'series': series_path, 'folds': folds_csv, 'fold': fold,
            'all_flights': all_flights, 'exclude_days': exclude_days,
            'crown_folds': crown_folds, 'crown_fold': crown_fold,
            'split_x': split_x, 'test_side': test_side,
            'nc4': list(nc4s), 'min_pixels': min_pixels,
            'min_clear_frac': min_clear_frac, 'min_crown_area': min_crown_area,
            'label_days': label_days, 'window': window, 'max_obs': max_obs,
            'pixels': n_pix, 'd_model': d_model, 'layers': layers,
            'epochs': epochs, 'batch_size': batch_size, 'lr': lr,
            'weight_decay': weight_decay, 'pos_weight': pos_weight,
            'date_dropout': date_dropout, 'max_crowns': max_crowns, 'seed': seed,
            'train_flights': sorted(str(pd.Timestamp(f).date()) for f in train_flights),
            'test_flights': sorted(str(pd.Timestamp(f).date()) for f in test_flights),
            'history': history,
        }, fh, indent=1)
    click.echo(f'wrote {outdir}/predictions.csv ({np.isfinite(p).sum()} predicted)')


if __name__ == '__main__':
    main()
