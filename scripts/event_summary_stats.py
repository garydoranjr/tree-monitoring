#!/usr/bin/env python
import click
import numpy as np
import pandas as pd

from fit_empirical_count_models import EmpiricalCountModel


def get_doy(df, field='event_peak'):
    return np.array([
        float(p.dayofyear)
        for p in pd.to_datetime(df[field])
    ])


def get_obs_probabilities(model, peak, duration):
    model_idx = (peak - 1).astype(int)
    probs = [
        model.capture_prob(i, [d])[0]
        for i, d in zip(model_idx, duration)
    ]
    return probs


def compute_comparison(model, df_s):

    peak = get_doy(df_s)
    dl = df_s['event_length']
    probs = get_obs_probabilities(model, peak, dl)
    return np.average(probs)


@click.command()
@click.argument('modelfile')
@click.argument('eventfile')
@click.argument('outputfile')
def main(modelfile, eventfile, outputfile):

    model = EmpiricalCountModel.load(modelfile)

    df = pd.read_csv(eventfile)
    unique_species = sorted(np.unique(df['species']))

    data = []
    for sp in unique_species:
        df_s = df.loc[df['species'] == sp]

        frac = compute_comparison(model, df_s)
        data.append({
            'species': sp,
            'frac_obs': frac,
            'n': len(df_s),
        })

    df_out = pd.DataFrame(data)
    df_out.to_csv(outputfile, index=False)


if __name__ == '__main__':
    main()
