#!/usr/bin/env python
import click
import numpy as np
import pandas as pd

from fit_count_models import PoissonMixtureModel
from calc_decid_resolution import CadenceInterp
from plot_decid_summary import SPECIES


def get_doy(df, field='event_peak'):
    return np.array([
        float(p.dayofyear)
        for p in pd.to_datetime(df[field])
    ])


def get_obs_probabilities(models, peak, duration):
    model_idx = (peak - 1).astype(int)
    probs = [
        models[i].capture_prob([d])[0]
        for i, d in zip(model_idx, duration)
    ]
    return probs


def compute_comparison(models, df_s):

    peak = get_doy(df_s)
    dl = df_s['event_length']
    probs = get_obs_probabilities(models, peak, dl)
    return np.average(probs)


@click.command()
@click.argument('modelfile')
@click.argument('eventfile')
@click.argument('outputfile')
def main(modelfile, eventfile, outputfile):

    data = np.load(modelfile)
    models = PoissonMixtureModel.from_dict(data)

    df = pd.read_csv(eventfile)
    unique_species = sorted(np.unique(df['species']))

    data = []
    for sp in unique_species:
        df_s = df.loc[df['species'] == sp]

        frac = compute_comparison(models, df_s)
        data.append({
            'species': sp,
            'frac_obs': frac,
            'n': len(df_s),
        })

    df_out = pd.DataFrame(data)
    df_out.to_csv(outputfile, index=False)


if __name__ == '__main__':
    main()
