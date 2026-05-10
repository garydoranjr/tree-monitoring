#!/usr/bin/env python
import click
import numpy as np
import pandas as pd

from fit_empirical_count_models import EmpiricalCountModel
from event_summary_stats import compute_comparison


@click.command()
@click.argument('modelfile')
@click.argument('decidfile')
@click.argument('outputfile')
def main(modelfile, decidfile, outputfile):

    model = EmpiricalCountModel.load(modelfile)

    df = pd.read_csv(decidfile).rename(columns={
        'decid_peak': 'event_peak',
        'decid_length': 'event_length',
    })
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
