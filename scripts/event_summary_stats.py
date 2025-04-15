#!/usr/bin/env python
import click
import numpy as np
import pandas as pd

from calc_decid_resolution import CadenceInterp
from plot_decid_summary import SPECIES


def get_doy(df, field='event_peak'):
    return np.array([
        float(p.dayofyear)
        for p in pd.to_datetime(df[field])
    ])


def compute_comparison(cintp, df_s):

    peak = get_doy(df_s)
    v = cintp(peak)
    dl = df_s['event_length']
    return np.average(dl > v)


@click.command()
@click.argument('cadencefile')
@click.argument('decidfile')
@click.argument('outputfile')
def main(cadencefile, decidfile, outputfile):

    cdata = np.load(cadencefile)
    med_idx = np.where(cdata['percentiles'] == 50.)[0][0]
    cintp = CadenceInterp(cdata['dates'], cdata['cadence'][:, med_idx])

    df = pd.read_csv(decidfile)
    unique_species = sorted(np.unique(df['species']))
    #unique_species = sorted(SPECIES)

    data = []
    for sp in unique_species:
        df_s = df.loc[df['species'] == sp]

        frac = compute_comparison(cintp, df_s)
        data.append({
            'species': sp,
            'frac_obs': frac,
            'n': len(df_s),
        })

    df_out = pd.DataFrame(data)
    df_out.to_csv(outputfile, index=False)


if __name__ == '__main__':
    main()
