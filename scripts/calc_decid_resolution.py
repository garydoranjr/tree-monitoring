#!/usr/bin/env python
import click
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import scoreatpercentile


PCS = [10., 50., 90.]


class CadenceInterp:


    def __init__(self, dates, values):
        days = (dates - np.min(dates)).astype('timedelta64[D]')
        days = np.hstack([days, [360]]).astype(float)
        values = np.hstack([values, [values[0]]])
        self.f = interp1d(days, values)


    def __call__(self, doy):
        doy %= 360.
        return self.f(doy)


def sample_cadence(intp, mean, std, pcs=PCS, size=1000):
    doy = mean.dayofyear
    doy_samp = np.random.normal(loc=doy, scale=std, size=size)
    cadence_samp = intp(doy_samp)
    return np.array([
        scoreatpercentile(cadence_samp, pc)
        for pc in pcs
    ])


@click.command()
@click.argument('cadencefile')
@click.argument('decidfile')
@click.argument('outputfile')
def main(cadencefile, decidfile, outputfile):

    cdata = np.load(cadencefile)
    med_idx = np.where(cdata['percentiles'] == 50.)[0][0]
    cintp = CadenceInterp(cdata['dates'], cdata['cadence'][:, med_idx])

    decid_df = pd.read_excel(decidfile)

    rows = []

    for i, row in decid_df.iterrows():
        spcode = row['spcode']
        start_mean = row['StartDate_mean']
        start_sd = row['StartDate_sd']
        end_mean = row['EndDate_mean']
        end_sd = row['EndDate_sd']

        # N/A check
        if pd.isna(spcode): continue
        if pd.isna(start_mean): continue
        if pd.isna(end_mean): continue
        if pd.isna(start_sd): continue
        if pd.isna(end_sd): continue

        row = { 'spcode': spcode }
        start_pcs = sample_cadence(cintp, start_mean, start_sd, PCS)
        end_pcs = sample_cadence(cintp, end_mean, end_sd, PCS)
        for prefix, pcs in zip(('start', 'end'), (start_pcs, end_pcs)):
            for p, pc in zip(PCS, pcs):
                key = f'{prefix}_pc{int(p):02d}'
                row[key] = pc

        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(outputfile, index=False)


if __name__ == '__main__':
    main()
