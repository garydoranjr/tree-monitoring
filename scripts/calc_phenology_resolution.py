#!/usr/bin/env python
import click
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import scoreatpercentile, vonmises


from calc_decid_resolution import CadenceInterp


PCS = [10., 50., 90.]


def sample_cadence(intp, mean, kappa, pcs=PCS, size=1000):
    rvs = vonmises.rvs(kappa, loc=mean, size=size)
    doy_samp = (365 * rvs / (2 * np.pi)) % 365.
    cadence_samp = intp(doy_samp)
    return np.array([
        scoreatpercentile(cadence_samp, pc)
        for pc in pcs
    ])


@click.command()
@click.argument('cadencefile')
@click.argument('phenofile')
@click.argument('outputfile')
def main(cadencefile, phenofile, outputfile):

    cdata = np.load(cadencefile)
    med_idx = np.where(cdata['percentiles'] == 50.)[0][0]
    cintp = CadenceInterp(cdata['dates'], cdata['cadence'][:, med_idx])

    pheno_df = pd.read_csv(phenofile)
    pheno_df = pheno_df.loc[pheno_df['model'] == 'M2A']

    rows = []

    for i, row in pheno_df.iterrows():
        spcode = row['sp']
        ptype = row['type']
        mean = row['mean1']
        kappa = row['kappa1']

        row = {
            'spcode': spcode,
            'type': ptype,
        }
        pheno_pcs = sample_cadence(cintp, mean, kappa, PCS)
        for p, pc in zip(PCS, pheno_pcs):
            key = f'cadence_pc{int(p):02d}'
            row[key] = pc

        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(outputfile, index=False)


if __name__ == '__main__':
    main()
