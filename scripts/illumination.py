#!/usr/bin/env python
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from calc_decid_resolution import (
    CadenceInterp
)


def select(df, start_str, end_str):
    df['date'] = pd.to_datetime(
        df['date'], dayfirst=True, yearfirst=False,
    )
    start = pd.to_datetime(start_str, dayfirst=True, yearfirst=False)
    end = pd.to_datetime(end_str, dayfirst=True, yearfirst=False)
    return df.loc[(df['date'] >= start) & (df['date'] <= end)]


def select_hour(df, hour):
    return df.loc[df['datetime'].str.endswith(hour)]


@click.command()
@click.argument('cadencefile')
@click.argument('minfile')
@click.argument('meanfile')
@click.argument('maxfile')
def main(cadencefile, minfile, meanfile, maxfile):
    cdata = np.load(cadencefile)
    med_idx = np.where(cdata['percentiles'] == 50.)[0][0]
    cintp = CadenceInterp(cdata['dates'], cdata['cadence'][:, med_idx])

    df_min = pd.read_csv(minfile)
    df_mean = pd.read_csv(meanfile)
    df_max = pd.read_csv(maxfile)

    df_min = select(df_min, '01/01/2022', '31/12/2022')
    df_mean = select(df_mean, '01/01/2022', '31/12/2022')
    df_max = select(df_max, '01/01/2022', '31/12/2022')

    df_min = select_hour(df_min, '10:00:00')
    df_mean = select_hour(df_mean, '10:00:00')
    df_max = select_hour(df_max, '10:00:00')

    rel = (df_mean['sr'] - df_min['srmn']) / (df_max['srmx'] - df_min['srmn'])

    dd = np.linspace(0, 364., 365)
    v = cintp(dd)

    #plt.plot(df_mean['date'], v)
    #plt.plot(df_mean['date'], rel)
    #plt.plot(df_mean['date'], rel.ewm(span=10).mean())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_mean['date'], df_mean['sr'], 'gray')
    #plt.plot(df_mean['date'], df_mean['sr'].ewm(span=30).mean())
    ax.set_ylabel('Avg. Radiation at 10am (W/m$^2$)', fontsize=16)
    ax2 = ax.twinx()
    ax2.plot(df_mean['date'], (1 / v), 'r')
    ax.set_xlabel('Date', fontsize=16)
    ax2.set_ylabel('Observations / Day', fontsize=16)

    plt.show()
    exit()

    plt.figure()
    plt.plot(df_mean['date'], 1000 * (1 / v))
    plt.plot(df_mean['date'], df_mean['sr'])
    plt.plot(df_mean['date'], df_mean['sr'].ewm(span=30).mean())

    plt.figure()
    plt.plot(df_max['date'], df_max['srmx'])
    plt.plot(df_max['date'], df_max['srmx'].ewm(span=30).mean())

    plt.show()

    print(df_mean)


if __name__ == '__main__':
    main()
