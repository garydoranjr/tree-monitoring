#!/usr/bin/env python
import click
import numpy as np
import pandas as pd
from datetime import timedelta, timezone
import matplotlib.pyplot as plt
from pysolar.solar import get_altitude
from pysolar.radiation import get_radiation_direct

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
    df['datetime'] = pd.to_datetime(
        df['datetime'], dayfirst=True, yearfirst=False,
    )
    return df.loc[df['datetime'].dt.hour == hour]


@click.command()
@click.argument('cadencefile')
@click.argument('emeanfile')
@click.argument('wmeanfile')
def main(cadencefile, emeanfile, wmeanfile):
    cdata = np.load(cadencefile)
    med_idx = np.where(cdata['percentiles'] == 50.)[0][0]
    cintp = CadenceInterp(cdata['dates'], cdata['cadence'][:, med_idx])

    df_mean = pd.concat([
        pd.read_csv(emeanfile),
        pd.read_csv(wmeanfile),
    ], ignore_index=True)

    df_10 = select_hour(df_mean, 10)
    daily_avg = df_10.groupby(df_10['datetime'].dt.dayofyear)['sr'].mean()

    dd = np.arange(0, 365.)
    v = cintp(dd)

    latlon = (9.161750812689062, -79.83767928034351)

    datetimes = pd.date_range(
        start='2020-01-01 15:00:00',
        periods=365,
        freq='D',
        tz='UTC'  # Ensures timestamps are timezone-aware
    )
    altitude = [
        get_altitude(latlon[0], latlon[1], dt)
        for dt in datetimes
    ]
    radiation = np.array([
        get_radiation_direct(d, a)
        for d, a in zip(datetimes, altitude)
    ])

    relative = daily_avg.values[:-1] / radiation

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dd, daily_avg.values[:-1], 'gray')
    #ax.plot(dd, radiation, 'r-')
    #plt.plot(df_mean['date'], df_mean['sr'].ewm(span=30).mean())
    ax.set_ylabel('Avg. Radiation at 10am (W/m$^2$)', fontsize=16)
    ax2 = ax.twinx()
    ax2.plot(dd, (1 / v), 'r')
    ax.set_xlabel('Date', fontsize=16)
    ax2.set_ylabel('Observations / Day', fontsize=16)

    plt.show()

    exit()

    #df_min = select(df_min, '01/01/2022', '31/12/2022')
    df_mean = select(df_mean, '01/01/2022', '31/12/2022')
    #df_max = select(df_max, '01/01/2022', '31/12/2022')

    #df_min = select_hour(df_min, '10:00:00')
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
    ax.plot(df_mean['date'], radiation, 'r-')
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
