#!/usr/bin/env python
import click
import numpy as np
import pandas as pd
from datetime import timedelta, timezone
import matplotlib.pyplot as plt
from pysolar.solar import get_altitude
from pysolar.radiation import get_radiation_direct
from pysolar.util import diffuse_underclear, diffuse_underovercast
from scipy.stats import linregress
from scipy.ndimage import convolve1d


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
@click.argument('countfile')
@click.argument('emeanfile')
@click.argument('wmeanfile')
@click.argument('outputfile')
def main(countfile, emeanfile, wmeanfile, outputfile):
    data = np.load(countfile)
    v = np.average(data['counts'], axis=1) / data['window_size']

    df_mean = pd.concat([
        pd.read_csv(emeanfile),
        pd.read_csv(wmeanfile),
    ], ignore_index=True)

    df_10 = select_hour(df_mean, 10)
    daily_avg = df_10.groupby(df_10['datetime'].dt.dayofyear)['sr'].mean()

    dd = np.arange(0, 365.) + 1

    latlon = (9.161750812689062, -79.83767928034351)

    datetimes = pd.date_range(
        start='2020-01-01 15:00:00',
        periods=365,
        freq='D',
        tz='UTC'  # Ensures timestamps are timezone-aware
    )
    radiation = np.array([
        diffuse_underclear(latlon[0], latlon[1], dt)
        for dt in datetimes
    ])

    relative = (radiation - daily_avg.values[:-1]) / radiation
    relative = daily_avg.values[:-1] / radiation

    N = int(data['window_size'])
    smoothed = convolve1d(daily_avg.values[:-1], np.ones(N) / N, mode='wrap')

    fig, ax = plt.subplots(figsize=(10, 5))

    #ax.plot(dd, daily_avg.values[:-1] / radiation, 'gray')
    ax.plot(dd, smoothed / radiation, 'k')
    ax.set_ylim(0.35, 1.0)
    #ax.plot(dd, radiation, 'r-')
    #plt.plot(df_mean['date'], df_mean['sr'].ewm(span=30).mean())
    #ax.set_ylabel('Avg. Radiation at 10am (W/m$^2$)', fontsize=16)
    ax.set_ylabel('Fraction of Expected Radiation', fontsize=16)
    ax2 = ax.twinx()
    ax2.plot(dd, v, 'r')
    ax.set_xlabel('Date', fontsize=16)
    ax2.set_ylabel('Observations / Day', fontsize=16, color='red')

    ax2.set_ylim(-0.3, 1.2)
    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ymin, ymax = ax2.get_ylim()

    result = linregress(v, smoothed / radiation)
    print(f'R = {result.rvalue:.2f}')
    y1min = result.slope * ymin + result.intercept
    y1max = result.slope * ymax + result.intercept
    ax.set_ylim(y1min, y1max)

    ticks = []
    for month in range(1, 13):
        ticks.append(pd.to_datetime(f'2021-{month:02d}-01').dayofyear)
    labels = [
        'Jan', 'Feb', 'Mar',
        'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep',
        'Oct', 'Nov', 'Dec',
    ]

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(1, 365)

    fig.savefig(outputfile)


if __name__ == '__main__':
    main()
