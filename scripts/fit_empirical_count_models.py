#!/usr/bin/env python
import click
import numpy as np
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

from windowed_obs_counts import setup_inputs, get_rate_pcs


class EmpiricalCountModel:


    def __init__(self, dates, window_sizes, probablities):
        self.dates = dates
        self.window_sizes = window_sizes
        self.probablities = probablities
        self.interp = RegularGridInterpolator(
            (self.window_sizes, np.arange(len(dates))),
            self.probablities
        )


    def save(self, outputfile):
        np.savez_compressed(outputfile, **{
            'dates': self.dates,
            'window_sizes': self.window_sizes,
            'probablities': self.probablities,
        })


    def load(inputfile):
        data = np.load(inputfile)
        return EmpiricalCountModel(
            data['dates'],
            data['window_sizes'],
            data['probablities'],
        )


    def capture_prob(self, doy, durations):
        return self.interp((durations, doy))


def get_obs_rate(dates, visible, sample, half_window):
    return np.average(get_rate_pcs(dates, visible, sample, half_window) > 0)


@click.command()
@click.argument('inputfile')
@click.argument('outputfile')
def main(inputfile, outputfile):

    N = 365

    dates, visible, samples = setup_inputs(inputfile)

    window_sizes = np.arange(1, N)

    rates = np.array([
        [
            get_obs_rate(
                dates, visible, s,
                np.timedelta64((24*d) // 2, 'h')
            )
            for s in samples
        ] for d in tqdm(window_sizes, 'Windows')
    ])

    model = EmpiricalCountModel(samples, window_sizes, rates)
    model.save(outputfile)


if __name__ == '__main__':
    main()
