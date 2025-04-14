#!/usr/bin/env python
import click
import numpy as np
from tqdm import tqdm
from scipy.stats import poisson
from scipy.optimize import minimize, LinearConstraint
from collections import defaultdict
import matplotlib.pyplot as plt


class PoissonMixtureModel:

    def __init__(self, k=3, window_size=None, alpha=None, mu=None):
        self.k = k

        self.window_size = window_size
        self._alpha = alpha
        self._mu = mu


    @staticmethod
    def to_dict(models):
        data = defaultdict(list)
        for model in models:
            data['k'].append(model.k)
            data['window_size'].append(model.window_size)
            data['alpha'].append(model._alpha)
            data['mu'].append(model._mu)
        return { k: np.vstack(v) for k, v in data.items() }


    @staticmethod
    def from_dict(data):
        it = zip(
            data['k'],
            data['window_size'],
            data['alpha'],
            data['mu'],
        )
        return [
            PoissonMixtureModel(k=ki, window_size=wi, alpha=ai, mu=mi)
            for ki, wi, ai, mi in it
        ]


    def pmf(self, counts, alpha=None, mu=None):
        if alpha is None: alpha = self._alpha
        if mu is None: mu = self._mu
        if alpha is None or mu is None:
            raise ValueError('Must call "fit" before "pmf"')

        assert len(alpha) == len(mu) and len(alpha) == self.k

        wpmf = 0
        for ai, mi in zip(alpha, mu):
            wpmf += ai * poisson.pmf(counts, mi)

        return wpmf


    def fit(self, counts, window_size):
        self.window_size = window_size
        k = self.k

        def loss(params):
            alpha = params[:k]
            mu = params[k:]
            return -np.average(np.log(self.pmf(counts, alpha, mu)))

        alpha0 = np.full((k,), 1. / k)
        mu = np.average(counts)
        mu0 = np.array([0.5 * mu, mu, 1.5 * mu])
        mu0 = np.linspace(0.5 * mu, 2.0 * mu, k)
        p0 = np.hstack([alpha0, mu0])

        I = np.eye(2*self.k)
        lb = np.zeros(2*k)
        ub = np.hstack([np.ones(k), np.full((k,), np.inf)])

        result = minimize(loss, p0,
            constraints=[
                LinearConstraint(I, lb, ub),
                LinearConstraint(np.hstack([np.ones(k), np.zeros(k)]), 1.0, 1.0),
            ],
        )
        if not result.success:
            raise ValueError(f'Error minimizing loss: {result}')

        self._alpha = result.x[:k]
        self._mu = result.x[k:]

        return self


    def capture_prob(self, durations):
        E = np.exp(-np.outer(self._mu, durations) / self.window_size)
        return 1 - np.dot(self._alpha, E)


@click.command()
@click.argument('inputfile')
@click.argument('outputfile')
def main(inputfile, outputfile):
    data = np.load(inputfile)
    counts = data['counts']
    win_size = data['window_size']

    models = [
        PoissonMixtureModel(k=3).fit(ci, win_size)
        for ci in tqdm(counts)
    ]

    results = PoissonMixtureModel.to_dict(models)
    results['dates'] = data['dates']

    np.savez_compressed(outputfile, **results)


if __name__ == '__main__':
    main()
