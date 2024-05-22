#!/usr/bin/env python
import os
import json
import click
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

from util import load_config
from planet_coreg import stem


def load_successes(coregfile):
    with open(coregfile, 'r') as f:
        results = json.load(f)

    successes = {}
    key = stem(coregfile)
    for tgt, r in results.items():
        successes[key, tgt] = r['success']

    return successes


def successes_to_matrix(successes):
    keys = set([])
    for src, tgt in successes.keys():
        keys |= set([src, tgt])
    keys = sorted(keys)
    n = len(keys)

    S = np.full((n, n), np.nan)
    for i, k1 in tqdm(list(enumerate(keys)), 'Converting'):
        for j, k2 in enumerate(keys):
            key = (k1, k2)
            if k1 == k2:
                S[i, j] = 1
            elif key not in successes:
                continue
            else:
                S[i, j] = 1 if successes[key] else -1

    return S


@click.command()
@click.argument('coregdir')
@click.argument('outputfile')
def main(coregdir, outputfile):

    files = sorted(glob(os.path.join(coregdir, '*.json')))

    successes = {}
    for f in tqdm(files, 'Loading'):
        successes.update(load_successes(f))

    S = successes_to_matrix(successes)

    idx = np.argsort(np.sum(S, axis=0))
    S = S[idx].T[idx]

    fig, ax = plt.subplots(figsize=(24, 24))

    ax.set_facecolor('gray')
    ax.imshow(S, cmap='RdYlGn', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(outputfile, bbox_inches='tight')



if __name__ == '__main__':
    main()
