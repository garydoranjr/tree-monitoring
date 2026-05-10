#!/usr/bin/env python
import os
import click
import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt


def get_peak(dates, decid, n):
    w = np.ones(n)
    conv = convolve1d(decid, w, mode='wrap')
    return dates.iloc[np.argmax(conv)]


@click.command()
@click.argument('inputfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
def main(inputfile, outputfile):
    df = pd.read_csv(inputfile, index_col=0)

    tags = sorted(np.unique(df['tag']))

    lengths = defaultdict(list)
    peaks = defaultdict(list)
    data = []
    for tag in tags:
        t = df.loc[df['tag'] == tag]

        s = np.unique(t['spcode'])
        assert len(s) == 1
        species = s[0]

        decid = t['predicted_branch']
        l = np.sum(decid > 40.)
        if l > 0:
            peak = get_peak(t['date'], decid, l)
            data.append({
                'tag': tag,
                'species': species,
                'event_length': l,
                'event_peak': peak,
            })
            lengths[species].append(l)
            peaks[species].append(peak)

    df_out = pd.DataFrame(data)
    df_out.to_csv(outputfile, index=False)



if __name__ == '__main__':
    main()
