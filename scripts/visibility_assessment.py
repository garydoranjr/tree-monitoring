import os
import json
import click
import numpy as np
from tqdm import tqdm
from glob import glob
from skops.io import load
from collections import defaultdict


def assess(model, embedding_file):
    raw = np.load(embedding_file)

    files = raw['image_ids']
    E = raw['embeddings']
    conf = model.predict_proba(E)[:, 1]
    return files, conf


@click.command()
@click.argument('model_file')
@click.argument('embedding_folder')
@click.argument('output_file')
def main(model_file, embedding_folder, output_file):

    embedding_files = glob(os.path.join(embedding_folder, '*'))

    model = load(model_file)

    values = defaultdict(lambda: defaultdict(lambda: np.nan))

    all_tags = set([])
    all_files = set([])

    for ef in tqdm(embedding_files, 'Assessments'):
        tag_id = os.path.splitext(os.path.basename(ef))[0]
        all_tags.add(tag_id)
        files, conf = assess(model, ef)
        all_files |= set(files)
        for f, c in zip(files, conf):
            values[f][tag_id] = float(c)

    all_tags = sorted(all_tags)
    all_files = sorted(all_files)

    n = len(all_tags)
    m = len(all_files)

    V = np.full((n, m), np.nan)

    for i, t in enumerate(all_tags):
        for j, f in enumerate(all_files):
            V[i, j] = values[f][t]

    np.savez_compressed(output_file,
        tags=np.array(all_tags),
        files=np.array(all_files),
        values=V,
    )


if __name__ == '__main__':
    main()
