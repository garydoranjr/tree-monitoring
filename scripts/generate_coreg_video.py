#!/usr/bin/env python
import os
import json
import click
import shutil
from tqdm import tqdm
from ffmpeg import FFmpeg
from random import seed, shuffle
from itertools import combinations
from PIL import Image, ImageEnhance
from tempfile import TemporaryDirectory
from werkzeug.security import safe_join


def reformat_image(inputfile, outputdir):

    base = os.path.basename(inputfile)
    outputfile = safe_join(outputdir, base)

    with Image.open(inputfile) as im:
        w, h = im.size
        rsz = im.resize((2*w, 2*h))
        out = ImageEnhance.Brightness(rsz).enhance(3.0)
        out.save(outputfile)
        return outputfile


def generate_video(images, outputfile):

    # Get common extension
    exts = set([
        os.path.splitext(i)[1]
        for i in images
    ])
    assert len(exts) == 1
    ext = exts.pop()

    with TemporaryDirectory() as tmpdir:
        reformatted = [
            reformat_image(i, tmpdir)
            for i in images
        ]

        tmpmov = safe_join(
            tmpdir,
            os.path.basename(outputfile)
        )

        ffmpeg = FFmpeg().input(
            safe_join(tmpdir, f'*{ext}'),
            pattern_type='glob', framerate=1
        ).output(
            tmpmov,
            {"codec:v": "libx264"}
        )
        ffmpeg.execute()

        shutil.move(tmpmov, outputfile)


@click.command()
@click.argument('coregfile')
@click.argument('imagedir')
@click.argument('outputdir')
@click.option('-n', '--nvids', default=50)
def main(coregfile, imagedir, outputdir, nvids):

    with open(coregfile, 'r') as f:
        keys = [
            item['key']
            for item in json.load(f)
        ]

    keys = [
        key for key in keys if
        os.path.exists(safe_join(imagedir, f'{key}.tif'))
    ]

    combos = list(combinations(keys, 2))

    seed(0)
    shuffle(combos)

    pairs = combos[:nvids]

    jobs = [
        (
            (
                safe_join(imagedir, f'{src}.tif'),
                safe_join(imagedir, f'{tgt}.tif'),
            ),
            safe_join(outputdir, f'{src}-{tgt}.mp4')
        )
        for src, tgt in pairs
    ]

    remaining = [j for j in jobs if not os.path.exists(j[-1])]

    for job in tqdm(remaining):
        generate_video(*job)


if __name__ == '__main__':
    main()
