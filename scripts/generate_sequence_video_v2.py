#!/usr/bin/env python
import os
import yaml
import click
import shutil
from glob import glob
from ffmpeg import FFmpeg
from datetime import datetime
from PIL import Image, ImageDraw
from tempfile import TemporaryDirectory
from werkzeug.security import safe_join


from generate_sequence_video import generate_video


def parse_info(filename):
    base = os.path.basename(filename)
    directory = os.path.basename(
        os.path.dirname(filename)
    )
    parts1 = directory.split('_')
    parts2 = os.path.splitext(base)[0].split('_')
    return {
        'genus': parts1[0],
        'species': parts1[1],
        'crown_id': int(parts1[2]),
        'date': datetime(*map(int, parts2)),
    }


@click.command()
@click.argument('inputdir')
@click.argument('configfile')
@click.argument('outputfile')
def main(inputdir, configfile, outputfile):

    with open(configfile, 'r') as f:
        config = yaml.safe_load(f)

    ext = config['input_ext']

    images = glob(safe_join(inputdir, f'*_*_*.{ext}'))

    generate_video(images, config, outputfile, parse_info)


if __name__ == '__main__':
    main()
