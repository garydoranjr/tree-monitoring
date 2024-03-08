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


def parse_date(filename):
    parts = os.path.splitext(filename)[0].split('_')
    y, m, d = map(int, parts[1:])
    return datetime(y, m, d)


def reformat_image(inputfile, config, outputdir):
    video_size = config['video_size']
    date_fmt = config['date_format']
    padding = config['padding']
    left, right, top, bottom = (
        padding.get(side, 0)
        for side in ('left', 'right', 'top', 'bottom')
    )

    base = os.path.basename(inputfile)
    date = parse_date(base)
    datestr = date.strftime(date_fmt['format'])
    outputfile = os.path.join(outputdir, base)

    total_size = (
        video_size[0] + left + right,
        video_size[1] + top + bottom
    )

    out = Image.new('RGB', total_size)

    # Draw original image
    with Image.open(inputfile) as im:
        w, h = im.size
        xoff = ((video_size[0] - w) // 2) + left
        yoff = ((video_size[1] - h) // 2) + top
        out.paste(im, (xoff, yoff))

    draw = ImageDraw.Draw(out)
    draw.text(
        date_fmt['xy'],
        datestr,
        **date_fmt.get('draw_kwargs', {})
    )

    out.save(outputfile)
    return outputfile


@click.command()
@click.argument('inputdir')
@click.argument('sequence_id', type=int)
@click.argument('configfile')
@click.argument('outputfile')
def main(inputdir, sequence_id, configfile, outputfile):

    with open(configfile, 'r') as f:
        config = yaml.safe_load(f)

    ext = config['input_ext']

    images = glob(os.path.join(inputdir, f'{sequence_id}*.{ext}'))

    with TemporaryDirectory() as tmpdir:
        reformatted = [
            reformat_image(i, config, tmpdir)
            for i in images
        ]

        tmpmov = os.path.join(
            tmpdir,
            os.path.basename(outputfile)
        )

        ffmpeg = (
            FFmpeg()
            .input(
                os.path.join(tmpdir, f'*.{ext}'),
                pattern_type='glob', framerate=1
            )
            .output(
                tmpmov,
                {"codec:v": "libx264"}
            )
        )
        ffmpeg.execute()

        shutil.move(tmpmov, outputfile)


if __name__ == '__main__':
    main()
