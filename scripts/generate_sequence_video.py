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


def parse_info(filename):
    parts = os.path.splitext(filename)[0].split('_')
    return {
        'genus': parts[0],
        'species': parts[1],
        'crown_id': int(parts[2]),
        'date': datetime(*map(int, parts[-3:])),
    }


def reformat_image(inputfile, config, outputdir):
    video_size = config['video_size']
    date_fmt = config.get('date_format', None)
    species_fmt = config.get('species_format', None)
    padding = config['padding']
    left, right, top, bottom = (
        padding.get(side, 0)
        for side in ('left', 'right', 'top', 'bottom')
    )

    base = os.path.basename(inputfile)
    info = parse_info(base)
    outputfile = safe_join(outputdir, base)

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

    if date_fmt is not None:
        datestr = info['date'].strftime(date_fmt['format'])
        draw = ImageDraw.Draw(out)
        draw.text(
            date_fmt['xy'],
            datestr,
            **date_fmt.get('draw_kwargs', {})
        )

    if species_fmt is not None:
        specstr = species_fmt['format'].format(**info)
        draw = ImageDraw.Draw(out)
        draw.text(
            species_fmt['xy'],
            specstr,
            **species_fmt.get('draw_kwargs', {})
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

    images = glob(safe_join(inputdir, f'*_{sequence_id}_*_*_*.{ext}'))

    with TemporaryDirectory() as tmpdir:
        reformatted = [
            reformat_image(i, config, tmpdir)
            for i in images
        ]

        tmpmov = safe_join(
            tmpdir,
            os.path.basename(outputfile)
        )

        ffmpeg = (
            FFmpeg()
            .input(
                safe_join(tmpdir, f'*.{ext}'),
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
