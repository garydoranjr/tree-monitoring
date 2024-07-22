#!/usr/bin/env python
import os
import json
import yaml
import click
from glob import glob
from collections import defaultdict
from werkzeug.security import safe_join


from generate_sequence_video import parse_info


def get_frame_date_dict(inputdir, pattern):
    images = sorted(glob(safe_join(inputdir, pattern)))
    dates = defaultdict(list)
    for image in images:
        info = parse_info(os.path.basename(image))
        datestr = info['date'].strftime('%Y-%m-%d')
        dates[info['crown_id']].append(datestr)

    return dict(dates.items())


@click.command()
@click.argument('inputdirs', nargs=-1)
@click.argument('configfile')
@click.argument('outputfile')
def main(inputdirs, configfile, outputfile):

    with open(configfile, 'r') as f:
        config = yaml.safe_load(f)

    ext = config['input_ext']
    pattern = f'*_*_*_*_*.{ext}'

    dates = {}
    for inputdir in inputdirs:
        dates |= get_frame_date_dict(inputdir, pattern)

    with open(outputfile, 'w') as f:
        json.dump(dates, f, indent=2)


if __name__ == '__main__':
    main()
