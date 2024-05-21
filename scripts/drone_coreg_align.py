import click
import numpy as np

from arosics import COREG


@click.command()
@click.argument('dronefile')
@click.argument('planetfile')
@click.argument('outputfile')
def main(dronefile, planetfile, outputfile):

    coreg = COREG(
        dronefile, planetfile, ws=(200, 200),
        path_out=outputfile, fmt_out='GTIFF',
        align_grids=True, max_shift=10,
        ignore_errors=True, q=True,
    )
    coreg.calculate_spatial_shifts()
    result = coreg.correct_shifts()


if __name__ == '__main__':
    main()
