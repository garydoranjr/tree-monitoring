import json
import click

from arosics import COREG
from planet_coreg import stem


@click.command()
@click.argument('dronefile')
@click.argument('planetfile')
@click.argument('outputfile')
def main(dronefile, planetfile, outputfile):

    coreg = COREG(
        dronefile, planetfile, ws=(200, 200),
        align_grids=True, max_shift=10,
        ignore_errors=True, q=True,
    )
    coreg.calculate_spatial_shifts()
    result = coreg.coreg_info

    output = {
        'coreg_info': result,
        'drone_map': stem(dronefile),
        'planet_map': stem(planetfile),
    }

    with open(outputfile, 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == '__main__':
    main()
