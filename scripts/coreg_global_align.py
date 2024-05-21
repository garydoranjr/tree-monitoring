#!/usr/bin/env python
import os
import json
import click
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from arosics import COREG
from py_tools_ds.geo.coord_trafo import imXY2mapXY
import arosics.geometry as GEO

from coreg_global import (
    load_offsets, offsets_to_matrix,
    load_offset_matrix, iterate,
)


def spoof_shift(self, offset):
    self.x_shift_px, self.y_shift_px = offset
    new_originX, new_originY = imXY2mapXY((self.x_shift_px, self.y_shift_px), self.shift.gt)
    self.x_shift_map, self.y_shift_map = new_originX - self.shift.gt[0], new_originY - self.shift.gt[3]
    self.vec_length_map = float(np.sqrt(self.x_shift_map ** 2 + self.y_shift_map ** 2))
    self.vec_angle_deg = GEO.angle_to_north((self.x_shift_px, self.y_shift_px)).tolist()[0]
    self._get_updated_map_info()
    self.success = True


@click.command()
@click.argument('coregdir')
@click.argument('imagedir')
@click.argument('outputdir')
def main(coregdir, imagedir, outputdir):

    files = sorted(glob(os.path.join(coregdir, '*.json')))

    keys, xy = load_offset_matrix(files)

    offset = np.zeros((len(xy), 2))

    for i in range(50):
        prev = np.array(offset)
        offset = iterate(offset, xy)
        print(np.sqrt(np.nanmean(np.square(prev - offset))))


    for k, dxy in tqdm(list(zip(keys, offset)), 'Shifting'):
        if np.any(np.isnan(dxy)): continue
        inputfile = os.path.join(imagedir, k + '.tif')
        outputfile = os.path.join(outputdir, k + '_aligned.tif')
        if not os.path.exists(inputfile):
            raise ValueError(f'Missing file {inputfile}')
        try:
            coreg = COREG(
                inputfile, inputfile,
                path_out=outputfile,
                fmt_out='GTIFF',
                align_grids=True,
                q=True,
            )
            spoof_shift(coreg, -dxy)
            coreg.correct_shifts()
        except RuntimeError as e:
            print(e)


if __name__ == '__main__':
    main()
