#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import argparse
from pathlib import Path
from os.path import split as psplit
from os.path import join as pjoin
import re
import pandas as pd
import numpy as np
from linumpy.io.zarr import read_omezarr, save_omezarr
from linumpy.utils_images import apply_xy_shift
from linumpy.utils.io import add_overwrite_arg, assert_output_exists
import dask.array as da


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_mosaics_dir',
                   help='Directory containing mosaics to bring to common space.')
    p.add_argument('in_shifts',
                   help='Spreadsheet containing xy shifts (.csv).')
    p.add_argument('out_directory',
                   help='Output directory containing the aligned mosaics.')
    add_overwrite_arg(p)
    return p


def compute_common_shape(slice_ids, mosaic_shapes, dx_cumsum, dy_cumsum):
    # Compute the volume shape
    xmin = []
    xmax = []
    ymin = []
    ymax = []

    for i, shape in zip(slice_ids, mosaic_shapes):
        # Get the cumulative shift
        dx = dx_cumsum[i]
        dy = dy_cumsum[i]
        xmin.append(-dx)
        xmax.append(-dx + shape[-2])
        ymin.append(-dy)
        ymax.append(-dy + shape[-1])

    # Get the volume shape
    x0 = min(xmin)
    y0 = min(ymin)
    x1 = max(xmax)
    y1 = max(ymax)
    nx = int((x1 - x0))
    ny = int((y1 - y0))

    return nx, ny, x0, y0


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_output_exists(args.out_directory, parser, args)

    # create output directory
    Path(args.out_directory).mkdir(parents=True)

    # get all .ome.zarr files in in_mosaics_dir
    in_mosaics_dir = Path(args.in_mosaics_dir)
    mosaics_files = [p for p in in_mosaics_dir.glob('*.ome.zarr')]
    mosaics_files.sort()
    pattern = r".*z(\d+)_.*"
    slice_ids = []
    for f in mosaics_files:
        foo = re.match(pattern, f.name)
        id = int(foo.groups()[0])
        slice_ids.append(id)

    # Load cvs containing the shift values for each slice
    df = pd.read_csv(args.in_xy_shifts)

    # We load the shifts in mm, but we need to convert them to pixels
    dx_list = np.array(df["x_shift_mm"].tolist())
    dy_list = np.array(df["y_shift_mm"].tolist())

    # assuming the resolution is the same for all slices
    img, res = read_omezarr(mosaics_files[slice_ids[0]])
    dx_list /= res[-2]
    dy_list /= res[-1]

    dx_cumsum = np.append([0], np.cumsum(dx_list))
    dy_cumsum = np.append([0], np.cumsum(dy_list))

    mosaic_shapes = []
    for id in slice_ids:
        img, _ = read_omezarr(mosaics_files[id])
        mosaic_shapes.append(img.shape)

    nx, ny, x0, y0 = compute_common_shape(slice_ids, mosaic_shapes, dx_cumsum, dy_cumsum)

    for id, mosaic_file in zip(slice_ids, mosaics_files):
        img, res = read_omezarr(mosaic_file)
        reference = np.zeros((img.shape[0], nx, ny), dtype=img.dtype)

        dx = dx_cumsum[id] + x0
        dy = dy_cumsum[id] + y0

        aligned = apply_xy_shift(img[:], reference, dy, dx)

        _, file = psplit(mosaic_file)
        outfile = pjoin(args.out_directory, file)
        save_omezarr(da.from_array(aligned), outfile, res,
                     chunks=img.chunks)


if __name__ == '__main__':
    main()
