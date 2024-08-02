#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Stack 2D mosaics into a single volume."""

import argparse
import re
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas
import zarr
from tqdm.auto import tqdm

from linumpy.utils_images import apply_xy_shift


# TODO: add option to give a folder


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_images", nargs="+",
                   help="Full path to a 2D mosaic grid image (nifti files). Expects this format: '.*z(\d+)_.*' to extract the slice number.")
    p.add_argument("output_volume",
                   help="Assembled volume filename (must be a .zarr)")
    p.add_argument("--xy_shifts", required=False, default=None,
                   help="CSV file containing the xy shifts for each slice")
    p.add_argument("--resolution_xy", type=float, default=1.0,
                   help="Lateral (xy) resolution in micron. (default=%(default)s)")
    p.add_argument("--resolution_z", type=float, default=1.0,
                   help="Axial (z) resolution in micron, corresponding to the z distance between images in the stack. (default=%(default)s)")
    return p


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Parameters
    zarr_file = Path(args.output_volume)
    assert zarr_file.suffix == ".zarr", "Output volume must be a zarr file."

    # Detect the slices ids
    files = [Path(x) for x in args.input_images]
    files.sort()
    pattern = r".*z(\d+)_.*"
    slice_ids = []
    for f in files:
        foo = re.match(pattern, f.name)
        slice_ids.append(int(foo.groups()[0]))
    n_slices = np.max(slice_ids) - np.min(slice_ids) + 1

    if args.xy_shifts is None:
        dx_list = np.zeros(len(files))
        dy_list = np.zeros(len(files))
    else:
        # Load cvs containing the shift values for each slice
        df = pandas.read_csv(args.xy_shifts)
        dx_list = np.array(df["x_shift"].tolist())
        dy_list = np.array(df["y_shift"].tolist())

    # Compute the volume shape
    xmin = []
    xmax = []
    ymin = []
    ymax = []

    for i, f in enumerate(files):
        # Get this volume shape
        img = nib.load(f)
        shape = img.shape

        # Get the cumulative shift
        if i == 0:
            xmin.append(0)
            xmax.append(shape[1])
            ymin.append(0)
            ymax.append(shape[0])
        else:
            dx = np.cumsum(dx_list)[i - 1]
            dy = np.cumsum(dy_list)[i - 1]
            xmin.append(-dx)
            xmax.append(-dx + shape[1])
            ymin.append(-dy)
            ymax.append(-dy + shape[0])

    # Get the volume shape
    x0 = min(xmin)
    y0 = min(ymin)
    x1 = max(xmax)
    y1 = max(ymax)
    nx = int((x1 - x0))
    ny = int((y1 - y0))
    volume_shape = (n_slices, ny, nx)

    # Create the zarr persistent array
    process_sync_file = str(zarr_file).replace(".zarr", ".sync")
    synchronizer = zarr.ProcessSynchronizer(process_sync_file)
    mosaic = zarr.open(zarr_file, mode="w", shape=volume_shape, dtype=np.float32, chunks=(1, 256, 256),
                       synchronizer=synchronizer)

    # Loop over the slices
    for i in tqdm(range(len(files)), unit="slice", desc="Stacking slices"):
        # Load the slice
        f = files[i]
        z = slice_ids[i]
        img = nib.load(f).get_fdata()

        # Get the shift values for the slice
        if i == 0:
            dx = x0
            dy = y0
        else:
            dx = np.cumsum(dx_list)[i - 1] + x0
            dy = np.cumsum(dy_list)[i - 1] + y0

        # Apply the shift
        img = apply_xy_shift(img, mosaic[0, :, :], dx, dy)

        # Add the slice to the volume
        mosaic[z, :, :] = img

        del img

    # Removing the synchronizer file
    shutil.rmtree(process_sync_file)

if __name__ == "__main__":
    main()
