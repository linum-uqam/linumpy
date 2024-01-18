#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Stack 2D mosaics into a single volume."""

import argparse
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas
from tqdm.auto import tqdm

from linumpy.utils_images import apply_xy_shift


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_images", nargs="+",
                   help="Full path to a 2D mosaic grid image.")
    p.add_argument("output_volume",
                   help="Assembled volume filename (must be a nii or nii.gz)")
    p.add_argument("--xy_shifts", required=True,
                   help="CSV file containing the xy shifts for each slice. (default=%(default)s)")
    p.add_argument("--resolution_xy", type=float, default=1.0,
                   help="Lateral (xy) resolution in micron. (default=%(default)s)")
    p.add_argument("--resolution_z", type=float, default=1.0,
                   help="Axial (z) resolution in micron, corresponding to the z distance between images in the stack. (default=%(default)s)")
    return p


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Detect the slices ids
    files = [Path(x) for x in args.input_images]
    files.sort()
    pattern = r".*z(\d+)_.*"
    slice_ids = []
    for f in files:
        foo = re.match(pattern, f.name)
        slice_ids.append(int(foo.groups()[0]))
    n_slices = np.max(slice_ids) - np.min(slice_ids) + 1

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
    volume_shape = (ny, nx, n_slices)

    # Create the volume
    volume = np.zeros(volume_shape, dtype=np.float32)

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
        img = apply_xy_shift(img, volume[:, :, 0], dx, dy)

        # Add the slice to the volume
        volume[:, :, z] = img

    # Save this volume
    affine = np.eye(4)
    affine[0, 0] = args.resolution_xy / 1000.0  # x resolution in mm
    affine[1, 1] = args.resolution_xy / 1000.0  # y resolution in mm
    affine[2, 2] = args.resolution_z / 1000.0  # z resolution in mm
    img = nib.Nifti1Image(volume, affine)
    output_file = Path(args.output_volume)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    nib.save(img, output_file)


if __name__ == "__main__":
    main()
