#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""Compute z shift between slices"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from linumpy.stitching import shift_oct
from linumpy.utils import data_io
from linumpy.io.zarr import read_omezarr


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )

    # Mandatory parameters
    p.add_argument("input", nargs="+", help="List of slice filenames to process")
    p.add_argument(
        "output", help="Output file for the computed shift between slices (.json)"
    )

    # Optional parameters
    p.add_argument(
        "-f",
        "--factor",
        default=4,
        type=int,
        help="Downsampling factor to accelerate computation (default=%(default)s)",
    )
    p.add_argument(
        "--dz",
        default=1,
        type=int,
        help="Number of slices used by the registration (default=%(default)s)",
    )
    p.add_argument(
        "--metric",
        choices=["CC", "MI"],
        default="CC",
        help="Similarity metric for the registration (default=%(default)s)",
    )
    p.add_argument(
        "--use_gradient",
        action="store_true",
        help="Use gradient instead of intensity to compute the shift.",
    )
    p.add_argument(
        "--use_log",
        action="store_true",
        help="Use log of intensity. (default=%(default)s)",
    )
    p.add_argument(
        "--align_xy",
        action="store_true",
        help="To align the slices together in the XY direction before z shift computation.",
    )
    p.add_argument(
        "--slice_indexes",
        default=None,
        type=str,
        help="Slice indexes to process (default: all slices)",
    )

    return p


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Loading filenames
    slice_list = args.input
    slice_list.sort()

    # Detect slice index
    if args.slice_indexes is not None:
        slice_z = [int(i) for i in args.slice_indexes.split(",")]
    else:
        # Get the slice indices from the filenames
        slice_z = data_io.getSliceListIndices(slice_list)

    slice_files = {}
    for i, f in zip(slice_z, slice_list):
        slice_files[i] = f

    # Prepare the slices
    nz = len(slice_list)

    bottom_slices = slice_z[0 : nz - 1]
    top_slices = slice_z[1:nz]
    slice_pairs = [
        [slice1, slice2] for slice1, slice2 in zip(bottom_slices, top_slices)
    ]

    dz_list = list()
    for z1, z2 in tqdm(slice_pairs, desc="Slice pairs z shift estimation"):

        # Loading the slices
        vol1, _ = read_omezarr(slice_files[z1], level=0)
        vol2, _ = read_omezarr(slice_files[z2], level=0)

        # Change the order of the axes from (z, x, y) to (x, y, z)
        vol1 = np.transpose(vol1, (1, 2, 0))
        vol2 = np.transpose(vol2, (1, 2, 0))

        if args.use_log:
            mask_1 = vol1 > 0.0
            mask_2 = vol2 > 0.0
            vol1[mask_1] = np.log(vol1[mask_1])
            vol2[mask_2] = np.log(vol2[mask_2])
        options = {
            "dz": args.dz,
            "factor": args.factor,
            "metric": args.metric,
            "useGradient": args.use_gradient,
            "align_xy": args.align_xy,
        }

        delta_z = shift_oct.compute_z_shift(vol1, vol2, **options)
        dz_list.append({"z1": int(z1), "z2": int(z2), "dz": int(delta_z)})

    # Save the output
    output_file = Path(args.output)

    output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dz_list, f)


if __name__ == "__main__":
    main()
