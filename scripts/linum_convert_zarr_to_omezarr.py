#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert a zarr file to an ome-zarr file"""

import argparse

import zarr

from linumpy.io.zarr import save_zarr
from pathlib import Path


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input",
                   help="Full path to a zarr file (.zarr)")
    p.add_argument("output",
                   help="Full path to the output ome-zarr file (.ome-zarr)")
    p.add_argument("-r", "--resolution", nargs="+", type=float, default=1.0,
                   help="Resolution of the image in microns. (default=%(default)s)")

    return p


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Parameters
    input_file = Path(args.input)
    output_file = Path(args.output)
    resolution = args.resolution  # in microns

    assert len(resolution) in [1, 3], "Resolution must be a single value or a tuple of 3 values"

    # Convert the resolution to mm
    scales = []
    if len(resolution) == 1:
        scales = [resolution[0] * 1e-3] * 3
    else:
        scales = [r * 1e-3 for r in resolution]

    foo = zarr.open(input_file, mode="r")
    save_zarr(foo, output_file, scales=scales, overwrite=True)


if __name__ == "__main__":
    main()
