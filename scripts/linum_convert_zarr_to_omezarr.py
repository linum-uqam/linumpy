#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert a zarr file to an ome-zarr file"""

import argparse

import zarr

from linumpy.io.zarr import save_zarr


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input", nargs="+",
                   help="Full path to a 2D mosaic grid image.")
    p.add_argument("output",
                   help="Flatfield filename (must be a .nii or .nii.gz file).")
    p.add_argument("-r", "--resolution", type=float, default=1.0,
                   help="Resolution of the image in microns. (default=%(default)s)")

    return p


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Parameters
    input_file = args.input
    output_file = args.output
    resolution = args.resolution  # in microns

    foo = zarr.open(input_file, mode="r")
    scales = [resolution * 1e-3] * 3  # convert to mm
    save_zarr(foo, output_file, scales=scales, overwrite=True)


if __name__ == "__main__":
    main()
