#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Compute the average intensity projection of a 3D zarr volume."""

import argparse
from pathlib import Path

import numpy as np
import zarr


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_zarr",
                   help="Full path to the zarr volume.")
    p.add_argument("output_image", default=None,
                   help="Full path to the output zarr image")

    return p


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Parameters
    input_file = Path(args.input_zarr)
    output_file = Path(args.output_image)

    # Open the zarr volume
    vol = zarr.open(input_file, mode="r")

    # Prepare the output
    shape = vol.shape[1:3]
    aip = zarr.open(output_file, mode="w", shape=shape, dtype=np.float32, chunks=vol.chunks[1:3])

    # Process every tile
    tile_shape = vol.chunks
    nx = vol.shape[1] // tile_shape[1]
    ny = vol.shape[2] // tile_shape[2]
    for i in range(nx):
        for j in range(ny):
            rmin = i * tile_shape[1]
            rmax = (i + 1) * tile_shape[1]
            cmin = j * tile_shape[2]
            cmax = (j + 1) * tile_shape[2]
            tile = vol[:, rmin:rmax, cmin:cmax].mean(axis=0)
            aip[rmin:rmax, cmin:cmax] = tile


if __name__ == "__main__":
    main()
