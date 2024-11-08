#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Compute the average intensity projection of a 3D zarr volume."""

import argparse
from pathlib import Path
import dask.array as da

import numpy as np
import zarr
from linumpy.io.zarr import save_zarr, read_omezarr

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
    vol, resolution = read_omezarr(input_file, level=0)

    # Prepare the output
    shape = vol.shape[1:3]
    zarr_store = zarr.TempStore(suffix='.zarr')
    aip = zarr.open(zarr_store, mode="w", shape=shape,
                    dtype=np.float32, chunks=vol.chunks[1:3])

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

    out_dask = da.from_zarr(aip)
    save_zarr(out_dask, output_file, resolution[1:], tile_shape[1:])


if __name__ == "__main__":
    main()
