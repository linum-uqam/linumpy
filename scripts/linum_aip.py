#!/usr/bin/env python3

"""Compute the average intensity projection of a 3D zarr volume."""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse
from pathlib import Path

import dask.array as da
import numpy as np
import zarr

from linumpy.io.zarr import create_tempstore, read_omezarr, save_omezarr


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_zarr", type=Path, help="Full path to the zarr volume.")
    p.add_argument("output_image", type=Path, default=None, help="Full path to the output zarr image")

    return p


def main() -> None:
    """Run the average intensity projection script."""
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
    zarr_store = create_tempstore(suffix=".zarr")
    _aip = zarr.open(zarr_store, mode="w", shape=shape, dtype=np.float32, chunks=vol.chunks[1:3])
    assert isinstance(_aip, zarr.Array)
    aip = _aip

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
            tile = np.asarray(vol[:, rmin:rmax, cmin:cmax]).mean(axis=0)
            aip[rmin:rmax, cmin:cmax] = tile

    out_dask = da.from_zarr(aip)
    save_omezarr(out_dask, output_file, resolution[1:], tile_shape[1:])


if __name__ == "__main__":
    main()
