#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Detect and fix the lateral illumination inhomogeneities for each 3D tiles of a mosaic grid"""

import argparse
import shutil
from pathlib import Path

import zarr
from pybasic.shading_correction import BaSiC
from tqdm.auto import tqdm


# TODO: add option to export the flatfields and darkfields
# TODO: add option to estimate the darkfield
# TODO: add default parameter to control the number of CPU for this task.
# TODO: parallelize the processing of the tiles

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_zarr",
                   help="Full path to the input zarr file")
    p.add_argument("output_zarr",
                   help="Full path to the output zarr file")

    return p


def process_tile(params: dict):
    """Process a tile and add it to the output mosaic"""
    vol = params["vol"]
    vol_output = params["vol_output"]
    z = params["z"]

    # Get the number of tiles
    tile_shape = vol.chunks
    nx = vol.shape[1] // tile_shape[1]
    ny = vol.shape[2] // tile_shape[2]

    # Extract the tiles for this slice
    tiles = []
    for i in range(nx):
        for j in range(ny):
            rmin = i * tile_shape[1]
            rmax = (i + 1) * tile_shape[1]
            cmin = j * tile_shape[2]
            cmax = (j + 1) * tile_shape[2]
            tiles.append(vol[z, rmin:rmax, cmin:cmax])

    # Estimate the illumination bias
    optimizer = BaSiC(tiles, estimate_darkfield=True)
    optimizer.working_size = 64
    optimizer.prepare()
    optimizer.run()

    # Apply the correction to all the tiles
    tiles_corrected = []
    for i in range(len(tiles)):
        tile = tiles[i]
        tiles_corrected.append(optimizer.normalize(tile))

    # Fill the output mosaic
    for i in range(nx):
        for j in range(ny):
            rmin = i * tile_shape[1]
            rmax = (i + 1) * tile_shape[1]
            cmin = j * tile_shape[2]
            cmax = (j + 1) * tile_shape[2]
            vol_output[z, rmin:rmax, cmin:cmax] = tiles_corrected[i * ny + j]


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Parameters
    input_zarr = Path(args.input_zarr)
    output_zarr = Path(args.output_zarr)

    vol = zarr.open(input_zarr, mode="r")

    # Prepare the output zarr
    process_sync_file = str(output_zarr).replace(".zarr", ".sync")
    synchronizer = zarr.ProcessSynchronizer(process_sync_file)
    vol_output = zarr.open(output_zarr, mode="w", shape=vol.shape, dtype=vol.dtype, chunks=vol.chunks,
                           synchronizer=synchronizer)

    # Prepare the data for each process
    for z in tqdm(range(vol.shape[0])):
        params = {
            "z": z,
            "vol": vol,
            "vol_output": vol_output,
        }
        process_tile(params)

    # Remove the process sync file
    shutil.rmtree(process_sync_file)

if __name__ == "__main__":
    main()
