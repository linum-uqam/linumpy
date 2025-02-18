#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detect and fix the lateral illumination inhomogeneities
for each 3D tiles of a mosaic grid
"""

from os import environ

environ["OMP_NUM_THREADS"] = "1"

import argparse
import multiprocessing
import shutil
import tempfile
from pathlib import Path

import dask.array as da

import zarr
from pybasic.shading_correction import BaSiC
from tqdm.auto import tqdm
import imageio as io
import numpy as np
from pqdm.processes import pqdm
from linumpy.io.zarr import save_zarr, read_omezarr

# TODO: add option to export the flatfields and darkfields


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
    file = params["slice_file"]
    z = params["z"]
    tile_shape = params["tile_shape"]
    vol = io.v3.imread(str(file))
    file_output = Path(file).parent / file.name.replace(".tiff", "_corrected.tiff")

    # Get the number of tiles
    nx = vol.shape[0] // tile_shape[0]
    ny = vol.shape[1] // tile_shape[1]

    # Extract the tiles for this slice
    tiles = []
    for i in range(nx):
        for j in range(ny):
            rmin = i * tile_shape[0]
            rmax = (i + 1) * tile_shape[0]
            cmin = j * tile_shape[1]
            cmax = (j + 1) * tile_shape[1]
            tiles.append(vol[rmin:rmax, cmin:cmax])

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
    vol_output = np.zeros_like(vol)
    for i in range(nx):
        for j in range(ny):
            rmin = i * tile_shape[0]
            rmax = (i + 1) * tile_shape[0]
            cmin = j * tile_shape[1]
            cmax = (j + 1) * tile_shape[1]
            vol_output[rmin:rmax, cmin:cmax] = tiles_corrected[i * ny + j]

    io.imsave(str(file_output), vol_output)

    return z, file_output


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Parameters
    input_zarr = Path(args.input_zarr)
    output_zarr = Path(args.output_zarr)
    n_cpus = multiprocessing.cpu_count() - 1

    # Prepare the data for the parallel processing
    vol, resolution = read_omezarr(input_zarr, level=0)
    n_slices = vol.shape[0]
    tmp_dir = tempfile.TemporaryDirectory(suffix="_linum_fix_illumination_3d_slices", dir=output_zarr.parent)
    params_list = []
    for z in tqdm(range(n_slices), "Preprocessing slices"):
        slice_file = Path(tmp_dir.name) / f"slice_{z:03d}.tiff"
        img = vol[z]
        io.imsave(str(slice_file), img)
        params = {
            "z": z,
            "slice_file": slice_file,
            "tile_shape": vol.chunks[1:],
        }
        params_list.append(params)

    # Process the tiles in parallel
    corrected_files = pqdm(params_list, process_tile, n_jobs=n_cpus, desc="Processing tiles")

    # Retrieve the results and fix the volume
    temp_store = zarr.TempStore(suffix=".zarr")
    process_sync_file = temp_store.path.replace(".zarr", ".sync")
    synchronizer = zarr.ProcessSynchronizer(process_sync_file)
    vol_output = zarr.open(temp_store, mode="w", shape=vol.shape, dtype=vol.dtype,
                           chunks=vol.chunks, synchronizer=synchronizer)

    # TODO: Rebuilding volume step could be faster
    for z, f in tqdm(corrected_files, "Rebuilding volume"):
        slice_vol = io.v3.imread(str(f))
        vol_output[z] = slice_vol[:]

    out_dask = da.from_zarr(vol_output)
    save_zarr(out_dask, output_zarr, voxel_size=resolution,
              chunks=vol.chunks)

    # Remove the process sync file
    shutil.rmtree(process_sync_file)

    # Remove the temporary slice files used by the parallel processes
    tmp_dir.cleanup()


if __name__ == "__main__":
    main()
