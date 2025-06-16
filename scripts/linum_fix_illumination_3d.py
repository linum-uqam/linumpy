#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detect and fix the lateral illumination inhomogeneities for each
3D tiles of a mosaic grid.
"""

from os import environ

environ["OMP_NUM_THREADS"] = "1"

import argparse
import tempfile
from pathlib import Path

import dask.array as da

import zarr
from pybasic.shading_correction import BaSiC
from tqdm.auto import tqdm
import imageio as io
import numpy as np
from pqdm.processes import pqdm
from linumpy.io.zarr import save_omezarr, read_omezarr
from linumpy.utils.io import add_processes_arg, parse_processes_arg

# TODO: add option to export the flatfields and darkfields


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_zarr",
                   help="Full path to the input zarr file")
    p.add_argument("output_zarr",
                   help="Full path to the output zarr file")
    add_processes_arg(p)
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
    # Check if tiles contain complex values
    if np.iscomplexobj(tiles[0]):
        # Separate real and imaginary parts
        try:
            tiles_real = [t.real for t in tiles]
            tiles_imag = [t.imag for t in tiles]

            # Store the original signs before applying BaSic as it requires positive values
            sign_real = [np.sign(t) for t in tiles_real]
            sign_imag = [np.sign(t) for t in tiles_imag]

            # Run BaSiC
            optimizer = BaSiC(np.abs(tiles).astype(np.float64), estimate_darkfield=True)
            optimizer.working_size = 64
            optimizer.prepare()
            optimizer.run()
            
            # Apply correction and reconstruct complex result with original signs
            tiles_corrected = [
                (optimizer.normalize(t_real) * s_real)
                + 1j * (optimizer.normalize(t_imag) * s_imag)
                for t_real, t_imag, s_real, s_imag in zip(
                    tiles_real, tiles_imag, sign_real, sign_imag
                )
            ]
        except TypeError as e:
            print(f"Error processing complex tiles: {e}")
            
    else:
        # Process normally if tiles are real
        optimizer = BaSiC(tiles, estimate_darkfield=True)
        optimizer.working_size = 64
        optimizer.prepare()
        optimizer.run()

        # Apply correction
        tiles_corrected = [optimizer.normalize(t) for t in tiles]

    # Fill the output mosaic
    vol_output = np.zeros_like(vol)
    for i in range(nx):
        for j in range(ny):
            rmin = i * tile_shape[0]
            rmax = (i + 1) * tile_shape[0]
            cmin = j * tile_shape[1]
            cmax = (j + 1) * tile_shape[1]
            t = tiles_corrected[i * ny + j]
            if np.isnan(t).any():
                print(f"NaN values found in tile {i}, {j} at z={z}. Replacing with zeros.")
                t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            vol_output[rmin:rmax, cmin:cmax] = t

    io.imsave(str(file_output), vol_output)

    return z, file_output


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Parameters
    input_zarr = Path(args.input_zarr)
    output_zarr = Path(args.output_zarr)
    n_cpus = parse_processes_arg(args.n_processes)

    # Prepare the data for the parallel processing
    vol, resolution = read_omezarr(input_zarr, level=0)
    n_slices = vol.shape[0]
    tmp_dir = tempfile.TemporaryDirectory(
        suffix="_linum_fix_illumination_3d_slices", dir=output_zarr.parent)
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

    if n_cpus > 1:
        # Process the tiles in parallel
        corrected_files = pqdm(params_list, process_tile, n_jobs=n_cpus,
                               desc="Processing tiles", exception_behaviour='immediate')
    else:  # process sequentially
        corrected_files = []
        for param in tqdm(params_list):
            corrected_files.append(process_tile(param))

    # Retrieve the results and fix the volume
    temp_store = zarr.TempStore(suffix=".zarr")
    vol_output = zarr.open(temp_store, mode="w", shape=vol.shape,
                           dtype=vol.dtype, chunks=vol.chunks)

    # TODO: Rebuilding volume step could be faster
    for z, f in tqdm(corrected_files, "Rebuilding volume"):
        slice_vol = io.v3.imread(str(f))
        vol_output[z] = slice_vol[:]

    out_dask = da.from_zarr(vol_output)
    save_omezarr(out_dask, output_zarr, voxel_size=resolution,
              chunks=vol.chunks)

    # Remove the temporary slice files used by the parallel processes
    tmp_dir.cleanup()


if __name__ == "__main__":
    main()
