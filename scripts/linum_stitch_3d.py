#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stitch a 3D mosaic grid.
"""

import argparse
from pathlib import Path

import numpy as np
import zarr
import dask.array as da

from linumpy.io.zarr import read_omezarr, save_zarr
from linumpy.utils.mosaic_grid import addVolumeToMosaic


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_volume",
                   help="Full path to a 3D mosaic grid volume.")
    p.add_argument("input_transform",
                   help="Transform file (.npy format)")
    p.add_argument("output_volume",
                   help="Stitched mosaic filename (zarr)")
    p.add_argument("--blending_method", type=str, default="diffusion", choices=["none", "average", "diffusion"],
                   help="Blending method. (default=%(default)s)")
    p.add_argument("--complex_input", default=False,
                   help="Full path to the output zarr image")
    return p


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Parameters
    input_file = Path(args.input_volume)
    input_transform = Path(args.input_transform)
    output_file = Path(args.output_volume)
    blending_method = args.blending_method

    # Check the output filename extensions
    assert output_file.name.endswith(".zarr"), "output_image must be a .zarr file"

    # Load the image
    volume, resolution = read_omezarr(input_file, level=0)
    tile_shape = volume.chunks

    # Load the transform
    transform = np.load(str(input_transform))

    # Compute the position for every tile
    nx = volume.shape[1] // tile_shape[1]
    ny = volume.shape[2] // tile_shape[2]
    positions = []
    for i in range(nx):
        for j in range(ny):
            pos = np.dot(transform, [i, j]).astype(int)
            positions.append(pos)

    # Get the pos min and max
    posx_min = min([pos[0] for pos in positions])
    posx_max = max([pos[0] + tile_shape[0] for pos in positions])
    posy_min = min([pos[1] for pos in positions])
    posy_max = max([pos[1] + tile_shape[1] for pos in positions])
    mosaic_shape = [volume.shape[0], posx_max - posx_min, posy_max - posy_min]

    # Stitch the mosaic
    temp_store = zarr.TempStore(suffix='.zarr')
    mosaic = zarr.open(
        temp_store,
        mode="w",
        shape=mosaic_shape,
        dtype=np.complex64 if args.complex_input else np.float32,
        chunks=(100, 100, 100),
    )
    for i in range(nx):
        for j in range(ny):
            # Compute the tile position in the input
            rmin = i * tile_shape[1]
            rmax = (i + 1) * tile_shape[1]
            cmin = j * tile_shape[2]
            cmax = (j + 1) * tile_shape[2]
            tile = volume[:, rmin:rmax, cmin:cmax]

            # Get the position within the mosaic
            pos = positions[i * ny + j]
            pos[0] -= posx_min
            pos[1] -= posy_min
            mosaic = addVolumeToMosaic(tile, pos, mosaic, blendingMethod=blending_method)
    
    out_dask = da.from_zarr(mosaic)
    save_zarr(out_dask, output_file, resolution)


if __name__ == "__main__":
    main()
