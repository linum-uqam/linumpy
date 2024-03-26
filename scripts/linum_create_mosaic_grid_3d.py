#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert 3D OCT tiles to a 3D mosaic grid"""

import argparse
import multiprocessing
from pathlib import Path

import numpy as np
import zarr
from skimage.transform import resize

from linumpy import reconstruction
from linumpy.microscope.oct import OCT


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("tiles_directory",
                   help="Full path to a directory containing the tiles to process")
    p.add_argument("output_directory",
                   help="Full path to a directory where to save the output tiff stack")
    p.add_argument("-r", "--resolution", type=float, default=10.0,
                   help="Output isotropic resolution in micron per pixel. (default=%(default)s)")
    p.add_argument("--basename", default="mosaic_grid_3d_",
                   help="Basename of the output file (default=%(default)s)")
    p.add_argument("-z", "--slice", type=int, default=0,
                   help="Slice to process (default=%(default)s)")
    p.add_argument("--keep_galvo_return", action="store_true",
                   help="Keep the galvo return signal (default=%(default)s)")

    return p


def preprocess_volume(vol: np.ndarray) -> np.ndarray:
    """Preprocess the volume by rotating and flipping it."""
    vol = np.rot90(vol, k=3, axes=(1, 2))
    vol = np.flip(vol, axis=1)
    return vol


def process_tile(params: dict):
    """Process a tile and add it to the mosaic"""
    f = params["file"]
    mx, my, mz = params["tile_pos"]
    crop = params["crop"]
    tile_size = params["tile_size"]
    mosaic = params["mosaic"]

    # Load the tile
    oct = OCT(f)
    vol = oct.load_image(crop=crop)
    vol = preprocess_volume(vol)

    # Rescale the volume
    vol = resize(vol, tile_size, anti_aliasing=True, order=1, preserve_range=True)

    # Compute the tile position
    rmin = mx * vol.shape[1]
    cmin = my * vol.shape[2]
    rmax = rmin + vol.shape[1]
    cmax = cmin + vol.shape[2]
    mosaic[:, rmin:rmax, cmin:cmax] = vol


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Parameters
    tiles_directory = Path(args.tiles_directory)
    output_directory = Path(args.output_directory)
    basename = args.basename
    z = args.slice
    output_resolution = args.resolution
    crop = not args.keep_galvo_return
    n_cpus = multiprocessing.cpu_count() - 1

    # Analyze the tiles
    tiles, tiles_pos = reconstruction.get_tiles_ids(tiles_directory, z=z)
    mx = [tiles_pos[i][0] for i in range(len(tiles_pos))]
    my = [tiles_pos[i][1] for i in range(len(tiles_pos))]
    mx_min = min(mx)
    mx_max = max(mx)
    my_min = min(my)
    my_max = max(my)
    n_mx = mx_max - mx_min + 1
    n_my = my_max - my_min + 1

    # Prepare the mosaic_grid
    oct = OCT(tiles[0])
    vol = oct.load_image(crop=crop)
    vol = preprocess_volume(vol)
    resolution = [oct.resolution[2], oct.resolution[0], oct.resolution[1]]

    # Compute the rescaled tile size based on the minimum target output resolution
    if output_resolution == -1:
        tile_size = vol.shape
    else:
        tile_size = [int(vol.shape[i] * resolution[i] * 1000 / output_resolution) for i in range(3)]
    mosaic_shape = [tile_size[0], n_mx * tile_size[1], n_my * tile_size[2]]

    # Generate a file name that contains info about the resolution and tiles shape
    if output_resolution == -1:
        filename = f"{basename}_z{z:02d}_res_{resolution}_tiles_{tile_size[0]}x{tile_size[1]}x{tile_size[2]}.zarr"
    else:
        filename = f"{basename}_z{z:02d}_res{output_resolution:.1f}um_tiles_{tile_size[0]}x{tile_size[1]}x{tile_size[2]}.zarr"

    # Create the zarr persistent array
    zarr_file = output_directory / filename
    synchronizer = zarr.ProcessSynchronizer('mosaic_grid_3d.sync')
    mosaic = zarr.open(zarr_file, mode="w", shape=mosaic_shape, dtype=np.float32, chunks=tile_size,
                       synchronizer=synchronizer)

    # Create a params dictionary for every tile
    params = []
    for i in range(len(tiles)):
        params.append({
            "file": tiles[i],
            "tile_pos": tiles_pos[i],
            "crop": crop,
            "tile_size": tile_size,
            "mosaic": mosaic
        })

    # Process the tiles in parallel
    with multiprocessing.Pool(n_cpus) as pool:
        pool.map(process_tile, params)


if __name__ == "__main__":
    main()
