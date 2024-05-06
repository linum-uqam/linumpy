#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert 3D OCT tiles to a 2D mosaic grid"""

import argparse
import multiprocessing
import shutil
from pathlib import Path

import imageio as io
import numpy as np
import zarr
from pqdm.processes import pqdm
from skimage.transform import resize

from linumpy import reconstruction
from linumpy.microscope.oct import OCT


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("tiles_directory",
                   help="Full path to a directory containing the tiles to process")
    p.add_argument("output_file",
                   help="Full path to the output file (tiff or zarr)")
    p.add_argument("-r", "--resolution", type=float, default=-1,
                   help="Output isotropic resolution in micron per pixel. (Use -1 to keep the original resolution). (default=%(default)s)")
    p.add_argument("-z", "--slice", type=int, default=0,
                   help="Slice to process (default=%(default)s)")
    p.add_argument("--keep_galvo_return", action="store_true",
                   help="Keep the galvo return signal (default=%(default)s)")
    p.add_argument("--n_cpus", type=int, default=-1,
                   help="Number of CPUs to use for parallel processing (default=%(default)s). If -1, all CPUs - 1 are used.")

    return p


def preprocess_volume(vol: np.ndarray) -> np.ndarray:
    """Preprocess the volume by rotating and flipping it and computing the average intensity projection"""
    vol = np.rot90(vol, k=3, axes=(1, 2))
    vol = np.flip(vol, axis=1)
    img = np.mean(vol, axis=0)
    return img


def process_tile(params: dict):
    """Process a tile and add it to the mosaic"""
    f = params["file"]
    rmin, rmax, cmin, cmax = params["tile_pos_px"]
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
    mosaic[rmin:rmax, cmin:cmax] = vol


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Parameters
    tiles_directory = Path(args.tiles_directory)
    output_file = Path(args.output_file)
    assert output_file.suffix in [".tiff", ".zarr"], "The output file must be a .tiff or .zarr file."
    if output_file.suffix == ".zarr":
        zarr_file = output_file
    else:
        zarr_file = output_file.with_suffix(".zarr")
    z = args.slice
    output_resolution = args.resolution
    crop = not args.keep_galvo_return
    n_cpus = args.n_cpus
    if n_cpus == -1:
        n_cpus = multiprocessing.cpu_count() - 1

    # Analyze the tiles
    tiles, tiles_pos = reconstruction.get_tiles_ids(tiles_directory, z=z)
    if len(tiles) == 0:
        print(f"No tiles found in the directory for the slice z={z}.")
        return
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
    resolution = [oct.resolution[0], oct.resolution[1]]

    # Compute the rescaled tile size based on the minimum target output resolution
    if output_resolution == -1:
        tile_size = vol.shape
    else:
        tile_size = [int(vol.shape[i] * resolution[i] * 1000 / output_resolution) for i in range(2)]
    mosaic_shape = [n_mx * tile_size[0], n_my * tile_size[1]]

    # Compute the tile position in pixel within the mosaic
    tile_size = (tile_size[0], tile_size[1])
    tile_pos_px = []
    for i in range(len(tiles_pos)):
        mx, my, mz = tiles_pos[i]
        rmin = (mx - mx_min) * tile_size[0]
        rmax = rmin + tile_size[0]
        cmin = (my - my_min) * tile_size[1]
        cmax = cmin + tile_size[1]
        tile_pos_px.append((rmin, rmax, cmin, cmax))

    # Create the zarr persistent array
    process_sync_file = str(zarr_file).replace(".zarr", ".sync")
    synchronizer = zarr.ProcessSynchronizer(process_sync_file)
    mosaic = zarr.open(zarr_file, mode="w", shape=mosaic_shape, dtype=np.float32, chunks=tile_size,
                       synchronizer=synchronizer)

    # Create a params dictionary for every tile
    params = []
    for i in range(len(tiles)):
        params.append({
            "file": tiles[i],
            "tile_pos_px": tile_pos_px[i],
            "crop": crop,
            "tile_size": tile_size,
            "mosaic": mosaic
        })

    # Process the tiles in parallel
    pqdm(params, process_tile, n_jobs=n_cpus, desc="Processing tiles")

    # Remove the process sync file
    shutil.rmtree(process_sync_file)

    # Convert the mosaic to a tiff file
    if output_file.suffix == ".tiff":
        img = mosaic[:]
        io.imsave(output_file, img)
        shutil.rmtree(zarr_file)


if __name__ == "__main__":
    main()
