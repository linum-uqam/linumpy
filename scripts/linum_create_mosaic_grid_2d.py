#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert 3D OCT tiles to a 2D mosaic grid

Notes
-----
- jpg output should only be used for visualization purposes due to loss of data from the 8bit conversion.
"""

import argparse
import json
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
                   help="Full path to the output file (jpg, tiff, or zarr)")
    p.add_argument("-r", "--resolution", type=float, default=-1,
                   help="Output isotropic resolution in micron per pixel. (Use -1 to keep the original resolution). (default=%(default)s)")
    p.add_argument("-z", "--slice", type=int, default=0,
                   help="Slice to process (default=%(default)s)")
    p.add_argument("--n_cpus", type=int, default=-1,
                   help="Number of CPUs to use for parallel processing (default=%(default)s). If -1, all CPUs - 1 are used.")
    p.add_argument("--normalize", action="store_true",
                   help="Normalize the mosaic (default=%(default)s)")
    p.add_argument("--saturation", type=float, default=99.9,
                   help="Saturation value for the normalization (default=%(default)s)")
    p.add_argument("-c", "--config", type=str, default=None,
                   help="JSON mosaic configuration file (default=%(default)s)")

    return p


def get_volume(filename: str, config: dict = None) -> np.ndarray:
    """Load and preprocess an OCT volume

    Parameters
    ----------
    filename : str
        Path to the OCT file
    config : dict
        Loading and preprocessing configuration. The expected keys are :
            crop : bool
            fix_shift : bool
            shift : int (if fix_shift is true)
            n_rots : int
            flip_alines : bool
            flip_bscans : bool
    """

    # Get the loading options
    if config is None:
        config = {}
    crop = config.get("crop", True)
    fix_shift = config.get("fix_shift", True)
    if fix_shift:
        fix_shift = config.get("shift",
                               True)  # Either a precomputed shift, or a True value to compute it during loading.

    # Load the volume
    vol = OCT(filename).load_image(crop=crop, fix_shift=fix_shift)

    # Rotation and flips
    n_rots = config.get("n_rots", 0)
    if n_rots != 0:
        vol = np.rot90(vol, k=n_rots, axes=(1, 2))

    if config.get("flip_alines", False):
        vol = np.flip(vol, axis=1)

    if config.get("flip_bscans", False):
        vol = np.flip(vol, axis=2)

    # Compute AIP
    img = np.mean(vol, axis=0)

    return img


def process_tile(params: dict):
    """Process a tile and add it to the mosaic"""
    f = params["file"]
    rmin, rmax, cmin, cmax = params["tile_pos_px"]
    tile_size = params["tile_size"]
    mosaic = params["mosaic"]
    config = params["config"]

    # Load the tile
    img = get_volume(f, config)

    # Rescale the volume (temporary fix)
    if img.shape != tile_size:
        img = resize(img, tile_size, anti_aliasing=True, order=1, preserve_range=True)

    # Add to the mosaic
    mosaic[rmin:rmax, cmin:cmax] = img


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Load the JSON config file
    if args.config is not None:
        mosaic_config = json.load(open(args.config))
    else:
        mosaic_config = {}

    # Parameters
    tiles_directory = Path(args.tiles_directory)
    output_file = Path(args.output_file)
    assert output_file.suffix in [".jpg", ".tiff", ".zarr"], "The output file must be .jpg, .tiff, or .zarr file."
    if output_file.suffix == ".zarr":
        zarr_file = output_file
    else:
        zarr_file = output_file.with_suffix(".zarr")
    z = args.slice
    output_resolution = args.resolution
    n_cpus = args.n_cpus
    if n_cpus == -1:
        n_cpus = multiprocessing.cpu_count() - 2

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
    f = tiles[0]
    oct = OCT(f)
    vol = get_volume(f, config=mosaic_config)
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
    mosaic = zarr.open(zarr_file, mode="w", shape=mosaic_shape,
                       dtype=np.float32, chunks=tile_size)

    # Create a params dictionary for every tile
    params = []
    for i in range(len(tiles)):
        params.append({
            "file": tiles[i],
            "tile_pos_px": tile_pos_px[i],
            "tile_size": tile_size,
            "mosaic": mosaic,
            "config": mosaic_config,
        })

    # Process the tiles in parallel
    pqdm(params, process_tile, n_jobs=n_cpus, desc="Processing tiles")

    # Normalize the mosaic
    if args.normalize:
        imin = np.min(mosaic)
        imax = np.percentile(mosaic, args.saturation)
        mosaic = (mosaic - imin) / (imax - imin)
        mosaic[mosaic < 0] = 0
        mosaic[mosaic > 1] = 1

    # Convert the mosaic to a tiff file
    if output_file.suffix == ".tiff":
        img = mosaic[:]
        io.imsave(output_file, img)
        shutil.rmtree(zarr_file)

    if output_file.suffix == ".jpg":
        imin = np.min(mosaic)
        imax = np.percentile(mosaic, args.saturation)
        mosaic = (mosaic - imin) / (imax - imin)
        mosaic[mosaic < 0] = 0
        mosaic[mosaic > 1] = 1
        mosaic = (mosaic * 255).astype(np.uint8)
        img = mosaic[:]
        io.imsave(output_file, img)
        shutil.rmtree(zarr_file)

if __name__ == "__main__":
    main()
