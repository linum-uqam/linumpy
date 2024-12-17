#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Uses the BaSiC algorithm to estimate and compensate illumination inhomogeneities in a 2D mosaic grid"""

import argparse
from pathlib import Path

import SimpleITK as sitk
import numpy as np

from linumpy.utils.mosaic_grid import MosaicGrid

# TODO: Adapt the script to use multiple mosaic grids
# TODO: Optimize performance for large tile numbers

# Global Parameters
log_epsilon = 1e-8


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_image",
                   help="Full path to a 2D mosaic grid image.")
    p.add_argument("output_image", nargs='?', default=None,
                   help="Full path to a 2D mosaic grid image with the fixed illumination. If not provided, a new file with the same name as the input + `_compensated` suffix will be created.")
    p.add_argument("--flatfield", required=True, help="Full path to precomputed flatfield")
    p.add_argument("--darkfield", required=True, help="Full path to precomputed darkfield ")
    p.add_argument("-t", "--tile_shape", nargs="+", type=int, default=400,
                   help="Tile shape in pixel. You can provide both the row and col shape if different. Additional "
                        "shapes will be ignored. (default=%(default)s)")
    return p


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Parameters
    input_file = Path(args.input_image)
    if args.output_image is not None :
        output_file = Path(args.output_image)
    else :
        output_file = input_file.parent / Path(input_file.stem + "_compensated" + input_file.suffix)
    flatfield_file = Path(args.flatfield)
    darkfield_file = Path(args.darkfield)

    # Get the tile shape
    tile_shape = args.tile_shape
    if isinstance(tile_shape, int):
        tile_shape = [tile_shape] * 2
    elif len(tile_shape) == 1:
        tile_shape = [tile_shape[0]] * 2
    elif len(tile_shape) > 2:
        tile_shape = tile_shape[0:2]

    # Load the image and convert to a mosaic grid
    image = sitk.GetArrayFromImage(sitk.ReadImage(str(input_file)))
    mosaic = MosaicGrid(image, tile_shape=tile_shape)
    tiles, tile_pos = mosaic.get_tiles()

    # Load the flat and dark fields
    flatfield = sitk.GetArrayFromImage(sitk.ReadImage(flatfield_file))
    darkfield = sitk.GetArrayFromImage(sitk.ReadImage(darkfield_file))

    # Prepare the BaSiC object
    # optimizer = BaSiC(tiles)
    # optimizer.set_flatfield(flatfield)
    # optimizer.set_darkfield(darkfield)

    # Apply shading correction.
    # epsilon = 1e-6
    epsilon = 0.0
    clip = True
    for tile, pos in zip(tiles, tile_pos):
        if np.all(tile == 0):  # Ignoring empty tiles
            continue
        fixed_tile = (tile.astype(np.float64) - darkfield) / (flatfield + epsilon)
        # if clip and not(tile.dtype in [np.float32, np.float64]):
        #    fixed_tile[fixed_tile < np.iinfo(tile.dtype).min] = np.iinfo(tile.dtype).min
        #    fixed_tile[fixed_tile > np.iinfo(tile.dtype).max] = np.iinfo(tile.dtype).max

        mosaic.set_tile(x=pos[0], y=pos[1], tile=fixed_tile)

    # Preserve initial range
    fixed_image = mosaic.get_image()
    # fixed_image = fixed_image / fixed_image.mean() * image.mean()

    # Save the output
    output_file.parent.mkdir(exist_ok=True, parents=True)
    sitk.WriteImage(sitk.GetImageFromArray(fixed_image), str(output_file))

if __name__ == "__main__":
    main()
