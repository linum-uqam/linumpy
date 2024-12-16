#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Uses the BaSiC algorithm to estimate the illumination inhomogeneities in a mosaic grid"""

import argparse
import random

import SimpleITK as sitk
from pathlib import Path
from linumpy.utils.mosaic_grid import MosaicGrid
from pybasic.shading_correction import BaSiC
import numpy as np


# Global Parameters
log_epsilon = 1e-8

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_images", nargs="+",
                   help="Full path to a 2D mosaic grid image.")
    p.add_argument("output_flatfield",
                   help="Flatfield filename (must be a .nii or .nii.gz file).")
    p.add_argument("--output_darkfield", default=None,
                   help="Optional darkfield filename (if none is given, the darkfield won't be estimated). (must be a .nii or .nii.gz file).")
    p.add_argument("-t", "--tile_shape", nargs="+", type=int, default=512,
                   help="Tile shape in pixel. You can provide both the row and col shape if different. Additional "
                        "shapes will be ignored. (default=%(default)s)")
    p.add_argument("--n_samples", type=int, default=512,
                   help="Maximum number of tiles to use for the optimization. (default=%(default)s)")
    p.add_argument("--use_log", action="store_true",
                   help="Perform optimization and correction in log space.")
    p.add_argument("--working_size", type=int, default=128)

    return p

def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Parameters
    input_images = args.input_images
    if isinstance(input_images, str):
        input_images = [input_images]

    tile_shape = args.tile_shape
    if isinstance(tile_shape, int):
        tile_shape = [tile_shape]*2
    elif len(tile_shape) > 2:
        tile_shape = tile_shape[0:2]

    tiles = []
    # Load all input images and get their tiles
    for input_file in input_images:
        image = sitk.GetArrayFromImage(sitk.ReadImage(str(input_file)))

        # Apply a log transformation
        if args.use_log:
            image = np.log(image + log_epsilon)
            log_imin = image.min()
            log_imax = image.max()
            image = (image - log_imin) / (log_imax - log_imin)

        mosaic = MosaicGrid(image, tile_shape=tile_shape)

        # Convert the image into a stack of ndarrays of shape N_Images x Height x Width
        these_tiles, _ = mosaic.get_tiles()

        # Add to the list of tiles
        tiles.extend([these_tiles[i,...] for i in range(these_tiles.shape[0])])

    n_tiles = len(tiles)
    tiles_ids = list(range(n_tiles))
    if n_tiles > args.n_samples:
        random.shuffle(tiles_ids)
        tiles_ids = tiles_ids[0:args.n_samples]
    tiles_sample = [tiles[i] for i in tiles_ids]

    # Perform the basic optimization
    estimate_darkfield = False
    if args.output_darkfield is not None:
        estimate_darkfield = True
    optimizer = BaSiC(tiles_sample, estimate_darkfield=estimate_darkfield)
    optimizer.working_size = args.working_size
    optimizer.prepare()
    optimizer.run()

    # Save the estimated fields (only if the profiles were estimated)
    flatfield_name = Path(args.output_flatfield).resolve()
    flatfield_name.parent.mkdir(parents=True, exist_ok=True)
    flatfield = optimizer.flatfield_fullsize
    flatfield = flatfield / flatfield.mean() # normalization
    sitk.WriteImage(sitk.GetImageFromArray(flatfield), str(flatfield_name))

    if args.output_darkfield is not None:
        darkfield_name = Path(args.output_darkfield).resolve()
        darkfield_name.parent.mkdir(parents=True, exist_ok=True)
        darkfield = optimizer.darkfield_fullsize
        sitk.WriteImage(sitk.GetImageFromArray(darkfield), str(darkfield_name))

if __name__ == "__main__":
    main()