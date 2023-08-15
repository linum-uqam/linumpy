#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Stitch a 2D mosaic grid.
"""

import argparse
from pathlib import Path

import SimpleITK as sitk
import numpy as np

from linumpy.utils.mosaic_grid import MosaicGrid


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_image",
                   help="Full path to a 2D mosaic grid image.")
    p.add_argument("input_transform",
                   help="Transform file (.npy format)")
    p.add_argument("output_image",
                   help="Stitched mosaic filename (must be a nii or nii.gz)")
    p.add_argument("--blending_method", type=str, default="none", choices=["none", "average", "diffusion"],
                   help="Blending method. (default=%(default)s)")
    p.add_argument("-t", "--tile_shape", nargs="+", type=int, default=512,
                   help="Tile shape in pixel. You can provide both the row and col shape if different. Additional "
                        "shapes will be ignored. (default=%(default)s)")
    return p

def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Parameters
    input_image = Path(args.input_image)
    input_transform = Path(args.input_transform)
    output_image = Path(args.output_image)
    blending_method = args.blending_method
    tile_shape = args.tile_shape
    if isinstance(tile_shape, int):
        tile_shape = [tile_shape]*2
    elif len(tile_shape) > 2:
        tile_shape = tile_shape[0:2]

    # Check the output filename extensions
    assert "".join(output_image.suffixes) in [".nii", ".nii.gz"], "output_image must be a .nii or .nii.gz file"

    # Load the image
    image = sitk.GetArrayFromImage(sitk.ReadImage(str(input_image)))

    # Load the transform
    transform = np.load(str(input_transform))

    # Create the mosaic grid object and set the transform
    mosaic = MosaicGrid(image, tile_shape=tile_shape)
    mosaic.affine = transform

    # Stitch the mosaic
    img = mosaic.get_stitched_image(blending_method=blending_method)
    output_image.parent.mkdir(exist_ok=True, parents=True)

    # Save the grid and mask
    sitk.WriteImage(sitk.GetImageFromArray(img), str(output_image))

if __name__ == "__main__":
    main()
