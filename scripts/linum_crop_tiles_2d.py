#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Crop the tiles given a 2D mosaic grid."""

import argparse
from pathlib import Path
import SimpleITK as sitk

from linumpy.utils.mosaic_grid import MosaicGrid


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_image",
                   help="Full path to a 2D mosaic grid image.")
    p.add_argument("output_image", default=None,
                   help="Full path to the cropped mosaic grid image (must be .tiff or .tif)")
    p.add_argument("--xmin", type=int, default=0,
                   help="Minimum x limit in pixel (default=%(default)s)")
    p.add_argument("--xmax", type=int, default=-1,
                   help="Minimum x limit in pixel (default=%(default)s)")
    p.add_argument("--ymin", type=int, default=0,
                   help="Minimum y limit in pixel (default=%(default)s)")
    p.add_argument("--ymax", type=int, default=-1,
                   help="Minimum y limit in pixel (default=%(default)s)")
    p.add_argument("-t", "--tile_shape", nargs="+", type=int, default=400,
                   help="Tile shape in pixel. You can provide both the row and col shape if different. Additional "
                        "shapes will be ignored. (default=%(default)s)")

    return p


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Parameters
    input_image = Path(args.input_image)
    output_image = Path(args.output_image)
    extension = ""
    if output_image.name.endswith(".tiff"):
        extension = ".tiff"
    elif output_image.name.endswith(".tif"):
        extension = ".tif"
    assert extension in [".tiff", ".tif"], "The output file must be a .tiff or .tif file."
    xlim = (args.xmin, args.xmax)
    ylim = (args.ymin, args.ymax)
    tile_shape = args.tile_shape
    if isinstance(tile_shape, int):
        tile_shape = [tile_shape]*2
    elif len(tile_shape) > 2:
        tile_shape = tile_shape[0:2]

    # Load the image
    image = sitk.GetArrayFromImage(sitk.ReadImage(str(input_image)))

    # Create the mosaic grid object
    mosaic = MosaicGrid(image, tile_shape=tile_shape)

    # Crop the tiles
    mosaic.crop_tiles(xlim, ylim)

    # Save the cropped mosaic grid
    img = mosaic.image
    img = sitk.GetImageFromArray(img)
    output_image.parent.mkdir(exist_ok=True, parents=True)
    sitk.WriteImage(img, str(output_image))

if __name__ == "__main__":
    main()