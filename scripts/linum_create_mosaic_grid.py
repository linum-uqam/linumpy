#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Detects the tiles in a directory and generates a 2D AIP mosaic grid.

.. note::
    * The input directory must only contain the (reconstructed) volume tiles for a single slice.
    * The script expects the tile filename to contain the x , y and z position (in that order)
    * This script assumes that the tiles are volumes and that the last axis is the z axis for the average intensity projection (AIP)
"""

import argparse
from pathlib import Path

import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

from linumpy.stitching import FileUtils


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_directory",
                   help="Full path to a directory containing the tiles to assemble for a single slice.")
    p.add_argument("output_image",
                   help="Assembled mosaic grid filename (must be a nii or nii.gz)")
    p.add_argument("--output_mask", default=None,
                   help="Optional mosaic data mask filename (must be a nii or nii.gz)")
    p.add_argument("--rot", type=int, default=0,
                   help="Number of 90deg rotations to apply to the tiles. (default=%(default)s).")
    p.add_argument("--xmin", type=int, default=0,
                   help="Minimum x position (row) for the average intensity projection. (default=%(default)s)")
    p.add_argument("--xmax", type=int, default=-1,
                   help="Maximum x position (row) for the average intensity projection (-1 means last slice). (default=%(default)s)")
    p.add_argument("--ymin", type=int, default=0,
                   help="Minimum y position (col) for the average intensity projection. (default=%(default)s)")
    p.add_argument("--ymax", type=int, default=-1,
                   help="Maximum y position (col) for the average intensity projection (-1 means last slice). (default=%(default)s)")
    p.add_argument("--zmin", type=int, default=0,
                   help="Minimum z position for the average intensity projection. (default=%(default)s)")
    p.add_argument("--zmax", type=int, default=-1,
                   help="Maximum z position for the average intensity projection (-1 means last slice). (default=%(default)s)")
    p.add_argument("--flip_rows", action="store_true",
                   help="Flip the rows of each tile after cropping, but before applying the rotation.")
    p.add_argument("--flip_cols", action="store_true",
                   help="Flip the columns of each tile after cropping, but before applying the rotation.")

    return p


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Parameters
    input_directory = Path(args.input_directory)
    output_image = Path(args.output_image)
    output_mask = args.output_mask
    if output_mask is not None:
        output_mask = Path(args.output_mask)

    # Cropping limits
    xmin = args.xmin
    xmax = args.xmax
    ymin = args.ymin
    ymax = args.ymax
    zmin = args.zmin
    zmax = args.zmax

    # Flipping
    flip_rows = args.flip_rows
    flip_cols = args.flip_cols

    # Rotation
    n_rot90 = args.rot

    # Check the output filename extensions
    assert "".join(output_image.suffixes) in [".nii", ".nii.gz"], "output_image must be a .nii or .nii.gz file"
    if output_mask is not None:
        assert "".join(output_mask.suffixes) in [".nii", ".nii.gz"], "output_mask must be a .nii or .nii.gz file"

    # Sniff the directory
    data_info = FileUtils.dataSniffer(input_directory)
    # Create the data object
    data = FileUtils.SlicerData(input_directory,
                                gridshape=data_info["gridshape"],
                                prototype=data_info["prototype"],
                                extension=data_info["extension"])
    data.startIdx = data_info["startIdx"]
    print(data)
    # Load the first volume and get its shape
    data.checkVolShape()

    # Prepare the mosaic & mask (load if existing and updating)
    if xmin == 0 and xmax == -1:
        nrows = data.volshape[0]
        xmax = nrows
    else:
        nrows = xmax - xmin
    if ymin == 0 and ymax == -1:
        ncols = data.volshape[1]
        ymax = ncols
    else:
        ncols = ymax - ymin
    mosaic_nrows = data.gridshape[0]
    mosaic_ncols = data.gridshape[1]
    # If n_rots is odd, switch the nrows and nvols
    if n_rot90 % 2 == 1:
        nrows = data.volshape[1]
        ncols = data.volshape[0]
    mosaic = np.zeros((nrows * mosaic_nrows, ncols * mosaic_ncols))
    if output_mask is not None:
        mosaic_mask = np.zeros_like(mosaic)

    # Loop over the tiles and load them
    for vol, pos in tqdm(data.sliceIterator(z=0, returnPos=True), total=mosaic_nrows * mosaic_ncols):
        # Cropping and compute the average intensity projection (AIP)
        img = vol[xmin:xmax, ymin:ymax, zmin:zmax].mean(axis=2)

        # Flip the axis
        if flip_rows:
            img = np.flipud(img)
        if flip_cols:
            img = np.fliplr(img)

        # Perform in-plane rotations
        img = np.rot90(img, k=n_rot90)

        # Compute the mosaic grid position
        i = pos[0] * nrows
        j = pos[1] * ncols

        # Place in the mosaic grid
        mosaic[i:i + nrows, j:j + ncols] = img

        # Update the data mask
        if output_mask is not None:
            mosaic_mask[i:i + nrows, j:j + ncols] = 1

    # Create the parent directory for the output
    output_image.parent.mkdir(exist_ok=True, parents=True)
    if output_mask is not None:
        output_mask.parent.mkdir(exist_ok=True, parents=True)

    # Save the grid and mask
    sitk.WriteImage(sitk.GetImageFromArray(mosaic), str(output_image))
    if output_mask is not None:
        sitk.WriteImage(sitk.GetImageFromArray(mosaic_mask), str(output_mask))


if __name__ == "__main__":
    main()
