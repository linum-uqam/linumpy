#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert a nifti volume into a .zarr volume"""

import argparse

import dask.array as da
import nibabel as nib
import numpy as np

from linumpy.conversion import save_zarr


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input",
                   help="Full path to a 3D .nii file")
    p.add_argument("zarr_directory",
                   help="Full path to the .zarr directory")
    p.add_argument("--chunk_size", type=int, default=128,
                   help="Chunk size in pixel (default=%(default)s)")
    p.add_argument("--n_levels", type=int, default=5,
                   help="Number of levels in the pyramid (default=%(default)s)")
    return p


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Prepare the zarr information
    chunks = tuple([args.chunk_size] * 3)
    img = nib.load(str(args.input))
    resolution = np.array(img.header['pixdim'][1:4]) # Resolution in mm

    # Load the data
    vol = img.get_fdata()

    # Invert the x and z axis
    vol = np.moveaxis(vol, (0, 1, 2), (2, 1, 0))
    resolution = resolution[::-1]

    # Save the zarr
    save_zarr(da.from_array(vol, chunks=chunks), args.zarr_directory, scales=resolution, chunks=chunks, n_levels=args.n_levels)


if __name__ == "__main__":
    main()
