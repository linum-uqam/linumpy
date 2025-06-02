#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert a nifti volume into a .zarr volume"""

import argparse

import dask.array as da
import nibabel as nib
import numpy as np

from linumpy.io.zarr import save_omezarr


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
                   help="Number of levels in the pyramid. "
                        " (default=%(default)s)")
    p.add_argument("--normalize", action="store_true",
                   help="Normalize the data (default=%(default)s)")
    return p


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Prepare the zarr information
    chunks = tuple([args.chunk_size] * 3)
    img = nib.load(str(args.input))

    # Resolution in mm
    resolution = np.array(img.header['pixdim'][1:4])

    # Load the data
    # Neuroglancer doesn't support float64
    vol = img.get_fdata(dtype=np.float32)

    # Normalize the data
    if args.normalize:
        vol -= vol.min()
        vol /= vol.max()

    # Invert the x and z axis
    vol = np.moveaxis(vol, (0, 1, 2), (2, 1, 0))
    vol = da.from_array(vol, chunks=chunks)
    resolution = resolution[::-1]

    # Save the zarr
    save_omezarr(vol, args.zarr_directory,
              voxel_size=resolution,
              chunks=chunks, n_levels=args.n_levels)


if __name__ == "__main__":
    main()
