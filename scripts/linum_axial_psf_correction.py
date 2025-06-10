#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Model-free axial PSF correction."""

import argparse

import numpy as np
import dask.array as da
from skimage.filters import threshold_otsu

from linumpy.io.zarr import save_zarr, read_omezarr
import zarr

import matplotlib.pyplot as plt


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_zarr",
                   help="Path to file (.ome.zarr) containing the 3D mosaic grid.")
    p.add_argument('input_mask',
                   help='Path to brain mask (.ome.zarr).')
    p.add_argument("output_zarr",
                   help="Corrected 3D mosaic grid file path (.ome.zarr).")
    p.add_argument('--dont_mask_output', action='store_true',
                   help='Option for disabling masking of the corrected output.')
    p.add_argument('--n_levels', type=int, default=5,
                   help='Number of levels in pyramid representation.')
    return p


def main():
    # Parse the arguments
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load ome-zarr data
    vol, res = read_omezarr(args.input_zarr, level=0)
    mask, res = read_omezarr(args.input_mask, level=0)

    # Extract the tile shape from the filename
    tile_shape = vol.chunks

    # otsu threshold for identifying agarose voxels
    bg = vol[:]
    bg = np.ma.masked_array(bg, mask[:] > 0)
    bg_curve = np.mean(bg, axis=(1, 2))

    temp_store = zarr.TempStore()
    vol_corr = zarr.open(temp_store, mode="w", shape=vol.shape,
                         dtype=np.float32, chunks=tile_shape)

    vol_corr[:] = vol[:]
    vol_corr[:] /= (np.reshape(bg_curve, (-1, 1, 1)))
    vol_corr[:] *= np.mean(bg_curve)
    if not args.dont_mask_output:
        vol_corr[:] = vol_corr[:] * mask[:]

    # save to ome-zarr
    dask_arr = da.from_zarr(vol_corr)
    save_zarr(dask_arr, args.output_zarr, scales=res, chunks=tile_shape,
              n_levels=args.n_levels)


if __name__ == "__main__":
    main()
