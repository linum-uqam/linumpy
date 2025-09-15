#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Detect and fix the focal curvature in a 3D mosaic grid"""

import argparse

import numpy as np
import dask.array as da
from basicpy import BaSiC

from linumpy.io.zarr import save_omezarr, read_omezarr, create_tempstore
from linumpy.preproc.xyzcorr import findTissueInterface
import zarr


# TODO: Replace by interpolation using deformation field
# TODO: optimize for full resolution data
# TODO: parallelize the correction

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_zarr",
                   help="Path to file (.ome.zarr) "
                        "containing the 3D mosaic grid.")
    p.add_argument("output_zarr",
                   help="Corrected 3D mosaic grid file path (.ome.zarr).")
    p.add_argument('--n_levels', type=int, default=5,
                   help='Number of levels in pyramid representation.')
    p.add_argument("--sigma_xy", type=float, default=3.0,
                   help="Gaussian smoothing sigma in X and Y before interface detection [%(default)s]")
    p.add_argument("--sigma_z", type=float, default=2.0,
                   help="Gaussian smoothing sigma in Z before interface detection [%(default)s]")
    p.add_argument("--use_log", action="store_true",
                   help="Apply log transform before gradient detection")
    return p


def main():
    # Parse the arguments
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Parameters
    input_zarr = args.input_zarr
    output_zarr = args.output_zarr

    # Load ome-zarr data
    vol, res = read_omezarr(input_zarr, level=0)
    dtype = vol.dtype
    data = np.moveaxis(vol, 0, -1)
    # Estimate the water-tissue interface
    z0 = findTissueInterface(np.abs(data), s_xy=args.sigma_xy,
                             s_z=args.sigma_z, useLog=args.use_log)

    # Extract the tile shape from the filename
    tile_shape = vol.chunks

    # Extract the tiles from the z0 map
    nx = vol.shape[1] // tile_shape[1]
    ny = vol.shape[2] // tile_shape[2]
    nz = vol.shape[0]
    tiles = []
    for i in range(nx):
        for j in range(ny):
            rmin = i * tile_shape[1]
            rmax = (i + 1) * tile_shape[1]
            cmin = j * tile_shape[2]
            cmax = (j + 1) * tile_shape[2]
            tile = z0[rmin:rmax, cmin:cmax]
            tile = tile.astype(float) / nz  # Normalize the depth
            tiles.append(tile)

    # Perform the basic optimization
    optimizer = BaSiC(get_darkfield=False, smoothness_flatfield=1)
    optimizer.fit(np.asarray(tiles))

    # Save the estimated fields (only if the profiles were estimated)
    flatfield = optimizer.flatfield

    # Apply the correction to a tile
    corr = ((flatfield - 1) * z0.mean()).astype(int)

    temp_store = create_tempstore()
    vol_corr = zarr.open(temp_store, mode="w", shape=vol.shape,
                         dtype=dtype, chunks=tile_shape)

    for i in range(nx):
        for j in range(ny):
            rmin = i * tile_shape[1]
            rmax = (i + 1) * tile_shape[1]
            cmin = j * tile_shape[2]
            cmax = (j + 1) * tile_shape[2]
            tile = vol[:, rmin:rmax, cmin:cmax]

            # Apply the correction (shift the focal plane to flatten it)
            for m in range(tile.shape[1]):
                for n in range(tile.shape[2]):
                    tile[:, m, n] = np.roll(tile[:, m, n], -corr[m, n])

            vol_corr[:, rmin:rmax, cmin:cmax] = tile

    # save to ome-zarr
    dask_arr = da.from_zarr(vol_corr)
    save_omezarr(dask_arr, output_zarr,
                 voxel_size=res,
                 chunks=tile_shape,
                 n_levels=args.n_levels)


if __name__ == "__main__":
    main()
