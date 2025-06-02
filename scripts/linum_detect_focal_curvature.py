#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Detect and fix the focal curvature in a 3D mosaic grid"""

import argparse

import numpy as np
import dask.array as da
from pybasic.shading_correction import BaSiC
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from linumpy.io.zarr import save_zarr, read_omezarr
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
    return p


def findTissueInterface(vol, sigma=5, sigma_xy=1, useLog=False):
    """Detects the tissue interface."""
    # Apply a Gaussian filter
    vol_p = gaussian_filter(vol, sigma=(0, sigma_xy, sigma_xy))

    if useLog:
        vol_p[vol_p > 0] = np.log(vol_p[vol_p > 0])

    # Detect the tissue interface
    vol_g = gaussian_filter1d(vol_p, sigma, order=1, axis=0)
    z0 = np.ceil(vol_g.argmax(axis=0) + sigma * 0.5).astype(int)

    return z0


def main():
    # Parse the arguments
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Parameters
    input_zarr = args.input_zarr
    output_zarr = args.output_zarr

    # Load ome-zarr data
    vol, res = read_omezarr(input_zarr, level=0)

    # Estimate the water-tissue interface
    z0 = findTissueInterface(vol, sigma=2, useLog=True)

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
    optimizer = BaSiC(tiles, estimate_darkfield=False)
    optimizer.working_size = 64
    optimizer.prepare()
    optimizer.run()

    # Save the estimated fields (only if the profiles were estimated)
    flatfield = optimizer.flatfield_fullsize

    # Apply the correction to a tile
    corr = ((flatfield - 1) * z0.mean()).astype(int)

    temp_store = zarr.TempStore()
    vol_corr = zarr.open(temp_store, mode="w", shape=vol.shape,
                         dtype=np.float32, chunks=tile_shape)

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
    save_zarr(dask_arr, output_zarr, voxel_size=res, chunks=tile_shape,
              n_levels=args.n_levels)


if __name__ == "__main__":
    main()
