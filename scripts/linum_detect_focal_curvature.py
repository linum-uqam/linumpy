#!/usr/bin/env python3

"""Detect and fix the focal curvature in a 3D mosaic grid."""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import zarr
from basicpy import BaSiC

from linumpy.geometry.interface import find_tissue_interface
from linumpy.gpu import GPU_AVAILABLE, to_cpu
from linumpy.io.zarr import create_tempstore, read_omezarr, save_omezarr

# TODO: Replace integer roll by sub-voxel interpolation using a deformation field


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_zarr", type=Path, help="Path to file (.ome.zarr) containing the 3D mosaic grid.")
    p.add_argument("output_zarr", type=Path, help="Corrected 3D mosaic grid file path (.ome.zarr).")
    p.add_argument("--n_levels", type=int, default=5, help="Number of levels in pyramid representation.")
    p.add_argument(
        "--sigma_xy",
        type=float,
        default=3.0,
        help="Gaussian smoothing sigma in X and Y before interface detection [%(default)s]",
    )
    p.add_argument(
        "--sigma_z", type=float, default=2.0, help="Gaussian smoothing sigma in Z before interface detection [%(default)s]"
    )
    p.add_argument("--use_log", action="store_true", help="Apply log transform before gradient detection")
    p.add_argument(
        "--use_gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GPU acceleration for the per-tile focal-plane shift if available [%(default)s].",
    )
    return p


def main() -> None:
    """Run the focal curvature detection script."""
    # Parse the arguments
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Parameters
    input_zarr = args.input_zarr
    output_zarr = args.output_zarr

    # Load ome-zarr data
    vol, res = read_omezarr(input_zarr, level=0)
    dtype = vol.dtype
    data = np.moveaxis(np.asarray(vol), 0, -1)
    # Estimate the water-tissue interface
    z0 = find_tissue_interface(
        np.abs(data),
        s_xy=args.sigma_xy,
        s_z=args.sigma_z,
        use_log=args.use_log,
        use_gpu=args.use_gpu,
    )

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

    use_gpu = args.use_gpu and GPU_AVAILABLE
    if use_gpu:
        import cupy as cp

        xp: Any = cp
    else:
        xp = np

    nz_full = vol.shape[0]
    z_arange = xp.arange(nz_full)[:, None, None]
    corr_xp = xp.asarray(corr)
    # Per-(m, n) circular shift along Z, vectorised via take_along_axis.
    # Equivalent to: tile[:, m, n] = roll(tile[:, m, n], -corr[m, n]).

    temp_store = create_tempstore()
    vol_corr_ = zarr.open(temp_store, mode="w", shape=vol.shape, dtype=dtype, chunks=tile_shape)
    assert isinstance(vol_corr_, zarr.Array)
    vol_corr = vol_corr_

    for i in range(nx):
        for j in range(ny):
            rmin = i * tile_shape[1]
            rmax = (i + 1) * tile_shape[1]
            cmin = j * tile_shape[2]
            cmax = (j + 1) * tile_shape[2]
            tile_np = np.asarray(vol[:, rmin:rmax, cmin:cmax])
            tile_xp = xp.asarray(tile_np) if use_gpu else tile_np
            # corr is a single per-tile correction (shape == tile_shape[1:]) shared
            # across all tile positions, so broadcast it against every tile rather
            # than slicing into volume coordinates.
            z_idx_tile = (z_arange + corr_xp[None, :, :]) % nz_full
            tile_rolled = xp.take_along_axis(tile_xp, z_idx_tile, axis=0)
            vol_corr[:, rmin:rmax, cmin:cmax] = to_cpu(tile_rolled) if use_gpu else tile_rolled

    # save to ome-zarr
    dask_arr = da.from_zarr(vol_corr)
    save_omezarr(dask_arr, output_zarr, voxel_size=res, chunks=tile_shape, n_levels=args.n_levels)


if __name__ == "__main__":
    main()
