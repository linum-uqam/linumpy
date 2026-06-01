#!/usr/bin/env python3

"""Detect and fix the focal curvature in a 3D mosaic grid.

The script estimates a per-tile water-tissue interface, fits a smooth
flat-field with BaSiC to recover the systematic focal-plane curvature, then
applies a sub-voxel circular roll along Z to each A-line. The roll is
vectorised via ``take_along_axis`` with linear interpolation between the
neighbouring depth indices, so a fractional correction map is preserved
instead of being truncated to integer voxels.
"""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
from basicpy import BaSiC

from linumpy.geometry.interface import find_tissue_interface
from linumpy.gpu import GPU_AVAILABLE, print_gpu_info, to_cpu
from linumpy.io.zarr import read_omezarr, save_omezarr


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
    p.add_argument("--verbose", action="store_true", help="Print GPU information.")
    return p


def _apply_subvoxel_roll(tile_xp: Any, corr_xp: Any, z_arange: Any, nz_full: int, xp: Any) -> Any:
    """Circular sub-voxel roll along axis 0 with linear interpolation.

    Equivalent to ``tile[:, m, n] = roll(tile[:, m, n], -corr[m, n])`` for
    integer ``corr``, generalised to fractional shifts by linearly interpolating
    between the two neighbouring integer offsets.
    """
    corr_floor = xp.floor(corr_xp).astype(xp.int64)
    frac = (corr_xp - corr_floor).astype(xp.float32)
    z_idx0 = (z_arange + corr_floor[None, :, :]) % nz_full
    z_idx1 = (z_arange + corr_floor[None, :, :] + 1) % nz_full
    v0 = xp.take_along_axis(tile_xp, z_idx0, axis=0).astype(xp.float32)
    v1 = xp.take_along_axis(tile_xp, z_idx1, axis=0).astype(xp.float32)
    return (1.0 - frac[None, :, :]) * v0 + frac[None, :, :] * v1


def main() -> None:
    """Run the focal curvature detection script."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    use_gpu = args.use_gpu and GPU_AVAILABLE

    if args.verbose:
        print_gpu_info()
        print(f"Using GPU: {use_gpu}")
        if args.use_gpu and not GPU_AVAILABLE:
            print("GPU requested but not available, falling back to CPU")

    # Load ome-zarr data once and reuse the in-memory copy for both the
    # interface detection and the per-tile roll.
    vol, res = read_omezarr(args.input_zarr, level=0)
    dtype = vol.dtype
    tile_shape = vol.chunks
    vol_data = np.asarray(vol)  # (Z, Y, X)
    nz_full, ny_full, nx_full = vol_data.shape

    # find_tissue_interface expects axes (Y, X, Z).
    data_yxz = np.moveaxis(vol_data, 0, -1)
    z0 = find_tissue_interface(
        np.abs(data_yxz),
        s_xy=args.sigma_xy,
        s_z=args.sigma_z,
        use_log=args.use_log,
        use_gpu=args.use_gpu,
    )
    del data_yxz

    # Extract per-tile interface depth maps for BaSiC.
    n_tiles_y = ny_full // tile_shape[1]
    n_tiles_x = nx_full // tile_shape[2]
    tiles = []
    for i in range(n_tiles_y):
        for j in range(n_tiles_x):
            rmin = i * tile_shape[1]
            rmax = (i + 1) * tile_shape[1]
            cmin = j * tile_shape[2]
            cmax = (j + 1) * tile_shape[2]
            tile = z0[rmin:rmax, cmin:cmax].astype(float) / nz_full
            tiles.append(tile)

    optimizer = BaSiC(get_darkfield=False, smoothness_flatfield=1)
    optimizer.fit(np.asarray(tiles))
    flatfield = optimizer.flatfield

    # Fractional per-tile correction; kept as float for sub-voxel rolling.
    corr = (flatfield - 1.0) * float(z0.mean())

    if use_gpu:
        import cupy as cp

        xp: Any = cp
    else:
        xp = np

    z_arange = xp.arange(nz_full, dtype=xp.int64)[:, None, None]
    corr_xp = xp.asarray(corr, dtype=xp.float32)

    # Per-tile loop bounds peak memory; the correction map is shared across
    # tile positions and broadcast against each in-memory tile.
    vol_corr = np.empty_like(vol_data)
    for i in range(n_tiles_y):
        for j in range(n_tiles_x):
            rmin = i * tile_shape[1]
            rmax = (i + 1) * tile_shape[1]
            cmin = j * tile_shape[2]
            cmax = (j + 1) * tile_shape[2]
            tile_np = vol_data[:, rmin:rmax, cmin:cmax]
            tile_xp = xp.asarray(tile_np) if use_gpu else tile_np
            tile_rolled = _apply_subvoxel_roll(tile_xp, corr_xp, z_arange, nz_full, xp)
            tile_rolled = to_cpu(tile_rolled) if use_gpu else tile_rolled
            vol_corr[:, rmin:rmax, cmin:cmax] = tile_rolled.astype(dtype, copy=False)

    # Write directly without an intermediate temp zarr store.
    dask_arr = da.from_array(vol_corr, chunks=tile_shape)
    save_omezarr(dask_arr, args.output_zarr, voxel_size=res, chunks=tile_shape, n_levels=args.n_levels)


if __name__ == "__main__":
    main()
