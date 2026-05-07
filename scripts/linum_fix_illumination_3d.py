#!/usr/bin/env python3
"""Detect and fix the lateral illumination inhomogeneities of a 3D mosaic grid.

A single BaSiC flat-/dark-field model is fit on tiles pooled across all axial
(Z) planes of the input mosaic, then applied uniformly to every plane. This
removes the per-plane flatfield jitter that produced visible tile-period
banding in stitched volumes when the model was fit independently per Z.

GPU acceleration is used through BaSiCPy (PyTorch backend) when available.
"""

import linumpy.config.threads  # noqa: F401

import argparse
from pathlib import Path

import dask.array as da
import numpy as np
import zarr
from basicpy import BaSiC
from tqdm.auto import tqdm

from linumpy.cli.args import add_processes_arg
from linumpy.io.zarr import create_tempstore, read_omezarr, save_omezarr


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_zarr", help="Full path to the input zarr file")
    p.add_argument("output_zarr", help="Full path to the output zarr file")
    p.add_argument(
        "--max_iterations",
        type=int,
        default=500,
        help="Maximum number of iterations for BaSiC. [%(default)s]",
    )
    p.add_argument(
        "--percentile_max",
        type=float,
        help="Values above this percentile will be clipped when\nestimating the flatfield (inside range [0-100]).",
    )
    p.add_argument(
        "--n_levels",
        type=int,
        default=5,
        help="Number of levels in pyramid representation. [%(default)s]",
    )
    p.add_argument(
        "--use_darkfield",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Estimate a per-tile darkfield (additive offset) in addition to\n"
        "the flatfield. Recommended for OCT where small per-tile DC offsets\n"
        "otherwise leak into the flatfield estimate. [%(default)s]",
    )
    p.add_argument(
        "--fit_max_samples",
        type=int,
        default=2000,
        help="Upper bound on the number of tile samples drawn (uniformly\n"
        "across axial planes) to fit BaSiC. Caps memory at\n"
        "fit_max_samples * tile_shape * 4 bytes. [%(default)s]",
    )
    p.add_argument(
        "--use_gpu",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Kept for backward compatibility. BaSiCPy auto-selects the\n"
        "available device; this flag has no effect. [%(default)s]",
    )
    add_processes_arg(p)
    return p


def _split_into_tiles(plane: np.ndarray, tile_shape: tuple[int, int]) -> np.ndarray:
    """Split a (Y, X) plane into a (N, ty, tx) stack of tiles in row-major order."""
    ty, tx = tile_shape
    ny = plane.shape[0] // ty
    nx = plane.shape[1] // tx
    tiles = np.empty((ny * nx, ty, tx), dtype=plane.dtype)
    for i in range(ny):
        for j in range(nx):
            tiles[i * nx + j] = plane[i * ty : (i + 1) * ty, j * tx : (j + 1) * tx]
    return tiles


def _assemble_from_tiles(tiles: np.ndarray, plane_shape: tuple[int, int], tile_shape: tuple[int, int]) -> np.ndarray:
    """Inverse of `_split_into_tiles`: stitch tiles back into a (Y, X) plane."""
    ty, tx = tile_shape
    ny = plane_shape[0] // ty
    nx = plane_shape[1] // tx
    out = np.zeros(plane_shape, dtype=tiles.dtype)
    for i in range(ny):
        for j in range(nx):
            out[i * ty : (i + 1) * ty, j * tx : (j + 1) * tx] = tiles[i * nx + j]
    return out


def main() -> None:
    """Run function operation."""
    from linumpy.config.threads import configure_all_libraries

    # configure_all_libraries() also caps PyTorch intra-/inter-op threads
    # (used by BaSiCPy). apply_threadpool_limits() alone misses torch and
    # lets it default to ~half the host cores, oversubscribing the node.
    configure_all_libraries()

    p = _build_arg_parser()
    args = p.parse_args()

    input_zarr = Path(args.input_zarr)
    output_zarr = Path(args.output_zarr)

    vol, resolution = read_omezarr(input_zarr, level=0)
    n_axial = vol.shape[0]
    plane_shape = vol.shape[1:]
    tile_shape = tuple(vol.chunks[1:])
    is_complex = np.iscomplexobj(vol[0])

    ny = plane_shape[0] // tile_shape[0]
    nx = plane_shape[1] // tile_shape[1]
    tiles_per_plane = ny * nx
    if tiles_per_plane == 0:
        msg = f"Tile shape {tile_shape} does not fit in plane shape {plane_shape}."
        raise ValueError(msg)

    # Choose which axial planes to draw fit samples from. Pool across Z so the
    # flatfield is informed by N_axial * tiles_per_plane samples instead of just
    # tiles_per_plane (which was severely under-constraining the model and
    # producing a different flatfield per axial plane -> tile-period banding
    # after attenuation correction).
    fit_max_samples = max(args.fit_max_samples, tiles_per_plane)
    n_planes_for_fit = min(n_axial, max(1, fit_max_samples // tiles_per_plane))
    fit_z = np.arange(n_axial) if n_planes_for_fit >= n_axial else np.linspace(0, n_axial - 1, n_planes_for_fit, dtype=int)

    fit_pool = []
    for z in tqdm(fit_z, desc="Loading fit samples"):
        plane = np.asarray(vol[int(z)])
        if is_complex:
            plane = np.abs(plane).astype(np.float64)
        fit_pool.append(_split_into_tiles(plane, tile_shape))
    fit_pool_arr = np.concatenate(fit_pool, axis=0)
    del fit_pool

    if args.percentile_max is not None:
        p_upper = np.percentile(fit_pool_arr, args.percentile_max)
        fit_pool_arr = np.clip(fit_pool_arr, None, p_upper)

    print(
        f"Fitting BaSiC (darkfield={args.use_darkfield}) on "
        f"{fit_pool_arr.shape[0]} tile samples drawn from {len(fit_z)} / {n_axial} axial planes."
    )
    optimizer = BaSiC(get_darkfield=args.use_darkfield, max_iterations=args.max_iterations)
    optimizer.fit(fit_pool_arr)
    del fit_pool_arr

    # Apply the single fitted model to every axial plane.
    temp_store = create_tempstore(suffix=".zarr")
    vol_output = zarr.open(temp_store, mode="w", shape=vol.shape, dtype=vol.dtype, chunks=vol.chunks)

    for z in tqdm(range(n_axial), desc="Applying flatfield"):
        plane = np.asarray(vol[z])
        if is_complex:
            real_tiles = _split_into_tiles(plane.real.astype(np.float64), tile_shape)
            imag_tiles = _split_into_tiles(plane.imag.astype(np.float64), tile_shape)
            sign_real = np.sign(real_tiles)
            sign_imag = np.sign(imag_tiles)
            try:
                real_corr = np.asarray(optimizer.transform(np.abs(real_tiles))) * sign_real
                imag_corr = np.asarray(optimizer.transform(np.abs(imag_tiles))) * sign_imag
            except RuntimeError:
                print(f"BaSiC transform failed at z={z}; leaving plane uncorrected.")
                real_corr, imag_corr = real_tiles, imag_tiles
            tiles_corrected = real_corr + 1j * imag_corr
        else:
            tiles = _split_into_tiles(plane, tile_shape)
            try:
                tiles_corrected = np.asarray(optimizer.transform(tiles))
            except RuntimeError:
                print(f"BaSiC transform failed at z={z}; leaving plane uncorrected.")
                tiles_corrected = tiles

        if np.isnan(tiles_corrected).any():
            tiles_corrected = np.nan_to_num(tiles_corrected, nan=0.0, posinf=0.0, neginf=0.0)

        vol_output[z] = _assemble_from_tiles(tiles_corrected, plane_shape, tile_shape).astype(vol.dtype)

    out_dask = da.from_zarr(vol_output)
    min_value = out_dask.min().compute()
    if min_value < 0:
        print(f"Minimum value in the output volume is {min_value}. Clipping at 0.")
        out_dask = da.clip(out_dask, 0.0, None)

    save_omezarr(out_dask, output_zarr, voxel_size=resolution, chunks=vol.chunks, n_levels=args.n_levels)


if __name__ == "__main__":
    main()
