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
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Estimate an additive darkfield (per-pixel offset) in addition to\n"
        "the multiplicative flatfield. We do NOT use BaSiC's built-in\n"
        "`get_darkfield` because on OCT data it diverges (flatfield ends up\n"
        "all NaN). Instead the darkfield is estimated as a per-pixel low\n"
        "percentile of the (non-empty) tile pool, subtracted, then BaSiC is\n"
        "fit on the residual for the flatfield. [%(default)s]",
    )
    p.add_argument(
        "--darkfield_percentile",
        type=float,
        default=5.0,
        help="Per-pixel percentile across the tile pool used to estimate the\n"
        "additive darkfield when --use_darkfield is set. [%(default)s]",
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
        "--smoothness_flatfield",
        type=float,
        default=1.0,
        help="BaSiC regularization weight for the flatfield (higher = smoother,\n"
        "less spatial detail). BaSiCPy default is 1.0. Lower values (e.g. 0.05)\n"
        "allow the flatfield to capture finer spatial structure at the cost of\n"
        "overfitting noise. [%(default)s]",
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

    # Drop out-of-tile zero-padded slots. In our mosaic_grid format a missing
    # tile position is filled with exact zeros; if those get into BaSiC's fit
    # pool they bias the flat/darkfield estimate so the model ends up subtracting
    # the median signal away (negative -> clipped -> entire volume zeroed).
    n_before_filter = fit_pool_arr.shape[0]
    nonzero_frac = np.mean(fit_pool_arr != 0, axis=(1, 2))
    keep = nonzero_frac > 0.5
    fit_pool_arr = fit_pool_arr[keep]
    print(
        f"Filtered fit pool: {n_before_filter} -> {fit_pool_arr.shape[0]} tiles "
        f"(dropped {n_before_filter - fit_pool_arr.shape[0]} all-zero/padding tiles)."
    )
    if fit_pool_arr.shape[0] == 0:
        msg = "No non-empty tiles available for BaSiC fit; mosaic grid is empty."
        raise RuntimeError(msg)

    if args.percentile_max is not None:
        p_upper = np.percentile(fit_pool_arr, args.percentile_max)
        fit_pool_arr = np.clip(fit_pool_arr, None, p_upper)

    # Estimate darkfield first (per-pixel low percentile of the non-empty
    # tile pool), subtract, then fit BaSiC on the residual for the flatfield.
    # BaSiC's own get_darkfield=True diverges to NaN on OCT data, so we do not
    # use it.
    if args.use_darkfield:
        darkfield = np.percentile(fit_pool_arr, args.darkfield_percentile, axis=0).astype(np.float32)
        fit_pool_arr = np.clip(fit_pool_arr - darkfield[None, :, :], 0.0, None)
        print(
            f"Estimated darkfield (p{args.darkfield_percentile}): "
            f"min={darkfield.min():.4g} max={darkfield.max():.4g} mean={darkfield.mean():.4g}"
        )
    else:
        darkfield = None

    print(
        f"Fitting BaSiC (flatfield only; darkfield_pre_subtracted={args.use_darkfield}) on "
        f"{fit_pool_arr.shape[0]} tile samples drawn from {len(fit_z)} / {n_axial} axial planes."
    )
    optimizer = BaSiC(get_darkfield=False, max_iterations=args.max_iterations, smoothness_flatfield=args.smoothness_flatfield)
    optimizer.fit(fit_pool_arr)
    del fit_pool_arr

    flatfield = np.asarray(optimizer.flatfield)
    if np.isnan(flatfield).any() or flatfield.max() <= 0:
        msg = f"BaSiC flatfield fit failed (min={flatfield.min()}, max={flatfield.max()}). Refusing to proceed."
        raise RuntimeError(msg)
    ff_stats = f"flatfield: min={flatfield.min():.4g} max={flatfield.max():.4g} mean={flatfield.mean():.4g}"
    df_stats = (
        f"darkfield: min={darkfield.min():.4g} max={darkfield.max():.4g} mean={darkfield.mean():.4g}"
        if darkfield is not None
        else "darkfield: disabled"
    )
    print(f"Fit done. {ff_stats}  {df_stats}")

    # Apply the model to every axial plane
    # rather than optimizer.transform(): BaSiC's transform also re-fits a
    # per-image baseline which is inappropriate here (tiles are spatial
    # patches of one image, not a time series), and we want to use our own
    # darkfield estimate rather than BaSiC's (which diverges on OCT data).
    temp_store = create_tempstore(suffix=".zarr")
    vol_output = zarr.open(temp_store, mode="w", shape=vol.shape, dtype=vol.dtype, chunks=vol.chunks)
    ff = flatfield[None, :, :].astype(np.float32)
    df = darkfield[None, :, :].astype(np.float32) if darkfield is not None else None

    def _apply(tiles_abs: np.ndarray) -> np.ndarray:
        x = tiles_abs.astype(np.float32, copy=False)
        if df is not None:
            x = x - df
        return x / ff

    for z in tqdm(range(n_axial), desc="Applying flatfield"):
        plane = np.asarray(vol[z])
        if is_complex:
            real_tiles = _split_into_tiles(plane.real.astype(np.float64), tile_shape)
            imag_tiles = _split_into_tiles(plane.imag.astype(np.float64), tile_shape)
            sign_real = np.sign(real_tiles)
            sign_imag = np.sign(imag_tiles)
            real_corr = _apply(np.abs(real_tiles)) * sign_real
            imag_corr = _apply(np.abs(imag_tiles)) * sign_imag
            tiles_corrected = real_corr + 1j * imag_corr
        else:
            tiles = _split_into_tiles(plane, tile_shape)
            empty_mask = np.all(tiles == 0, axis=(1, 2))
            tiles_corrected = _apply(tiles)
            if empty_mask.any():
                tiles_corrected[empty_mask] = 0.0

        if np.isnan(tiles_corrected).any():
            tiles_corrected = np.nan_to_num(tiles_corrected, nan=0.0, posinf=0.0, neginf=0.0)

        vol_output[z] = _assemble_from_tiles(tiles_corrected, plane_shape, tile_shape).astype(vol.dtype)

    out_dask = da.from_zarr(vol_output)
    # Sanity-check the corrected volume before saving. A collapsed (~all-zero)
    # output usually means BaSiC over-fit the darkfield because the fit pool
    # included padding tiles, or the percentile clip nuked the dynamic range.
    out_min = float(out_dask.min().compute())
    out_max = float(out_dask.max().compute())
    nonzero_frac = float((out_dask != 0).mean().compute())
    print(f"Corrected volume stats: min={out_min:.4g} max={out_max:.4g} nonzero_frac={nonzero_frac:.4f}")
    if nonzero_frac < 0.01 or out_max <= 0:
        msg = (
            f"Illumination correction collapsed the volume "
            f"(nonzero_frac={nonzero_frac:.4f}, max={out_max:.4g}). "
            f"Likely cause: BaSiC over-fit (often with --use_darkfield and "
            f"too few non-empty tiles). Refusing to write all-zero output."
        )
        raise RuntimeError(msg)
    if out_min < 0:
        print(f"Minimum value in the output volume is {out_min}. Clipping at 0.")
        out_dask = da.clip(out_dask, 0.0, None)

    save_omezarr(out_dask, output_zarr, voxel_size=resolution, chunks=vol.chunks, n_levels=args.n_levels)


if __name__ == "__main__":
    main()
