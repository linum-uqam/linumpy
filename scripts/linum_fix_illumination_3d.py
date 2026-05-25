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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import dask.array as da
import numpy as np
import zarr
from basicpy import BaSiC
from scipy.ndimage import gaussian_filter
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
        "--tile_fov_mm",
        type=float,
        default=0.0,
        help="Acquisition tile field-of-view in millimetres. When > 0, the tile\n"
        "size for BaSiC is computed as round(tile_fov_mm / pixel_size_mm)\n"
        "instead of using the zarr chunk size. Use the same value as\n"
        "params.tile_fov_mm in the Nextflow config. [%(default)s]",
    )
    p.add_argument(
        "--per_z_fit",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Fit a separate BaSiC flatfield model for each axial (Z/depth) plane\n"
        "using only tiles from that plane. This captures depth-dependent\n"
        "illumination variation caused by focal curvature. When disabled (default)\n"
        "a single model is fit across all Z planes (faster, less tile jitter). [%(default)s]",
    )
    p.add_argument(
        "--darkfield_smooth_sigma",
        type=float,
        default=0.0,
        help="Gaussian sigma for spatially smoothing the estimated darkfield image.\n"
        "Reduces pixel-level noise in the per-pixel percentile estimate. 0 disables. [%(default)s]",
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


def _assemble_from_tiles(
    tiles: np.ndarray,
    plane_shape: tuple[int, int],
    tile_shape: tuple[int, int],
    background: np.ndarray | None = None,
) -> np.ndarray:
    """Inverse of `_split_into_tiles`: stitch tiles back into a (Y, X) plane.

    Pixels outside the last complete tile row/column (remainder when
    plane_shape is not divisible by tile_shape) are filled with *background*
    when provided, otherwise with zeros.
    """
    ty, tx = tile_shape
    ny = plane_shape[0] // ty
    nx = plane_shape[1] // tx
    out = np.zeros(plane_shape, dtype=tiles.dtype) if background is None else np.array(background, dtype=tiles.dtype)
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
    is_complex = np.iscomplexobj(vol[0])

    if args.tile_fov_mm > 0:
        pixel_size_mm = float(resolution[1])
        tile_px = round(args.tile_fov_mm / pixel_size_mm)
        tile_shape: tuple[int, int] = (tile_px, tile_px)
        print(f"tile_fov_mm={args.tile_fov_mm}: tile_size_px={tile_px} (pixel_size={pixel_size_mm}mm/px)")
    else:
        tile_shape = tuple(vol.chunks[1:])

    ny = plane_shape[0] // tile_shape[0]
    nx = plane_shape[1] // tile_shape[1]
    tiles_per_plane = ny * nx
    if tiles_per_plane == 0:
        msg = f"Tile shape {tile_shape} does not fit in plane shape {plane_shape}."
        raise ValueError(msg)

    def _filter_tiles(tiles: np.ndarray) -> np.ndarray:
        """Remove all-zero padding tiles from a (N, ty, tx) stack."""
        keep = np.mean(tiles != 0, axis=(1, 2)) > 0.5
        return tiles[keep]

    def _prep_fit_pool(tiles_arr: np.ndarray, darkfield_ref: np.ndarray | None) -> np.ndarray:
        """Clip at percentile_max and subtract darkfield_ref if provided."""
        if args.percentile_max is not None:
            p_upper = np.percentile(tiles_arr, args.percentile_max)
            tiles_arr = np.clip(tiles_arr, None, p_upper)
        if darkfield_ref is not None:
            tiles_arr = np.clip(tiles_arr - darkfield_ref[None, :, :], 0.0, None)
        return tiles_arr

    def _fit_basic(tiles_arr: np.ndarray) -> np.ndarray | None:
        """Fit BaSiC on tiles_arr; return flatfield or None on failure."""
        opt = BaSiC(get_darkfield=False, max_iterations=args.max_iterations, smoothness_flatfield=args.smoothness_flatfield)
        opt.fit(tiles_arr)
        ff = np.asarray(opt.flatfield)
        if np.isnan(ff).any() or ff.max() <= 0:
            return None
        return ff

    def _apply_flatfield(tiles: np.ndarray, ff: np.ndarray, df: np.ndarray | None) -> np.ndarray:
        x = tiles.astype(np.float32, copy=False)
        if df is not None:
            x = x - df[None, :, :]
        return x / ff[None, :, :]

    temp_store = create_tempstore(suffix=".zarr")
    vol_output = zarr.open(temp_store, mode="w", shape=vol.shape, dtype=vol.dtype, chunks=vol.chunks)

    if not args.per_z_fit:
        # ── Global fit: pool tiles from all axial planes, fit once ──────────
        # This avoids per-plane jitter that causes tile-period banding after
        # attenuation correction, at the cost of ignoring depth-dependent
        # illumination variation due to focal curvature.
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

        n_before_filter = fit_pool_arr.shape[0]
        fit_pool_arr = _filter_tiles(fit_pool_arr)
        print(
            f"Filtered fit pool: {n_before_filter} -> {fit_pool_arr.shape[0]} tiles "
            f"(dropped {n_before_filter - fit_pool_arr.shape[0]} all-zero/padding tiles)."
        )
        if fit_pool_arr.shape[0] == 0:
            msg = "No non-empty tiles available for BaSiC fit; mosaic grid is empty."
            raise RuntimeError(msg)

        if args.use_darkfield:
            darkfield = np.percentile(fit_pool_arr, args.darkfield_percentile, axis=0).astype(np.float32)
            if args.darkfield_smooth_sigma > 0:
                darkfield = gaussian_filter(darkfield, sigma=args.darkfield_smooth_sigma).astype(np.float32)
            print(
                f"Estimated darkfield (p{args.darkfield_percentile}, sigma={args.darkfield_smooth_sigma}): "
                f"min={darkfield.min():.4g} max={darkfield.max():.4g} mean={darkfield.mean():.4g}"
            )
        else:
            darkfield = None

        fit_pool_arr = _prep_fit_pool(fit_pool_arr, darkfield)

        print(
            f"Fitting BaSiC (global; darkfield_pre_subtracted={args.use_darkfield}) on "
            f"{fit_pool_arr.shape[0]} tile samples drawn from {len(fit_z)} / {n_axial} axial planes."
        )
        flatfield = _fit_basic(fit_pool_arr)
        del fit_pool_arr
        if flatfield is None:
            msg = "BaSiC flatfield fit failed. Refusing to proceed."
            raise RuntimeError(msg)

        ff_stats = f"flatfield: min={flatfield.min():.4g} max={flatfield.max():.4g} mean={flatfield.mean():.4g}"
        df_stats = (
            f"darkfield: min={darkfield.min():.4g} max={darkfield.max():.4g} mean={darkfield.mean():.4g}"
            if darkfield is not None
            else "darkfield: disabled"
        )
        print(f"Fit done. {ff_stats}  {df_stats}")

        for z in tqdm(range(n_axial), desc="Applying flatfield"):
            plane = np.asarray(vol[z])
            if is_complex:
                real_tiles = _split_into_tiles(plane.real.astype(np.float64), tile_shape)
                imag_tiles = _split_into_tiles(plane.imag.astype(np.float64), tile_shape)
                sign_real = np.sign(real_tiles)
                sign_imag = np.sign(imag_tiles)
                real_corr = _apply_flatfield(np.abs(real_tiles), flatfield, darkfield) * sign_real
                imag_corr = _apply_flatfield(np.abs(imag_tiles), flatfield, darkfield) * sign_imag
                tiles_corrected = real_corr + 1j * imag_corr
            else:
                tiles = _split_into_tiles(plane, tile_shape)
                empty_mask = np.all(tiles == 0, axis=(1, 2))
                tiles_corrected = _apply_flatfield(tiles, flatfield, darkfield)
                if empty_mask.any():
                    tiles_corrected[empty_mask] = 0.0

            if np.isnan(tiles_corrected).any():
                tiles_corrected = np.nan_to_num(tiles_corrected, nan=0.0, posinf=0.0, neginf=0.0)

            vol_output[z] = _assemble_from_tiles(tiles_corrected, plane_shape, tile_shape, background=plane).astype(vol.dtype)

    else:
        # ── Per-Z fit: fit a separate BaSiC model per axial (depth) plane ───
        # Captures depth-dependent illumination variation caused by focal
        # curvature.  Each plane's tile pool is used for both the darkfield
        # estimate and the flatfield fit.  Planes with too few non-empty tiles
        # (e.g. near the bottom of the tissue) fall back to the uncorrected
        # plane.
        print(f"Per-Z BaSiC fit: {n_axial} planes, {tiles_per_plane} tiles/plane ({tiles_per_plane} tile samples per fit).")
        min_tiles_for_fit = max(4, tiles_per_plane // 4)
        n_workers = args.n_processes if args.n_processes and args.n_processes > 0 else 1
        print(f"  Using {n_workers} parallel worker(s).")

        def _process_plane_perz(z: int) -> tuple[int, np.ndarray]:
            plane = np.asarray(vol[z])
            plane_abs = np.abs(plane).astype(np.float64) if is_complex else plane

            tiles = _split_into_tiles(plane_abs, tile_shape)
            tiles_fit = _filter_tiles(tiles)

            if tiles_fit.shape[0] < min_tiles_for_fit:
                return z, plane

            if args.use_darkfield:
                darkfield_z = np.percentile(tiles_fit, args.darkfield_percentile, axis=0).astype(np.float32)
                if args.darkfield_smooth_sigma > 0:
                    darkfield_z = gaussian_filter(darkfield_z, sigma=args.darkfield_smooth_sigma).astype(np.float32)
            else:
                darkfield_z = None

            tiles_fit = _prep_fit_pool(tiles_fit.astype(np.float32), darkfield_z)
            flatfield_z = _fit_basic(tiles_fit)

            if flatfield_z is None:
                return z, plane

            if is_complex:
                real_tiles = _split_into_tiles(plane.real.astype(np.float64), tile_shape)
                imag_tiles = _split_into_tiles(plane.imag.astype(np.float64), tile_shape)
                sign_real = np.sign(real_tiles)
                sign_imag = np.sign(imag_tiles)
                real_corr = _apply_flatfield(np.abs(real_tiles), flatfield_z, darkfield_z) * sign_real
                imag_corr = _apply_flatfield(np.abs(imag_tiles), flatfield_z, darkfield_z) * sign_imag
                tiles_corrected = real_corr + 1j * imag_corr
            else:
                empty_mask = np.all(tiles == 0, axis=(1, 2))
                tiles_corrected = _apply_flatfield(tiles.astype(np.float32), flatfield_z, darkfield_z)
                if empty_mask.any():
                    tiles_corrected[empty_mask] = 0.0

            if np.isnan(tiles_corrected).any():
                tiles_corrected = np.nan_to_num(tiles_corrected, nan=0.0, posinf=0.0, neginf=0.0)

            return z, _assemble_from_tiles(tiles_corrected, plane_shape, tile_shape, background=plane).astype(vol.dtype)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_process_plane_perz, z): z for z in range(n_axial)}
            for fut in tqdm(as_completed(futures), total=n_axial, desc="Per-Z fit+apply"):
                z_idx, corrected_plane = fut.result()
                vol_output[z_idx] = corrected_plane

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
