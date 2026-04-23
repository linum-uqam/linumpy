#!/usr/bin/env python3
"""Detect and fix the lateral illumination inhomogeneities for each 3D tile of a mosaic grid.

GPU acceleration is used through BaSiCPy (PyTorch backend) when available (--use_gpu, default on).
When GPU is not available or --no-use_gpu is passed, BaSiCPy runs on CPU and
multiple processes (--n_processes) can be used to parallelize over Z-planes.
"""

import linumpy.config.threads  # noqa: F401

import os
from pathlib import Path

# When using multiprocessing with pqdm, we need to limit threads per worker
# to prevent thread oversubscription.
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import tempfile

import dask.array as da
import imageio as io
import numpy as np
import zarr
from basicpy import BaSiC
from tqdm.auto import tqdm

from linumpy.cli.args import add_processes_arg, parse_processes_arg
from linumpy.io.zarr import create_tempstore, read_omezarr, save_omezarr

# TODO: add option to export the flatfields and darkfields


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_zarr", help="Full path to the input zarr file")
    p.add_argument("output_zarr", help="Full path to the output zarr file")
    p.add_argument("--max_iterations", type=int, default=500, help="Maximum number of iterations for BaSiC. [%(default)s]")
    p.add_argument(
        "--percentile_max",
        type=float,
        help="Values above this percentile will be clipped when\nestimating the flatfield (inside range [0-100]).",
    )
    p.add_argument(
        "--use_gpu",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable GPU acceleration via BaSiCPy (PyTorch backend).\n"
        "When enabled, tiles are processed sequentially (CUDA cannot be\n"
        "forked across processes). When disabled, --n_processes is honoured.\n"
        "[%(default)s]",
    )
    add_processes_arg(p)
    return p


def process_tile(params: dict) -> tuple:
    """Process a tile and add it to the output mosaic."""
    from linumpy.config.threads import apply_threadpool_limits

    apply_threadpool_limits()

    file = params["slice_file"]
    z = params["z"]
    tile_shape = params["tile_shape"]
    max_iterations = params["max_iterations"]
    p_upper = params["p_upper"]
    vol = io.v3.imread(str(file))
    file_output = Path(file).parent / file.name.replace(".tiff", "_corrected.tiff")

    nx = vol.shape[0] // tile_shape[0]
    ny = vol.shape[1] // tile_shape[1]

    tiles = []
    for i in range(nx):
        for j in range(ny):
            rmin = i * tile_shape[0]
            rmax = (i + 1) * tile_shape[0]
            cmin = j * tile_shape[1]
            cmax = (j + 1) * tile_shape[1]
            tiles.append(vol[rmin:rmax, cmin:cmax])

    tiles_for_fit = np.asarray(tiles)
    if np.iscomplexobj(tiles[0]):
        tiles_for_fit = np.abs(tiles_for_fit).astype(np.float64)
    if p_upper is not None:
        tiles_for_fit = np.clip(tiles_for_fit, None, p_upper)
    optimizer = BaSiC(get_darkfield=False, max_iterations=max_iterations)
    optimizer.fit(tiles_for_fit)

    try:
        # TODO: Hasn't been validated for complex input since basicpy has replaced pybasic
        if np.iscomplexobj(tiles[0]):
            tiles_real = [t.real for t in tiles]
            tiles_imag = [t.imag for t in tiles]
            sign_real = [np.sign(t) for t in tiles_real]
            sign_imag = [np.sign(t) for t in tiles_imag]
            tiles_real_corr = optimizer.transform(np.asarray(tiles_real))
            tiles_imag_corr = optimizer.transform(np.asarray(tiles_imag))
            tiles_corrected = [
                (t_real * s_real) + 1j * (t_imag * s_imag)
                for t_real, t_imag, s_real, s_imag in zip(tiles_real_corr, tiles_imag_corr, sign_real, sign_imag, strict=False)
            ]
        else:
            tiles_corrected = optimizer.transform(np.asarray(tiles))
    except RuntimeError:
        print(f"Got runtime error at z={z}")
        tiles_corrected = np.asarray(tiles)

    vol_output = np.zeros_like(vol)
    for i in range(nx):
        for j in range(ny):
            rmin = i * tile_shape[0]
            rmax = (i + 1) * tile_shape[0]
            cmin = j * tile_shape[1]
            cmax = (j + 1) * tile_shape[1]
            t = tiles_corrected[i * ny + j]
            if np.isnan(t).any():
                print(f"NaN values found in tile {i}, {j} at z={z}. Replacing with zeros.")
                t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            vol_output[rmin:rmax, cmin:cmax] = t

    io.imsave(str(file_output), vol_output)
    return z, file_output


def main() -> None:
    """Run function operation."""
    p = _build_arg_parser()
    args = p.parse_args()

    input_zarr = Path(args.input_zarr)
    output_zarr = Path(args.output_zarr)
    n_cpus = parse_processes_arg(args.n_processes)

    if not args.use_gpu:
        os.environ["JAX_PLATFORMS"] = "cpu"

    try:
        import jax  # ty: ignore[unresolved-import]

        devices = jax.devices()
        has_gpu = any("cuda" in str(d).lower() for d in devices)
        if has_gpu:
            print(f"JAX GPU acceleration enabled: {devices}")
        else:
            print(f"JAX running on CPU: {devices}")
    except Exception as e:
        print(f"JAX device check failed: {e}")

    vol, resolution = read_omezarr(input_zarr, level=0)
    p_upper = None
    if args.percentile_max is not None:
        p_upper = np.percentile(vol[:], args.percentile_max)
    n_slices = vol.shape[0]

    tmp_dir = tempfile.TemporaryDirectory(suffix="_linum_fix_illumination_3d_slices", dir=output_zarr.parent)
    params_list = []
    for z in tqdm(range(n_slices), "Preprocessing slices"):
        slice_file = Path(tmp_dir.name) / f"slice_{z:03d}.tiff"
        img = vol[z]
        io.imsave(str(slice_file), img)
        params_list.append(
            {
                "z": z,
                "slice_file": slice_file,
                "tile_shape": vol.chunks[1:],
                "max_iterations": args.max_iterations,
                "p_upper": p_upper,
            }
        )

    if args.use_gpu and n_cpus > 1:
        print(f"Note: GPU mode uses sequential processing (ignoring n_processes={n_cpus})")

    if args.use_gpu or n_cpus <= 1:
        corrected_files = [process_tile(param) for param in tqdm(params_list, desc="Processing tiles")]
    else:
        from pqdm.processes import pqdm

        corrected_files = pqdm(
            params_list, process_tile, n_jobs=n_cpus, desc="Processing tiles", exception_behaviour="immediate"
        )

    temp_store = create_tempstore(suffix=".zarr")
    vol_output = zarr.open(temp_store, mode="w", shape=vol.shape, dtype=vol.dtype, chunks=vol.chunks)

    # TODO: Rebuilding volume step could be faster
    for z, f in tqdm(corrected_files, "Rebuilding volume"):
        slice_vol = io.v3.imread(str(f))
        vol_output[z] = slice_vol[:]

    out_dask = da.from_zarr(vol_output)
    min_value = out_dask.min().compute()
    if min_value < 0:
        print(f"Minimum value in the output volume is {min_value}. Clipping at 0.")
        out_dask = da.clip(out_dask, 0.0, None)

    save_omezarr(out_dask, output_zarr, voxel_size=resolution, chunks=vol.chunks)
    tmp_dir.cleanup()


if __name__ == "__main__":
    main()
