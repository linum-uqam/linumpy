#!/usr/bin/env python3

"""Compute the average intensity projection of a 3D zarr volume.

Falls back to CPU if GPU is not available or --no-use_gpu is passed.
"""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import zarr

from linumpy.gpu import GPU_AVAILABLE, is_cupy_array, print_gpu_info, to_cpu
from linumpy.io.zarr import create_tempstore, read_omezarr, save_omezarr


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_zarr", type=Path, help="Full path to the zarr volume.")
    p.add_argument("output_image", type=Path, default=None, help="Full path to the output zarr image")
    p.add_argument(
        "--use_gpu",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use GPU acceleration if available. [%(default)s]",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Print GPU information.")
    return p


def _compute_aip(vol: Any, use_gpu: bool) -> tuple[zarr.Array, tuple]:
    """Compute the per-tile mean and store it into a temporary 2D zarr array."""
    shape = vol.shape[1:3]
    tile_shape = vol.chunks
    zarr_store = create_tempstore(suffix=".zarr")
    _aip = zarr.open(zarr_store, mode="w", shape=shape, dtype=np.float32, chunks=vol.chunks[1:3])
    assert isinstance(_aip, zarr.Array)
    aip = _aip

    nx = vol.shape[1] // tile_shape[1]
    ny = vol.shape[2] // tile_shape[2]
    for i in range(nx):
        for j in range(ny):
            rmin = i * tile_shape[1]
            rmax = (i + 1) * tile_shape[1]
            cmin = j * tile_shape[2]
            cmax = (j + 1) * tile_shape[2]
            tile = vol[:, rmin:rmax, cmin:cmax]
            if use_gpu:
                import cupy as cp

                # Tile is already cupy when the reader is inside gpu_zarr_context;
                # otherwise transfer it once.
                tile_gpu = tile if is_cupy_array(tile) else cp.asarray(np.asarray(tile))
                aip[rmin:rmax, cmin:cmax] = to_cpu(cp.mean(tile_gpu.astype(cp.float32), axis=0))
                del tile_gpu
            else:
                aip[rmin:rmax, cmin:cmax] = np.asarray(tile).mean(axis=0)
    return aip, tile_shape


def main() -> None:
    """Run the average intensity projection script."""
    p = _build_arg_parser()
    args = p.parse_args()

    input_file = Path(args.input_zarr)
    output_file = Path(args.output_image)

    use_gpu = args.use_gpu and GPU_AVAILABLE

    if args.verbose:
        print_gpu_info()
    if args.use_gpu and not GPU_AVAILABLE:
        print("WARNING: GPU requested but not available, falling back to CPU")
    elif use_gpu:
        print("GPU: ENABLED")
    else:
        print("GPU: DISABLED (using CPU)")

    if use_gpu:
        from linumpy.gpu.zarr_io import gpu_zarr_context

        with gpu_zarr_context():
            vol, resolution = read_omezarr(input_file, level=0)
            aip, tile_shape = _compute_aip(vol, use_gpu=True)
    else:
        vol, resolution = read_omezarr(input_file, level=0)
        aip, tile_shape = _compute_aip(vol, use_gpu=False)

    out_dask = da.from_zarr(aip)
    save_omezarr(out_dask, output_file, resolution[1:], tile_shape[1:])


if __name__ == "__main__":
    main()
