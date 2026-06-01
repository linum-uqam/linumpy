#!/usr/bin/env python3

"""Compute an Average Intensity Projection (AIP) from a 3D mosaic grid and save as PNG.

The AIP is computed by averaging voxel intensities along the Z-axis, producing a 2D
image at full XY resolution (1 data pixel = 1 output pixel). The result is saved as
a 16-bit PNG for QC visualization.

Falls back to CPU if GPU is not available or --no-use_gpu is passed.
"""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from skimage.io import imsave

from linumpy.gpu import GPU_AVAILABLE, print_gpu_info, to_cpu
from linumpy.io.zarr import read_omezarr


def compute_aip(vol: Any, use_gpu: bool = True) -> np.ndarray:
    """Compute the AIP of a mosaic grid volume tile-by-tile.

    Parameters
    ----------
    vol:
        Dask array of shape (Z, Y, X) from read_omezarr.
    use_gpu:
        Whether to use GPU acceleration for the averaging.

    Returns
    -------
    np.ndarray
        2D float32 AIP array of shape (Y, X).
    """
    tile_shape = vol.chunks
    nx = vol.shape[1] // tile_shape[1]
    ny = vol.shape[2] // tile_shape[2]

    aip = np.empty((vol.shape[1], vol.shape[2]), dtype=np.float32)

    for i in range(nx):
        for j in range(ny):
            rmin = i * tile_shape[1]
            rmax = (i + 1) * tile_shape[1]
            cmin = j * tile_shape[2]
            cmax = (j + 1) * tile_shape[2]

            tile = vol[:, rmin:rmax, cmin:cmax]

            if use_gpu:
                import cupy as cp

                # Slices may already be cupy when the read happens inside
                # ``gpu_zarr_context`` (no extra H→D copy). Otherwise we
                # transfer the host tile once.
                tile_gpu = tile if isinstance(tile, cp.ndarray) else cp.asarray(np.asarray(tile))
                aip[rmin:rmax, cmin:cmax] = to_cpu(cp.mean(tile_gpu.astype(cp.float32), axis=0))
                del tile_gpu
            else:
                aip[rmin:rmax, cmin:cmax] = np.asarray(tile).mean(axis=0)

    if use_gpu:
        try:
            import cupy as cp

            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    return aip


def save_aip_png(aip: np.ndarray, output_path: Path) -> None:
    """Normalize and save an AIP array as a 16-bit PNG.

    Intensities are clipped to the 0.1-99.9 percentile range and mapped
    to the full uint16 range. Spatial resolution is preserved: each data
    pixel maps to exactly one output pixel.

    Parameters
    ----------
    aip:
        2D float32 array.
    output_path:
        Destination PNG file path.
    """
    vmin = np.percentile(aip, 0.1)
    vmax = np.percentile(aip, 99.9)
    aip_norm = np.clip((aip - vmin) / (vmax - vmin), 0, 1) if vmax > vmin else np.zeros_like(aip)
    imsave(output_path, (aip_norm * 65535).astype(np.uint16))


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_zarr", help="Full path to the input mosaic grid OME-Zarr volume.")
    p.add_argument("output_png", help="Full path to the output PNG file.")
    p.add_argument(
        "--use_gpu",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use GPU acceleration if available. [%(default)s]",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Print GPU information.")
    return p


def main() -> None:
    """Run function."""
    p = _build_arg_parser()
    args = p.parse_args()

    input_file = Path(args.input_zarr)
    output_file = Path(args.output_png)

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
        # Open the array inside the GPU context so each tile slice is read
        # directly into device memory (no host round-trip per tile).
        from linumpy.gpu.zarr_io import gpu_zarr_context

        with gpu_zarr_context():
            vol, _ = read_omezarr(input_file, level=0)
            aip = compute_aip(vol, use_gpu=True)
    else:
        vol, _ = read_omezarr(input_file, level=0)
        aip = compute_aip(vol, use_gpu=False)
    save_aip_png(aip, output_file)


if __name__ == "__main__":
    main()
