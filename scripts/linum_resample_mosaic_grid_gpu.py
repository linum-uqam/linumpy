#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Resample a mosaic grid to a new isotropic resolution.

GPU-accelerated version using CuPy for:
- Volume resampling/rescaling (5-12x speedup)

Falls back to CPU if GPU is not available.
"""

# Configure thread limits before numpy/scipy imports
import linumpy._thread_config  # noqa: F401

import argparse
import itertools
import time

import numpy as np
from tqdm import tqdm

from linumpy.gpu import GPU_AVAILABLE, print_gpu_info
from linumpy.gpu.interpolation import resize
from linumpy.io import read_omezarr, OmeZarrWriter


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_mosaic',
                   help='Input mosaic grid in .ome.zarr.')
    p.add_argument('out_mosaic',
                   help='Output resampled mosaic .ome.zarr.')
    p.add_argument('--resolution', '-r', type=float, default=10.0,
                   help='Isotropic resolution for resampling in microns.')
    p.add_argument('--n_levels', type=int, default=5,
                   help='Number of levels in pyramid decomposition [%(default)s].')
    p.add_argument('--use_gpu', default=True,
                   action=argparse.BooleanOptionalAction,
                   help='Use GPU acceleration if available. [%(default)s]')
    p.add_argument('--verbose', '-v', action='store_true',
                   help='Print GPU information and timing')
    return p


def rescale_gpu(image, scale, order=1, use_gpu=True):
    """
    Rescale an image by a scale factor using GPU acceleration.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D or 3D)
    scale : float or tuple
        Scale factor(s) for each axis
    order : int
        Interpolation order (1=linear)
    use_gpu : bool
        Whether to use GPU acceleration

    Returns
    -------
    np.ndarray
        Rescaled image
    """
    # Convert scalar scale to tuple
    if np.isscalar(scale):
        scale = tuple([scale] * image.ndim)
    else:
        scale = tuple(scale)

    # Compute output shape
    output_shape = tuple(int(round(s * sc)) for s, sc in zip(image.shape, scale))

    # Use GPU-accelerated resize
    return resize(image, output_shape, order=order, anti_aliasing=True, use_gpu=use_gpu)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Determine GPU usage
    use_gpu = args.use_gpu and GPU_AVAILABLE

    if args.verbose:
        print_gpu_info()

    if args.use_gpu and not GPU_AVAILABLE:
        print("WARNING: GPU requested but not available, falling back to CPU")
    elif use_gpu:
        print("GPU: ENABLED")
        # Try to verify CUDA is working
        try:
            import cupy as cp
            device = cp.cuda.Device()
            print(f"  Device: {device.id} - {cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode()}")
            mem_info = device.mem_info
            print(f"  Memory: {mem_info[1] / 1e9:.1f} GB total, {mem_info[0] / 1e9:.1f} GB free")
        except Exception as e:
            print(f"  Warning: Could not query GPU info: {e}")
    else:
        print("GPU: DISABLED (using CPU)")

    start_time = time.time()

    # Read input mosaic
    print(f"Loading: {args.in_mosaic}")
    vol, source_res = read_omezarr(args.in_mosaic)
    target_res = args.resolution / 1000.0  # conversion um to mm

    tile_shape = vol.chunks
    scaling_factor = np.asarray(source_res) / target_res

    print(f"  Volume shape: {vol.shape}")
    print(f"  Tile shape: {tile_shape}")
    print(f"  Source resolution: {[f'{r*1000:.2f}' for r in source_res]} µm")
    print(f"  Target resolution: {args.resolution} µm")
    print(f"  Scale factor: {scaling_factor}")

    # Load and process first tile to get output shape
    tile_00 = np.asarray(vol[:tile_shape[0], :tile_shape[1], :tile_shape[2]])
    out_tile00 = rescale_gpu(tile_00, scaling_factor, order=1, use_gpu=use_gpu)
    out_tile_shape = out_tile00.shape

    nx = vol.shape[1] // tile_shape[1]
    ny = vol.shape[2] // tile_shape[2]
    total_tiles = nx * ny

    out_shape = (out_tile_shape[0], nx * out_tile_shape[1], ny * out_tile_shape[2])
    print(f"  Output shape: {out_shape} ({total_tiles} tiles)")

    out_zarr = OmeZarrWriter(args.out_mosaic, out_shape, out_tile_shape,
                             dtype=vol.dtype, overwrite=True)

    # Write first tile (already processed)
    out_zarr[:, 0:out_tile_shape[1], 0:out_tile_shape[2]] = out_tile00

    # Free memory
    del tile_00, out_tile00
    if use_gpu:
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    # Process remaining tiles with progress bar
    tile_iter = list(itertools.product(range(nx), range(ny)))
    tile_iter = [(i, j) for i, j in tile_iter if not (i == 0 and j == 0)]

    for i, j in tqdm(tile_iter, desc="Resampling tiles", unit="tile"):
        current_vol = np.asarray(vol[:, i * tile_shape[1]:(i + 1) * tile_shape[1],
                                     j * tile_shape[2]:(j + 1) * tile_shape[2]])

        resampled = rescale_gpu(current_vol, scaling_factor, order=1, use_gpu=use_gpu)

        out_zarr[:, i * out_tile_shape[1]:(i + 1) * out_tile_shape[1],
                 j * out_tile_shape[2]:(j + 1) * out_tile_shape[2]] = resampled

        # Free GPU memory periodically
        if use_gpu and (i * ny + j) % 10 == 0:
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

    print("Building pyramid...")
    out_zarr.finalize([target_res] * 3, args.n_levels)

    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.1f}s ({total_tiles / elapsed:.1f} tiles/s)")


if __name__ == '__main__':
    main()
