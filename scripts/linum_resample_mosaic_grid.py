#!/usr/bin/env python3
"""Resample a mosaic grid to a new isotropic resolution.

GPU acceleration is used when available (--use_gpu, default on) for
volume resampling/rescaling (5-12x speedup). Falls back to CPU if no GPU
is detected or --no-use_gpu is passed.
"""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse
import itertools
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from tqdm import tqdm

from linumpy.geometry.resampling import resolution_is_mm
from linumpy.gpu import GPU_AVAILABLE, print_gpu_info
from linumpy.gpu.interpolation import resize
from linumpy.io import OmeZarrWriter, read_omezarr


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("in_mosaic", help="Input mosaic grid in .ome.zarr.")
    p.add_argument("out_mosaic", help="Output resampled mosaic .ome.zarr.")
    p.add_argument("--resolution", "-r", type=float, default=10.0, help="Isotropic resolution for resampling in microns.")
    p.add_argument("--n_levels", type=int, default=5, help="Number of levels in pyramid decomposition [%(default)s].")
    p.add_argument(
        "--use_gpu",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use GPU acceleration if available. [%(default)s]",
    )
    p.add_argument(
        "--prefetch",
        type=int,
        default=8,
        help=(
            "Number of input tiles to read concurrently while the GPU resizes the current tile. "
            "Increases host RAM use by ~prefetch * tile_size. "
            "On sharded zstd inputs, raising from 1 to 8 yields a ~3x end-to-end speedup "
            "because zarr decode is the bottleneck and parallelises well. [%(default)s]"
        ),
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Print GPU information and timing.")
    return p


def rescale(image: Any, scale: float | Sequence[float], order: int = 1, use_gpu: bool = True) -> Any:
    """Rescale an image by a scale factor.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D or 3D).
    scale : float or tuple
        Scale factor(s) for each axis.
    order : int
        Interpolation order (1=linear).
    use_gpu : bool
        Whether to use GPU acceleration.

    Returns
    -------
    np.ndarray
        Rescaled image.
    """
    scale_tuple = tuple([float(scale)] * image.ndim) if isinstance(scale, (int, float)) else tuple(scale)
    # Clamp to >=1 so heavy downsampling of small axes (e.g. Z=5 by factor 0.1)
    # doesn't produce a zero-sized output that triggers ZeroDivisionError downstream.
    output_shape = tuple(max(1, round(s * sc)) for s, sc in zip(image.shape, scale_tuple, strict=False))
    return resize(image, output_shape, order=order, anti_aliasing=True, use_gpu=use_gpu)


def _read_tile(vol: Any, i: Any, j: Any, tile_shape: Any) -> Any:
    """Read one tile from the input zarr array (I/O stage of the pipeline)."""
    return np.asarray(vol[:, i * tile_shape[1] : (i + 1) * tile_shape[1], j * tile_shape[2] : (j + 1) * tile_shape[2]])


def _run_pipelined(
    vol: Any,
    out_zarr: Any,
    tile_iter: Any,
    tile_shape: Any,
    out_tile_shape: Any,
    scaling_factor: float,
    use_gpu: bool,
    prefetch: int = 4,
) -> None:
    """Process tiles with a depth-``prefetch`` read pipeline.

    Background reader threads keep up to ``prefetch`` input tiles decoding in
    parallel while the main thread runs GPU resize and writes the current
    tile to the output zarr.

    On sharded zstd inputs, zarr decode is the per-tile bottleneck and
    parallelises well across worker threads, so this is where the real
    end-to-end speedup comes from (~2-3x going from depth 1 to 4-8 on a
    typical mosaic). The GPU compute (gauss + zoom + H↔D) for a downsampled
    output is small enough to be fully hidden behind the read.
    """
    if not tile_iter:
        return

    cp: Any = None
    cupy_available = False
    if use_gpu:
        try:
            import cupy as cp

            cupy_available = True
        except Exception:
            pass

    workers = max(1, min(prefetch, len(tile_iter)))
    with ThreadPoolExecutor(max_workers=workers) as prefetch_executor:
        in_flight = [prefetch_executor.submit(_read_tile, vol, i, j, tile_shape) for i, j in tile_iter[:workers]]

        for k, (i, j) in enumerate(tqdm(tile_iter, desc="Resampling tiles", unit="tile")):
            tile = in_flight[k].result()

            nxt = k + workers
            if nxt < len(tile_iter):
                ni, nj = tile_iter[nxt]
                in_flight.append(prefetch_executor.submit(_read_tile, vol, ni, nj, tile_shape))

            resampled = rescale(tile, scaling_factor, order=1, use_gpu=use_gpu)
            out_zarr[
                :, i * out_tile_shape[1] : (i + 1) * out_tile_shape[1], j * out_tile_shape[2] : (j + 1) * out_tile_shape[2]
            ] = resampled

            if cupy_available and cp is not None and k % 10 == 9:
                cp.get_default_memory_pool().free_all_blocks()


def main() -> None:
    """Run function."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    use_gpu = args.use_gpu and GPU_AVAILABLE

    if args.verbose:
        print_gpu_info()

    if args.use_gpu and not GPU_AVAILABLE:
        print("WARNING: GPU requested but not available, falling back to CPU")
    elif use_gpu:
        print("GPU: ENABLED")
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

    print(f"Loading: {args.in_mosaic}")
    vol, source_res = read_omezarr(args.in_mosaic)
    source_in_mm = resolution_is_mm(source_res)
    target_res = args.resolution / 1000.0 if source_in_mm else float(args.resolution)

    tile_shape = vol.chunks
    scaling_factor = np.asarray(source_res) / target_res

    print(f"  Volume shape: {vol.shape}")
    print(f"  Tile shape: {tile_shape}")
    source_um = [r * 1000 for r in source_res] if source_in_mm else list(source_res)
    print(f"  Source resolution: {[f'{r:.2f}' for r in source_um]} µm")
    print(f"  Target resolution: {args.resolution} µm")
    print(f"  Scale factor: {scaling_factor}")

    out_tile_shape = tuple(max(1, round(s * sc)) for s, sc in zip(tile_shape, scaling_factor, strict=False))

    nx = vol.shape[1] // tile_shape[1]
    ny = vol.shape[2] // tile_shape[2]
    total_tiles = nx * ny

    out_shape = (out_tile_shape[0], nx * out_tile_shape[1], ny * out_tile_shape[2])
    print(f"  Output shape: {out_shape} ({total_tiles} tiles)")

    out_zarr = OmeZarrWriter(args.out_mosaic, out_shape, out_tile_shape, dtype=vol.dtype, overwrite=True)

    tile_iter = list(itertools.product(range(nx), range(ny)))
    _run_pipelined(vol, out_zarr, tile_iter, tile_shape, out_tile_shape, scaling_factor, use_gpu, prefetch=args.prefetch)

    print("Building pyramid...")
    out_res = [target_res] * 3
    out_zarr.finalize(out_res, args.n_levels)

    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.1f}s ({total_tiles / elapsed:.1f} tiles/s)")


if __name__ == "__main__":
    main()
