#!/usr/bin/env python3
"""Generate Average Intensity Projection (AIP) PNG previews from mosaic grid OME-Zarr files.

Computes the AIP (mean over the Z-axis) for each mosaic grid found in the input
directory and saves the 2D results as 16-bit PNG files in the output directory.
Spatial resolution is preserved: each data pixel maps to exactly one output pixel.

AIP images are useful for QC visualization and for checking tile layout after
preprocessing. GPU acceleration is used when available (falls back to CPU).

Example usage:
    # Process all mosaic grids in a directory
    linum_generate_mosaic_aips.py /path/to/mosaics /path/to/aips

    # Force CPU fallback
    linum_generate_mosaic_aips.py /path/to/mosaics /path/to/aips --no-use_gpu

    # Use a downsampled pyramid level for faster processing
    linum_generate_mosaic_aips.py /path/to/mosaics /path/to/aips --level 1
"""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from skimage.io import imsave
from tqdm.auto import tqdm

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
    p.add_argument("input", help="Input directory containing mosaic grid OME-Zarr files\n(mosaic_grid_3d_z*.ome.zarr).")
    p.add_argument("output", help="Output directory where AIP PNG files will be saved.")
    p.add_argument(
        "--level",
        type=int,
        default=0,
        help="Pyramid level of the input mosaic grids to use.\n"
        "Higher levels are downsampled and faster to process.\n"
        "Default: 0 (full resolution)",
    )

    gpu_group = p.add_argument_group("GPU Options")
    gpu_group.add_argument(
        "--use_gpu",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use GPU acceleration if available. [%(default)s]",
    )
    gpu_group.add_argument("--gpu_id", type=int, default=0, help="GPU device ID to use. [%(default)s]")
    gpu_group.add_argument("--verbose", "-v", action="store_true", help="Print GPU information.")
    return p


def main() -> None:
    """Run function."""
    p = _build_arg_parser()
    args = p.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = args.use_gpu and GPU_AVAILABLE

    if args.verbose:
        print_gpu_info()

    if args.use_gpu and not GPU_AVAILABLE:
        print("WARNING: GPU requested but not available, falling back to CPU")
    elif use_gpu:
        print("GPU: ENABLED")
        try:
            import cupy as cp

            cp.cuda.Device(args.gpu_id).use()
            device = cp.cuda.Device(args.gpu_id)
            mem_info = device.mem_info
            print(f"  Device: {args.gpu_id} - {cp.cuda.runtime.getDeviceProperties(args.gpu_id)['name'].decode()}")
            print(f"  Memory: {mem_info[1] / 1e9:.1f} GB total, {mem_info[0] / 1e9:.1f} GB free")
        except Exception as e:
            print(f"  Warning: Could not select GPU {args.gpu_id}: {e}. Using default.")
    else:
        print("GPU: DISABLED (using CPU)")

    mosaic_files = sorted(input_dir.glob("mosaic_grid_3d_z*.ome.zarr"))
    if not mosaic_files:
        raise FileNotFoundError(
            f"No mosaic grid files found in {input_dir}.\nExpected files matching 'mosaic_grid_3d_z*.ome.zarr'."
        )

    if use_gpu:
        # Open each volume inside the GPU context so tile slices land on
        # device memory directly (no host round-trip per tile).
        from linumpy.gpu.zarr_io import gpu_zarr_context

        for mosaic_file in tqdm(mosaic_files, desc="Generating AIPs"):
            slice_id = mosaic_file.name[len("mosaic_grid_3d_z") : -len(".ome.zarr")]
            output_file = output_dir / f"aip_z{slice_id}.png"
            with gpu_zarr_context():
                vol, _ = read_omezarr(mosaic_file, level=args.level)
                aip = compute_aip(vol, use_gpu=True)
            save_aip_png(aip, output_file)
    else:
        for mosaic_file in tqdm(mosaic_files, desc="Generating AIPs"):
            slice_id = mosaic_file.name[len("mosaic_grid_3d_z") : -len(".ome.zarr")]
            output_file = output_dir / f"aip_z{slice_id}.png"
            vol, _ = read_omezarr(mosaic_file, level=args.level)
            aip = compute_aip(vol, use_gpu=False)
            save_aip_png(aip, output_file)


if __name__ == "__main__":
    main()
