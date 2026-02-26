#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Average Intensity Projection (AIP) PNG previews from mosaic grid OME-Zarr files.

Computes the AIP (mean over the Z-axis) for each mosaic grid found in the input
directory and saves the 2D results as 16-bit PNG files in the output directory.
Spatial resolution is preserved: each data pixel maps to exactly one output pixel.

AIP images are useful for QC visualization and for checking tile layout after
preprocessing.

Example usage:
    # Process all mosaic grids in a directory
    linum_generate_mosaic_aips.py /path/to/mosaics /path/to/aips

    # Use a downsampled pyramid level for faster processing
    linum_generate_mosaic_aips.py /path/to/mosaics /path/to/aips --level 1
"""
# Configure thread limits before numpy/scipy imports
import linumpy._thread_config  # noqa: F401

import argparse
from pathlib import Path

import numpy as np
from skimage.io import imsave
from tqdm.auto import tqdm

from linumpy.io.zarr import read_omezarr


def compute_aip(vol) -> np.ndarray:
    """Compute the AIP of a mosaic grid volume tile-by-tile.

    Parameters
    ----------
    vol:
        Dask array of shape (Z, X, Y) from read_omezarr.

    Returns
    -------
    np.ndarray
        2D float32 AIP array of shape (X, Y).
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
            aip[rmin:rmax, cmin:cmax] = np.asarray(
                vol[:, rmin:rmax, cmin:cmax]).mean(axis=0)
    return aip


def save_aip_png(aip: np.ndarray, output_path: Path) -> None:
    """Normalize and save an AIP array as a 16-bit PNG.

    Intensities are clipped to the 0.1–99.9 percentile range and mapped
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
    if vmax > vmin:
        aip_norm = np.clip((aip - vmin) / (vmax - vmin), 0, 1)
    else:
        aip_norm = np.zeros_like(aip)
    imsave(output_path, (aip_norm * 65535).astype(np.uint16))


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input",
                   help="Input directory containing mosaic grid OME-Zarr files\n"
                        "(mosaic_grid_3d_z*.ome.zarr).")
    p.add_argument("output",
                   help="Output directory where AIP PNG files will be saved.")
    p.add_argument("--level", type=int, default=0,
                   help="Pyramid level of the input mosaic grids to use.\n"
                        "Higher levels are downsampled and faster to process.\n"
                        "Default: 0 (full resolution)")
    return p


def main():
    p = _build_arg_parser()
    args = p.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    mosaic_files = sorted(input_dir.glob("mosaic_grid_3d_z*.ome.zarr"))
    if not mosaic_files:
        raise FileNotFoundError(
            f"No mosaic grid files found in {input_dir}.\n"
            "Expected files matching 'mosaic_grid_3d_z*.ome.zarr'.")

    for mosaic_file in tqdm(mosaic_files, desc="Generating AIPs"):
        slice_id = mosaic_file.name[len("mosaic_grid_3d_z"):-len(".ome.zarr")]
        output_file = output_dir / f"aip_z{slice_id}.png"
        vol, _ = read_omezarr(mosaic_file, level=args.level)
        aip = compute_aip(vol)
        save_aip_png(aip, output_file)


if __name__ == "__main__":
    main()
