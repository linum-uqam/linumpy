#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compute an Average Intensity Projection (AIP) from a 3D mosaic grid and save as PNG.

The AIP is computed by averaging voxel intensities along the Z-axis, producing a 2D
image at full XY resolution (1 data pixel = 1 output pixel). The result is saved as
a 16-bit PNG for QC visualization.
"""

# Configure thread limits before numpy/scipy imports
import linumpy._thread_config  # noqa: F401

import argparse
from pathlib import Path

import numpy as np
from skimage.io import imsave

from linumpy.io.zarr import read_omezarr


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_zarr",
                   help="Full path to the input mosaic grid OME-Zarr volume.")
    p.add_argument("output_png",
                   help="Full path to the output PNG file.")
    return p


def main():
    p = _build_arg_parser()
    args = p.parse_args()

    input_file = Path(args.input_zarr)
    output_file = Path(args.output_png)

    vol, _ = read_omezarr(input_file, level=0)

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

    vmin = np.percentile(aip, 0.1)
    vmax = np.percentile(aip, 99.9)
    if vmax > vmin:
        aip_norm = np.clip((aip - vmin) / (vmax - vmin), 0, 1)
    else:
        aip_norm = np.zeros_like(aip)
    imsave(output_file, (aip_norm * 65535).astype(np.uint16))


if __name__ == "__main__":
    main()
