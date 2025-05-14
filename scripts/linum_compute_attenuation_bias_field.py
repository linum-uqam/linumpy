#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Compute the tissue attenuation compensation bias field"""

import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from linumpy.io.zarr import read_omezarr, save_zarr


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    # Mandatory parameters
    p.add_argument("input",
                   help="Input attenuation (OME-zarr).")
    p.add_argument("output",
                   help="Output bias field (OME-zarr).")

    # Optional argument
    p.add_argument("--isInCM", action="store_true",
                   help="The provided attenuation map is in 1/cm")

    return p


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Loading the attn volume
    vol, res = read_omezarr(args.input, level=0)
    res_axial_microns = res[0] * 1000
    attn = np.moveaxis(vol, (0, 1, 2), (2, 1, 0))

    # Converting this in 1 / voxel if it was given in 1/cm
    if args.isInCM:
        attn = attn * 100 * 1.0e-6  # This is now in 1 / micron
        attn = attn * res_axial_microns  # This is now in 1 / voxel

    # Compute the attenuation bias field by integrating over 0 -> z for each A-Lines
    bias_field = cumulative_trapezoid(attn, axis=2, initial=0)
    bias_field = np.exp(-2 * bias_field)

    # Saving this bias field
    bias_field = np.moveaxis(bias_field, (0, 1, 2), (2, 1, 0))
    save_zarr(bias_field.astype(np.float32), args.output,
              scales=res, chunks=vol.chunks)


if __name__ == "__main__":
    main()
