#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Computes the tissue apparent attenuation coefficient map and then use the average
attenuation to compensate its effect in the OCT reflectivity data.
"""
# TODO: Keep the OCT pixel format (which is float32 ?)
import argparse

import numpy as np
from scipy.ndimage import gaussian_filter

from linumpy.preproc.icorr import getAttenuation_Vermeer2013
from linumpy.io.zarr import read_omezarr, save_zarr

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    # Mandatory parameters
    p.add_argument("input",
                   help="A single slice to process (ome-zarr).")
    p.add_argument("output",
                   help="Output attenuation map (ome-zarr).")

    # Optional argument
    p.add_argument("-m", "--mask", default=None,
                   help="Optional tissue mask (.ome.zarr)")
    p.add_argument("--s_xy", default=0.0, type=float,
                   help="Lateral smoothing sigma (default=%(default)s)")
    p.add_argument("--s_z", default=5.0, type=float,
                   help="Axial smoothing sigma (default=%(default)s)")

    return p


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Loading the data
    zarr_vol, res = read_omezarr(args.input, level=0)
    vol = np.moveaxis(zarr_vol, (0, 1, 2), (2, 1, 0))
    res_axial_microns = res[0] * 1000  # resolution is expected to be in microns

    if args.mask is None:
        mask = np.ones_like(vol, dtype=bool)
    else:
        mask_zarr, _ = read_omezarr(args.mask, level=0)
        mask = np.moveaxis(mask_zarr, (0, 1, 2), (2, 1, 0)).astype(bool)

    # Preprocessing
    vol = gaussian_filter(vol, sigma=(args.s_xy, args.s_xy, args.s_z))

    # Computing the attenuation using the Vermeer Method
    # TODO: If there is a 1.0e-6 multiplier it means dz is
    # expected to be given in meters. However, from docstring
    # the resolution appears to be expected in microns also.
    attn = getAttenuation_Vermeer2013(vol, dz=res_axial_microns*1.0e-6, mask=mask)
    # attn = get_extendedAttenuation_Vermeer2013(vol, mask, k=0, res=args.resolution)

    # Saving the attenuation
    attn = np.moveaxis(attn, (0, 1, 2), (2, 1, 0))
    save_zarr(attn, args.output, scales=res, chunks=zarr_vol.chunks)


if __name__ == "__main__":
    main()
