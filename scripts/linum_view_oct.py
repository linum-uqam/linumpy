#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""View an OCT tile in 3D using napari."""

import argparse

import napari
import numpy as np

from linumpy.microscope import oct


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input",
                   help="Full path to an OCT data directory.")
    p.add_argument("--no_crop", action="store_true",
                   help="Keep the full image (e.g. do not crop the galvo return)")
    p.add_argument("--no_shift_fix", action="store_true",
                   help="Do not apply the galvo shift correction")
    p.add_argument("--log", action="store_true",
                   help="Apply a log transform to the image")

    return p


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Parameters
    directory = args.input
    crop = not args.no_crop
    fix_shift = not args.no_shift_fix

    # Load an oct volume
    data = oct.OCT(directory)
    vol = data.load_image(crop=crop, fix_shift=fix_shift)

    # Log transform
    if args.log:
        vol = np.log(vol)

    imin = max(vol.min(), 0)
    imax = np.percentile(vol, 99.9)
    scale = (data.resolution[2], data.resolution[0], data.resolution[1])

    # Prepare the viewer
    viewer = napari.Viewer()
    viewer.add_image(vol, colormap='magma', contrast_limits=(imin, imax), name='oct', scale=scale, units="mm")

    # Run the viewer
    napari.run()


if __name__ == "__main__":
    main()
