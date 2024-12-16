#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""View a Zarr file with napari."""

import argparse

import napari
import zarr


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_zarr",
                   help="Full path to the Zarr file.")
    p.add_argument("-r", "--resolution",  nargs=3, type=float, default=[1.0]*3, metavar=('z', 'x', 'y'),
                   help="Resolution in micrometer in the Z, X, Y order. For an isotropic resolution, provide a single value. (default=%(default)s)")

    return p


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Parameters
    zarr_location = args.input_zarr
    resolution = args.resolution
    assert len(resolution) in [1, 3], "Resolution must be a single value or a tuple of 3 values"

    # Load the volume
    vol = zarr.open(zarr_location, mode="r")
    scales = []
    if len(resolution) == 1:
        scales = [resolution[0] * 1e-3] * 3
    else:
        scales = [r * 1e-3 for r in resolution]

    # Prepare the viewer
    viewer = napari.Viewer()
    viewer.add_image(vol, scale=scales, colormap="magma")
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"

    # Run the viewer
    napari.run()

if __name__ == "__main__":
    main()
