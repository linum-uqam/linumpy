#!/usr/bin/env python
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
    p.add_argument("-r", "--resolution", type=float, default=10.0,
                   help="Resolution in micrometer (default=%(default)s)")

    return p


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Parameters
    zarr_location = args.input_zarr
    resolution = args.resolution

    # Load the volume
    vol = zarr.open(zarr_location, mode="r")
    scale = (resolution, resolution, resolution)

    # Prepare the viewer
    viewer = napari.Viewer()
    viewer.add_image(vol, scale=scale, colormap="magma")
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"

    # Run the viewer
    napari.run()

if __name__ == "__main__":
    main()
