#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""View an ome-zarr file with napari."""

import argparse

import napari
import zarr
import numpy as np

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_zarr",
                   help="Full path to the OME_ZARR file.")

    return p


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Parameters
    zarr_location = args.input_zarr

    # Load the OME-Zarr file
    root = zarr.open(zarr_location, mode="r")
    n_scales = len(root)

    # Prepare the viewer
    viewer = napari.Viewer()
    layers = viewer.open(zarr_location, plugin="napari-ome-zarr")
    layers[0].colormap = "magma"
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "mm"

    # Set the color limits
    imin = max(root[n_scales-1][:].min(), 0)
    imax = np.percentile(root[n_scales-1][:], 99.9)
    layers[0].contrast_limits = [imin, imax]

    # Run the viewer
    napari.run()

if __name__ == "__main__":
    main()
