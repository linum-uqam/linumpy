#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""View an ome-zarr file with napari."""

import argparse
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from linumpy.io.zarr import read_omezarr
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

    # read the image data
    reader = Reader(parse_url(args.input_zarr))
    # nodes may include images, labels etc
    nodes = list(reader())

    # first node will be the image pixel data
    image_node = nodes[0]

    dask_data = image_node.data
    res = image_node.metadata["coordinateTransformations"][0][0]['scale']
    imin = max(dask_data[-1].min().compute(), 0)
    imax = dask_data[-1].max().compute()

    # Prepare the viewer
    viewer = napari.Viewer()
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = 'mm'
    viewer.add_image(dask_data, metadata=image_node.metadata, contrast_limits=(imin, imax), colormap='magma', scale=res)

    # Run the viewer
    napari.run()

if __name__ == "__main__":
    main()
