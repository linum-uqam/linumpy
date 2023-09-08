#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert a nifti volume into a .zarr volume"""

import nibabel as nib
import numpy as np
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
import zarr
from pathlib import Path
import argparse

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_image",
                   help="Full path to a 3D .nii image, with axis in ZYX order.")
    p.add_argument("zarr_directory",
                   help="Full path to the .zarr directory")
    p.add_argument("--resolution_xy", type=float, default=3.0,
                   help="Lateral (xy) resolution in micron. (default=%(default)s)")
    p.add_argument("--resolution_z", type=float, default=3.5,
                   help="Axial (z) resolution in micron. (default=%(default)s)")
    return p

def create_transformation_dict(scales, levels):
    """
    Create a dictionary with the transformation information for 3D images.

    :param scales: The scale of the image, in z y x order.
    :param levels: The number of levels in the pyramid.
    :return:
    """
    coord_transforms = []
    for i in range(levels):
        transform_dict = [{
            "type": "scale",
            "scale": [scales[0] * (2 ** i), scales[1] * (2 ** i), scales[2] * (2 ** i)]
        }]
        coord_transforms.append(transform_dict)
    return coord_transforms

def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()
    path = Path(args.zarr_directory)
    path.parent.mkdir(exist_ok=True, parents=True)   
    scales = (args.resolution_xy, args.resolution_xy, args.resolution_z)

    # Load the data
    img=nib.load(args.input_image)
    img_array=img.get_fdata()

    # Prepare the chunk size
    dim = np.shape(img_array)
    chunk_size_z = 1; chunk_size_y = dim[1]; chunk_size_x = dim[2]    
    chunks = (chunk_size_z, chunk_size_y, chunk_size_x)

    # write the image data
    store = parse_url(path, mode="w").store
    root = zarr.open_group(path, mode="w")
    write_image(image=img_array, group=root, axes="zyx",
                coordinate_transformations=create_transformation_dict(scales, 5),
                storage_options=dict(chunks=chunks))

if __name__ == "__main__":
    main()