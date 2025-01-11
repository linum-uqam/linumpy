#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
import argparse
from pathlib import Path

""" Change the axis from XYZ order to ZYX, necessary before converting to .zarr format
"""

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_image",
                   help="Full path to a .nii image, with axis in XYZ order.")
    p.add_argument("output_image",
                   help="Full path to the output .nii image, with axis in ZYX order")
    p.add_argument("--resolution_xy", type=float, default=3.0,
                   help="Lateral (xy) resolution in micron. (default=%(default)s)")
    p.add_argument("--resolution_z", type=float, default=200,
                   help="Axial (z) resolution in microns. (default=%(default)s)")
    return p

def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Load the input image
    img = nib.load(Path(args.input_image))
    img_array = img.get_fdata()
    print("input array dimension :", np.shape(img_array))
    # Check the number of dimensions
    num_dimensions = img_array.ndim
    if num_dimensions == 4:
        # If there's a 4th axis (singleton dimension), remove it
        img_array = np.squeeze(img_array, axis=3)
    
    # Change the order of the axis to ZYX
    output_array = np.transpose(img_array, (2, 1, 0))
    print("output array dimension :", np.shape(output_array))
    
    # Save the image
    affine = np.eye(4)
    affine[0, 0] = args.resolution_xy / 1000.0  # x resolution in mm
    affine[1, 1] = args.resolution_xy / 1000.0  # y resolution in mm
    affine[2, 2] = args.resolution_z / 1000.0  # z resolution in mm

    img = nib.Nifti1Image(output_array, affine)
    output_file = Path(args.output_image)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    nib.save(img, output_file)

if __name__ == "__main__":
    main()