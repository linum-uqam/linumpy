#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Segment the brain from a 3D volume using a threshold and morphological operations."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import median_filter
from skimage.filters import threshold_otsu

from linumpy import segmentation


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_volume",
                   help="Full path to the input volume (.nii or .nii.gz)")
    p.add_argument("output_mask",
                   help="Full path to the output mask (.nii or .nii.gz)")
    p.add_argument("--median-size", type=int, default=5,
                   help="Size of the median filter (default=%(default)s)")

    return p


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Parameters
    volume_filename = Path(args.input_volume)
    mask_filename = Path(args.output_mask)

    # Check the extension
    assert volume_filename.suffix in [".nii", ".nii.gz"], "The input volume must be a .nii or .nii.gz file."
    assert mask_filename.suffix in [".nii", ".nii.gz"], "The output mask must be a .nii or .nii.gz file."

    # Load the volume
    img = nib.load(str(volume_filename))
    vol = img.get_fdata()

    # Create a data mask
    mask = np.zeros_like(vol, dtype=bool)
    mask[vol > 0] = True

    # Compute the threahold only in the data
    threshold = threshold_otsu(vol[mask])

    # Update the mask
    mask[vol < threshold] = False

    # Fill the holes
    mask = segmentation.fillHoles_2Dand3D(mask)

    # Filter to remove some noise
    mask = median_filter(mask, size=args.median_size)

    # Save the mask
    img_mask = nib.Nifti1Image(mask.astype(int), img.affine, img.header)
    nib.save(img_mask, str(mask_filename))


if __name__ == "__main__":
    main()
