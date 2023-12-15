#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Stack 2D mosaics into a single volume.
"""

import SimpleITK as sitk
import numpy as np
import argparse
from pathlib import Path
import re
import nibabel as nib


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_images", nargs="+",
                   help="Full path to a 2D mosaic grid image.")
    p.add_argument("output_volume",
                   help="Assembled volume filename (must be a nii or nii.gz)")
    p.add_argument("--resolution_xy", type=float, default=1.0,
                   help="Lateral (xy) resolution in micron. (default=%(default)s)")
    p.add_argument("--resolution_z", type=float, default=1.0,
                   help="Axial (z) resolution in micron, corresponding to the z distance between images in the stack. (default=%(default)s)")
    return p


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Detect the slices ids
    files = [Path(x) for x in args.input_images]
    pattern = r".*z(\d+)_.*"
    slice_ids = []
    for f in files:
        foo = re.match(pattern, f.name)
        slice_ids.append(int(foo.groups()[0]))

    # Preparing the volume
    n_slices = len(slice_ids)

    # Detect the mosaic shape (not all mosaic grid will have the same size)
    n_rows = 0
    n_cols = 0
    for f in files:
        img = sitk.GetArrayFromImage(sitk.ReadImage(f))
        if img.shape[0] > n_rows:
            n_rows = img.shape[0]
        if img.shape[1] > n_cols:
            n_cols = img.shape[1]
    volume = np.zeros((n_rows, n_cols, n_slices), dtype=img.dtype)

    # Add the slices to the volume
    for z, f in zip(slice_ids, files):
        img = sitk.GetArrayFromImage(sitk.ReadImage(f))

        # Zero padding
        pad_r_0 = (n_rows - img.shape[0]) // 2
        pad_r_1 = (n_rows - img.shape[0] - pad_r_0)
        pad_c_0 = (n_cols - img.shape[1]) // 2
        pad_c_1 = (n_cols - img.shape[1] - pad_c_0)
        volume[:, :, z] = np.pad(img, ((pad_r_0, pad_r_1), (pad_c_0, pad_c_1)))

    # Save this volume
    affine = np.eye(4)
    affine[0, 0] = args.resolution_xy / 1000.0  # x resolution in mm
    affine[1, 1] = args.resolution_xy / 1000.0  # y resolution in mm
    affine[2, 2] = args.resolution_z / 1000.0  # z resolution in mm
    img = nib.Nifti1Image(volume, affine)
    output_file = Path(args.output_volume)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    nib.save(img, output_file)


if __name__ == "__main__":
    main()
