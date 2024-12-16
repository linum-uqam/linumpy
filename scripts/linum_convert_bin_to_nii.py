#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert OCT raw binary data to nifti
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from linumpy.microscope.oct import OCT


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input",
                   help="Input OCT directory. This should contain image_*.bin and info.txt files")
    p.add_argument("output",
                   help="Output nifti filename")

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Prepare the output directory
    output = Path(args.output)
    extension = ""
    if output.name.endswith(".nii"):
        extension = ".nii"
    elif output.name.endswith(".nii.gz"):
        extension = ".nii.gz"
    assert extension in [".nii", ".nii.gz"], "The output file must be a nifti file."
    output.absolute()
    output.parent.mkdir(exist_ok=True, parents=True)

    # Load the oct data
    oct = OCT()
    vol = oct.load_image(args.input)

    # Swap axes to have XYZ instead of ZXY
    vol = np.moveaxis(vol, (0, 1, 2), (2, 0, 1))

    # Prepare the affine matrix
    res_x_um = oct.info['width'] / oct.info['nx']
    res_y_um = oct.info['height'] / oct.info['ny']
    res_z_um = 3.5  # TODO: add the axial resolution to the oct scan info file.
    affine = np.eye(4)
    affine[0, 0] = res_x_um
    affine[1, 1] = res_y_um
    affine[2, 2] = res_z_um

    # Save the output file
    img = nib.Nifti1Image(vol, affine)
    img.header.set_xyzt_units(xyz="micron")
    nib.save(img, str(output))


if __name__ == "__main__":
    main()