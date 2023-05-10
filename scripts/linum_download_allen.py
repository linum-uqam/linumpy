#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Download the Allen mouse brain template, and setting the correct RAS+ direction and spacing.
"""

import argparse
from pathlib import Path

import SimpleITK as sitk

from linumpy.io import allen


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("output",
                   help="Output nifti filename")
    p.add_argument("-r", "--resolution", default=100, type=int, choices=allen.AVAILABLE_RESOLUTIONS,
                   help="Template resolution in micron. Default=%(default)s")

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

    vol = allen.download_template(args.resolution, cache=True)

    # Preparing the affine to align the template in the RAS+
    r_mm = args.resolution / 1e3  # Convert the resolution from micron to mm
    vol.SetSpacing([r_mm] * 3)  # Set the spacing in mm

    # Apply the transform to RAS
    vol = sitk.PermuteAxes(vol, (2, 0, 1))
    vol = sitk.Flip(vol, (False, False, True))
    vol.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])

    # Save the volume
    sitk.WriteImage(vol, str(output))


if __name__ == "__main__":
    main()
