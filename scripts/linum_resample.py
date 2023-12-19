#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Resample a nifti volume to a given resolution."""

from pathlib import Path

import argparse
import SimpleITK as sitk
import numpy as np

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_volume",
                   help="Full path to a nifti volume.")
    p.add_argument("output_volume", default=None,
                   help="Full path to the output nifti volume (must be .nii or .nii.gz)")
    p.add_argument("resolution", type=float, default=25.0,
                     help="Output resolution in micron (default=%(default)s)")

    return p

def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Parameters
    input_volume = Path(args.input_volume)
    output_volume = Path(args.output_volume)
    extension = ""
    if output_volume.name.endswith(".nii"):
        extension = ".nii"
    elif output_volume.name.endswith(".nii.gz"):
        extension = ".nii.gz"
    assert extension in [".nii", ".nii.gz"], "The output file must be a .nii or .nii.gz file."
    resolution = args.resolution

    # Load the nifti volume
    vol = sitk.ReadImage(str(input_volume))

    # Set the scaling factor
    transform = np.eye(3)
    transform[0, 0] = resolution / vol.GetSpacing()[0]
    transform[1, 1] = resolution / vol.GetSpacing()[1]
    transform[2, 2] = resolution / vol.GetSpacing()[2]

    # Compute the output volume shape
    old_shape = vol.GetSize()
    new_shape = (int(old_shape[0] / transform[0, 0]),
                 int(old_shape[1] / transform[1, 1]),
                 int(old_shape[2] / transform[2, 2]))
    new_spacing = (resolution, resolution, resolution)

    # Create the sampler
    sampler = sitk.ResampleImageFilter()
    sampler.SetSize(new_shape)
    sampler.SetOutputOrigin(vol.GetOrigin())
    sampler.SetOutputDirection(vol.GetDirection())
    sampler.SetOutputSpacing(new_spacing)
    sampler.SetOutputPixelType(sitk.sitkFloat32)
    sampler.SetInterpolator(sitk.sitkLinear)
    sampler.SetDefaultPixelValue(0)
    warped = sampler.Execute(vol)

    # Save the output volume
    output_volume.parent.mkdir(exist_ok=True, parents=True)
    sitk.WriteImage(warped, str(output_volume))

if __name__ == "__main__":
    main()

