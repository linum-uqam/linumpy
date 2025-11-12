#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert an ome-zarr volume into a nifti volume at a given resolution.
"""
import argparse

import SimpleITK as sitk
import numpy as np
from linumpy.io.zarr import read_omezarr

import nibabel as nib


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input",
                   help="Full path to an OME-ZARR directory")
    p.add_argument("output",
                   help="Full path to the output nifti file")
    p.add_argument("-r", "--resolution", type=float, default=10.0,
                   help="Output resolution in micron (default=%(default)s)")
    p.add_argument("-i", "--isotropic", action="store_true",
                   help="Interpolate the volume to isotropic resolution")
    p.add_argument("--save_mm", action='store_true',
                   help='Save nifti header in mm.')
    return p


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Load the ome-zarr volume and choose the scale
    vol, zarr_resolution = read_omezarr(args.input)

    # zarr_resolution is already in mm but resolution given in microns
    if args.save_mm:
        out_resolution = args.resolution / 1000.0
    else:
        out_resolution = args.resolution
        zarr_resolution = [1000*r for r in zarr_resolution]

    # Set the scaling factor
    transform = np.eye(3)
    transform[0, 0] = out_resolution / (zarr_resolution[2])
    transform[1, 1] = out_resolution / (zarr_resolution[1])
    if args.isotropic:
        transform[2, 2] = out_resolution / (zarr_resolution[0])

    # Compute the output volume shape
    old_shape = vol.shape
    new_shape = (int(old_shape[2] / transform[0, 0]),
                 int(old_shape[1] / transform[1, 1]),
                 int(old_shape[0] / transform[2, 2]))
    if args.isotropic:
        new_spacing = (out_resolution, out_resolution, out_resolution)
    else:
        new_spacing = (out_resolution, out_resolution, zarr_resolution[0])

    # Prepare the output
    input_volume = sitk.GetImageFromArray(vol[:])
    # conversion mm to um
    input_volume.SetSpacing((zarr_resolution[2], zarr_resolution[1], zarr_resolution[0]))

    # Create the sampler
    sampler = sitk.ResampleImageFilter()
    sampler.SetSize(new_shape)
    sampler.SetOutputSpacing(new_spacing)
    sampler.SetOutputPixelType(sitk.sitkFloat32)
    sampler.SetInterpolator(sitk.sitkLinear)
    sampler.SetDefaultPixelValue(0)
    warped = sampler.Execute(input_volume)

    # Save the output volume
    warped_np = sitk.GetArrayFromImage(warped)
    nib.save(nib.Nifti1Image(warped_np.astype(np.float32),
                             np.diag(new_spacing + (1,))),
             args.output)


if __name__ == "__main__":
    main()
