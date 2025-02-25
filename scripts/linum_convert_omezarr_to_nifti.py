#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert an ome-zarr volume into a nifti volume at a given resolution."""

import argparse
from pathlib import Path

import SimpleITK as sitk
import numpy as np
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from linumpy.io.zarr import read_omezarr


# TODO: Read the units (mm, micron, etc) from the zarr file metadata


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
    return p


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Load the ome-zarr volume and choose the scale
    # reader = Reader(parse_url(args.input))
    vol, zarr_resolution = read_omezarr(args.input)

    # Set the scaling factor
    transform = np.eye(3)
    transform[0, 0] = args.resolution / (1000 * zarr_resolution[2])
    transform[1, 1] = args.resolution / (1000 * zarr_resolution[1])
    if args.isotropic:
        transform[2, 2] = args.resolution / (1000 * zarr_resolution[0])

    # Compute the output volume shape
    old_shape = vol.shape
    new_shape = (int(old_shape[2] / transform[0, 0]),
                 int(old_shape[1] / transform[1, 1]),
                 int(old_shape[0] / transform[2, 2]))
    if args.isotropic:
        new_spacing = (args.resolution, args.resolution, args.resolution)
    else:
        new_spacing = (args.resolution, args.resolution, zarr_resolution[0] * 1000)

    # Prepare the output
    input_volume = sitk.GetImageFromArray(vol[:])
    input_volume.SetSpacing((zarr_resolution[2] * 1000, zarr_resolution[1] * 1000, zarr_resolution[0] * 1000))

    # Create the sampler
    sampler = sitk.ResampleImageFilter()
    sampler.SetSize(new_shape)
    sampler.SetOutputSpacing(new_spacing)
    sampler.SetOutputPixelType(sitk.sitkFloat32)
    sampler.SetInterpolator(sitk.sitkLinear)
    sampler.SetDefaultPixelValue(0)
    warped = sampler.Execute(input_volume)

    # Save the output volume
    output_volume = Path(args.output)
    output_volume.parent.mkdir(exist_ok=True, parents=True)
    sitk.WriteImage(warped, str(output_volume))


if __name__ == "__main__":
    main()
