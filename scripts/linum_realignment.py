#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import SimpleITK as sitk
from tqdm import tqdm
from skimage.filters import threshold_yen
from scipy.ndimage import median_filter
from scipy.ndimage import binary_fill_holes
import numpy as np
import nibabel as nib

""" Realign the ROI in different slices """


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_volume",
                   help="Full path to the nifti volume which needs realignement")
    p.add_argument("output_volume",
                   help="Full path to the output nifty volume")
    return p


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()
    # Load the input_volume and check it is not a 2D image
    img = nib.load(args.input_volume)
    input_volume = sitk.GetImageFromArray(img.get_fdata())
    dimension = input_volume.GetSize()

    assert len(dimension) > 2 and dimension[2] > 1, "The input must be a volume with at least 2 slices"
    n_slice = dimension[2]
    # The first slice of the volume has to be saved here because it will be the reference for the first realignement
    output_volume = sitk.Image(dimension, sitk.sitkFloat32)
    output_volume[:, :, 0] = input_volume[:, :, 0]

    for z in tqdm(range(n_slice - 1), unit="slice", desc="Registration"):
        fixed_image = input_volume[:, :, z]
        moving_image = input_volume[:, :, z + 1]

        # Normalize the images
        img1 = sitk.GetArrayFromImage(fixed_image)
        img1 = (img1 - img1.min()) / (np.percentile(img1, 99.7) - img1.min())
        img1[img1 > 1] = 1
        fixed_image = sitk.GetImageFromArray(img1)

        img2 = sitk.GetArrayFromImage(moving_image)
        img2 = (img2 - img2.min()) / (np.percentile(img2, 99.7) - img2.min())
        img2[img2 > 1] = 1
        moving_image = sitk.GetImageFromArray(img2)

        # Mask the data (ignore empty tiles and agarose)
        fixed = sitk.GetArrayFromImage(fixed_image)
        moving = sitk.GetArrayFromImage(moving_image)
        mask_fixed = fixed > 0
        mask_moving = moving > 0

        # Remove the agarose from the mask
        mask_fixed = fixed > threshold_yen(fixed[mask_fixed])
        mask_moving = moving > threshold_yen(moving[mask_moving])

        # Clean the mask to remove noise, fill holes and get a part of background
        mask_fixed = median_filter(mask_fixed, 15)
        mask_fixed = binary_fill_holes(mask_fixed)

        mask_moving = median_filter(mask_moving, 15)
        mask_moving = binary_fill_holes(mask_moving)

        # Set up the Registration Method
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMattesMutualInformation()
        R.SetOptimizerAsRegularStepGradientDescent(4.0, 0.01, 300)
        R.SetInitialTransform(sitk.TranslationTransform(fixed_image.GetDimension()))
        R.SetInterpolator(sitk.sitkLinear)
        #R.SetMetricMovingMask(sitk.GetImageFromArray(mask_moving.astype(int)))
        #R.SetMetricFixedMask(sitk.GetImageFromArray(mask_fixed.astype(int)))

        outTx = R.Execute(fixed_image, moving_image)
        print("n_slice = ", z)
        print("-------")
        # print(outTx)
        print(f"Parameters: {outTx.GetParameters()}")
        print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
        print(f" Iteration: {R.GetOptimizerIteration()}")
        print(f" Metric value: {R.GetMetricValue()}")

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(outTx)

        moving_image = resampler.Execute(moving_image)
        output_volume[:, :, z + 1] = moving_image

    affine = img.affine
    img2 = nib.Nifti1Image(sitk.GetArrayFromImage(output_volume), affine)
    output_file = Path(args.output_volume)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    nib.save(img2, output_file)


if __name__ == "__main__":
    main()
