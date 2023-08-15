# -*- coding: utf-8 -*-

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

""" Realign the ROI in different slices
"""

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("fixed_image", 
                   help="Full path to the reference 2D image.")
    p.add_argument("moving_image", nargs="+",
                   help="Full path to the 2D image(s) that need realignment.")
    return p

def main():

    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()
    path_fixed_image = Path(args.fixed_image)
    path_moving_image = [Path(x) for x in args.moving_image]

    fixed_image = sitk.ReadImage(path_fixed_image, sitk.sitkFloat32)

    for path_image in path_moving_image :
        moving_image = sitk.ReadImage(path_image, sitk.sitkFloat32)
        output_image = path_image.parent / Path(path_image.stem + "_aligned" + path_image.suffix)
        print ("Path moving image", path_image)
        # Make sure the images are 2D
        if fixed_image.GetDimension() == 3 and fixed_image.GetSize()[2] == 1:
            fixed_image = fixed_image[:, :, 0]
        if moving_image.GetDimension() == 3 and moving_image.GetSize()[2] == 1:
            moving_image = moving_image[:, :, 0]

        print("Fixed image size : ", fixed_image.GetSize()  )
        print("Moving image size : ", moving_image.GetSize())

        # Convert the SimpleITK images to NumPy arrays
        fixed_array = sitk.GetArrayFromImage(fixed_image)
        moving_array = sitk.GetArrayFromImage(moving_image)

        # Get the minimum and maximum pixel values from both images
        min_intensity = min(fixed_array.min(), moving_array.min())
        max_intensity = max(fixed_array.max(), moving_array.max())

        # Perform histogram stretching (scaling the pixel intensities to [0, 255])
        fixed_array_stretch = ((fixed_array - min_intensity) / (max_intensity - min_intensity)) * 255
        moving_array_stretch = ((moving_array - min_intensity) / (max_intensity - min_intensity)) * 255

        # Convert the stretched arrays back to SimpleITK images
        fixed_image_stretch = sitk.GetImageFromArray(fixed_array_stretch)
        moving_image_stretch = sitk.GetImageFromArray(moving_array_stretch)

        # Set up the Registration Method
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMeanSquares()
        R.SetOptimizerAsRegularStepGradientDescent(4.0, 0.01, 300)
        R.SetInitialTransform(sitk.TranslationTransform(fixed_image.GetDimension()))
        R.SetInterpolator(sitk.sitkLinear)

        outTx = R.Execute(fixed_image_stretch, moving_image_stretch)

        print("-------")
        print(outTx)
        print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
        print(f" Iteration: {R.GetOptimizerIteration()}")
        print(f" Metric value: {R.GetMetricValue()}")

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(outTx)

        moving_image = resampler.Execute(moving_image)
        sitk.WriteImage(moving_image,output_image)
        fixed_image = moving_image

if __name__ == "__main__":
    main()