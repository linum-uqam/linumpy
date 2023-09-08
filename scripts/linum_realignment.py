#!/usr/bin/env python
# -*- coding: utf-8 -*-

import SimpleITK as sitk
import argparse
from pathlib import Path
from tqdm import tqdm

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
    input_volume = sitk.ReadImage(Path(args.input_volume), sitk.sitkFloat32)
    dimension = input_volume.GetSize()
    print(len(dimension))
    assert len(dimension) > 2 and dimension[2]>1, "The input must be a volume with at least 2 slices"
    n_slice = dimension[2]
    # The first slice of the volume has to be saved here because it will be the reference for the first realignement
    output_volume = sitk.Image(dimension, sitk.sitkFloat32)
    output_volume[:,:,0] = input_volume[:,:,0]
    print("n slice", n_slice)

    for z in tqdm(range(n_slice-1)):

        fixed_image = input_volume[:,:,z]
        moving_image = input_volume[:,:,z+1]

        # Set up the Registration Method
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMeanSquares()
        R.SetOptimizerAsRegularStepGradientDescent(4.0, 0.01, 300)
        R.SetInitialTransform(sitk.TranslationTransform(fixed_image.GetDimension()))
        R.SetInterpolator(sitk.sitkLinear)

        outTx = R.Execute(fixed_image, moving_image)
        print("n_slice = ", z)
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
        output_volume[:,:,z+1] = moving_image
    
    sitk.WriteImage(output_volume,Path(args.output_volume))

if __name__ == "__main__":
    main()