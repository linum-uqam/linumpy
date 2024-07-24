#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage.transform import rescale
from scipy.ndimage import median_filter, binary_fill_holes
from skimage.morphology import binary_dilation
from skimage.morphology import disk
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_yen
from tqdm import tqdm
import csv

from linumpy.utils_images import normalize, get_overlay, apply_xy_shift

# TODO: export the shifts to a text file for easier editing in case of failure
# TODO: use the shifts to then stitch the brain slices together

# Parameters
directory = Path("/home/joel/data/2023-07-31-alexia-multiorientation-f1/reconstruction_2d")
n_slices = len(list(directory.glob("*stitched.nii")))
scaling_factor = 1/16



def align(z):
    f1 = directory / f"mosaic_grid_z{z:02d}_stitched.nii"
    f2 = directory / f"mosaic_grid_z{z+1:02d}_stitched.nii"

    # Load and display the images
    img1_hr = nib.load(f1).get_fdata()
    img2_hr = nib.load(f2).get_fdata()
    #display_overlap(img1_hr, img2_hr, title="Original")

    img1 = normalize(img1_hr)
    img2 = normalize(img2_hr)
    img1 = rescale(img1, scaling_factor)
    img2 = rescale(img2, scaling_factor)
    print(img1.shape, img2.shape)
    #overlay = get_overlay(img1, img2)

    # Get a mask of the data
    mask1 = img1 > img1.min()
    mask1 = binary_fill_holes(img1 > threshold_yen(img1[mask1]))
    mask1 = median_filter(mask1, 15)
    mask1 = binary_dilation(mask1, disk(10))
    mask2 = img2 > img2.min()
    mask2 = binary_fill_holes(img2 > threshold_yen(img2[mask2]))
    mask2 = median_filter(mask2, 15)
    mask2 = binary_dilation(mask2, disk(10))
    #display_overlap(mask1, mask2, title="Mask")

    # Register the images
    img1_g = np.abs(gaussian_filter(img1 * mask1, sigma=3, order=1))
    img2_g = np.abs(gaussian_filter(img2 * mask2, sigma=3, order=1)) #* combined_mask
    #display_overlap(img1_g, img2_g, title="Gaussian", do_normalization=True)

    fixed_image = sitk.GetImageFromArray(img1)
    moving_image = sitk.GetImageFromArray(img2)

    # Set up the Registration Method
    R = sitk.ImageRegistrationMethod()
    #R.SetMetricAsMattesMutualInformation()
    #R.SetMetricAsCorrelation()
    R.SetMetricAsMeanSquares()
    R.SetOptimizerAsRegularStepGradientDescent(4.0, 0.01, 300)
    R.SetInitialTransform(sitk.TranslationTransform(fixed_image.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)
    R.SetMetricFixedMask(sitk.GetImageFromArray(mask1.astype(int)))
    R.SetMetricMovingMask(sitk.GetImageFromArray(mask2.astype(int)))
    outTx = R.Execute(fixed_image, moving_image)

    # Print the results
    print(f"Parameters: {outTx.GetParameters()}")
    print(f"Fixed Parameters: {outTx.GetFixedParameters()}")
    print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    print(f" Iteration: {R.GetOptimizerIteration()}")
    print(f" Metric value: {R.GetMetricValue()}")

    # Apply the transform to the full res image
    parameters_lr = outTx.GetParameters()
    parameters_hr = [p / scaling_factor for p in parameters_lr]
    dx, dy = parameters_hr
    img2_hr_warped = apply_xy_shift(img2_hr, img1_hr, dx, dy)

    # Display the results
    plt.figure(figsize=(12, 12))
    plt.subplot(1, 2, 1)
    foo = apply_xy_shift(img2_hr, img1_hr, 0, 0)
    plt.imshow(get_overlay(img1_hr, img2_hr))
    plt.axis("off")
    plt.title(f"Original (z={z:02d})")
    plt.subplot(1, 2, 2)
    plt.imshow(get_overlay(img1_hr, img2_hr_warped))
    plt.axis("off")
    plt.title(f"Warped ({parameters_hr})")
    plt.tight_layout()
    plt.show()

    return parameters_hr

fixed_ids = []
moving_ids = []
x_shifts = []
y_shift = []
for z in tqdm(range(n_slices-1)):
    transform = align(z)
    fixed_ids.append(z)
    moving_ids.append(z+1)
    x_shifts.append(transform[0])
    y_shift.append(transform[1])

# Save the shifts to a csv file
shifts = np.array([fixed_ids, moving_ids, x_shifts, y_shift]).T
with open(directory / "shifts_xy.csv", "w") as csv_file:
    writer = csv.writer(csv_file, delimiter=",")
    writer.writerow(["fixed_id", "moving_id", "x_shift", "y_shift"])
    writer.writerows(shifts)