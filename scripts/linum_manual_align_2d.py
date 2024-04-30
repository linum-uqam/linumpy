#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Manual alignment of two images using a GUI"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
from matplotlib.widgets import Button, Slider
from skimage.transform import rescale
import SimpleITK as sitk
from linumpy.utils_images import normalize, get_overlay, apply_xy_shift

matplotlib.use('Qt5Agg')


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("fixed_image",
                   help="Full path to the fixed image")
    p.add_argument("moving_image",
                   help="Full path to the moving image")
    p.add_argument("output_transform",
                   help="Full path to the output transform file (json file)")
    p.add_argument("--scaling_factor", type=float, default=1 / 16,
                   help="Scaling factor to apply to the images (default=%(default)s)")
    p.add_argument("--shift_limits", nargs=2, type=int, default=[-100, 100],
                   help="Limits for the shifts in pixels (default=%(default)s)")

    return p


def main():
    # Parse argumentsà
    p = _build_arg_parser()
    args = p.parse_args()

    # Parameters
    file_fixed = Path(args.fixed_image)
    file_moving = Path(args.moving_image)
    ouptut_transform_file = Path(args.output_transform)
    scaling_factor = args.scaling_factor
    shift_limits = args.shift_limits
    transform = [0, 0]  # Initialize the transform

    # Check the extension
    assert ouptut_transform_file.suffix in [".json"], "The output transform file must be a .json file."

    # Load and display the images
    #im1 = nib.load(file_fixed).get_fdata()
    im1_itk = sitk.ReadImage(str(file_fixed))
    im1 = sitk.GetArrayFromImage(im1_itk)
    #im2 = nib.load(file_moving).get_fdata()
    im2_itk = sitk.ReadImage(str(file_moving))
    im2 = sitk.GetArrayFromImage(im2_itk)

    im1 = normalize(im1)
    im2 = normalize(im2)
    im1 = rescale(im1, scaling_factor)
    im2 = rescale(im2, scaling_factor)

    def f(im1, im2, dx, dy):
        # Apply the transform
        img2_shifted = apply_xy_shift(im2, im1, dx, dy)

        # Get the overlay
        overlay = get_overlay(im1, img2_shifted)

        return overlay

    initial_dx = 0.0
    initial_dy = 0.0

    # Create the figure and the image that we will manipulate
    fig, ax = plt.subplots()
    image = ax.imshow(f(im1, im2, initial_dx, initial_dy))
    ax.set_title("Overlay")

    # Adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # Make the horizontal slider to control the horizontal shift
    ax_dx = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    dx_slider = Slider(
        ax=ax_dx,
        label="dx",
        valmin=shift_limits[0],
        valmax=shift_limits[1],
        valinit=initial_dx,
    )

    # Make the vertical slider to control the vertical shift
    ax_dy = fig.add_axes([0.1, 0.15, 0.0225, 0.63])
    dy_slider = Slider(
        ax=ax_dy,
        label="dy",
        valmin=shift_limits[0],
        valmax=shift_limits[1],
        valinit=initial_dy,
        orientation="vertical"
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        image.set_data(f(im1, im2, dx_slider.val, dy_slider.val))
        fig.canvas.draw_idle()

    # Add a button to acceprt or refect the transform
    ax_accept = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(ax_accept, 'Accept', color='lightgoldenrodyellow', hovercolor='0.975')

    def accept(event):
        transform[0] = dx_slider.val
        transform[1] = dy_slider.val
        plt.close()

    # register the update function with each slider
    dx_slider.on_changed(update)
    dy_slider.on_changed(update)
    button.on_clicked(accept)
    plt.show()

    # Get the transform
    transform_hr = [p / scaling_factor for p in transform]
    print(f"Transform: {transform_hr}")

    # Save the transform
    with open(ouptut_transform_file, "w") as f:
        json.dump({"dx": transform_hr[0], "dy": transform_hr[1]}, f)


if __name__ == "__main__":
    main()
