#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Orientation Distribution Function Visualization"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d


# Function developed by Philippe Lemieux, 2021
# TODO: Add option to compute average ODF instead of average dominant direction ODF
def odf_grid_roi_overlay(img, angle, nbins=15, mask=None,
                         odf_scale=1.0, sigma=1, roi_size=None, rotate_90=True,
                         imshow_options={'cmap': 'gray'},
                         odf_options={"color": "b", "linewidth": 2}):
    """ Create plot of image with polar graph of calculated angles on top
    Parameters
    ----------
    img : ndarray
        Image to process.
    angle : ndarray
        Local orientation (in radians) between 0 and pi
    mask: ndarray
        Optional mask. False value are ignored and no ODF are generated for these regions.
    odf_scale: float
        Relative scale of the ODF compared to its ROI.
    sigma: float
        ODF angular gaussian smoothing sigma
    roi_size: int
        ROI size in pixel to create the ODF grid.
    rotate_90: bool
        Rotate the ODF by 90 degrees (useful to display the max gradient with the structure)
    imshow_options: dict
        Options passed to imshow (for the image display)
    odf_options: dict
        Options passed to plot and fill_between (for the odf overlays)
    Returns
    -------
    fig:
        Reference to the figure
    ax_image:
        Reference to the image axes
    ax_odfs: list
        List of axes reference for each ODF overlay
    """

    # Create a default mask
    if mask is None:
        mask = np.ones_like(img, dtype=bool)

    # Display the image
    ratio = img.shape[1] / img.shape[0]
    fig = plt.figure(figsize=(5 * ratio, 5))
    ax_image = fig.add_axes([0, 0, 1, 1], label="ax image")
    ax_image.imshow(img, **imshow_options)
    ax_image.axis('off')

    # Compute the number of patches
    if roi_size is None:
        nx_patches = 1
        ny_patches = 1
    else:
        nx_patches = img.shape[1] // roi_size
        if nx_patches * roi_size < img.shape[1]:
            nx_patches += 1
        ny_patches = img.shape[0] // roi_size
        if ny_patches * roi_size < img.shape[0]:
            ny_patches += 1

    # Generate the x and y ROI limits
    x_pos = np.linspace(0, img.shape[1], nx_patches + 1, dtype=int)
    y_pos = np.linspace(0, img.shape[0], ny_patches + 1, dtype=int)

    # Display the ROI limits
    for x in x_pos:
        ax_image.axvline(x, ymin=0, ymax=1, color="r", alpha=0.5)
    for y in y_pos:
        ax_image.axhline(y, xmin=0, xmax=1, color="r", alpha=0.5)

    # Pixel positions
    x, y = np.meshgrid(range(img.shape[1]), range(img.shape[0]))

    # Loop over regions
    ax_odfs = []
    bins = np.linspace(0, np.pi, nbins)
    theta = 0.5 * (bins[1::] + bins[0:-1])
    theta = [*theta, *[i + np.pi for i in theta], theta[0]]
    if rotate_90:
        theta = [i + np.pi / 2 for i in theta]
    for i in tqdm(range(nx_patches)):
        for j in range(ny_patches):
            # Get the ROI limites
            xmin = x_pos[i]
            xmax = x_pos[i + 1]
            ymin = y_pos[j]
            ymax = y_pos[j + 1]

            # Update the mask
            roi_mask = (x >= xmin) * (x <= xmax) * (y >= ymin) * (y <= ymax)
            if not np.all(mask[roi_mask]):
                continue

            # Extract the orientations for this ROI
            roi_angles = angle[roi_mask]

            # Get unique angles
            r, _ = np.histogram(roi_angles, bins=bins)

            # Put angle over 2 PI instead of 1
            r = [*r, *r, r[0]]

            # ODF smoothing
            if sigma > 0:
                r = gaussian_filter1d(r, sigma=sigma, mode="wrap")

            width = (xmax - xmin) / img.shape[1]
            height = (ymax - ymin) / img.shape[0]
            left = xmin / img.shape[1] + width * (1 - odf_scale) / 2
            bottom = 1 - ymin / img.shape[0] - height + height * (1 - odf_scale) / 2
            this_plotdim = (left, bottom, width * odf_scale, height * odf_scale)

            ax_polar = fig.add_axes(this_plotdim, projection='polar')
            ax_polar.plot(theta, r, **odf_options)
            ax_polar.fill_between(theta, r, alpha=0.5, **odf_options)
            ax_polar.set_theta_zero_location("S")  # 0 angle points south
            ax_polar.set_theta_direction(-1)  # Counter clockwise
            ax_polar.axis('off')
            ax_odfs.append(ax_polar)
        ax_image.set_xlim((0, img.shape[1]))
        ax_image.set_ylim((img.shape[0], 0))

    return (fig, ax_image, ax_odfs)


def odf_grid(img, odf, angles, distance=8, odf_scale=1, normalize=True, rotate_90=True,
             imshow_options={'cmap': 'gray'},
             odf_options={"color": "b", "linewidth": 1}):
    """ Single-voxel ODF overlay grid
        Parameters
        ----------
        img : ndarray of shape NxM
            Image to process.
        odf : ndarray of shape KxNxM
            2D oriented energy (ODF) for each pixel.
        angles: list of length K
            Orientation (in radian) of the 2D oriented energy
        distance: int
            Distance between each ODF
        odf_scale: float
            ODF scale, between 0 and 1
        normalize: bool
            If set to True, the ODF size will be normalize between 0 and 1 for each pixel
        rotate_90: bool
            Rotate the ODF by 90 degrees (useful to display the max gradient with the structure)
        imshow_options: dict
            Options passed to imshow (for the image display)
        odf_options: dict
            Options passed to plot and fill_between (for the odf overlays)
    """

    # Display the image
    ratio = img.shape[1] / img.shape[0]
    fig = plt.figure(figsize=(5 * ratio, 5))
    ax_image = fig.add_axes([0, 0, 1, 1], label="ax image")
    ax_image.imshow(img, **imshow_options)
    ax_image.axis('off')

    # Get the r and c coordinates for the ODFs
    rows = np.arange(distance // 2, img.shape[0], distance, dtype=int)
    cols = np.arange(distance // 2, img.shape[1], distance, dtype=int)

    # Display all odfs
    ax_odfs = []
    for r in rows:
        for c in cols:
            # ax_image.scatter(r,c, color="r")  # Debugging

            this_odf = np.array([*odf[:, r, c], *odf[:, r, c]])
            this_angle = [*angles, *[i + np.pi for i in angles]]
            if rotate_90:
                this_angle = [i + np.pi / 2 for i in this_angle]

            width = distance / img.shape[1]
            height = distance / img.shape[0]
            left = c / img.shape[1] - width / 2 * odf_scale
            bottom = 1 - r / img.shape[0] - height / 2 * odf_scale
            this_plotdim = (left, bottom, width * odf_scale, height * odf_scale)

            ax_polar = fig.add_axes(this_plotdim, projection='polar', label="ax polar")
            ax_polar.set_theta_zero_location("S")  # 0 angle points south
            ax_polar.set_theta_direction(-1)  # Counter clockwise
            ax_polar.plot(this_angle, this_odf, **odf_options)
            ax_polar.fill_between(this_angle, this_odf, alpha=0.5, **odf_options)
            ax_polar.axis('off')
            if not normalize:
                ax_polar.set_ylim((odf.min(), odf.max()))
            ax_odfs.append(ax_polar)
