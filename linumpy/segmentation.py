#! /usr/bin/env python
# -*- coding: utf-8 -*-

import SimpleITK as sitk
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.ndimage import gaussian_filter, binary_fill_holes, distance_transform_edt
from skimage.filters import threshold_otsu, median
from skimage.measure import label
from skimage.morphology import remove_small_objects, disk, ball, binary_opening, binary_closing, local_maxima
from skimage.segmentation import watershed


def segmentOCT3D(vol: np.ndarray, k: int = 5, useLog: bool = True, thresholdMethod: str = "otsu") -> np.ndarray:
    """To segment an S-OCT brain in 3D using thresholding and morphological watershed
    Parameters
    ----------
    vol
        The OCT brain to segment
    k
        Median smoothing kernel size in pixel
    useLog
        Transform the pixel intensity with a log before computing mask
    thresholdMethod
        'ostu', 'triangle'
    Returns
    -------
    ndarray
        The brain mask
    """
    vol_p = np.copy(vol)
    if useLog:
        vol_p[vol > 0] = np.log(vol_p[vol > 0])

    # Creating a sitk image + smoothing
    img = sitk.GetImageFromArray(vol_p)
    img = sitk.Median(img, [k, k, k])

    # Segmenting using an Otsu threshold
    if thresholdMethod == "otsu":
        marker_img = ~sitk.OtsuThreshold(img)
    elif thresholdMethod == "triangle":
        marker_img = ~sitk.TriangleThreshold(img)
    else:
        marker_img = ~sitk.OtsuThreshold(img)

    # Using a watershed algorithm to optimize the mask
    ws = sitk.MorphologicalWatershedFromMarkers(img, marker_img)

    # Separating into foreground / background
    seg = sitk.ConnectedComponent(ws != ws[0, 0, 0])

    # Filling holes and returning the mask
    mask = fillHoles_2Dand3D(sitk.GetArrayFromImage(seg))

    return mask


def fillHoles_2Dand3D(mask: np.ndarray) -> np.ndarray:
    """Fill holes in a 2D or 3D mask
    Parameters
    ----------
    mask
        The mask to fill
    Returns
    -------
    ndarray
        The filled mask
    """
    # Filling holes and returning the mask
    mask = binary_fill_holes(mask)

    # Fill holes (in 2D)
    nx, ny, nz = mask.shape
    for x in range(nx):
        mask[x, :, :] = binary_fill_holes(mask[x, :, :])
    for y in range(ny):
        mask[:, y, :] = binary_fill_holes(mask[:, y, :])
    for z in range(nz):
        mask[:, :, z] = binary_fill_holes(mask[:, :, z])

    # Refill holes in 3D (in case some were missed)
    mask = binary_fill_holes(mask)
    return mask


def removeBottom(mask: np.ndarray, k: int = 10, axis: int = 2, inverse: bool = False,
                 fillHoles: bool = False) -> np.ndarray:
    """Remove the bottom side of the mask.
    Parameters
    ----------
    mask
        Mask to modify. The 3rd axis is assumed to be the dimension direction to modify.
    k
        Number of pixel to erode
    axis
        Axis to erode
    inverse
        Inverse the operation
    fillHoles
        Fill holes in the mask
    Returns
    -------
    ndarray
        Modified mask
    """
    assert axis >= 0 and axis <= 2, "axis must be between 0 and 2"
    if axis == 0:
        kernel = np.zeros((2 * k, 1, 1), dtype=bool)
    elif axis == 1:
        kernel = np.zeros((1, 2 * k, 1), dtype=bool)
    elif axis == 2:
        kernel = np.zeros((1, 1, 2 * k), dtype=bool)
    if inverse:
        kernel[0:k] = True
    else:
        kernel[k::] = True
    if fillHoles:
        mask_p = binary_erosion(fillHoles_2Dand3D(mask), kernel)
        mask_p = mask_p * mask
    else:
        mask_p = binary_erosion(mask, kernel)
    return mask_p


def create_mask(image: np.ndarray, sigma: float = 5.0, selem_radius: int = 1, min_size: int = 100,
                normalize: bool = True) -> np.ndarray:
    """
    Create a mask for the given image using normalization, smoothing, thresholding, morphological operations,
    distance transform, and watershed segmentation. Not dependent on SimpleITK.

    Parameters:
    - image: np.ndarray
        The input image to create a mask for.
    - sigma: float
        The standard deviation for Gaussian smoothing.
    - selem_radius: int
        The radius of the structuring element for morphological operations.
    - min_size: int
        The minimum size of objects to keep in the final mask.
    - normalize: bool
        Whether to normalize the image before processing.

    Returns:
    - mask: np.ndarray
    """
    if normalize:
        # Normalize image
        image = np.copy(image)
        image -= np.percentile(image[image > 0], 0.5)
        image /= np.percentile(image, 99.5)
        image = np.clip(image, 0, 1)

    # Smoothing
    image = gaussian_filter(image, sigma=sigma)
    image = median(image)  # simple denoising

    # threshold
    threshold = threshold_otsu(image)
    mask = image > threshold

    selem = disk(selem_radius) if image.ndim == 2 else ball(selem_radius)
    mask = binary_opening(mask, selem)
    mask = binary_closing(mask, selem)
    mask = binary_fill_holes(mask)
    mask = remove_small_objects(mask, min_size=min_size)

    # compute distance transform and use local maxima as markers
    dist = distance_transform_edt(mask)
    peaks = local_maxima(dist)
    markers = label(peaks)

    # watershed on -distance to segment foreground objects; mask restricts to bw
    labels = watershed(-dist, markers=markers, mask=mask)

    mask = labels > 0
    mask = binary_fill_holes(mask)
    mask = remove_small_objects(mask, min_size=min_size)

    return mask
