#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_erosion, binary_fill_holes


def segmentOCT3D(vol, k=5, useLog=True, thresholdMethod="otsu"):
    """To segment an OCT brain in 3D using Otsu and morphological watershed
    Parameters
    ----------
    vol : ndarray
        The OCT brain to segment
    k : int
        Median smoothing kernel size in pixel
    useLog : bool
        Transform the pixel intensity with a log before computing mask
    thresholdMethod : str
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


def fillHoles_2Dand3D(mask):
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


def removeBottom(mask, k=10, axis=2, inverse=False, fillHoles=False):
    """Remove the bottom side of the mask.
    Parameters
    ----------
    mask : ndarray
        Mask to modify. The 3rd axis is assumed to be the dimension direction to modify.
    k : int
        Number of pixel to erode
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