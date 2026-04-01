#! /usr/bin/env python
"""Brain segmentation utilities."""

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, binary_fill_holes


def segment_oct_3d(vol: np.ndarray, k: int = 5, use_log: bool = True, threshold_method: str = "otsu") -> np.ndarray:
    """To segment an S-OCT brain in 3D using thresholding and morphological watershed.

    Parameters
    ----------
    vol
        The OCT brain to segment
    k
        Median smoothing kernel size in pixel
    use_log
        Transform the pixel intensity with a log before computing mask
    threshold_method
        'ostu', 'triangle'

    Returns
    -------
    ndarray
        The brain mask
    """
    vol_p = np.copy(vol)
    if use_log:
        vol_p[vol > 0] = np.log(vol_p[vol > 0])

    # Creating a sitk image + smoothing
    img = sitk.GetImageFromArray(vol_p)
    img = sitk.Median(img, [k, k, k])

    # Segmenting using an Otsu threshold
    if threshold_method == "otsu":
        marker_img = ~sitk.OtsuThreshold(img)
    elif threshold_method == "triangle":
        marker_img = ~sitk.TriangleThreshold(img)
    else:
        marker_img = ~sitk.OtsuThreshold(img)

    # Using a watershed algorithm to optimize the mask
    ws = sitk.MorphologicalWatershedFromMarkers(img, marker_img)

    # Separating into foreground / background
    seg = sitk.ConnectedComponent(ws != ws[0, 0, 0])

    # Filling holes and returning the mask
    mask = fill_holes_2d_and_3d(sitk.GetArrayFromImage(seg))

    return mask


def fill_holes_2d_and_3d(mask: np.ndarray) -> np.ndarray:
    """Fill holes in a 2D or 3D mask.

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


def remove_bottom(mask: np.ndarray, k: int = 10, axis: int = 2, inverse: bool = False, fill_holes: bool = False) -> np.ndarray:
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
    fill_holes
        Fill holes in the mask

    Returns
    -------
    ndarray
        Modified mask
    """
    assert axis >= 0 and axis <= 2, "axis must be between 0 and 2"
    kernel: np.ndarray = np.empty(0)
    if axis == 0:
        kernel = np.zeros((2 * k, 1, 1), dtype=bool)
    elif axis == 1:
        kernel = np.zeros((1, 2 * k, 1), dtype=bool)
    else:  # axis == 2
        kernel = np.zeros((1, 1, 2 * k), dtype=bool)
    if inverse:
        kernel[0:k] = True
    else:
        kernel[k::] = True
    if fill_holes:
        mask_p = binary_erosion(fill_holes_2d_and_3d(mask), kernel)
        mask_p = mask_p * mask
    else:
        mask_p = binary_erosion(mask, kernel)
    return mask_p
