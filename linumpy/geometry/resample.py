"""Volume resampling and downsampling using SimpleITK."""

from collections.abc import Sequence

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter


def resample_itk(vol: np.ndarray, newshape: Sequence[int] | int, interpolator: str = "linear") -> np.ndarray:
    """Resamples a volume / image using ITK.

    Parameters
    ----------
    vol : ndimage
        Array to resample
    newshape : tuple of int or int
        New shape of the array, or resampling factor (if a single integer is given)
    interpolator : str
        Interpolation method to use. Available are:
         - 'NN' (NearestNeighbor)
         - 'linear'

    Returns
    -------
    ndarray
        Resampled array

    """
    resample = sitk.ResampleImageFilter()

    # Computing newshape if a factor is given
    if isinstance(newshape, int):
        newshape = np.round(np.array(vol.shape) / float(newshape)).astype(int)
    else:
        newshape = tuple(int(x) for x in newshape)

    if vol.dtype == bool:
        is_bool = True
        vol = 255 * vol.astype(np.uint8)
    else:
        is_bool = False

    if vol.ndim == 3 and vol.shape[2] == 1:
        vol = np.squeeze(vol, axis=(2,))
        newshape = newshape[0:2]

    # Use a small positive default value instead of zero to avoid black dots
    nonzero_vals = vol[vol > 0]
    default_val = float(np.percentile(nonzero_vals, 1)) if len(nonzero_vals) > 0 else 0.0
    resample.SetDefaultPixelValue(default_val)

    if vol.ndim == 2:
        nx, ny = vol.shape
        ox, oy = newshape
        resample.SetSize([oy, ox])
        resample.SetOutputSpacing([(ny - 1) / float(oy), (nx - 1) / float(ox)])
        if nx / float(ox) > 1 or ny / float(oy) > 1:  # Smoothing if downsampling
            vol = gaussian_filter(vol, sigma=[nx / float(2 * ox), ny / float(2 * oy)])

    elif vol.ndim == 3:
        nx, ny, nz = vol.shape
        ox, oy, oz = newshape
        resample.SetSize([oz, oy, ox])
        resample.SetOutputSpacing([(nz - 1) / float(oz), (ny - 1) / float(oy), (nx - 1) / float(ox)])
        if nx / float(ox) > 1 or ny / float(oy) > 1 or nz / float(oz) > 1:  # Smoothing if downsampling
            vol = gaussian_filter(vol, sigma=[nx / float(2 * ox), ny / float(2 * oy), nz / float(2 * oz)])

    if interpolator == "NN":
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    elif interpolator == "linear":
        resample.SetInterpolator(sitk.sitkLinear)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    vol_itk = sitk.GetImageFromArray(vol)
    output_itk = resample.Execute(vol_itk)

    if is_bool:
        vol_p = sitk.GetArrayFromImage(output_itk)
        vol_p = vol_p > vol_p.max() * 0.5
    else:
        vol_p = sitk.GetArrayFromImage(output_itk)

    return vol_p


def shrink(
    vol: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    res: tuple[float, float, float] = (10.0, 10.0, 10.0),
) -> np.ndarray:
    """Shrink volume up to a given resolution (in each dimension).

    Parameters
    ----------
    vol : ndarray
        Volume to shrink
    spacing : (3,) list
        Voxel spacing of the original volume
    res : (3,) list
        Output resolution / spacing

    Returns
    -------
    ndarray
        Shrinked volume.
    """
    nx, ny, nz = vol.shape[:]
    dx, dy, dz = spacing[:]
    rx, ry, rz = res[:]

    # First compute output size
    output_size = (
        np.floor(dz * nz / (1.0 * rz)).astype(int),
        np.floor(dy * ny / (1.0 * ry)).astype(int),
        np.floor(dx * nx / (1.0 * rx)).astype(int),
    )

    # Apply a gaussian filter first
    vol = gaussian_filter(vol, sigma=(rx / (4.0 * dx), ry / (4.0 * dy), rz / (4.0 * dz)))

    # Creating a resampling filter using Sitk
    img = sitk.GetImageFromArray(vol)
    img.SetSpacing((dz, dy, dx))

    # Creating a resampling filter
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputSpacing([rz, ry, rx])
    resample.SetSize(output_size)

    # Use a small positive default value instead of zero to avoid black dots
    nonzero_vals = vol[vol > 0]
    default_val = float(np.percentile(nonzero_vals, 1)) if len(nonzero_vals) > 0 else 0.0
    resample.SetDefaultPixelValue(default_val)

    # Resampling
    return sitk.GetArrayFromImage(resample.Execute(img))
