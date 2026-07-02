"""Image transform helpers (normalization, padding, XY shifts)."""

import numpy as np
import SimpleITK as sitk


def normalize(img: np.ndarray, saturation: float = 99.7) -> np.ndarray:
    """Normalize an image between 0 and 1.

    Parameters
    ----------
    img : np.ndarray
        The image to normalize.
    saturation : float, optional
        The saturation value for the normalization

    Returns
    -------
    np.ndarray
        The normalized image.
    """
    imin = img.min()
    imax = np.percentile(img, saturation)
    img = (img.astype(np.float32) - imin) / (imax - imin)
    img[img > 1] = 1
    return img


def match_shape(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match the shape of two images by padding the smallest one.

    Parameters
    ----------
    img1 : np.ndarray
        The first image.
    img2 : np.ndarray
        The second image.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The two images with the same shape.
    """
    nr1, nc1 = img1.shape
    nr2, nc2 = img2.shape
    n_rows = max(nr1, nr2)
    n_cols = max(nc1, nc2)

    padded_images = []
    for img in [img1, img2]:
        pad_r_0 = max((n_rows - img.shape[0]) // 2, 0)
        pad_r_1 = max((n_rows - img.shape[0] - pad_r_0), 0)
        pad_c_0 = max((n_cols - img.shape[1]) // 2, 0)
        pad_c_1 = max((n_cols - img.shape[1] - pad_c_0), 0)
        padded_images.append(np.pad(img, ((pad_r_0, pad_r_1), (pad_c_0, pad_c_1))))

    return padded_images[0], padded_images[1]


def apply_xy_shift(img: np.ndarray, reference: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Apply a shift to the image in the xy plane.

    Parameters
    ----------
    img : np.ndarray
        The image to shift.
    reference : np.ndarray
        The reference image.
    dx : int
        The shift in x.
    dy : int
        The shift in y.
    """
    fixed = sitk.GetImageFromArray(reference)
    moving = sitk.GetImageFromArray(img)

    translation = [0.0] * fixed.GetDimension()
    translation[0] = dx
    translation[1] = dy

    transform = sitk.TranslationTransform(fixed.GetDimension())
    transform.SetParameters(translation)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)
    warped_moving_image = resampler.Execute(moving)
    img_warped = sitk.GetArrayFromImage(warped_moving_image)
    return img_warped
