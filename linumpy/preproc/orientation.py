import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_laplace
from skimage import filters
from skimage.morphology import reconstruction

import logging


def normalize(data):
    """ Normalizes array values

    Parameters
    ----------
    data : ndarray
        Array to be normalized

    Returns
    -------
    ndarray
       Array of normalized values
    """

    return (data - np.min(data)) / (np.max(data) - np.min(data))


def remove_background(volume):
    """
    Removes the background from a 3d image using a grayscale morphological
    reconstruction.

    Parameters
    ----------
    volume : ndarray
       3d image

    Returns
    -------
    ndarray
       3d image with its background removed
    """

    seed = volume - 0.92
    dilation = reconstruction(seed, volume, method='dilation')
    volume = volume - dilation
    return volume


def denoise(volume):
    """ Denoises a 3d image using a curvature filter

    Parameters
    ----------
    volume : ndarray
       3d image

    Returns
    -------
    ndarray
       Denoised 3d image
    """

    # Convertir en  format sitk
    vol_sitk = sitk.GetImageFromArray(volume)

    # Debruiter l'image avec curvature filter
    curv_filter = sitk.CurvatureFlowImageFilter()
    curv_filter.SetNumberOfIterations(2)
    curv_filter.SetTimeStep(0.1)
    vol_sitk = curv_filter.Execute(vol_sitk)

    # Reconvertir en numpy array
    volume = sitk.GetArrayFromImage(vol_sitk)
    return volume


def increase_contrast(volume):
    """ Increases the contrast of a 3d image using a LoG filter

    Parameters
    ----------
    volume : ndarray
       3d image

    Returns
    -------
    ndarray
       3d image
    """

    # Augmentation du contraste avec filtre LoG
    volume = gaussian_laplace(volume, sigma=1.2)
    volume = np.abs(volume)
    return volume


def segmentation(volume):
    """ Segments a 3d image with the Otsu method

    Parameters
    ----------
    volume : ndarray
       3d image

    Returns
    -------
    ndarray
       binary mask

    ndarray
       segmented 3d image

    """

    # Segmentation avec threshold de Otsu
    otsu = np.zeros(volume.shape)
    for z in range(volume.shape[2]):
        try:
            threshold = filters.threshold_otsu(volume[:, :, z])
            otsu[:, :, z] = volume[:, :, z] > threshold
        except ValueError:
            logging.warning("Erreur de segmentation, z=" + str(z))
    volume = np.where(otsu, volume, 0)
    return (otsu, volume)


def modulation(orientation, volume):
    """ Modulates orientation values with their corresponding intensity

    Parameters
    ----------
    orientation : ndarray
        Array containing 3d orientation values

    volume : ndarray
        Array containing 3d intensity values.
        Its shape must be equal to the shape of the orientation array.

    Returns
    -------
    ndarray
       Array of normalized values
    """

    orientation = np.stack([volume, volume, volume], axis=3) * orientation
    orientation = orientation/np.max(orientation)
    return orientation


def preprocess(volume):
    """ Preprocesses the volume.

    Parameters
    ----------
    volume : ndarray
        3d array

    Returns
    -------
    ndarray
       Segmentation mask

    ndarray
        Preprocessed volume

    """
    volume_c = normalize(volume.copy())
    volume_rb = remove_background(volume_c)
    volume_dn = denoise(volume_rb)
    volume_ic = increase_contrast(volume_dn)
    mask, volume_seg = segmentation(volume_ic)
    return (mask, volume_seg)
