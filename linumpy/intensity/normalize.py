"""Intensity normalization, equalization and histogram matching."""

from typing import Any, Literal, overload

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

from linumpy.mosaic.overlap import get_overlap


def eqhist(image: np.ndarray, nbins: int = 32) -> np.ndarray:
    """Apply histogram equalisation on the input image.

    Parameters
    ----------
    image : ndarray
        Input image
    nbins : int
        Number of histogram bins to use

    Returns
    -------
    ndarray
        Equalized image
    """
    Imax = image.max()
    Imin = image.min()
    Hnorm, bin_edges = np.histogram(np.ravel(image), bins=nbins, density=True)
    Hnorm = np.insert(Hnorm, 0, 0.0)  # bin_edges : intervals
    Hnorm_cs = np.cumsum(Hnorm) * bin_edges[1]  # Cumulative sum of the normalized histogram.
    F = interp1d(bin_edges, Hnorm_cs)
    im_eq = np.reshape(np.abs((Imax - Imin + 1) * F(np.ravel(image))) - 1, image.shape)
    return im_eq


def normalize(image: np.ndarray, low_thresh: float = 0.0, high_thresh: float = 99.5) -> np.ndarray:
    """Normalize an image using low and high intensity thresholds.

    Parameters
    ----------
    image : ndarray
        Image / volume to normalize
    low_thresh : float
        Low intensity threshold to saturate (in percentile)
    high_thresh : float
        High intensity threshold to saturate (in percentile)

    Returns
    -------
    ndarray
        Normalized image / volume
    """
    imax = np.percentile(image, high_thresh)
    imin = np.percentile(image, low_thresh)

    image_p = (image - imin) / float(imax - imin)
    image_p[image_p > 1.0] = 1.0
    image_p[image_p < 0.0] = 0.0
    return image_p


@overload
def match_histogram(im1: np.ndarray, im2: np.ndarray, return_transforms: Literal[False] = ...) -> np.ndarray: ...
@overload
def match_histogram(im1: np.ndarray, im2: np.ndarray, return_transforms: Literal[True]) -> tuple[Any, Any]: ...


def match_histogram(im1: np.ndarray, im2: np.ndarray, return_transforms: bool = False) -> np.ndarray | tuple[Any, Any]:
    """Match im2 and im1 histograms.

    Parameters
    ----------
    im1: ndarray
        Reference image used as target

    im2 : ndarray
        Image to be adjusted
    return_transforms : bool
        If set to True, the transform functions will be returned instead of the adjusted image.

    Returns
    -------
    ndarray
        If returnTransform=False, returns adjusted image (im2)

    list(interpolator functions)
        If returnTransform=True, returns the function used to adjust im2 intensity to fit the im1 histogram. (V_inv, and T)

    Notes
    -----
        The returned interpolators V_inv and T need to be applied in chain. Example :

        >> V_inv, T = match_histogram(im1, im2, returnTransform=True)
        >> im2p = V_inv(T(im2))

    """
    # Computing histogram for im1 and im2
    h1, bin_edges1 = np.histogram(im1, bins=100, density=True)
    h2, bin_edges2 = np.histogram(im2, bins=100, density=True)

    # Histogram CDF (Cumulative Distribution Function)
    f1 = np.cumsum(h1)
    f2 = np.cumsum(h2)
    f1 /= f1.max()  # Normalizing this cumsum
    f2 /= f2.max()  # Normalizing this cumsum

    # Computing the f2 interpolator T
    T = interp1d(bin_edges2[1::], f2, bounds_error=False, fill_value=0)

    # Computing the inverse of F1 interpolator V_inv
    V_inv = interp1d(f1, bin_edges1[1::], bounds_error=False, fill_value=im1.min())

    if return_transforms:
        return V_inv, T
    else:
        # Correcting im2
        im2p = V_inv(T(im2))
        return im2p


def match_histogram_sequentially(data: Any, preproc_data: Any, abspos: np.ndarray, z: int, overwrite: bool = False) -> None:
    """Match neighbor tiles histograms sequentially.

    Parameters
    ----------
    data : data object
        Used for iteration and to load/save volumes.
    preproc_data : data object
        Output data object used to save preprocessed volumes.
    abspos : ndarray
        Absolute positions for each tile.
    z : int
        Slice index to process.
    overwrite : bool
        If True, overwrite existing data.
    """
    first_vol = True
    for vol1, vol2, pos1, pos2 in data.single_pass_neighbor_slice_iterator((1, 1), z, method="dfs"):
        real_pos1 = abspos[pos1[0] - 1, pos1[1] - 1, :]
        real_pos2 = abspos[pos2[0] - 1, pos2[1] - 1, :]
        ov1, ov2, _, _ = get_overlap(vol1, vol2, real_pos1, real_pos2)
        V_inv, T = match_histogram(ov1, ov2, return_transforms=True)
        vol2p = V_inv(T(vol2))

        if first_vol:
            preproc_data.saveVolume(vol1, pos1, overwrite)
            first_vol = False

        preproc_data.saveVolume(vol2p, pos2, overwrite)


def get_smooth_intensity_transition(vol: np.ndarray, slices_start: list[int]) -> np.ndarray:
    """Use a regularization function to get a smooth intensity transition between adjacent slices.

    Parameters
    ----------
    vol : ndarray
        Volume containing the slice to adjust
    slices_start : list of int
        List of slice positions, corresponding to the slice transition location

    compensateAttenuation : bool
        If true, compensation Beer-Lambert attenuation before the regularization
        (using a division by a low-pass version of the slice).

    Returns
    -------
    ndarray
        Adjusted volume.


    References
    ----------
    * Wang, H., et al. (2014). Serial optical coherence scanner for large-scale brain imaging at microscopic resolution.
      NeuroImage, 84, 1007–1017. http://doi.org/10.1016/j.neuroimage.2013.09.063

    """
    volume = np.copy(vol)
    epsilon = 1e-3
    nx, ny, nz = volume.shape

    # Get position and size of each slices
    n_slices = len(slices_start)
    slice_range = np.zeros((n_slices, 3), dtype=int)
    slice_range[:, 0] = slices_start
    slice_range[0:-1, 1] = slice_range[1::, 0]
    slice_range[-1, 1] = nz
    slice_range[:, 2] = slice_range[:, 1] - slice_range[:, 0]

    # Loop over slice transitions
    for i in range(n_slices):
        z1 = slice_range[i, 0]
        z2 = slice_range[i, 1]

        # Creating the regularization function L(z)
        a_current = gaussian_filter(volume[:, :, z1], sigma=5)  # First z of current slice local intensity average
        if i + 1 == n_slices:
            a_next = gaussian_filter(volume[:, :, z2 - 1], sigma=5)  # First z of next slice local intensity average
        else:
            a_next = gaussian_filter(volume[:, :, z2], sigma=5)  # First z of next slice local intensity average

        b_current = gaussian_filter(volume[:, :, z2 - 1], sigma=5)  # Last z of current slice local intensity average
        b_previous = (
            gaussian_filter(volume[:, :, z1], sigma=5)  # Last z of current slice local intensity average
            if i == 0
            else gaussian_filter(volume[:, :, z1 - 1], sigma=5)  # Last z of previous slice local intensity average
        )

        # Depth position
        z = np.linspace(0, z2 - z1, z2 - z1)
        nz = len(z)
        z = np.tile(z, (nx, ny, 1))
        factor = np.zeros((nx, ny, nz))
        L = np.zeros((nx, ny, nz))

        # If z <= N/2
        nz_1 = factor[:, :, 0 : nz / 2].shape[2]
        f1 = np.log((a_current + b_previous) / (2.0 * a_current + epsilon))
        f1 = np.tile(np.reshape(f1, (nx, ny, 1)), (1, 1, nz_1))
        L1 = np.exp(-(2 * z[:, :, 0 : nz / 2] - nz) / (1.0 * nz) * f1)

        # If z > N/2
        nz_2 = factor[:, :, nz / 2 : :].shape[2]
        f2 = np.log((a_next + b_current) / (2.0 * b_current + epsilon))
        f2 = np.tile(np.reshape(f2, (nx, ny, 1)), (1, 1, nz_2))
        L2 = np.exp((2 * z[:, :, nz / 2 : :] - nz) / (1.0 * nz) * f2)

        L[:, :, 0 : nz / 2] = L1
        L[:, :, nz / 2 : :] = L2

        # Apply correction to this slice
        volume[:, :, z1:z2] = L * volume[:, :, z1:z2]

    volume[np.isnan(volume)] = 0
    return volume.astype(vol.dtype)
