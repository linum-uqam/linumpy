#! /usr/bin/env python
# -*- coding: utf-8 -*-
""" Collection of functions to fix intensity-related artefacts in raw data """

import itertools
import multiprocessing

import numpy as np
import SimpleITK as sitk
from dipy.segment.mask import median_otsu
from scipy.interpolate import interp1d, interpn
from scipy.ndimage.filters import (
    gaussian_filter,
    gaussian_filter1d,
    median_filter,
    uniform_filter,
)
from scipy.ndimage.morphology import binary_erosion, binary_fill_holes
from scipy.optimize import curve_fit, minimize
from skimage.filters import threshold_li
from sklearn import linear_model

from linumpy.preproc import xyzcorr
from linumpy.stitching.stitch_utils import getOverlap


def eqhist(image, nbins=32):
    """Apply histogram equalisation on the input image

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
    Hnorm, binEdges = np.histogram(np.ravel(image), bins=nbins, density=True)
    Hnorm = np.insert(Hnorm, 0, 0.0)  # binEdges : intervalles
    Hnorm_cs = (
        np.cumsum(Hnorm) * binEdges[1]
    )  # Somme cumulative de l'histogramme normalisé.
    F = interp1d(binEdges, Hnorm_cs)
    im_eq = np.reshape(np.abs((Imax - Imin + 1) * F(np.ravel(image))) - 1, image.shape)
    return im_eq


def normalize(image, lowThresh=0.0, highThresh=99.5):
    """Normalize an image using low and high intensity thresholds.

    Parameters
    ----------
    image : ndarray
        Image / volume to normalize
    lowThresh : float
        Low intensity threshold to saturate (in percentile)
    highThresh : float
        High intensity threshold to saturate (in percentile)

    Returns
    -------
    ndarray
        Normalized image / volume
    """
    imax = np.percentile(image, highThresh)
    imin = np.percentile(image, lowThresh)

    image_p = (image - imin) / float(imax - imin)
    image_p[image_p > 1.0] = 1.0
    image_p[image_p < 0.0] = 0.0
    return image_p


def get_averageVolume(data, z, mask=None, s=0):
    """Computes an average volume for a specific slice.

    :param data: (data object) The dataset for which the average volume is computed
    :param z: (int) Slice number.
    :param mask: (ndarray) Mask specifying which volume contributes to the computation.
    :param s: (int) Smoothing kernel size in pixel (default=0, no smoothing)

    :returns: Average volume.

    """
    average_vol = np.zeros(data.volshape, dtype=data.format)
    if mask is not None:
        nVol = mask[:, :, z - 1].sum()
    else:
        nX, nY, nZ = data.gridshape[:]
        nVol = nX * nY
        mask = np.ones((nX, nY, nZ))

    # Loop over volumes
    for vol in data.sliceIterator(z, mask=mask):
        if s > 0:  # Smoothing volume in xy
            vol = gaussian_filter(vol, sigma=(s, s, 0))

        average_vol += vol / (1.0 * nVol)

    return average_vol


def find_focalDepth(vol):
    """Detects the focal plane depth in a volume.

    Parameters
    ----------
    vol : ndarray
        Volume in which the focal plane is detected

    Returns
    -------
    int
        The focal plane depth

    """
    # Averaging intensity slice-by-slice
    intensityProfile = np.mean(np.mean(vol, axis=0), axis=0)

    # Focal plane depth
    fz = np.argmax(intensityProfile)

    return fz


def iProfilePieceWiseModel(z, I0, Imax, z0, zf, s, mu, k):
    iProfile = np.zeros(z.shape)
    z1 = z <= z0
    z2 = (z > z0) * (z <= zf)
    z3 = z > zf

    # Water above tissue
    iProfile[z1] = I0 * np.exp(-k * z[z1])

    # Tissue -> Focal plane area
    iProfile[z2] = Imax * np.exp(-((z[z2] - zf) ** 2) / s**2)

    # Attenuation area
    iProfile[z3] = Imax * np.exp(-mu * (z[z3] - zf))

    return iProfile


def glmVolumeNormalization(vol, average_vol):
    """Volume intensity normalization using GLM fit

    Parameters
    ----------
    vol : ndarray
        Volume to normalize

    average_vol : ndarray
        Average volume representing the background or objective profile without tissue

    Returns
    -------
    ndarray
        Normalized volume

    """
    nx, ny, nz = vol.shape
    vol_p = np.zeros_like(vol)

    # Preparing the input variables
    X = np.zeros((nx * ny * nz, 4))
    xx, yy, zz = np.meshgrid(
        list(range(nx)), list(range(ny)), list(range(nz)), indexing="ij"
    )
    X[:, 0] = np.reshape(xx, (nx * ny * nz,))
    X[:, 1] = np.reshape(yy, (nx * ny * nz,))
    X[:, 2] = np.reshape(zz, (nx * ny * nz,))
    X[:, 3] = np.reshape(average_vol, (nx * ny * nz,))
    y = np.reshape(vol, (nx * ny * nz,))

    regr = linear_model.BayesianRidge()
    regr.fit(X, y)
    meanv_p = np.reshape(regr.predict(X), (nx, ny, nz))
    vol_p = vol / meanv_p

    #     iMax = vol.max()
    #     iMin = vol.min()
    #     iMax_p = vol_p.max()
    #     iMin_p = vol_p.min()
    #
    #     vol_p = (iMax-iMin)*(vol_p - iMin_p)/(iMax_p-iMin_p) + iMin

    #     # Loop over each volume slices
    #     for z in range(nz):
    #         X[:,2] = np.reshape(average_vol[:,:,z], (nx*ny,))
    #         y = np.reshape(vol[:,:,z], (nx*ny,))
    #         #regr = linear_model.BayesianRidge()
    #         regr = linear_model.LinearRegression()
    #         regr.fit(X, y)
    #         meanv_p = np.reshape(regr.predict(X), (nx,ny))
    #
    #         vol_p[:,:,z] = vol[:,:,z]/meanv_p
    #
    #         iMax = vol[:,:,z].max()
    #         iMin = vol[:,:,z].min()
    #         iMax_p = vol_p[:,:,z].max()
    #         iMin_p = vol_p[:,:,z].min()
    #
    #         vol_p[:,:,z] = (iMax-iMin)*(vol_p[:,:,z] - iMin_p)/(iMax_p-iMin_p) + iMin

    return vol_p


def volumeNormalization(vol, average_vol, epsilon=0.05):
    """Volume intensity normalization.

    Parameters
    ----------
    vol : ndarray
        Volume to normalize

    average_vol : ndarray
        Average volume representing the background or objective profile without tissue

    epsilon: float (optional, 0 < epsilon < 1)
        Small constant to prevent zero-division

    Returns
    -------
    ndarray
        Normalized volume.

    """

    # Adjust volume intensity to be in range [0,1]
    # vol = (vol - vol.min())/(vol.max() - vol.min())

    # Adjust average volume intensity to be in range [0,1]
    # meanv = (average_vol - average_vol.min())/(average_vol.max() - average_vol.min())
    meanv = average_vol

    # Average intensity in first slices should be the same in both volumes (we assume that there are no structures in the first slices)
    # i0_v = vol[:,:,0:15].mean()
    # i0_mv = meanv[:,:,0:15].mean()
    # meanv = (meanv/i0_mv)*i0_v # Normalized average volume in same I scale as vol

    # Apply normalization (division)
    vol = vol / (meanv + epsilon) * meanv.mean()  #

    return vol


# Defining the radiometric transformation function
def T_r(p, x, y):
    """Radiometric transformation function

    Parameters
    ----------
    p : tuple
        Radiometic function parameters to evaluate. (Should be a (6,) tuple)

    x : int or ndarray
        X position where the function is evaluated (int or result of meshgrid)

    y : int or ndarray
        Y position where the function is evaluated (int or result of meshgrid)

    Returns
    -------
    int or ndarray
        Function evaluated at given positoin

    """
    return p[0] * x**2 + p[1] * x * y + p[2] * y**2 + p[3] * x + p[4] * y + p[5]


# Defining objective function f_r
def f_r(x, data, z, pos, Imean):
    """Global radiometric objective function.

    Parameters
    ----------
    x : tuple
        Radiometic function parameters to evaluate. (Should be a (6,) tuple)

    data : (slicecode.utils.FileUtils.data object)
        Data object used to iterate over volumes or images.

    z : int
        Slice number over which the function is evaluated.

    pos : ndarray
        Volume absolute positions computed with the shift_oct module

    Imean : (float)
        Average volume / image intensity across all tiles.

    Returns
    -------
    float
        Objective function evaluation for parameters x.

    Notes
    -----
        This method is an implementation of [Sun2006](http://onlinelibrary.wiley.com/doi/10.1111/j.1365-2818.2006.01687.x/full)

    """
    f = 0  # Initial value

    # Transform function
    nx, ny = data.volshape[:2]
    xx, yy = np.meshgrid(list(range(nx)), list(range(ny)), indexing="ij")
    T = T_r(x, xx, yy)

    # Loop over overlaps
    for vol1, vol2, pos1, pos2 in data.neighborSliceIterator(z, returnPos=True):
        # Getting overlap regions
        realPos1 = pos[pos1[0] - 1, pos1[1] - 1, :]
        realPos2 = pos[pos2[0] - 1, pos2[1] - 1, :]
        ov1, ov2, pov1, pov2 = getOverlap(vol1, vol2, realPos1, realPos2)

        # Getting AIP of overlap regions
        im1 = np.squeeze(ov1.mean(axis=2))
        im2 = np.squeeze(ov2.mean(axis=2))

        # Evaluation T for overlap regions
        T_im1 = T[pov1[0] : pov1[2], pov1[1] : pov1[3]]
        T_im2 = T[pov2[0] : pov2[2], pov2[1] : pov2[3]]

        # Updating function evaluation
        f += np.sum(np.abs(im1 * T_im1 - im2 * T_im2))

    # Loop over tiles
    for vol in data.sliceIterator(z):
        im = np.squeeze(vol.mean(axis=2))
        f += np.sum(np.abs(im * T - Imean))

    return f


def matchHistogram(im1, im2, returnTransforms=False):
    """Match im2 and im1 histograms

    Parameters
    ----------
    im1: ndarray
        Reference image used as target

    im2: ndarray
        Image to be adjusted

    returnTransforms: bool
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

        >> V_inv, T = matchHistogram(im1, im2, returnTransform=True)
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

    if returnTransforms:
        return V_inv, T
    else:
        # Correcting im2
        im2p = V_inv(T(im2))
        return im2p


def matchHistogramSequentially(data, preproc_data, abspos, z, overwrite=False):
    """Match neighbor tiles histograms sequentially

    Parameters
    ----------
    data : data object
        Used for iteration and to
    """
    firstVol = True
    for vol1, vol2, pos1, pos2 in data.singlePassNeighborSliceIterator(
        (1, 1), z, method="dfs"
    ):
        realPos1 = abspos[pos1[0] - 1, pos1[1] - 1, :]
        realPos2 = abspos[pos2[0] - 1, pos2[1] - 1, :]
        ov1, ov2, _, _ = getOverlap(vol1, vol2, realPos1, realPos2)
        V_inv, T = matchHistogram(ov1, ov2, returnTransforms=True)
        vol2p = V_inv(T(vol2))

        if firstVol:
            preproc_data.saveVolume(vol1, pos1, overwrite)
            firstVol = False

        preproc_data.saveVolume(vol2p, pos2, overwrite)


def getSmoothIntensityTransition(vol, slicesStart):
    """Uses a regularization function to get a smooth intensity transition between adjacent slices.

    Parameters
    ==========
    vol : ndarray
        Volume containing the slice to adjust

    slicesStart : list(int)
        List of slice position, corresponding to the slice transition location

    compensateAttenuation : bool
        If true, compensation Beer-Lambert attenuation before the regularization (using a division by a low-pass version of the slice).

    Returns
    =======
    ndarray
        Adjusted volume.


    References
    ==========
    * Wang, H., et al. (2014). Serial optical coherence scanner for large-scale brain imaging at microscopic resolution. NeuroImage, 84, 1007–1017. http://doi.org/10.1016/j.neuroimage.2013.09.063

    """
    volume = np.copy(vol)
    epsilon = 1e-3
    # volume += epsilon # To remove division by zero errors
    nx, ny, nz = volume.shape

    # Get position and size of each slices
    nSlices = len(slicesStart)
    sliceRange = np.zeros((nSlices, 3), dtype=int)
    sliceRange[:, 0] = slicesStart
    sliceRange[0:-1, 1] = sliceRange[1::, 0]
    sliceRange[-1, 1] = nz
    sliceRange[:, 2] = sliceRange[:, 1] - sliceRange[:, 0]

    # Loop over slice transitions
    for i in range(nSlices):
        z1 = sliceRange[i, 0]
        z2 = sliceRange[i, 1]

        # Creating the regularization function L(z)
        a_current = gaussian_filter(
            volume[:, :, z1], sigma=5
        )  # First z of current slice local intensity average
        if i + 1 == nSlices:
            a_next = gaussian_filter(
                volume[:, :, z2 - 1], sigma=5
            )  # First z of next slice local intensity average
        else:
            a_next = gaussian_filter(
                volume[:, :, z2], sigma=5
            )  # First z of next slice local intensity average

        b_current = gaussian_filter(
            volume[:, :, z2 - 1], sigma=5
        )  # Last z of current slice local intensity average
        if i == 0:  # If first slice
            b_previous = gaussian_filter(
                volume[:, :, z1], sigma=5
            )  # Last z of current slice local intensity average
        else:
            b_previous = gaussian_filter(
                volume[:, :, z1 - 1], sigma=5
            )  # Last z of previous slice local intensity average

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


def getAttenuation_Vermeer2013(vol, dz=6.5e-6, mask=None, C=None):
    """Estimates the attenuation coefficient using the Vermeer2013 model.

    Parameters
    ----------
    vol : ndarray
        3D Reflectivity OCT data
    dz : float
        Axial resolution (in microns/pixel)
    mask : ndarray
        Tissue mask (optional). Every pixel above the mask will be attributed null attenuation,
        and pixel under the mask will be set to the last computed Aline attenuation.
    C: ndarray, int or float
        The bottom constant (DEV)

    Return
    ------
    ndarray
        Estimated attenuation coefficient map (same size as vol)

    Notes
    -----
    - This algorithm is inspired by an ultrasound attenuation compensation method.

    References
    ----------
    - Vermeer et al. Depth-resolved model-based reconstruction of attenuation
      coefficients in optical coherence tomography. Biomed. Opt. Exp., vol5,
      no1, pp332-337, 2013

    """
    # Prepare the bottom constant (to better consider the finite Bscan dimension)
    if C is None:
        C = np.zeros(vol.shape)
    elif isinstance(C, int) or isinstance(C, float):
        C = np.ones(vol, dtype=float) * C
    elif C.ndim == 2:
        C = np.tile(np.reshape(C, (C.shape[0], C.shape[1], 1)), (1, 1, vol.shape[2]))

    # Use mask to ignore tissue layers above / under the mask
    if mask is None:
        mask = np.ones_like(vol, dtype=bool)

    # Compensation profile with depth for each A-line
    vol_p = np.ma.masked_array(vol, ~mask)
    profile = np.cumsum(vol_p[:, :, ::-1], axis=-1) + C
    profile = profile[:, :, ::-1]

    # Estimating the attenuation coefficient for each A-line
    mu = np.zeros_like(vol)
    mu[profile > 0] = vol[profile > 0] / profile[profile > 0]
    mu[profile > 0] = np.log(1 + mu[profile > 0]) / (2.0 * dz)

    # Convert attenuation in cm-1
    mu /= 100.0

    # Masking the result
    if mask is not None:
        interface = xyzcorr.getInterfaceDepthFromMask(mask)
        mask_aboveInterface = ~xyzcorr.maskUnderInterface(
            mask, interface, returnMask=True
        )
        mu[mask_aboveInterface] = 0

        # Find bottom interface
        nx, ny, nz = mask.shape
        bottom_interface = (
            nz - xyzcorr.getInterfaceDepthFromMask(mask[:, :, ::-1]) - 1
        ).astype(int)
        xx, yy = np.meshgrid(list(range(nx)), list(range(ny)), indexing="ij")
        bottom_mu = mu[xx, yy, bottom_interface]
        bottom_mu = np.tile(np.reshape(bottom_mu, (nx, ny, 1)), (1, 1, nz))
        bottom_mask = xyzcorr.maskUnderInterface(
            mask, bottom_interface, returnMask=True
        )
        mu[bottom_mask] = bottom_mu[bottom_mask]

    return mu


def get_extendedAttenuation_Vermeer2013(vol, mask=None, k=10, sigma=5,
                                        sigma_bottom=3, dz=1, res=6.5,
                                        zshift=3, fillHoles=False):
    """Compute the local effective tissue attenuation using
    the extended Vermeer model.

    Parameters
    ----------
    vol: ndarray
        OCT volume to process
    mask: ndarray
        Optional tissue mask. If none is given the water/tissue
        interface will be detected from the data.
    k: int
        Median filter kernel size (px) applied before the attenuation
        coefficient computation (applied in the XY direction).
    sigma: int
        Gaussian filter kernel size (px) applied axially before the
        exponential signal fit used to extend the Alines for the extended
        Vermeer signal evalution.
    dz: int
        Number of axial pixel to consider when computing the bottom slice
        signal for the signal extension.
    res: float
        Axial resolution in micron / pixel
    zshift: int
        Number of pixel under the water-tissue interface to ignore while
        fitting the exponential function for signal extension.

    Returns
    -------
    ndarray
        Computed attenuation coefficients.
    """
    # First the slice is denoised with a small median filter
    if k > 0:
        vol = sitk.GetArrayFromImage(
            sitk.Median(sitk.GetImageFromArray(vol), (0, k, k))
        )

    # Computing tissue mask
    if mask is None:
        # Detecting the water / tissue interface
        interface = xyzcorr.findTissueInterface(
            vol, s_xy=3, s_z=1, order=1, useLog=True
        )
        mask = xyzcorr.maskUnderInterface(vol, interface + zshift, returnMask=True)

    # Lets fit an exponential function on each Aline to extend the tissue slice.
    exp_fit = get_gradientAttenuation(gaussian_filter(vol, (0, 0, sigma)))
    exp_fit = np.ma.masked_array(exp_fit, ~mask).mean(axis=2)

    # Fill holes left by NaN values
    exp_fit[np.isnan(exp_fit)] = 0
    # exp_fit = gaussian_filter(exp_fit, sigma_bottom);

    # Get the signal at the interface for each Aline
    interface_bottom = (
        vol.shape[2] - xyzcorr.getInterfaceDepthFromMask(mask[:, :, ::-1]) - 1 - dz
    )
    mask_bottom = xyzcorr.maskUnderInterface(vol, interface_bottom, returnMask=True)
    mask_bottom = (mask_bottom * mask).astype(bool)
    i0 = np.ma.masked_array(vol, ~mask_bottom).mean(axis=2)
    # i0 = gaussian_filter(i0, sigma_bottom)

    # Compute the end-of-scan bias
    epsilon = 1e-3
    C = np.zeros_like(i0)
    C[exp_fit > epsilon] = i0[exp_fit > epsilon] / exp_fit[exp_fit > epsilon]
    C = gaussian_filter(C, sigma_bottom)

    # Compute the attenuation
    attn_cropped = getAttenuation_Vermeer2013(vol, dz=res * 1e-06, mask=mask, C=C)

    # Remove NaN
    attn_cropped[np.isnan(attn_cropped)] = 0

    # Only keep attn within the mask
    attn_cropped[~mask.astype(bool)] = 0

    # Fill holes
    if fillHoles:
        attn_cropped = sitk.GetArrayFromImage(
            sitk.GrayscaleFillhole(sitk.GetImageFromArray(attn_cropped))
        )

    return attn_cropped


def getAttenuation_Faber2004(vol, mask=None, dz=6.5e-6, N=4):
    """Estimates the attenuation coefficient using the Faber2004 model.
    Parameters
    ----------
    vol : ndarray
        3D Reflectivity OCT data
    mask : ndarray
        Tissue mask to control which points to use in each A-lines. (if None, the whole A-line is used)
    dz : float
        Axial resolution (in microns/pixel)
    N : int
        Size of the XY uniform filter used to average A-lines together

    Return
    ------
    ndarray
        Estimated attenuation coefficient map (computes a single mu_t per A-line)

    Notes
    -----
    - This algorithm uses the confocal PSF and an single-scattering photon model.
    - Assumes a 4X objective setup for now.

    References
    ----------
    - Faber et al. Quantitative measurement of attenuation coefficients of weakly
     scattering media using optical coherence tomography. Opt. Express 12, 4353–4365 (2004).
    """
    # Average the A-line together in the XY plane
    if N > 0:
        vol = uniform_filter(vol, size=(N, N, 0))

    # Computing the confocal psf parameters
    alpha = 2  # 1 : specular scattering; 2 : diffuse scattering
    n = 1.33  # Index of refraction
    w0 = 4.88e-6  # micron : Ligth beam width at the focal plane (need to validate)
    l0 = 1.030e-6  # micron : Ligth central wavelenth
    zr = np.pi * alpha * n * w0**2.0 / l0  # Apparent Rayleigh length (micron)

    # Defining the objective function for the minimization
    f = lambda x, y, z: np.sum(
        (y - octSignal_Faber2004Model(z, mu_t=x[2], zR=x[1], z0=x[0])) ** 2.0
    )

    # Loop over all A-lines and computing attenuation
    attn = np.zeros((vol.shape[0], vol.shape[1]))
    rLength = np.zeros((vol.shape[0], vol.shape[1]))
    z_focus = np.zeros((vol.shape[0], vol.shape[1]))
    z = np.arange(0.0, dz * vol.shape[2], dz)
    for x in range(vol.shape[0]):
        for y in range(vol.shape[1]):
            if mask is not None:
                mask_Aline = mask[x, y, :]
            else:
                mask_Aline = np.ones((vol.shape[2],)).astype(np.bool)

            if np.any(mask_Aline):
                p0 = [0.0, 100.0, 0.001]
                zp = np.where(mask_Aline)[0][0]
                data = vol[x, y, :][mask_Aline]
                data /= 1.0 * data.max()
                p_opt = minimize(
                    f,
                    p0,
                    args=(data, z[zp::] - z[zp]),
                    bounds=((None, None), (zr / 2.0, 2.0 * zr), (0.0, None)),
                )
                if p_opt.success:
                    attn[x, y] = p_opt.x[2]
                    rLength[x, y] = p_opt.x[1]
                    z_focus[x, y] = p_opt.x[0] + z[zp]
                else:
                    print(("No convergence for (x,y)=", x, y))

    return attn, rLength, z_focus


# Modele du signal utilisant la PSF confocale et single-scattering photons
def octSignal_Faber2004Model(z, mu_t=1.0, zR=200.0, z0=100.0):
    """Model the oct signal using a single-scattered photons and the confocal PSF
    Parameters
    ----------
    z : (N,) ndarray
        Depth Position along an A-line at which the signal must be computed (in micron)
    mu_t : float or (N,) ndarray
        Attenuation coefficient (either a single value for the whole column or N values for each position)
    zR : float
        Apparent Rayleigh length (in micron)
    z0 : float
        Focal plane depth (in micron)

    Returns
    -------
    ndarray
        OCT backscattering signal estimated at each z positions

    Notes
    -----
    - This is based on the single-scattering photon model from Faber2004.

    References
    ----------
    - https://www.osapublishing.org/oe/abstract.cfm?uri=oe-12-19-4353

    """
    psf = 1.0 / (((z - z0) / float(zR)) ** 2.0 + 1.0)
    iz = np.sqrt(psf * np.exp(-2 * mu_t * z))  # Should I fit the sqrt or not
    return iz


def _AlineFit(data):
    """Aline fit to extract the attenuation coefficient"""

    # Defining the attenuation model (biexponential) with :
    # x : [A, mu_t]; y : data; z : depths
    f_attn = lambda x, y, z: np.sum((y - x[0] * np.exp(-2 * x[1] * z)) ** 2.0)

    z = np.linspace(0, len(data), len(data))
    aline = np.array(data)
    p0 = [1.0, 0.001]  # Initial condition
    popt = minimize(f_attn, p0, args=(aline, z), bounds=((0, None), (0, None)))
    return popt.x[1]


def splitAline(data, mask):
    data_list = list()
    z_list = list()
    this_aline = list()
    this_z = list()
    for elem, m, z in zip(data, mask, list(range(len(data)))):
        if m:
            this_aline.append(elem)
            this_z.append(z)
        else:
            if len(this_aline) > 0:
                data_list.append(this_aline)
                this_aline = list()
                z_list.append(this_z)
                this_z = list()
    if len(this_aline) > 0:
        data_list.append(this_aline)
        z_list.append(this_z)

    return data_list, z_list


def _splitAlinesWorker(param):
    return splitAline(param[0], param[1])


def get_gradientAttenuation(
    vol,
    mask=None,
    return_mask=False,
    lowThresh=0.0,
    fillHoles=False,
    sz=3,
    sxy=0,
    res=1.0,
):
    vol_l = np.copy(vol)
    vol_l[vol > 0] = np.log(vol[vol > 0])
    attn = -0.5 * np.gradient(vol_l, res, axis=2)

    # Removing 0 values and masked values
    attn[vol == 0] = 0
    if mask is not None:
        attn[~mask] = 0

    # Removing negative attenuation values
    mask_attn = attn < 0
    attn[mask_attn] = 0

    # Removing also small attenuation values
    small_mask = attn < np.percentile(attn[attn > 0], lowThresh)
    mask_attn[small_mask] = True
    attn[mask_attn] = 0

    # Filling holes using morphological reconstruction.
    if fillHoles:
        vol_itk = sitk.GetImageFromArray(attn)
        cfilter = sitk.GrayscaleMorphologicalClosingImageFilter()
        cfilter.SetKernelType(sitk.sitkBall)
        cfilter.SetKernelRadius((sz, sxy, sxy))

        output_itk = cfilter.Execute(vol_itk)
        # output_recons_itk = sitk.ReconstructionByErosion(output_itk, vol_itk)
        attn = sitk.GetArrayFromImage(output_itk)

    if return_mask:
        return attn, mask_attn
    else:
        return attn


def getInterfaceMask(vol, s=0, maskTissue=True, maskWaterTissueInterface=True):
    nx, ny, nz = vol.shape
    mask = np.ones_like(vol).astype(bool)

    # Get tissue mask
    if maskTissue:
        tissueMask = binary_fill_holes(
            median_otsu(eqhist(vol.mean(axis=2)), median_radius=5.0)[1]
        )
        tissueMask = np.tile(np.reshape(tissueMask, (nx, ny, 1)), (1, 1, nz))
        mask *= tissueMask

    # Get water/tissue 3D interface mask
    if maskWaterTissueInterface:
        # Computing the gradient
        vol_f = median_filter(vol, 5)
        gradient = np.gradient(vol_f)
        gm = gradient[0] ** 2.0 + gradient[1] ** 2.0 + gradient[2] ** 2.0
        gm = normalize(gm, highThresh=99.5)

        # Thresholding the gradient
        thresh = threshold_li(gm)
        interfaces_g = gm >= thresh

        # Converting this into a water/tissue interface depth
        depths = np.zeros((nx, ny))
        for x, y in itertools.product(list(range(nx)), list(range(ny))):
            idx = np.where(interfaces_g[x, y, :])
            if len(idx[0]) > 0:
                depths[x, y] = idx[0][0]

        waterTissueMask = xyzcorr.maskUnderInterface(vol, depths, returnMask=True)
        mask *= waterTissueMask

    # Smoothing the volume
    if s > 0:
        vol = gaussian_filter(vol, sigma=(s, s, s))

    # Get tissue interface boundaries using a Canny filter
    vol_itk = sitk.GetImageFromArray(vol)
    canny_filter = sitk.CannyEdgeDetectionImageFilter()
    # canny_filter.SetVariance((3,3,3))
    edges = sitk.GetArrayFromImage(canny_filter.Execute(vol_itk)).astype(bool)
    mask *= ~edges

    return mask


def findInterfaceFromGradient(vol, f=0.005, removeSmooth=False):
    nx, ny = vol.shape[0:2]
    k = int(np.round(f * 0.5 * (nx + ny)))

    # Computing the gradient
    vol_f = gaussian_filter(vol, k)

    gradient = np.gradient(vol_f)
    gm = gradient[0] ** 2.0 + gradient[1] ** 2.0 + gradient[2] ** 2.0
    gm = normalize(gm, highThresh=99.5)

    # Thresholding the gradient
    thresh = threshold_li(gm)
    interfaces_g = gm >= thresh

    # Converting this into a water/tissue interface depth
    depths = np.argmax(interfaces_g, axis=2)

    if removeSmooth:
        depths += k + 1

    return depths


def getFlatAgaroseProfile(vol, returnMaskAndProfile=False):
    nx, ny, nz = vol.shape

    # Get agarose mask for this slice and its intensity profile
    tissueMask = binary_fill_holes(
        median_otsu(eqhist(vol.mean(axis=2)), median_radius=5.0)[1]
    )
    tissueMask = np.tile(np.reshape(tissueMask, (nx, ny, 1)), (1, 1, nz))
    mask = (~tissueMask).astype(bool)

    # Computing intensity profile for the agarose Alines
    iProfile = np.zeros((nz,))
    nAlines = np.sum(mask.max(axis=2))
    for x in range(nx):
        for y in range(ny):
            this_mask = mask[x, y, :]
            Aline = vol[x, y, :]
            if np.any(this_mask):
                iProfile += Aline / float(nAlines)

    # Correction profile
    agarose_norm = np.tile(np.reshape(iProfile, (1, 1, nz)), (nx, ny, 1))
    vol_p = vol / agarose_norm.astype(float)

    if returnMaskAndProfile:
        return vol_p, mask, iProfile
    else:
        return vol_p


def getSignalFromAttenuation(attn, i0=None, nz=120, mask=None, res=1.0):
    """Estimate the signal from the 2D A-Line attenuation map

    Parameters
    ----------
    attn : ndarray
        2D attenuation coefficient map for each A-line
    i0 : ndarray
        2D map of the top slice signal (optional)
    nz : int
        Number of Aline points
    mask : ndarray
        Data mask (to limit where the oct signal is simulated)
    res :

    Returns
    -------
    ndarray
        Estimated OCT signal

    """
    nx, ny = attn.shape
    attn_vol = np.zeros((nx, ny, nz))
    f_attn = lambda x, z: np.exp(-2 * x * z)
    for ix, iy in itertools.product(list(range(nx)), list(range(ny))):
        if mask is not None:
            this_mask = mask[ix, iy, :].astype(bool)
        else:
            this_mask = np.ones((nz,), dtype=bool)

        z0 = np.where(this_mask)
        if len(z0[0]) > 0:
            z0 = z0[0][0]
        else:
            z0 = 0

        if i0 is not None:
            A = i0[ix, iy]
        else:
            A = 1

        if np.any(this_mask):
            this_mu = attn[ix, iy]
            z = np.linspace(0, nz * res, nz) - z0 * res
            simProfile = A * f_attn(this_mu, z)
            attn_vol[ix, iy, this_mask] = simProfile[this_mask]

    return attn_vol


def confocalPSF(z, zf, zR, A=None):
    """Confocal PSF model using a gaussian beam.

    Parameters
    ----------
    z : ndarray
        Depths at which the psf is evaluated
    zf : float
        Focal plane depth
    zR : float
        Rayleigh Length
    A : float
        Normalization constant

    Return
    ------
    ndarray
        PSF evaluated at each z locations
    """
    psf = 1.0 / (((z - zf) / float(zR)) ** 2.0 + 1.0)
    if A is not None:
        psf *= A
    return psf


def get_SliceResolutionsFromPSF(
    zf, zr, nz=120, spacing=(6.5, 6.5, 6.5), N=512, l=1.030
):
    res = np.zeros((nz,))
    z = np.linspace(0, nz * spacing[2], nz)
    w0 = np.sqrt(zr * l / np.pi)
    wz = w0 * np.sqrt(1 + ((z - zf) / float(zr)) ** 2.0)

    for ii in range(nz):
        zmin = -wz[ii] * 4
        zmax = wz[ii] * 4
        x = np.linspace(zmin, zmax, N)
        edge = x < 0
        psf = np.exp(-(x**2.0) / wz[ii] ** 2.0)

        edge = np.convolve(edge, psf, mode="same")
        edge /= edge.max()

        x90 = x[np.where(edge > 0.90)[0][-1]]
        x10 = x[np.where(edge < 0.1)[0][0]]
        this_res = x10 - x90
        res[ii] = this_res

    return res


def estimatePSF(
    agarose,
    interface=None,
    dz=6.5,
    removeSaturation=False,
    maskInterface=False,
    zf=None,
    fitAttn=False,
):
    """Estimates the confocal PSF assuming a gaussian beam

    Parameters
    ----------
    agarose : ndarray
        Agarose volume
    interface : ndarray
        Water-Tissue interface map
    dz : float
        Z spacing in micron
    removeSaturation : Bool
    maskInterface : Bool

    Returns
    -------
    float
        Focal plane depth
    float
        Rayleigh length
    float
        Normalization constant

    Notes
    -----
    * If no interface is given, the whole volume is used for the psf regression
    """
    if agarose.ndim == 1:
        iProfile = np.copy(agarose)
    else:
        iProfile = agarose.mean(axis=(0, 1))
    nz = len(iProfile)
    z = np.linspace(0, len(iProfile) * dz, len(iProfile))

    # Only fit the tissue intensity (if interface is given)
    if interface is not None:
        z0 = np.median(interface)
        iProfile = iProfile[z0::]
        z = z[z0::]
    else:
        z0 = 0

    if maskInterface:
        m = gaussian_filter1d(iProfile, sigma=1, order=1)
        m = m < 0
        m = binary_erosion(m, iterations=1)

        iProfile = iProfile[m]
        z = z[m]

    # Remove outliers in the tissue
    if removeSaturation:
        i_diff = np.diff(iProfile)
        i_diff = np.insert(i_diff, 0, 0)
        z_diff = np.copy(z)

        med = np.median(i_diff)
        MAD = np.median(np.abs(i_diff - med))
        if MAD != 0:
            Zscore = np.abs(0.6745 * (i_diff - med) / MAD)
            outliers = Zscore > 3.5
        else:
            outliers = np.zeros_like(i_diff)

        iProfile = iProfile[~outliers]
        z = z_diff[~outliers]

    # Normalization
    iProfile /= iProfile.max()

    # Fitting model
    if zf is None:
        if fitAttn:
            fo_psf = lambda x, y, z: np.sum(
                (y - confocalPSF(z, x[0], x[1], x[2]) * np.exp(-2 * x[3] * (z - z[0])))
                ** 2.0
            )
        else:
            fo_psf = lambda x, y, z: np.sum(
                (y - confocalPSF(z, x[0], x[1], x[2])) ** 2.0
            )

        # 1st fit of the model
        zf = 0.5 * (nz * dz)
        zR = 250.0
        A = 1.0
        p0 = [zf, zR, A]  # Initial parameters
        param_bounds = [(0.0, zf * dz), (0.0, None), (0.0, 1.0)]
        if fitAttn:
            p0.append(0.0)
            param_bounds.append((0.0, None))
        popt_1 = minimize(fo_psf, p0, args=(iProfile, z), bounds=param_bounds)

        # Detect outliers
        if fitAttn:
            I_err = (
                iProfile
                - confocalPSF(z, popt_1.x[0], popt_1.x[1], popt_1.x[2])
                * np.exp(-2 * popt_1.x[3] * (z - z[0]))
            ) ** 2.0
        else:
            I_err = (
                iProfile - confocalPSF(z, popt_1.x[0], popt_1.x[1], popt_1.x[2])
            ) ** 2.0
        err_med = np.median(I_err)
        err_MAD = np.median(np.abs(I_err - err_med))
        if err_MAD != 0:
            err_Zscore = np.abs(0.6745 * (I_err - err_med) / err_MAD)
            err_outliers = err_Zscore > 3.5
        else:
            err_outliers = np.zeros_like(I_err)

        # 2nd fit without outliers
        p0 = popt_1.x
        popt_2 = minimize(
            fo_psf,
            p0,
            args=(np.array(iProfile)[~err_outliers], np.array(z)[~err_outliers]),
            bounds=param_bounds,
        )

        if fitAttn:
            return popt_2.x[0], popt_2.x[1], popt_2.x[2], popt_2.x[3]
        else:
            return popt_2.x[0], popt_2.x[1], popt_2.x[2]
    else:
        fo_psf = lambda x, y, z, zf: np.sum((y - confocalPSF(z, zf, x[0], x[1])) ** 2.0)

        # 1st fit of the model
        zR = 250.0
        A = 1.0
        p0 = [zR, A]  # Initial parameters
        popt_1 = minimize(
            fo_psf, p0, args=(iProfile, z, zf), bounds=((0.0, None), (0.0, 1.0))
        )

        # Detect outliers
        I_err = (iProfile - confocalPSF(z, zf, popt_1.x[0], popt_1.x[1])) ** 2.0
        err_med = np.median(I_err)
        err_MAD = np.median(np.abs(I_err - err_med))
        if err_MAD != 0:
            err_Zscore = np.abs(0.6745 * (I_err - err_med) / err_MAD)
            err_outliers = err_Zscore > 3.5
        else:
            err_outliers = np.zeros_like(I_err)

        # 2nd fit without outliers
        p0 = popt_1.x
        popt_2 = minimize(
            fo_psf,
            p0,
            args=(np.array(iProfile)[~err_outliers], np.array(z)[~err_outliers], zf),
            bounds=((0.0, None), (0.0, 1.0)),
        )

        return zf, popt_2.x[0], popt_2.x[1]


def get3DPSF(
    vol, interface, res=6.5, useAverageRayleigh=False, removeInterface=True, zf=None
):
    """Compute a 3D PSF from a given uniform volume (e.g. agarose)

    Parameters
    ----------
    vol : ndarray
        Agarose / Background volume to use for the PSF computation
    interface : ndarray
        Water/Tissue interface depth map (in pixel)
    res : float
        Z axis resolution (in micron / pixel)
    useAverageRayleigh : bool
        Use average Rayleigh length instead of the Rayleigh map.

    Returns
    -------
    ndarray
        PSF
    ndarray
        Focal depth map
    ndarray
        Rayleigh length map
    """

    nx, ny, nz = vol.shape
    zf_map = np.zeros((nx, ny))
    zr_map = np.zeros((nx, ny))

    print("Computing PSF parameters for each aline")
    for ix in range(nx):
        for iy in range(ny):
            Aline = np.array(vol[ix, iy, :])
            Aline_interface = np.squeeze(interface[ix, iy])
            try:
                if zf is None:
                    params = estimatePSF(
                        Aline,
                        interface=Aline_interface,
                        dz=res,
                        maskInterface=removeInterface,
                    )
                else:
                    params = estimatePSF(
                        Aline,
                        interface=Aline_interface,
                        dz=res,
                        maskInterface=removeInterface,
                        zf=zf[ix, iy],
                    )
                zf_map[ix, iy] = params[0]
                zr_map[ix, iy] = params[1]
            except:
                pass

    # Smoothing the zf and zr maps
    print("Smoothing the zf and zr maps")
    if zf is None:
        zf_map = gaussian_filter(zf_map, (nx * 0.1, ny * 0.1))
    zr_map = gaussian_filter(zr_map, (nx * 0.1, ny * 0.1))

    # Fit parabola on zf_map
    f = (
        lambda x, a, b, c, d, e, f: a * x[0] * x[1]
        + b * x[0] ** 2
        + c * x[1] ** 2
        + d
        + e * x[0]
        + f * x[1]
    )
    xx, yy = np.meshgrid(list(range(nx)), list(range(ny)), indexing="ij")
    xdata = (np.ravel(yy), np.ravel(xx))
    ydata = np.ravel(zf_map)
    popt, _ = curve_fit(f, xdata, ydata)
    a, b, c, d, e, f = popt
    print(popt)
    zf_map = np.reshape(
        a * xdata[0] * xdata[1]
        + b * xdata[0] ** 2.0
        + c * xdata[1] ** 2.0
        + d
        + e * xdata[0]
        + f * xdata[1],
        (nx, ny),
    )

    # Computing this PSF
    print("Creating the 3D PSF")
    z = np.linspace(0, nz * res, nz)
    psf = np.zeros_like(vol)
    zr_ave = np.mean(zr_map)
    for ix in range(nx):
        for iy in range(ny):
            if useAverageRayleigh:
                psf[ix, iy, :] = confocalPSF(z, zf_map[ix, iy], zr_ave, 1.0)
            else:
                psf[ix, iy, :] = confocalPSF(z, zf_map[ix, iy], zr_map[ix, iy], 1.0)

    return psf, zf_map, zr_map


def vignette_gauss(pos, x0, y0, sx, sy, a, b):
    return (
        np.exp(
            -((pos[0] - x0) ** 2) / (2.0 * sx**2.0)
            - (pos[1] - y0) ** 2 / (2.0 * sy**2.0)
        )
        * a
        + b
    )


def vignette_gauss_lin(pos, x0, y0, s, a, b, c):
    gauss_surf = np.exp(
        -((pos[0] - x0) ** 2) / (2.0 * s**2.0) - (pos[1] - y0) ** 2 / (2.0 * s**2.0)
    )
    lin_surf = pos[0] * a + pos[1] * b + c
    return gauss_surf * lin_surf


def vignette_quad(pos, a, b, c, d, e, f):
    x = pos[0]
    y = pos[1]
    return a * x + b * y + c * x * y + d * x**2 + e * y**2 + f


def get_vignette(vol, returnParams=False, mask_z=None, method="gauss"):
    if method == "gauss":
        f_opt = lambda x, y, pos: np.mean(
            (y - vignette_gauss(pos, x[0], x[2], x[1], x[3], x[4], x[5])) ** 2.0
        )
    elif method == "gauss_lin":
        f_opt = lambda x, y, pos: np.mean(
            (y - vignette_gauss_lin(pos, x[0], x[1], x[2], x[3], x[4], x[5])) ** 2.0
        )
    else:
        f_opt = lambda x, y, pos: np.mean(
            (y - vignette_quad(pos, x[0], x[1], x[2], x[3], x[4], x[5])) ** 2.0
        )

    # Computing position in this mosaic.
    xx, yy = np.meshgrid(
        np.linspace(-1, 1, vol.shape[0]),
        np.linspace(-1, 1, vol.shape[1]),
        indexing="ij",
    )
    pos = [xx, yy]

    # Find center before
    img = vol.mean(axis=2)
    img /= float(img.max())
    if method == "gauss":
        p0 = [0.0, 0.5, 0.0, 0.5, 1.0, 0.0]
    elif method == "gauss_lin":
        p0 = [0.0, 0.0, 0.5, 0.0, 0.0, 1.0]
    else:
        p0 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    popt_0 = minimize(f_opt, p0, args=(img, pos))

    print(popt_0)

    w_list = list()
    params_list = list()
    if mask_z is None:
        mask_z = np.ones((vol.shape[2],))
    for z in range(vol.shape[2]):
        if mask_z[z]:
            # p0 = [popt_0.x[1], popt_0.x[3], 1.0, 0.0]
            p0 = popt_0.x
            # param_bounds = ((None,None), (1e-3, None), (None, None), (1e-3, None), (None,None), (None,None))
            img = vol[:, :, z]
            img /= float(img.max())

            popt = minimize(f_opt, p0, args=(img, pos))

            params_list.append(popt.x)
            w_list.append(f_opt(popt.x, img, pos))

    optimized_vignetteParams = np.median(np.array(params_list), axis=0)
    print((np.array(params_list)))
    print(optimized_vignetteParams)

    if returnParams:
        output = {"method": method, "params": optimized_vignetteParams}
        return output
    else:
        if method == "gauss":
            x0, sx, y0, sy, a, b = optimized_vignetteParams
            vignette = vignette_gauss(pos, x0, y0, sx, sy, 1, 0)
        elif method == "gauss_lin":
            x0, y0, s, a, b, c = optimized_vignetteParams
            vignette = vignette_gauss_lin(pos, x0, y0, s, a, b, c)
        else:
            a, b, c, d, e, f = optimized_vignetteParams
            vignette = vignette_quad(pos, a, b, c, d, e, f)
        return vignette


def removeHFIntensityArtifact(vol, sigma=5, mask=None):
    nx, ny, nz = vol.shape
    maxI = vol.max()
    minI = vol.min()
    vol_zeros = vol == 0

    # Compute Intensity depth profile
    if mask is None:
        iProfile = vol.mean(axis=(0, 1))
    else:
        iProfile = np.zeros((nz,))
        if mask.ndim == 2:
            for z in range(nz):
                iProfile[z] = np.mean(vol[:, :, z][mask])

        else:
            iProfile = np.zeros((nz,))
            for z in range(nz):
                iProfile[z] = np.mean(vol[:, :, z][mask[:, :, z]])

    # Low pass filter of the intensity profile
    lp_profile = gaussian_filter(iProfile, sigma)
    hf_profile = iProfile - lp_profile

    # Removing the hf component from the original data
    hf_3dprofile = np.tile(np.reshape(hf_profile, (1, 1, nz)), (nx, ny, 1))
    vol_p = vol - hf_3dprofile
    vol_p = (maxI - minI) * (vol_p - vol_p.min()) / float(
        vol_p.max() - vol_p.min()
    ) + minI
    vol_p[vol_zeros] = 0

    return vol_p.astype(vol.dtype)


def remove_reflection(vol, z0, radius=3):
    vol_p = np.copy(vol)
    nx, ny, nz = vol.shape
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)

    # Compute the zmin and zmax index used for the interpolation
    zmin = z0 - radius
    zmax = z0 + radius
    zList = list(range(zmin, zmax))
    z = np.delete(z, zList)
    vol_p = np.delete(vol_p, zList, axis=2)

    # 3D Interpolation
    xx, yy, zz = np.meshgrid(x, y, zList, indexing="ij")
    new_pos = np.stack((xx, yy, zz), axis=3)
    # f = interp1d(z_p, vol_p, kind='linear', axis=-1)
    vol_roi = interpn((x, y, z), vol_p, new_pos, method="linear")

    # Update the vol_p
    vol_p = np.concatenate((vol_p[:, :, 0:zmin], vol_roi, vol_p[:, :, zmax::]), axis=-1)

    return vol_p


def convert_to_8bit(vol):
    return (255 * (vol - vol.min()) / float(vol.max() - vol.min())).astype(np.uint8)


def fit_TissueConfocalModel(
    iprofile,
    z0,
    zr_0=400.0,
    res=6.5,
    useBumpModel=False,
    returnParameters=False,
    return_fullModel=False,
    plotProfiles=False,
    fix_zr=False,
):
    nz = len(iprofile)
    z = np.linspace(0, nz * res, nz)

    # Removing water DC background
    dc = np.min(iprofile[0:z0])
    this_profile = iprofile - dc

    # Normalizing the intensity profile
    imax = this_profile.max()
    this_profile = this_profile / float(imax)

    # Fit an initial tissue model (sigmoid)
    def tissue_model(x, z):
        c, z0, a = x[:]
        signal = a / (1 + np.exp(-c * (z - z0) / float(z[-1] - z[0]))).astype(float)
        return signal

    fo_signal = lambda x, y, z: np.sqrt(
        np.sum((y - tissue_model(x, z)) ** 2.0) / float(y.size)
    )
    p0 = [50.0, z0 * res, 1.0]  # c, z0, a
    popt_tissue = minimize(fo_signal, x0=p0, args=(this_profile, z))
    c, z0, a = popt_tissue.x[:]
    syn_tissue = tissue_model([c, z0, 1.0], z)
    new_z0 = popt_tissue.x[1]

    # Fit a static PSF (fixed zr, moving zf)
    def confocal_model(x, z):
        zf, zr, a = x[:]
        return confocalPSF(z, zf, zr, a)

    fo_PSF = lambda x, y, z, tissue: np.sqrt(
        np.sum((y - tissue * confocal_model(x, z)) ** 2.0) / float(y.size)
    )
    p0 = [z[-1] * 0.5, zr_0, 1.0]  # zf, zr, a
    param_bounds = [[z[0], z[-1]], [zr_0, zr_0], [0.0, None]]
    popt_firstpsf = minimize(
        fo_PSF, x0=p0, args=(this_profile, z, syn_tissue), bounds=param_bounds
    )
    zf, zr = popt_firstpsf.x[0:2]
    psf1 = confocal_model([zf, zr, 1.0], z)

    # Detect the specular reflection bump at the water / tissue interface
    if useBumpModel:

        def bumpModel(signal, w, b):
            t_grad = -gaussian_filter1d(signal, w, order=2)
            t_grad[t_grad < 0] = 0
            if t_grad.max() > 0:
                try:
                    t_grad /= float(t_grad.max())
                except:
                    pass
            bump = b * t_grad
            return bump

        def bumpTissueModel(x, z):
            z0, c, w, a, b = x[:]
            signal = tissue_model([c, z0, a], z)
            return signal + bumpModel(signal, w, b)

        fo_btm = lambda x, y, z, psf: np.sqrt(
            np.sum((y - psf * bumpTissueModel(x, z)) ** 2.0) / float(y.size)
        )
        p0 = [new_z0, 60, 5, 1.0, 0.5]
        param_bounds = [[z[0], z[-1]], [0, 100], [1.0, 10], [0, None], [0, None]]
        popt_btm = minimize(
            fo_btm, x0=p0, args=(this_profile, z, psf1), bounds=param_bounds
        )
        z0, c, w, a, b = popt_btm.x[:]
        bumpTissue = bumpTissueModel(popt_btm.x, z)
        tissue = tissue_model([c, z0, a], z)
        bump = bumpModel(tissue, w, b)
        bump_mask = (bump - bump[-1]) <= 0.1 * (bump.max() - bump[-1])
        bump_mask[z <= new_z0] = 0
        limit = np.where(bump_mask)[0][0] * res
        this_mask = z >= limit

        # Optimize the PSF model (zf, zr)
        # Normalize the signal to fit the tissue model.
        def normalizeProfile(signal, psf):
            normalizedSignal = signal / psf
            return normalizedSignal

        fo_PSFNormalized = lambda x, y, z, tissue: np.sqrt(
            np.sum((tissue - normalizeProfile(y, confocal_model(x, z))) ** 2.0)
            / float(y.size)
        )
        p0 = popt_firstpsf.x  # zf, zr, a
        if fix_zr:
            zr_0 = p0[1]
            param_bounds = [[z[0], z[-1]], [zr_0, zr_0], [0.0, None]]
        else:
            param_bounds = [[z[0], z[-1]], [0, None], [0.0, None]]
        popt_psf = minimize(
            fo_PSFNormalized,
            x0=p0,
            args=(this_profile[this_mask], z[this_mask], bumpTissue[this_mask]),
            bounds=param_bounds,
        )
        psf_final = confocal_model(popt_psf.x, z)

        if plotProfiles:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 5))
            plt.plot(
                z,
                iprofile,
                marker="o",
                linestyle="dashed",
                alpha=2,
                label="Original Data",
            )
            plt.plot(
                z,
                imax * psf_final * bumpTissue + dc,
                label="Tissue model",
                linewidth=2,
                color="r",
            )
            plt.plot(
                z,
                imax * psf_final / psf_final.max() + dc,
                linewidth=2,
                linestyle="dashed",
                color="k",
                label="Confocal PSF",
            )
            plt.legend(loc="best", shadow=True)
            plt.grid("on")
            plt.show()

        output = {"psf": psf_final}
        if return_fullModel:
            output["tissue"] = imax * bumpTissue + dc
            output["tissue_psf"] = imax * psf_final * bumpTissue + dc

        if returnParameters:
            output["parameters"] = {
                "zf": popt_psf.x[0],
                "zr": popt_psf.x[1],
                "a": popt_psf.x[2],
            }
            output["parameters"]["z0"] = popt_btm.x[0]
            output["parameters"]["c"] = popt_btm.x[1]
            output["parameters"]["w"] = popt_btm.x[2]
            output["parameters"]["b1"] = popt_btm.x[3]
            output["parameters"]["b2"] = popt_btm.x[4]
    else:
        if plotProfiles:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 5))
            plt.plot(
                z,
                iprofile,
                marker="o",
                linestyle="dashed",
                alpha=0.6,
                label="Original Data",
            )
            plt.plot(
                z,
                imax * psf1 * syn_tissue * popt_firstpsf.x[2] + dc,
                label="Tissue model",
                linewidth=2,
                color="r",
            )
            plt.plot(
                z,
                imax * psf1 / psf1.max() + dc,
                linewidth=2,
                linestyle="dashed",
                color="k",
                label="Confocal PSF",
            )
            plt.plot(
                z,
                (iprofile - dc) * psf1.max() / psf1 + dc,
                marker="o",
                linewidth=2,
                linestyle="dashed",
                color="k",
                label="Compensated Data",
            )
            plt.legend(loc="best", shadow=True)
            plt.grid("on")
            plt.show()

        output = {"psf": psf1}
        if return_fullModel:
            output["tissue"] = imax * syn_tissue + dc
            output["tissue_psf"] = imax * psf1 * syn_tissue + dc

        if returnParameters:
            output["parameters"] = {
                "zf": popt_firstpsf.x[0],
                "zr": popt_firstpsf.x[1],
                "a": popt_firstpsf.x[2],
            }
            output["parameters"]["z0"] = popt_tissue.x[0]
            output["parameters"]["c"] = popt_tissue.x[1]
            output["parameters"]["w"] = popt_tissue.x[2]

    return output
