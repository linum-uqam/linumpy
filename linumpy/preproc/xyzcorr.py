#! /usr/bin/env python
# -*- coding: utf-8 -*-
""" Collection of functions to fix spatial-related artefacts in raw data """
import itertools

import numpy as np
import SimpleITK as sitk
from scipy.interpolate import interp1d
from scipy.ndimage import (
    gaussian_filter,
    gaussian_filter1d,
    gaussian_gradient_magnitude,
    uniform_filter,
)
from scipy.ndimage import label
from scipy.ndimage import binary_closing, binary_fill_holes
from scipy.optimize import curve_fit
from scipy.signal import argrelmax
from skimage.filters import threshold_li, threshold_otsu
from skimage.morphology import dilation, disk

from skimage.transform import resize
from skimage.metrics import normalized_mutual_information as nmi


def cropVolume(vol, xlim=[0, -1], ylim=[0, -1], zlim=[0, -1]):
    """Crops the given volume according to the range given as input

    Parameters
    ----------
    vol : ndarray
        Volume to crop
    xlim : (2,) list
        x range to keep
    ylim : (2,) list
        y range to keep
    zlim : (2,) list
        z range to keep

    Returns
    -------
    ndarray
        Cropped volume

    Notes
    -----
    * xlim=[0,-1] means that the whole volume in the x dimension will be returned.

    """
    nx, ny = vol.shape[:2]
    xlim = list(xlim)
    ylim = list(ylim)
    zlim = list(zlim)
    if xlim[1] == -1:
        xlim[1] = nx
    if ylim[1] == -1:
        ylim[1] = ny

    if vol.ndim == 3:
        nz = vol.shape[2]
        if zlim[1] == -1:
            zlim[1] = nz
        return vol[xlim[0]: xlim[1], ylim[0]: ylim[1], zlim[0]: zlim[1]]

    elif vol.ndim == 2:
        return vol[xlim[0]: xlim[1], ylim[0]: ylim[1]]


def resampleITK(vol, newshape, interpolator="linear"):
    """Resamples a volume / image using ITK

    Parameters
    ----------
    vol : ndimage
        Array to resample
    newshape : (2,) or (3,) tuple or an integer
        New shape of the array, or resampling factor (if a single integer is given)

    interpolation : str
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
        newshape = [int(x) for x in newshape]

    if vol.dtype == bool:
        isBool = True
        vol = 255 * vol.astype(np.uint8)
    else:
        isBool = False

    if vol.ndim == 3:
        if vol.shape[2] == 1:
            vol = np.squeeze(vol, axis=(2,))
            newshape = newshape[0:2]

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
        resample.SetOutputSpacing(
            [(nz - 1) / float(oz), (ny - 1) / float(oy), (nx - 1) / float(ox)]
        )
        if (
                nx / float(ox) > 1 or ny / float(oy) > 1 or nz / float(oz) > 1
        ):  # Smoothing if downsampling
            vol = gaussian_filter(
                vol, sigma=[nx / float(2 * ox), ny / float(2 * oy), nz / float(2 * oz)]
            )

    if interpolator == "NN":
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    elif interpolator == "linear":
        resample.SetInterpolator(sitk.sitkLinear)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    vol_itk = sitk.GetImageFromArray(vol)
    output_itk = resample.Execute(vol_itk)

    if isBool:
        vol_p = sitk.GetArrayFromImage(output_itk)
        # vol_p = vol_p == vol_p.max()
        vol_p = vol_p > vol_p.max() * 0.5
    else:
        vol_p = sitk.GetArrayFromImage(output_itk)

    return vol_p


def shrink(vol, spacing=(1.0, 1.0, 1.0), res=(10.0, 10.0, 10.0)):
    """Shrink volume up to a given resolution (in each dimension)

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
    outputSize = (
        np.floor(dz * nz / (1.0 * rz)).astype(int),
        np.floor(dy * ny / (1.0 * ry)).astype(int),
        np.floor(dx * nx / (1.0 * rx)).astype(int),
    )

    # Apply a gaussian filter first
    vol = gaussian_filter(
        vol, sigma=(rx / (4.0 * dx), ry / (4.0 * dy), rz / (4.0 * dz))
    )

    # Creating a resampling filter using Sitk
    img = sitk.GetImageFromArray(vol)
    img.SetSpacing((dz, dy, dx))

    # Creating a resampling filter
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputSpacing([rz, ry, rx])
    resample.SetSize(outputSize)

    # Resampling
    return sitk.GetArrayFromImage(resample.Execute(img))


def cropZ0WholeSlice(
        vol,
        dz=20.0,
        nz=200.0,
        voxdim=(1, 1, 1),
        z0=None,
        verbose=False,
        mask=None,
        returnZ0=False,
):
    """Crop whole slice in the z direction.

    Parameters
    ----------
    vol : ndarray
    dz : (microns) margin under interface to crop (to remove cutting deformations)
    nz : (microns) size of slice to crop
    voxdim : dimension of each voxel in micron/pixel
    z0 : (micron)

    Returns
    -------
    ndarray
        Cropped array
    """
    volshape = vol.shape

    if z0 is None:
        # Computing tissue mask
        if mask is not None:
            mask = vol.std(axis=2) > threshold_otsu(
                vol.std(axis=2)
            )  # Using otsu on A-line intensity std
            mask = binary_fill_holes(binary_closing(mask))  # Closing and filling holes
        else:
            mask = np.ones(vol.shape[0:2], dtype=bool)

        # Computing tissue interface
        interface = findTissueInterface(vol)

        # Use median interface
        z0 = np.median(interface)

        """
        # Computing depth histogram
        h, b = np.histogram(np.ravel(interface[mask]), bins=volshape[2], density=True)
        h = np.insert(h, 0, 0) # So that len(b) = len(h)
        h = np.cumsum(h)*(b[2]-b[1]) # Cumulative sum of depth histogram
        h /= h[-1] # Normalisation

        # Detecting z0 as 90% of interface
        z0 = np.round(b[np.where(h > 0.9)[0][0]]).astype(int)
        """

        # Computing intensity and gradient profiles
        # iProfile = np.mean(vol[:,:,0:volshape[2]/2], axis=(0,1))
        # gProfile = np.gradient(iProfile)
        # maxG = np.where(gProfile == gProfile.max())
        # z0 = maxG[0]
    else:
        # Compute z0 in pixel
        z0 = np.floor(z0 / (1.0 * voxdim[2])).astype(int)

    # Finding crop limits
    zmin = np.floor((z0 * voxdim[2] + dz) / (1.0 * voxdim[2])).astype(int)
    zmax = np.floor((zmin * voxdim[2] + nz) / (1.0 * voxdim[2])).astype(int)

    if verbose:
        print(
            (
                    "Crop limits are : [%.2f, %.2f] microns"
                    % (zmin * voxdim[2], zmax * voxdim[2])
            )
        )
        print(("Crop limits are : [%d, %d] pixels" % (zmin, zmax)))

    # Cropping
    if returnZ0:
        return cropVolume(vol, zlim=[zmin, zmax]), z0
    else:
        return cropVolume(vol, zlim=[zmin, zmax])


def findTissueDepth(vol, zmin=15, zmax=100, agaroseIntensity=5000):
    """Detects the tissue interface depth in given volume

    This algorithm first segments the volume into tissue vs background(agarose) using
    the Li thresholding method and user-defined agarose intensity value. It then
    computes the XZ mean projection of the tissue mask, and detects the main edge
    position of the tissue/water interface using morphological operations and relative
    maximum detection. Other operations are done on the data to reduce the effect
    of intensity noise and artefacts on the water/tissue interface depth detection.

    Parameters
    ----------
    vol : ndarray
        Volume to analyze
    zmin : int
        Minimum depth of interface in pixel
    zmax : int
        Maximum depth of interface in pixel
    agaroseIntensity : int
        Agarose mean intensity value used to restrict analysis to tissue voxels

    Returns
    ------
    int
        Tissue interface depth

    Notes
    -----
    * The default depth is 40 px

    """
    z0 = 0  # Default value

    try:
        nx, ny, nz = vol.shape

        # Removing agarose (threshold selected empirically)
        mip_agaroseMask = np.mean(vol[:, :, zmin:zmax], axis=2) < agaroseIntensity
        mip_agaroseMask = binary_fill_holes(mip_agaroseMask)
        mip_agaroseMask = np.reshape(mip_agaroseMask, (nx, ny, 1))
        agaroseMask = np.tile(mip_agaroseMask, [1, 1, nz])

        mask = vol > threshold_li(vol[:, :, zmin:zmax])
        mask[agaroseMask] = 0  # This is mostly background/agarose pixels.
        im = mask.max(axis=1)
        im = binary_fill_holes(im)

        # Labeling features and keeping the largest
        im_label, num_features = label(im)
        hist = list()
        for i in range(num_features):
            hist.append(np.sum(im_label == i))
        mainFeature = np.argmax(hist[1:]) + 1
        im[im_label != mainFeature] = 0

        # Find edges based on morphological dilation
        edges = dilation(im, disk(3)) - im
        edges[:, 0:zmin] = 0  # We don't want top slices
        edges[:, zmax: edges.shape[1]] = 0  # We don't want bottom slices either
        z_profile = edges.sum(axis=0)
        peaks = argrelmax(z_profile, order=20)
        if len(peaks[0]) > 0:
            z0 = peaks[0][0]
    except:
        pass
    return z0


def getInterfaceDepthFromMask(vol):
    """Computes the interface depths from a 3D tissue mask

    Parameters
    ----------
    vol : (NxMxK) ndarray
        Tissue mask

    Returns
    -------
    ndarray : (NxM)
        Interface depth (in pixel)

    """
    nx, ny, _ = vol.shape
    depths = np.zeros((nx, ny))
    for x, y in itertools.product(list(range(nx)), list(range(ny))):
        idx = np.where(vol[x, y, :])
        if len(idx[0]) > 0:
            depths[x, y] = idx[0][0]

    return depths


def findTissueInterface(
        vol, s_xy=15, s_z=2, useLog=True, mask=None, order=1, detectCuttingErrors=False
):
    """Detects the tissue interface.

    Parameters
    ----------
    vol : ndarray
        Containing the volume to analyze
    s_xy : int
        Uniform filter kernel size (xy)
    s_z : int
        1st order gaussian kernel size (z)
    useLog : bool

    mask : ndarray

    Returns
    -------
    ndarray
        Tissue interface depth

    """
    if useLog:
        vol_p = np.copy(vol)
        vol_p[vol > 0] = np.log(vol[vol > 0])
    else:
        vol_p = vol
    vol_p = uniform_filter(vol_p, (s_xy, s_xy, 0))
    if mask is not None:
        vol_g = np.zeros_like(vol_p)
        for x in range(vol_p.shape[0]):
            for y in range(vol_p.shape[1]):
                mask_Aline = mask[x, y, :]
                Aline = vol_p[x, y, :]
                vol_g[x, y, mask_Aline] = gaussian_filter1d(
                    Aline[mask_Aline], s_z, order=order
                )
    else:
        vol_g = gaussian_filter1d(vol_p, s_z, order=order)
    z0 = np.ceil(vol_g.argmax(axis=2) + s_z * 0.5).astype(int)

    # Check if tissue begins before the FOV
    if detectCuttingErrors:
        vol_p = gaussian_filter1d(vol_p, s_z, order=0)
        z0_p = np.abs(vol_p).argmax(axis=2)
        mask_max = z0_p < z0
        z0[mask_max] = z0_p[mask_max]

    return z0


def maskUnderInterface(vol, interface, returnMask=False):
    nx, ny, nz = vol.shape
    _, _, zz = np.meshgrid(
        list(range(nx)), list(range(ny)), list(range(nz)), indexing="ij"
    )
    interface_3d = np.tile(np.reshape(interface, (nx, ny, 1)), (1, 1, nz))
    mask = zz >= interface_3d
    if returnMask:
        return mask
    else:
        return vol * mask


def findCuttingPlane(vol, z0map, agarose_mean, agarose_std):
    """Find the cutting plane using agarose segmentation

    Parameters
    ==========
    vol : ndarray

    z0map : ndarray

    agarose_mean : float

    agarose_std : float

    Returns
    =======
    popt

    detectedInterface : ndarray

    z0 : int

    """
    # Computing agarose mask
    mask_tissue = vol >= agarose_mean + 3 * agarose_std
    mask_tissue = binary_fill_holes(mask_tissue)

    # Removing zero background
    agarose_mask = ~mask_tissue
    agarose_mask[vol == 0] = 0
    agarose_mask = agarose_mask.astype(bool)

    # Removing z0 outliers
    z0_median = np.median(z0map[agarose_mask])
    z0_MAD = np.median(np.abs(z0map[agarose_mask] - z0_median))
    if z0_MAD != 0:  # Only if z0_MAD is not 0.
        z0_Zscore = np.abs(0.6745 * (z0map - z0_median) / z0_MAD)
        z0_outliers = z0_Zscore > 3.5
        agarose_mask[z0_outliers] = 0

    xdata = np.where(agarose_mask)
    ydata = z0map[agarose_mask][:]

    popt, _ = curve_fit(_plane, xdata, ydata)

    # Getting surface fit array
    xx, yy = np.meshgrid(
        list(range(vol.shape[0])), list(range(vol.shape[1])), indexing="ij"
    )
    detectedInterface = xx * popt[0] + yy * popt[1] + popt[2]

    # Choosing z range for stitching
    z0 = (
            np.round(detectedInterface.max()) + 5
    )  # Making sure we are 5*6.5 = 32.5 microns below the interface (this is assuming that the cut was ok)

    return popt, detectedInterface, z0


# Fitting plane on agarose z0 values
def _plane(pos, a, b, c):
    x = pos[0]
    y = pos[1]
    return a * x + b * y + c


def removeZ0Outliers(z0map):
    data = np.ravel(z0map[0, 0, :])
    # Median Depth
    med = np.median(data)

    # Median absolute deviation
    MAD = np.median(np.abs(data - med))

    if MAD != 0:  # Only if MAD is not 0.
        dZscore = np.abs(0.6745 * (data - med) / MAD)
        outliers = dZscore > 3.0  # was 3.5

        # Replacing outliers by median depth
        z0map[:, :, outliers] = np.median(data)

        # Printing outliers for information
        print(("Z0 outliers were removed for the slices : ", np.where(outliers)[0]))

        return z0map
    else:
        print("MAD = 0. No outliers")
        return z0map


def applyInterfaceCorrection(
        vol, interface
):  # TODO: Test this algorithm to make sure it works well.
    """Apply interface depth correction using linear interpolation.

    :param vol: (ndarray) containing the volume to fix.
    :param interface: (ndarray) Tissue interface depth.

    :returns: (ndarray) Fixed volume.

    """
    nx, ny, nz = vol.shape
    zRange = np.around(interface.max() - interface.min())
    fixedVol = np.zeros((nx, ny, nz - zRange), dtype=vol.dtype)

    # Loop over XY
    for x in range(nx):
        for y in range(ny):
            z = interface[x, y]
            realZ = np.linspace(-z, -z + nz, nz)
            newZ = list(range(int(nz - zRange)))
            zInterp = interp1d(
                realZ, vol[x, y, :], fill_value=0, bounds_error=False, kind="quadratic"
            )
            fixedVol[x, y, :] = zInterp(newZ)

    return fixedVol


def fitInterface(interface, method="linear", returnCenter=False):
    """Fit a model on the given interface
    Parameters
    ----------
    interface : ndarray

    method : str
        'linear', 'quad', 'gauss', 'sph'
    """
    xdata = np.where(interface)
    ydata = np.ravel(interface)
    xx, yy = np.meshgrid(
        list(range(interface.shape[0])), list(range(interface.shape[1])), indexing="ij"
    )
    if method == "linear":
        popt, _ = curve_fit(_plane, xdata, ydata)

        # Getting surface fit array
        fittedInterface = xx * popt[0] + yy * popt[1] + popt[2]
        center = (interface.shape[0] / 2, interface.shape[1] / 2)
    elif method == "quad":
        popt, _ = curve_fit(quadraticInterface, xdata, ydata)

        # Choosing z range for stitching
        a, b, c, d, e, f, g, h = popt
        xx = xx - g
        yy = yy - h
        fittedInterface = a * xx + b * yy + c * xx * yy + d * xx ** 2 + e * yy ** 2 + f
        center = (g, h)

    elif method == "gauss":
        f = (
            lambda x, a, b, c, d, e, f: np.exp(
                -((x[0] - a) ** 2) / (2.0 * b ** 2.0)
                - (x[1] - c) ** 2 / (2.0 * d ** 2.0)
            )
                                        * e
                                        + f
        )
        popt, _ = curve_fit(f, xdata, ydata)
        a, b, c, d, e, f = popt
        fittedInterface = (
                np.exp(
                    -((xx - a) ** 2) / (2.0 * b ** 2.0) - (yy - c) ** 2 / (2.0 * d ** 2.0)
                )
                * e
                + f
        )
        center = (a, c)

    elif method == "sph":
        f = lambda x, a, b, c: c * (((x[0] - a) ** 2 + (x[1] - b) ** 2) ** 2.0) / 8.0
        popt, _ = curve_fit(f, xdata, ydata)
        fittedInterface = (
                popt[2] * (((xx - popt[0]) ** 2 + (yy - popt[1]) ** 2) ** 2.0) / 8.0
        )
        center = (popt[0], popt[1])

    if returnCenter:
        return fittedInterface, center
    else:
        return fittedInterface


# Quadratic model for interface fit
def quadraticInterface(pos, a, b, c, d, e, f, g, h):
    x = pos[0] - g
    y = pos[1] - h
    return a * x + b * y + c * x * y + d * x ** 2 + e * y ** 2 + f


def getQuadraticInterface(popt, volshape=(512, 512, 120)):
    xx, yy = np.meshgrid(
        list(range(volshape[0])), list(range(volshape[1])), indexing="ij"
    )
    tmp = quadraticInterface(
        [xx[:], yy[:]], popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
    )
    interface = np.zeros([volshape[0], volshape[1]])
    interface[xx[:], yy[:]] = tmp
    return interface


def linearHomogeneousProfile(z, z0, dz, I0, Ib, sigma):
    """Intensity profile based on an single homogeneous tissue Beer-Lambert model (covered by some amount of water)
    This will return the log(I).

    Parameters
    ----------
    z : ndarray
        Position where the intensity is evaluated
    z0 : float
        Water-tissue interface depth
    dz : float
        Interface Transition width
    I0 : float
        Top tissue slice intensity
    Ib : float
        water intensity
    sigma : float
        Tissue Attenuation coefficient

    Returns
    -------
    ndarray
        Log(I) evaluated at each position z.

    """
    z_underz0 = z < z0 - dz
    z_betweenz0 = (z >= z0 - dz) * (z < z0)
    z_overz0 = z >= z0
    I = np.zeros((len(z),))
    I[z_underz0] = Ib
    I[z_betweenz0] = (I0 - Ib) / (1.0 * dz) * (z[z_betweenz0] - (z0 - dz)) + Ib
    I[z_overz0] = I0 - sigma * (z[z_overz0] - z0)
    return I


def estimateLHProfileParameters(vol, s=25):
    """Estimates the linear-homogeneous intensity profile parameters

    Parameters
    ----------
    vol : ndarray
        Volume for which the LHP parameters are evaluated
    s : int
        Neighborhood used to average intensities at each depth

    Returns
    -------
    float
        z0 : Water-tissue interface depth
    float
        dz : Interface Transition width
    float
        I0 : Top tissue slice intensity
    float
        Ib : water intensity
    float
        sigma : Tissue Attenuation coefficient

    Note
    ----
    * This first version loops over all intensity profiles (x, y)

    """
    nx, ny, _ = vol.shape
    vol_p = np.log(vol + 1.1)  # 1.1 factor is to prevent log of 0
    vol_p = uniform_filter(
        vol_p, (s, s, 0)
    )  # Averaging intensities over a small XY neigborhood
    vol_f = gaussian_filter1d(
        vol_p, sigma=1, axis=2
    )  # Smoothing the intensity profiles in Z
    vol_g = gaussian_gradient_magnitude(
        vol_p, [0, 0, 1]
    )  # TODO: Computing gradient in z direction only ?

    # Finding max gradient position
    z0 = vol_g.argmax(axis=2)

    xx, yy = np.meshgrid(list(range(nx)), list(range(ny)), indexing="ij")
    I_gmax = vol_p[xx, yy, z0]

    test = np.zeros(vol_p.shape)
    test[xx, yy, z0] = 1
    import nibabel as nib

    nib.save(
        nib.Nifti1Image(test, np.eye(4)),
        "/home/local/LIOM/jlefebvre/tmp/interface_test.nii",
    )

    # Preparing variables
    z0 = np.zeros((nx, ny), dtype=np.uint)
    dz = np.zeros((nx, ny), dtype=np.uint)
    I0 = np.zeros((nx, ny))
    Ib = np.zeros((nx, ny))
    sigma = np.zeros((nx, ny))

    for x in range(nx):  # TODO: Accelerate this loop (multithreading ?)
        for y in range(ny):
            I = vol_p[x, y, :]
            If = vol_f[x, y, :]
            I_g = np.gradient(If)

            this_z0 = np.where(I_g == I_g.max())[0][0]
            I_gm = I[this_z0]
            indices = np.where(I_g / I_gm < 0.1)
            zlist_min = indices[0][indices[0] < this_z0]
            zlist_max = indices[0][indices[0] > this_z0]

            if len(zlist_min) > 0 and len(zlist_max) > 0:
                this_dz = zlist_max[0] - zlist_min[-1]
            else:
                this_dz = 1
            if len(zlist_max) > 0:
                this_z0 = zlist_max[0]

            zmax = np.where(If == If.max())[0][0]
            if zmax - this_z0 < -5:
                this_z0 = zmax

            this_I0 = I[this_z0]
            this_sigma = -np.median(I_g[this_z0::])

            if (this_z0 == 0) or (this_z0 - this_dz <= 0):
                this_Ib = 1
            else:
                this_Ib = np.median(I[0: this_z0 - this_dz])

            z0[x, y] = this_z0
            dz[x, y] = this_dz
            I0[x, y] = this_I0
            Ib[x, y] = this_Ib
            sigma[x, y] = this_sigma

    return z0, dz, I0, Ib, sigma


def detect_galvo_shift(aip: np.ndarray, n_pixel_return: int = 40) -> int:
    """Detects the galvo shift in the AIP.
    Parameters
    ----------
    aip : ndarray
        AIP of the OCT volume containing both the image and the galvo return. This assumes that the first axis is the
        A-line axis, and the second axis is the B-scan axis, and the average was taken over the depth axis.
    n_pixel_return : int
        Number of pixels used for the galvo returns.
    Returns
    -------
    int
        Shift in pixels
    """
    similarities = []

    n_alines, n_bscans = aip.shape
    for i in range(n_alines):
        aip_shifted = fix_galvo_shift(aip, shift=i, axis=0)

        # Extract the image and galvo return parts
        img = aip_shifted[0:-n_pixel_return, :]
        img_galvos_simulated = np.flipud(resize(img, (n_pixel_return, n_bscans)))
        img_galvos = aip_shifted[-n_pixel_return:, :]

        similarity = nmi(img_galvos_simulated, img_galvos)
        similarities.append(similarity)

    shift = np.argmax(similarities)
    return shift

def fix_galvo_shift(vol: np.ndarray, shift: int=0, axis:int=1) -> np.ndarray:
    """Fix the galvo shift in an OCT volume."""
    if shift == 0:
        return vol
    else:
        return np.roll(vol, shift, axis=axis)

