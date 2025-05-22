#! /usr/bin/env python
# -*- coding: utf-8 -*-
""" This modules contains all methods used to calculates coregistrate to adjacent tiles or volumes.

"""

import numpy as np
from skimage.feature import peak_local_max

from linumpy.registration import applyHanningWindow
from linumpy.stitching.stitch_utils import getOverlap


def mutualInformation(vol1, vol2, mask=None, N=128):
    # Joint histogram vol1-vol2
    if mask is not None:
        p12, _, _ = np.histogram2d(np.ravel(vol1[mask]), np.ravel(vol2[mask]), N)
    else:
        p12, _, _ = np.histogram2d(np.ravel(vol1), np.ravel(vol2), N)

    p12 /= float(p12.sum())

    p1 = p12.sum(axis=0)
    p1 /= float(p1.sum())

    p2 = p12.sum(axis=1)
    p2 /= float(p2.sum())

    p1 = np.tile(np.reshape(p1, (N, 1)), (1, N))
    p2 = np.tile(np.reshape(p2, (1, N)), (N, 1))

    m1 = p1 > 0
    m2 = p2 > 0
    m12 = p12 > 0
    m = m1 * m2 * m12

    # Mutual Information H(1) + H(2) - H(12)
    # MI = np.sum(p12[m]*np.log(p12[m]/(1.0*p1[m]*p2[m])))
    MI = (
        np.sum(p1[m] * np.log2(p1[m]))
        + np.sum(p2[m] * np.log2(p2[m]))
        - np.sum(p12[m] * np.log2(p12[m]))
    )
    return MI


def extPhaseCorrelation3d(
    im1, im2, nDim=(1, 1, 1), hanningWindow=False, returnCC=False
):
    """Find the translation shift between two volumes in 3D using phase correlation

    Parameters
    ----------
    im1 : NxMxO ndarray
        Array containing the 1st image (reference)

    im2 : NxMxO ndarray
        Array containing the 2nd image (moving)

    nDim : (3,) tuple
        Fractional padding to use in each dimension

    hanningWindow : Bool
        If True, a hanning window will be applied before computing the phase correlation.

    returnCC : Bool
        If True, the phase correlation value is returned in addition to the displacements

    Returns
    -------
    int
        deltax the displacement in x direction.
    int
        deltay the displacement in y direction.
    int
        deltaz the displacement in z direction.
    float
        cc the phase correlation value if returnCC = True

    .. note::
        * im1 and im2 must have the same shape.
        * Each image is assumed to be a 3d array
        * This is an implementation of `[Foroosh2002] <http://www.ncbi.nlm.nih.gov/pubmed/18244623>`_
        * The matlab implemenation was done by : Lulu Ardiansyah (halluvme@gmail.com)

    .. author:: Joël Lefebvre <joel.lefebvre<at>polymtl.ca>

    .. todo:: Test if this translation (see com1) is necessary, and comment accordingly

    """

    xSize, ySize, zSize = im1.shape

    if hanningWindow:
        im1 = applyHanningWindow(im1)
        im2 = applyHanningWindow(im2)

    # Phase Correlation (Kuglin & Hines, 1975)
    fft_shape = (nDim[0] * xSize, nDim[1] * ySize, nDim[2] * zSize)
    af = np.fft.fftn(im1, s=fft_shape, axes=(0, 1, 2))  # nd fft of im1
    bf = np.fft.fftn(im2, s=fft_shape, axes=(0, 1, 2))  # nd fft im2
    cp = np.divide(af * np.conj(bf), np.abs(af * np.conj(bf)))
    icp = np.fft.ifftn(cp, axes=(0, 1, 2))  # Back to the real world
    icp = np.abs(icp)
    mmax = np.amax(icp)  # max value in the whole volume

    # Find the main peak
    indices = np.where(icp == mmax)
    try:
        deltax = indices[0][0]
        deltay = indices[1][0]
        deltaz = indices[2][0]

        # Validate if translation shift is negative
        nx, ny, nz = icp.shape
        if deltax > nx / 2:
            deltax -= nx
        if deltay > ny / 2:
            deltay -= ny
        if deltaz > nz / 2:
            deltaz -= nz

        # TODO: com1 : check if translation is necessary
        if deltax > 0 and nDim[0] > 1:
            deltax -= nx / 2
        if deltay > 0 and nDim[1] > 1:
            deltay -= ny / 2
        if deltaz > 0 and nDim[2] > 1:
            deltaz -= nz / 2
    except:
        print("Problem with deltas, using 0,0,0")
        deltax = 0
        deltay = 0
        deltaz = 0

    if returnCC:
        return deltax, deltay, deltaz, mmax
    else:
        return deltax, deltay, deltaz


def extPhaseCorrelation2d(im1, im2, nDim=(1, 1), applyPadding=False):
    """Find the translation shift between two volumes in 2D using phase correlation

    :param im1: nxm array containing the 1st image
    :param im2: nxm array containing the 2nd image
    :param nDim: Fractional padding to use in each dimension (default is (1,1))
    :returns: deltax, deltay the displacement in each direction.
    :returns: mmax : The maximum phase correlation value

    .. note::
        * im1 and im2 must have the same shape.
        * Each image is assumed to be a 3d array

    .. author:: Joël Lefebvre <joel.lefebvre<at>polymtl.ca>

    """
    from scipy.ndimage.filters import gaussian_filter

    im1 = gaussian_filter(im1, sigma=5)
    im2 = gaussian_filter(im2, sigma=5)

    xSize, ySize = im1.shape

    if applyPadding:
        newshape = (im1.shape[0] * nDim[0], im1.shape[1] * nDim[1])
        im1 = apply2DMirrorPadding(im1, newshape)
        im2 = apply2DMirrorPadding(im2, newshape)

    # Phase Correlation (Kuglin & Hines, 1975)
    fft_shape = (nDim[0] * xSize, nDim[1] * ySize)
    af = np.fft.fft2(im1, s=fft_shape)  # nd fft of the 1st image
    bf = np.fft.fft2(im2, s=fft_shape)  # nd fft of the 2nd image
    cp = np.divide(af * np.conj(bf), np.abs(af * np.conj(bf)))
    icp = np.fft.ifft2(cp)  # Back to the real world
    icp = np.fft.fftshift(np.abs(icp))
    mmax = np.amax(icp)  # max value in the whole image

    # Find the main peak
    indices = np.where(icp == mmax)
    xx = np.arange(-np.int(xSize / 2 * nDim[0]) + 1, np.int(xSize / 2 * nDim[0]))
    yy = np.arange(-np.int(ySize / 2 * nDim[1]) + 1, np.int(ySize / 2 * nDim[1]))

    deltax = xx[indices[0][0]]
    deltay = yy[indices[1][0]]

    return deltax, deltay, mmax


def normalizedCrossCorrelation2d(
    im1, im2, nDim=(2, 2)
):  # TODO: test the normalized cross correlation algorithm.
    """Computes the translation shift between two volumes in 2D using normalized cross correlation

    :param im1: nxm array containing the 1st image
    :param im2: nxm array containing the 2nd image
    :param nDim: Fractional padding to use in each dimension (default is (2, 2))
    :returns: deltax, deltay the displacement in each direction.
    :returns: mmax : The maximum phase correlation value

    .. author:: Joël Lefebvre <joel.lefebvre<at>polymtl.ca>

    .. todo:: Test the normalized cross correlation algorithm.

    """

    nx, ny = im1.shape

    # Applying a window
    hx = np.tile(
        np.reshape(np.hanning(nx), (nx, 1)), (1, ny)
    )  # Volume containing the x hanning window
    hy = np.tile(
        np.reshape(np.hanning(ny), (1, ny)), (nx, 1)
    )  # Volume containing the y hanning window

    h2d = np.multiply(hx, hy)  # 2d hanning window
    del hx, hy  # Deleting these variables from memory

    im1 *= h2d
    im2 *= h2d

    # Normalized Cross Correlation (Yu2011)
    fft_shape = (nDim[0] * nx, nDim[1] * ny)
    af = np.fft.fft2(im1, s=fft_shape)  # nd fft of the 1st image
    bf = np.fft.fft2(im2, s=fft_shape)  # nd fft of the 2nd image
    ccf = af * np.conj(bf)  # Cross Correlation (in Fourier space)
    cc = np.fft.ifft2(ccf)  # cross correlation in image space

    # Normalizing CC
    N = nx * ny
    cc_n = cc - 1.0 / N * np.sum(im1) * np.sum(im2)
    cc_n /= np.sqrt(
        (np.sum(im1**2) - 1.0 / N * (np.sum(im1)) ** 2)
        * (np.sum(im2**2) - 1.0 / N * (np.sum(im2)) ** 2)
    )

    # Find the main peak
    mmax = np.amax(cc_n)  # max value in the whole image
    indices = np.where(cc_n == mmax)
    deltax = indices[0][0]
    deltay = indices[1][0]

    # Validate if translation shift is negative
    nx, ny = cc_n.shape
    if deltax > nx / 2:
        deltax -= nx
    if deltay > ny / 2:
        deltay -= ny

    return deltax, deltay, np.float(np.abs(mmax))


def hybridCorrelation3D(
    im1,
    im2,
    nDim=(2, 2),
    n=4,
    M=15,
    hanningWindow=False,
    mirrorPadding=False,
    returnCC=False,
    preWhitening=False,
    verbose=False,
):  # TODO: Implement the hybridcorrelation in 3D
    """PCNCC algorithm

    Parameters
    ----------
    im1 : ndarray (nxm)
        2D image used as reference

    im2 : ndarray (nxm)
        2D image used as moving matrix (should be the same size as im1)

    nDim : tuple (2,)
        Fractional padding to use in each dimension.

    n : int
        Size of neighborhood considered for NCC (2n+1 is total neig size). If n is None, the whole overlap regions will be used to compute the NCC.

    M : int
        Number of NCC candidates

    hanningWindow : Bool
        If True, the images im1 and im2 will be multiplied by a Hanning window before the correlation process

    mirrorPadding : Bool
        If True, the padding used is a mirror of the data (with a Hanning decrease).

    returnCC : Bool
        If true, the NCC value is returned in addition to dx and dy

    verbose : Bool
        If true, some text outputs are printed in terminal

    Returns
    -------
    int
        dx : X displacement of im2 in relation to im1
    int
        dy : Y displacement of im2 in relation to im1
    float
        NCC (if returnCC=True) : Normalized Cross-Correlation value

    """
    pass


def hybridCorrelation2D(
    im1,
    im2,
    nDim=(2, 2),
    n=4,
    M=15,
    hanningWindow=False,
    mirrorPadding=False,
    returnCC=False,
    preWhitening=False,
    verbose=False,
):
    """PCNCC algorithm

    Parameters
    ----------
    im1 : ndarray (nxm)
        2D image used as reference

    im2 : ndarray (nxm)
        2D image used as moving matrix (should be the same size as im1)

    nDim : tuple (2,)
        Fractional padding to use in each dimension.

    n : int
        Size of neighborhood considered for NCC (2n+1 is total neig size). If n is None, the whole overlap regions will be used to compute the NCC.

    M : int
        Number of NCC candidates

    hanningWindow : Bool
        If True, the images im1 and im2 will be multiplied by a Hanning window before the correlation process

    mirrorPadding : Bool
        If True, the padding used is a mirror of the data (with a Hanning decrease).

    returnCC : Bool
        If true, the NCC value is returned in addition to dx and dy

    verbose : Bool
        If true, some text outputs are printed in terminal

    Returns
    -------
    int
        dx : X displacement of im2 in relation to im1
    int
        dy : Y displacement of im2 in relation to im1
    float
        NCC (if returnCC=True) : Normalized Cross-Correlation value

    """
    # Variable initialization
    nx, ny = im1.shape
    maxNCC = 0
    deltax = 0
    deltay = 0

    # Apply hanning window directly on image (use only if the translation is not near the edges of each images)
    if hanningWindow:
        im1 = applyHanningWindow(im1)
        im2 = applyHanningWindow(im1)

    # Applying preWhitening to remove gaussian like distribution that competes with the pc peak (gonzalez2011)
    if preWhitening:
        im1 = np.log(im1)
        im2 = np.log(im2)

    # Mirror padding to reduce sinc component in fft (due to finite support) and to keep all information in image (Preibish2009)
    if mirrorPadding:
        newshape = np.round((nx * nDim[0], ny * nDim[1]))
        im1p = apply2DMirrorPadding(im1, newshape)
        im2p = apply2DMirrorPadding(im2, newshape)
    else:
        im1p = im1
        im2p = im2

    # Phase Correlation (Kuglin & Hines, 1975)
    fft_shape = np.round((nDim[0] * nx, nDim[1] * ny)).astype(np.int)
    af = np.fft.fft2(im1p, s=fft_shape)  # nd fft of the 1st image
    bf = np.fft.fft2(im2p, s=fft_shape)  # nd fft of the 2nd image
    cp = np.divide(
        af * np.conj(bf), np.abs(af * np.conj(bf))
    )  # Normalized phase correlation
    icp = np.abs(np.fft.ifft2(cp))  # Back to the real world
    icp = np.fft.fftshift(icp)

    # Finding peaks in PC map.
    coordinates = peak_local_max(
        icp, min_distance=5, num_peaks=M, exclude_border=True
    )  # max value in the whole image

    # Defining x and y grid to find directly the displacement.
    # xx = np.round(np.linspace(-np.int(nx/2*nDim[0])+1, np.int(nx/2*nDim[0]), icp.shape[0]))
    # yy = np.round(np.linspace(-np.int(ny/2*nDim[1])+1, np.int(ny/2*nDim[1]), icp.shape[1]))
    xx = np.round(np.linspace(-np.int(nx / 2), np.int(nx / 2) - 1, icp.shape[0]))
    yy = np.round(np.linspace(-np.int(ny / 2), np.int(ny / 2) - 1, icp.shape[1]))

    # Testing phase correlation masked projection
    # import matplotlib.pyplot as plt
    # plt.imshow(np.angle(np.fft.fftshift(cp)), cmap='gray')
    # plt.title('Phase correlation')
    # plt.show()

    R = 7
    dxmin = -50
    dxmax = 50
    dymin = -50
    dymax = 50

    ipcn = np.copy(icp)
    ipcn[xx <= dxmin - 0.5 * R, :] = 0
    ipcn[xx >= dxmax + 0.5 * R, :] = 0
    ipcn[:, yy <= dymin - 0.5 * R] = 0
    ipcn[:, yy >= dymax + 0.5 * R] = 0

    # plt.imshow(ipcn)
    # plt.show()

    pcn = np.fft.fft2(np.fft.fftshift(ipcn))
    # plt.imshow(np.angle(np.fft.fftshift(pcn)), cmap='gray')
    # plt.show()

    # Loop over all peaks and compute NCC
    for x, y in zip(coordinates[:, 0], coordinates[:, 1]):
        dx = xx[x]
        dy = yy[y]

        # Making sure both positions are positive numbers
        pos1 = [0, 0]
        pos2 = [pos1[0] + dx, pos1[1] + dy]
        minx = np.min([pos1[0], pos2[0]])
        miny = np.min([pos1[1], pos2[1]])
        pos1[0] -= minx
        pos2[0] -= minx
        pos1[1] -= miny
        pos2[1] -= miny

        # Extracting overlap regions for these displacements
        ov1, ov2, _, _ = getOverlap(im1, im2, pos1, pos2)

        if (ov1 is None) or (ov2 is None):
            continue

        # Only compute the NCC on a central region of specific size
        xmid1 = np.round(0.5 * ov1.shape[0])
        xmid2 = np.round(0.5 * ov2.shape[0])
        ymid1 = np.round(0.5 * ov1.shape[1])
        ymid2 = np.round(0.5 * ov2.shape[1])

        r1 = ov1[xmid1 - n : xmid1 + n, ymid1 - n : ymid1 + n]
        r2 = ov2[xmid2 - n : xmid2 + n, ymid2 - n : ymid2 + n]
        try:
            if n is None:
                NCC = np.sum((ov1 - ov1.mean()) * (ov2 - ov2.mean())) / np.sqrt(
                    np.sum((ov1 - ov1.mean()) ** 2) * np.sum((ov2 - ov2.mean()) ** 2)
                )
            else:
                NCC = np.sum((r1 - r1.mean()) * (r2 - r2.mean())) / np.sqrt(
                    np.sum((r1 - r1.mean()) ** 2) * np.sum((r2 - r2.mean()) ** 2)
                )
            if verbose:
                print(
                    (
                        "(dx, dy, cc^2, n) : "
                        + str((dx, dy, NCC**2, ov1.shape[0] * ov1.shape[1]))
                    )
                )
        except:
            pass

        # Updating the values
        if NCC**2 > maxNCC:
            maxNCC = NCC**2
            deltax = dx
            deltay = dy

    if returnCC:
        return deltax, deltay, maxNCC
    else:
        return deltax, deltay


def apply2DMirrorPadding(im, newshape):
    """Apply a 2D mirror padding to image

    Parameters
    ----------
    im : ndarray
        2D image to modify

    newshape : tuple (2,)
        Shape of image after the padding (should be bigger than the original shape)

    Returns
    -------
    ndarray
        Modified image.

    """
    # Apply a "mirror" padding
    px = np.int((newshape[0] - im.shape[0]) / 2)
    py = np.int((newshape[1] - im.shape[1]) / 2)

    # To make sure that the shape is good
    if np.mod(im.shape[0], 2) == 0:
        px_end = px
    else:
        px_end = px + 1

    if np.mod(im.shape[1], 2) == 0:
        py_end = py
    else:
        py_end = py + 1

    im_p = np.pad(im, ((px, px_end), (py, py_end)), mode="reflect")

    # Hanning window
    nx = newshape[0] - im.shape[0]
    ny = newshape[1] - im.shape[1]
    hx = np.hanning(nx)
    hy = np.hanning(ny)

    # Extend the window
    hx_extended = np.ones((newshape[0],))
    hy_extended = np.ones((newshape[1],))
    hx_extended[0 : np.int(nx / 2)] = hx[0 : np.int(nx / 2)]
    hx_extended[np.int(nx / 2 + im.shape[0]) : :] = hx[np.int(nx / 2) : :]
    hy_extended[0 : np.int(ny / 2)] = hy[0 : np.int(ny / 2)]
    hy_extended[np.int(ny / 2 + im.shape[1]) : :] = hy[np.int(ny / 2) : :]

    # 2D window map
    hx_extended = np.tile(np.reshape(hx_extended, (newshape[0], 1)), (1, newshape[1]))
    hy_extended = np.tile(np.reshape(hy_extended, (1, newshape[1])), (newshape[0], 1))
    h2d = hx_extended * hy_extended

    # Apply window to extended image
    im_p = im_p * h2d

    return im_p


def warp_rigid(vol, deltas):
    vol_ndim = vol.ndim
    if vol_ndim == 2:
        vol = np.reshape(vol, (vol.shape[0], vol.shape[1], 1))

    new_vol = np.zeros_like(vol)
    nx, ny, nz = new_vol.shape

    # Preparing ranges
    if len(deltas) == 2:
        dx, dy = deltas[:]
        dz = 0
    elif len(deltas) == 3:
        dx, dy, dz = deltas[:]

    # Position in new vol
    px_min = max(dx, 0)
    py_min = max(dy, 0)
    pz_min = max(dz, 0)

    px_max = min(dx + nx, nx)
    py_max = min(dy + ny, ny)
    pz_max = min(dz + nz, nz)

    # Position in old vol
    qx_min = max(-dx, 0)
    qy_min = max(-dy, 0)
    qz_min = max(-dz, 0)

    qx_max = qx_min + (px_max - px_min)
    qy_max = qy_min + (py_max - py_min)
    qz_max = qz_min + (pz_max - pz_min)

    new_vol[px_min:px_max, py_min:py_max, pz_min:pz_max] = vol[
        qx_min:qx_max, qy_min:qy_max, qz_min:qz_max
    ]

    if vol_ndim == 2:
        vol = np.squeeze(vol)
        new_vol = np.squeeze(new_vol)

    return new_vol
