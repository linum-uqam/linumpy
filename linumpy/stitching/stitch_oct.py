#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The module stitch_oct uses absolution tile positions to create a mosaic. Various
stitching tools and methods are available.

@author: Joel Lefebvre
"""

import numpy as np
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morpho
import SimpleITK as sitk
from skimage.morphology import ball, disk, medial_axis

from linumpy.postproc import brainMask
from linumpy.preproc import icorr, xyzcorr



def stitch_aipStack(
        aipStack, pos, blendingMethod="diffusion", mask=None, dimension=2, fov=500.0
):
    aipStack = aipStack.squeeze()
    pos = pos.squeeze()

    # Getting the gridshape and volshape based on the aipStack shape
    dimension = int(dimension)
    gridshape = aipStack.shape[0:-dimension]
    volshape = aipStack.shape[-dimension::]

    # Computing a reconstruction mask
    if mask is None:
        mask = np.ones(gridshape, dtype=bool)
    else:
        mask = mask.squeeze()

    # Estimate the resolution for the pos and volshape variables.
    # res = np.abs((pos[0,0,0] - pos[1,0,0])/volshape[0] * 1000)
    res = fov / float(volshape[0])  # mm

    # Convert the positions from (mm) to (px)
    xmin = pos[:, :, 0].min()
    xmax = pos[:, :, 0].max() + res * volshape[0] / 1.0e3
    ymin = pos[:, :, 1].min()
    ymax = pos[:, :, 1].max() + res * volshape[1] / 1.0e3
    mx = int(np.ceil((xmax - xmin) * 1e3 / res))
    my = int(np.ceil((ymax - ymin) * 1e3 / res))
    pos_px = np.copy(pos)
    pos_px[:, :, 0] = np.ceil((pos_px[:, :, 0] - xmin) * 1e3 / res)
    pos_px[:, :, 1] = np.ceil((pos_px[:, :, 1] - ymin) * 1e3 / res)
    pos_px = pos_px.astype(int)
    if dimension == 2:
        mz = 1
    else:
        mz = volshape[2]

    # Creating the mosaic
    mosaic = np.zeros((mx, my, mz), dtype=aipStack.dtype)
    for x in range(gridshape[0]):
        for y in range(gridshape[1]):
            aip = aipStack[x, y, :]
            if dimension == 2:
                aip = np.reshape(aip, (*volshape, mz))
            else:
                aip = np.reshape(aip, volshape)
            this_pos = pos_px[x, y, 0:2]
            if not (mask[x, y]) or np.all(aip == 0):
                continue
            mosaic = addVolumeToMosaic(
                aip, this_pos, mosaic, blendingMethod, factor=2, width=0.3
            )

    return mosaic.squeeze()


def stitch3D(
        data,
        z,
        abspos=None,
        blendingMethod="diffusion",
        mask=None,
        h5File=None,
        factor=3,
        width=1.0,
        feathering=5,
):
    """Create a 3D mosaic with blending for a single slice

    Parameters
    ==========
    data : data object
        Dataset used for iteration
    z : int
        Slice to stitch
    absPos : ndarray
        Matrix containing the position of each volume in the mosaic.
    blendingMethod : str
        Blending method to use (available : 'diffusion', 'medialAxis', 'linear'[buggy], 'none')
    mask : ndarray
        Mask to use during the iteration (1 is added to mosaic, 0 is ignored).
    h5File : h5 File handle
        H5 file handle use to create the mosaic (to use if the stitched slice is too big.)

    Returns
    =======
    ndarray
        Mosaic

    Notes
    =====
    - It is assumed that all volumes have been preprocessed before the stitching
    - No z displacement is used.

    """
    if abspos is None:
        xx, yy = np.meshgrid(
            list(range(data.gridshape[0])),
            list(range(data.gridshape[1])),
            indexing="ij",
        )
        abspos = np.zeros((data.gridshape[0], data.gridshape[1], 1, 2), dtype=int)
        abspos[:, :, 0, 0] = xx * data.volshape[0]
        abspos[:, :, 0, 1] = yy * data.volshape[1]
        abspos = np.tile(abspos, (1, 1, data.gridshape[2], 1))
    abspos = abspos.astype(int)

    # Creating the mosaic array
    nX = abspos[:, :, :, 0].max() + data.volshape[0]
    nY = abspos[:, :, :, 1].max() + data.volshape[1]
    if abspos.shape[3] == 3:
        nZ = abspos[:, :, :, 2].max() + data.volshape[2]
    elif abspos.shape[3] == 2:
        nZ = data.volshape[2]
    mosaic = np.zeros((nX, nY, nZ), dtype=data.format)

    # Loop over all volumes in slice Z
    for vol, pos in data.sliceIterator(z, returnPos=True, mask=mask):
        realPos = abspos[pos[0], pos[1], z, :]
        mosaic = addVolumeToMosaic(
            vol,
            realPos,
            mosaic,
            blendingMethod,
            factor=factor,
            width=width,
            feathering=feathering,
        )

    foo = data.loadFirstVolume()
    if h5File is not None:
        mosaic_h5 = h5File.create_array(h5File.root, "x", mosaic.astype(foo.dtype))
        return mosaic_h5
    else:
        return mosaic.astype(foo.dtype)


def stitch2D(
        data,
        z,
        absPos=None,
        projection="aip",
        blendingMethod="none",
        mask=None,
        eqVol=False,
        factor=3,
        width=1.0,
        zlim=(0, -1),
):
    """Create a simple 2D MIP mosaic without blending.

    Parameters
    ==========
    data : data object
        Dataset used for iteration
    z : int
        Slice to stitch
    absPos : ndarray
        Matrix containing the position of each volume in the mosaic.
    projection : str
        Projection type to use in z direction ('mip', 'aip')
    blendingMethod : str
        Blending method to use (available : 'diffusion', 'medialAxis', 'linear'[buggy], 'none')
    mask : ndarray
        Mask to use during the iteration (1 is added to mosaic, 0 is ignored).

    Returns
    =======
    ndarray
        Mosaic

    Notes
    =====
    - If the tile is 3D, a projection will be done in the z direction.

    """
    if absPos is None:
        xx, yy = np.meshgrid(
            list(range(data.gridshape[0])),
            list(range(data.gridshape[1])),
            indexing="ij",
        )
        absPos = np.zeros((data.gridshape[0], data.gridshape[1], 2), dtype=int)
        absPos[:, :, 0] = xx * data.volshape[0]
        absPos[:, :, 1] = yy * data.volshape[1]

    # Creating the mosaic array
    if absPos.ndim == 3:
        nX = int(absPos[:, :, 0].max() + data.volshape[0])
        nY = int(absPos[:, :, 1].max() + data.volshape[1])
    else:
        nX = int(absPos[:, :, 0, 0].max() + data.volshape[0])
        nY = int(absPos[:, :, 0, 1].max() + data.volshape[1])
    mosaic = np.zeros((nX, nY, 1), dtype=data.format)

    # Loop over all volumes in slice Z
    for vol, pos in data.sliceIterator(z, returnPos=True, mask=mask):
        # Computing MIP in z direction
        if vol.ndim == 3 and vol.shape[2] > 1:
            vol = xyzcorr.cropVolume(vol, zlim=list(zlim))
            if projection == "mip":
                im = np.reshape(
                    vol.max(axis=2), (data.volshape[0], data.volshape[1], 1)
                )
            elif projection == "aip":
                im = np.reshape(
                    vol.mean(axis=2), (data.volshape[0], data.volshape[1], 1)
                )
            else:
                im = np.reshape(
                    vol.max(axis=2), (data.volshape[0], data.volshape[1], 1)
                )
        else:
            im = np.reshape(vol, (data.volshape[0], data.volshape[1], 1))

        if eqVol:
            im = icorr.eqhist(im)

        # Adding vol to mosaic only if its positions is not (-1, -1) (default value for unknown position.)
        if absPos.ndim == 3:
            realPos = absPos[pos[0], pos[1], :]
        else:
            realPos = absPos[pos[0], pos[1], z, :]

        mosaic = addVolumeToMosaic(
            im, realPos, mosaic, blendingMethod, factor=factor, width=width
        )

        del vol

    return np.squeeze(mosaic)


def stitchFromGridMosaic(gridMosaic, pos, imshape=(512, 512)):
    """Transforms a grid mosaic into a stitched mosaic.
    Parameters
    ----------
    * gridMosaic
    * pos
    * imshape

    Output
    ------
    * stitched mosaic

    Notes
    -----
    * Only in 2D for now
    * Performing average blending

    """
    # Computing number of tiles
    frameX = pos.shape[0]
    frameY = pos.shape[1]

    # Creating the mosaic array
    nX = pos[:, :, 0].max() + imshape[0]
    nY = pos[:, :, 1].max() + imshape[1]
    mosaic = np.zeros((nX, nY))
    mosaic_mask = np.zeros((nX, nY))

    # Loop over tiles
    for x in range(frameX):
        for y in range(frameY):
            mPos = pos[x, y, :]
            if mPos[0] != -1 and mPos[1] != -1:
                this_im = gridMosaic[
                          x * imshape[0]: (x + 1) * imshape[0],
                          y * imshape[1]: (y + 1) * imshape[1],
                          ]
                mosaic[
                mPos[0]: mPos[0] + imshape[0], mPos[1]: mPos[1] + imshape[1]
                ] += this_im
                mosaic_mask[
                mPos[0]: mPos[0] + imshape[0], mPos[1]: mPos[1] + imshape[1]
                ] += 1

    mosaic[mosaic_mask > 0] = mosaic[mosaic_mask > 0] / mosaic_mask[mosaic_mask > 0]

    return mosaic


def addVolumeToMosaic(
        volume, pos, mosaic, blendingMethod="diffusion", factor=3, width=1.0, feathering=5
):
    """Add a single volume into a mosaic, using the specified blendingMethod

    Parameters
    ==========
    vol : ndarray
        Volume to add to the mosaic
    pos : (2,) tuple
        Position of this volume in mosaic coordinates
    mosaic : ndarray
        Mosaic in which the volume is stitched
    blendingMethod : str
        Blending method to use (available : 'diffusion', 'medialAxis', 'linear' [buggy], 'average', 'none')

    Returns
    =======
    ndarray
        Updated mosaic

    Notes
    =====
    - The default blending method is 'diffusion'

    """
    # Mask representing the overlap of the mosaic and the new volume
    if volume.ndim == 3:
        nx, ny, nz = volume.shape
    elif volume.ndim == 2:
        nx, ny = volume.shape
        nz = 1
        volume = np.reshape(volume, [nx, ny, nz])

    # Position of tile in mosaic reference frame
    wx = np.int(pos[0])
    wy = np.int(pos[1])

    if len(pos) == 3:
        wz = pos[2]
    else:
        wz = 0

    if mosaic.ndim == 3 and mosaic.shape[2] != 1:
        mask = (
                mosaic[wx: wx + nx, wy: wy + ny, wz: wz + nz].mean(axis=2) > 0
        )  # Todo : Use a 3D mask instead.
    else:
        mask = np.squeeze(mosaic[wx: wx + nx, wy: wy + ny, 0]) > 0

    # Computing the blending weights
    if np.any(mask):
        if blendingMethod == "diffusion":
            alpha = getDiffusionBlendingWeights(mask, factor=factor)

        elif blendingMethod == "medialAxis":
            alpha = getMedialAxisBlendingWeights(mask, feathering=feathering)

        elif blendingMethod == "linear":
            alpha = getLinearBlendingWeights(mask)

        elif blendingMethod == "average":
            alpha = getAverageBlendingWeights(mask)

        else:  # Either none of unknown blending method
            alpha = np.ones([nx, ny])

    else:  # No overlap between mosaic and volume.
        alpha = np.ones([nx, ny])

    if width > 0 and width < 1 and blendingMethod == "diffusion":
        lowThresh = 0.5 * (1.0 - width)
        highThresh = 1.0 - lowThresh
        alpha = (alpha - lowThresh) / float(highThresh - lowThresh)
        alpha[alpha > 1] = 1
        alpha[alpha < 0] = 0

    # Repeating the matrix for each z slice
    alpha = np.tile(np.reshape(alpha, [nx, ny, 1]), [1, 1, nz])

    # Adding the volume to the mosaic using the blending weights computed above
    try:
        mosaic[wx: wx + nx, wy: wy + ny, wz: wz + nz] = (
                volume * alpha
                + (1 - alpha) * mosaic[wx: wx + nx, wy: wy + ny, wz: wz + nz]
        )
    except:
        print("Unable to add volume")
    return mosaic


def stitch_from_deltas(im1, im2, dx, dy, multichannel=False):
    """Stitching two images into a single mosaic.

    Parameters
    ==========
    im1 : ndarray
        First image to blend

    im2 : ndarray
        Second image to blend

    dx : int
        X displacement of im2 IRW im1

    dy : int
        Y displacement of im2 IRW im1

    multichannel : Bool
        If True, im1 will be in R and im2 will be in G of a RGB image.

    Returns
    =======
    ndarray
        Blended image.

    """
    nx, ny = im1.shape

    # Getting position of each image based on dx, dy
    pos1 = [0, 0]
    pos2 = [dx, dy]
    xmin = np.min((0, dx))
    ymin = np.min((0, dy))

    # Position Normalisation (To make sure no position is negative)
    pos1[0] -= xmin
    pos1[1] -= ymin
    pos2[0] -= xmin
    pos2[1] -= ymin

    # Creating mosaic matrix
    xmax = int(np.max((pos1[0] + nx, pos2[0] + nx)))
    ymax = int(np.max((pos1[1] + ny, pos2[1] + ny)))

    if multichannel:  # im1 in R, im2 in G of a RGB image.
        mosaic = np.zeros((xmax, ymax, 3))
        # Filling the mosaic
        mosaic[pos1[0]: pos1[0] + nx, pos1[1]: pos1[1] + ny, 0] = (
                                                                          im1 - im1.min()
                                                                  ) / (im1.max() - im1.min())
        mosaic[pos2[0]: pos2[0] + nx, pos2[1]: pos2[1] + ny, 1] = (
                                                                          im2 - im2.min()
                                                                  ) / (im2.max() - im2.min())

    else:  # Gray level mosaic (no blending)
        mosaic = np.zeros((xmax, ymax), dtype=im1.dtype)
        # Filling the mosaic
        mosaic[pos1[0]: pos1[0] + nx, pos1[1]: pos1[1] + ny] = im1
        mosaic[pos2[0]: pos2[0] + nx, pos2[1]: pos2[1] + ny] = im2

    return mosaic


def getLinearBlendingWeights(mask):  # FIXME: Fix the problem with "L" shape overlap.
    """Computes the linear blending weights

    Parameters
    ==========
    mask : ndarray
        2D mask to use as basis for the blending weights

    Returns
    =======
    ndarray
        2D blending weights

    """
    nx, ny = mask.shape
    xx, yy = np.meshgrid(list(range(nx)), list(range(ny)), indexing="ij")
    tmpx = mask.sum(axis=0)
    tmpy = mask.sum(axis=1)
    xl = np.mean(tmpx[tmpx > 0])  # Size of overlap in x
    yl = np.mean(tmpy[tmpy > 0])  # Size of overlap in y

    if xl < yl:  # Long side is y
        dist = np.abs(xx - nx / 2.0)
    else:  # Long side is x
        dist = np.abs(yy - ny / 2.0)

    alpha = 1 - (dist - dist[mask].min()) / (dist[mask].max() - dist[mask].min())
    alpha[~mask] = 1

    return alpha


def getAverageBlendingWeights(mask):
    """Computes the average blending weights over the mask in ND

    Parameters
    ----------
    mask : ndarray
        ND array decribing the overlap.

    Returns
    -------
    ndarray
        ND blending weights
    """
    alpha = np.ones_like(mask, dtype=float)
    alpha[mask] = 0.5
    return alpha


def getDiffusionBlendingWeights(
        fixedMask, movingMask=None, factor=8, nSteps=5e2, convergence_threshold=1e-4, k=1
):
    """Computes the diffusion blending (based on laplace equation) in 2D or 3D.

    Parameters
    ----------
    fixedMask : ndarray
        Fixed volume mask to use as basis for the blending weights
    movingMask : ndarray
        Moving volume data mask. (If none is given, it assumes that the whole volume contains data.)
    nstep : int
        Number of diffusion step to do.

    Returns
    -------
    ndarray
        2D blending weights

    """

    def laplaceSolverStep(I, mask):
        dI = np.zeros_like(I)
        if I.ndim == 2:
            dI[1:-1, 1:-1] = (
                    I[0:-2, 1:-1]
                    + I[2::, 1:-1]
                    + I[1:-1, 0:-2]
                    + I[1:-1, 2::]
                    - 4 * I[1:-1, 1:-1]
            )
            dI *= mask
            return dI / 4.0
        elif I.ndim == 3:
            dI[1:-1, 1:-1, 1:-1] = (
                    I[0:-2, 1:-1, 1:-1]
                    + I[2::, 1:-1, 1:-1]
                    + I[1:-1, 0:-2, 1:-1]
                    + I[1:-1, 2::, 1:-1]
                    + I[1:-1, 1:-1, 0:-2]
                    + I[1:-1, 1:-1, 2::]
                    - 6 * I[1:-1, 1:-1, 1:-1]
            )
            dI *= mask
            return dI / 6.0

    if movingMask is None:
        movingMask = np.ones_like(fixedMask, dtype=bool)

    # Resampling
    old_shape = fixedMask.shape
    if factor > 1:
        new_shape = list(np.round(np.array(old_shape) / float(factor)).astype(int))
        small_fixedMask = xyzcorr.resampleITK(fixedMask, new_shape, interpolator="NN")
        small_movingMask = xyzcorr.resampleITK(movingMask, new_shape, interpolator="NN")
    else:
        new_shape = old_shape
        small_fixedMask = fixedMask
        small_movingMask = movingMask

    # Getting the boundary of the mask
    if fixedMask.ndim == 2:
        strel = disk(k)
    elif fixedMask.ndim == 3:
        strel = ball(k)

    small_mask = np.logical_and(small_fixedMask, small_movingMask)
    erodedMask = morpho.binary_erosion(small_mask, structure=strel)
    boundary_moving = np.logical_xor(
        small_movingMask, morpho.binary_erosion(small_movingMask, structure=strel)
    )
    boundary_fixed = np.logical_xor(
        small_fixedMask, morpho.binary_erosion(small_fixedMask, structure=strel)
    )
    boundary = np.logical_xor(
        small_mask, morpho.binary_erosion(small_mask, structure=strel)
    )

    # Getting the boundary conditions
    bc = boundary.copy()
    bc = bc * morpho.binary_erosion(small_fixedMask, strel)

    # bc = morpho.binary_erosion(small_fixedMask, strel)*boundary
    dilatedMask = morpho.binary_dilation(
        ~np.logical_or(small_fixedMask, small_mask), structure=strel
    )
    bc = np.zeros(new_shape)
    bc[boundary] = (~dilatedMask[boundary]) * 1.0
    # del dilatedMask

    # Initialize alpha using gaussian smoothing
    alpha = filters.gaussian_filter((bc == 1.0).astype(float), np.array(bc.shape) * 0.1)
    alpha = alpha * small_mask
    alpha = (alpha - alpha.min()) / float(alpha.max() - alpha.min())
    alpha[boundary] = bc[boundary]

    # Solve the Laplace Equation for this geometry
    rms = np.inf
    iStep = 0
    while rms > convergence_threshold and iStep < nSteps:
        dAlpha = laplaceSolverStep(alpha, erodedMask)
        try:
            if np.any(alpha[erodedMask] == 0):
                rms = np.inf
            else:
                rms = np.sqrt(np.mean((dAlpha[erodedMask] / alpha[erodedMask]) ** 2.0))
        except:
            rms = np.inf
        alpha += dAlpha
        iStep += 1

    # Resampling the blending weigths to the original resolution
    alpha[~morpho.binary_dilation(small_mask, strel)] = 1
    alpha[np.logical_xor(small_movingMask, small_mask)] = 0.0
    alpha[np.logical_xor(small_fixedMask, small_mask)] = 1.0
    alpha = 1.0 - alpha

    if factor > 1:
        alpha = xyzcorr.resampleITK(alpha, old_shape, interpolator="linear")

    return alpha


def getMedialAxisBlendingWeights(mask, feathering=5):
    """Computes the medial axis blending weights

    Parameters
    ----------
    mask : ndarray
        2D / 3D mask to use as basis for the blending weights

    featering : int
        Gaussian Kernel size to use to feather the edges of the mask (in pixel)

    Returns
    -------
    ndarray
        2D / 3D blending weights

    Note
    -----
    - 3D medial axis blending has not been tested yet.
    """

    # Adding a 1 pixel padding to make sure the medial axis consider a closed shape
    mask_pad = np.pad(mask, 1, mode="constant", constant_values=0)

    # Computing the medial axis of padded mask
    if mask.ndim == 2:
        mask_m = medial_axis(mask_pad)[1:-1, 1:-1]
    elif mask.ndim == 3:
        mask_itk = sitk.GetImageFromArray(mask_pad.astype(np.uint8))
        skeleton = sitk.BinaryThinning(mask_itk)
        mask_m = sitk.GetArrayFromImage(skeleton)[1:-1, 1:-1, 1:-1]
        # mask_m = medial_axis(mask_pad)[1:-1, 1:-1, 1:-1]

    # Getting the boundary of the mask
    if mask.ndim == 2:
        strel = np.ones([3, 3])
        strel[1, 1] = 0
    elif mask.ndim == 3:
        strel = np.ones([3, 3, 3])
        strel[1, 1, 1] = 0

    erodedMask = morpho.binary_erosion(mask, structure=strel)
    boundary = mask - erodedMask

    # Getting the boundary conditions
    dilatedMask = morpho.binary_dilation(~mask, structure=strel)
    bc = np.zeros(mask.shape)
    bc[boundary] = ~dilatedMask[boundary] * 1.0

    # Applying boundary conditions to medial axis
    mask_m[bc > 0] = 1

    # Filling medial axis and removing spurious edges
    alpha = morpho.binary_opening(morpho.binary_fill_holes(mask_m)).astype(np.float)

    # Apply feathering
    if feathering > 0:
        alpha = filters.gaussian_filter(alpha, (feathering, feathering))

    alpha = 1 - alpha

    return alpha


def addSliceToVolume(
        fixedVolume,
        movingVolume,
        z,
        maskMoving=None,
        blendingFactor=1,
        width=1.0,
        nSteps=5,
        fill_belowMask=True,
):
    """Add a 3D slice to a volume, with Diffusion (Laplacian) blending

    Parameters
    ==========
    fixedVolume : ndarray
        Whole volume in which the slice is added

    movingVolume : ndarray
        3D slice to add in the volume

    z : int
        Depth at which the slice_3d begins

    maskMoving : bool
        If true, the new volume data mask will be used by the blending.

    fill_belowMask : bool
        If true and moving mask is given, every pixel under the mask will be filled to compute the alpha map

    Returns
    =======
    ndarray
        Updated volume

    """
    nx, ny, nz = movingVolume.shape

    if maskMoving is None:
        this_mask = np.ones((nx, ny, nz), dtype=bool)
    elif fill_belowMask:
        interface = xyzcorr.getInterfaceDepthFromMask(maskMoving)
        this_mask = xyzcorr.maskUnderInterface(movingVolume, interface, returnMask=True)
        this_mask[
        :, :, nz - z::
        ] = True  # Every thing under this line is only covered by the moving volume
    else:
        maskMoving = (maskMoving > 0) # Converting to integer array
        this_mask = maskMoving

    if maskMoving is not None:
        maskMoving = (maskMoving > 0)
    # Compute alpha weights
    fixedMask = fixedVolume[:, :, z: z + nz] > 0
    if np.any(fixedMask):
        alpha = getDiffusionBlendingWeights(
            fixedMask, movingMask=this_mask, factor=blendingFactor, nSteps=nSteps
        )
    else:
        alpha = np.ones_like(fixedMask, dtype=bool)

    # Adjusting the width of the blending weights
    if width > 0 and width < 1:
        lowThresh = 0.5 * (1.0 - width)
        highThresh = 1.0 - lowThresh
        alpha = (alpha - lowThresh) / float(highThresh - lowThresh)
        alpha[alpha > 1] = 1
        alpha[alpha < 0] = 0

    if maskMoving is not None and fill_belowMask:
        fixedVolume[:, :, z: z + nz][maskMoving] = (
                alpha * movingVolume + (1 - alpha) * fixedVolume[:, :, z: z + nz]
        )[maskMoving]
    else:
        fixedVolume[:, :, z: z + nz][this_mask] = (
                alpha * movingVolume + (1 - alpha) * fixedVolume[:, :, z: z + nz]
        )[this_mask]

    return fixedVolume
