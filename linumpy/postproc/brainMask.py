# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:50:15 2015

@author: LIOM\acastonguay
"""
import os

import nibabel as nib
import numpy as np
from scipy.ndimage import measurements
from scipy.ndimage.filters import gaussian_filter, median_filter
from scipy.ndimage.morphology import binary_fill_holes
from scipy.signal import argrelmin
from scipy.stats import mode
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import ball, dilation, disk, erosion
from skimage.segmentation import watershed

from linumpy.postproc import filters
from linumpy.preproc import icorr, xyzcorr


def volumeMask(vol, dim=0, modality="oct", centerLineCoord=0):
    maskedVol = np.zeros(np.shape(vol))
    for i in range(0, np.shape(vol)[dim]):
        if dim == 0:
            if modality == "oct":
                out = slicerimageMask(vol[i, :, :])
            elif modality == "MRIexvivo":
                out = MRIexvivoMask(vol[i, :, :], centerLineCoord)
            maskedVol[i, :, :] = out
            # print(i)
        elif dim == 1:
            if modality == "oct":
                out = slicerimageMask(vol[:, i, :])
            elif modality == "MRIexvivo":
                out = MRIexvivoMask(vol[:, i, :], centerLineCoord)
            maskedVol[:, i, :] = out
            # print(i)
        elif dim == 2:
            if modality == "oct":
                out = slicerimageMask(vol[:, :, i])
            elif modality == "MRIexvivo":
                out = MRIexvivoMask(vol[:, :, i], centerLineCoord)

            maskedVol[:, :, i] = out
            # print(i)

    return maskedVol


def MRIexvivoMask(im, centerLineCoord):
    if centerLineCoord == 0:
        x = np.shape(im)[0] / 2
        y = np.shape(im)[1] / 2
    else:
        x = centerLineCoord[0]
        y = centerLineCoord[1]

    val = threshold_otsu(im)
    initialmask = im > val

    im = im * initialmask
    selem = disk(3)  ###disk size of 6 is optimal for oct data
    eroded_im = erosion(im, selem)
    dilated_im = dilation(im, selem)
    edge = dilated_im - eroded_im

    val = threshold_otsu(edge)
    mask = edge < val
    labels = label(mask).astype("float")
    l = labels == labels[x, y]
    l = dilation(l, disk(4))
    l = binary_fill_holes(l == 1)
    return l
    # l=labels==mode(labels[np.nonzero(labels)],axis=None)[0]


def slicerimageMask(im):
    # selem = disk(1)
    edge = im
    # edge=morph.closing(im, selem)
    # edge=im[10:np.shape(im)[0],10:np.shape(im)[1]]
    # Log scale (enhances countours)
    zero_values = edge[0::] == 0
    nonzero_values = edge[0::] > 0
    edge[zero_values] = np.mean(edge[nonzero_values]) / 2
    edge = np.log10(edge)
    edge = gaussian_filter(edge, 5)

    # Edge detection
    # selem = disk(6)    ###disk size of 6 is optimal for oct data
    # eroded_im = erosion(edge, selem)
    # dilated_im = dilation(edge, selem)
    # edge = dilated_im - eroded_im

    # Mask
    val = threshold_otsu(edge)
    mask = edge > val

    # val=threshold_otsu(edge[mask])
    # mask=edge<val
    # Segmentation
    # labels = label(mask)
    # labels = mask*labels
    # if np.max(labels)==0:
    #     l=labels
    # else:
    #    l=labels==mode(labels[np.nonzero(labels)],axis=None)[0]
    #
    # mask=l
    # Fill holes
    # mask = binary_fill_holes(mask==1)
    # mask = binary_fill_holes(mask==0, selem)

    # second segmentation to take out large countour made by large selem disk
    #    eroded_im = erosion(l, selem)
    #    dilated_im = dilation(l, selem)
    #    edge = dilated_im - eroded_im
    #    labels = label(edge)
    #    if np.max(labels)==0:
    #        l=labels
    #    else:
    #        l=labels==mode(labels[np.nonzero(labels)],axis=None)[0]
    #
    #    #thrid segmentation to take out large countour made by gaussian filtering
    #    eroded_im = erosion(l, selem)
    #    dilated_im = dilation(l, selem)
    #    edge = dilated_im - eroded_im
    #    labels = label(edge)
    #    if np.max(labels)==0:
    #        l=labels
    #    else:
    #        l=labels==mode(labels[np.nonzero(labels)],axis=None)[0]
    #
    #    mask = np.zeros(np.shape(im))
    #    mask[10:np.shape(im)[0],10:np.shape(im)[1]]=l

    # cleanimage = mask*im

    # tissueMask = mask[0::]==0
    # outsideMask=im[0::]!=0
    # combo = tissueMask*outsideMask
    # meanbckgrnd = np.mean(im[combo])
    # for i in range(0,np.shape(im)[0]):
    # for j in range(0,np.shape(im)[1]):
    # if cleanimage[i,j] < meanbckgrnd:
    # mask[i,j] = 0
    # mask = binary_fill_holes(mask==1)
    # Multiply mask with input image

    return mask


def MRIexvivo3d(vol, shrinkFactor=1):
    saveFlag = 0
    if isinstance(vol, str):
        fpath, fname = os.path.split(vol)
        root, ext = os.path.splitext(vol)
        saveFlag = 1
        im = nib.load(vol)
        vol = im.get_data()
        vol = np.squeeze(vol)
    smaller_vol = filters.shrink(vol, shrinkFactor)
    nx, ny, nz = smaller_vol.shape
    smaller_vol = gaussian_filter(smaller_vol, 2)
    # Edge detection
    eroded_vol = erosion(smaller_vol, ball(3))
    dilated_vol = dilation(smaller_vol, ball(3))
    edge_3d = dilated_vol - eroded_vol
    markers_3d = np.zeros_like(smaller_vol)
    markers_3d[edge_3d < edge_3d.max() * 0.05] = 1
    labels = label(markers_3d).astype("float")
    labels = labels * markers_3d
    hist = np.histogram(labels, np.max(labels))
    mainFeature = np.argmax(hist[0][1:]) + 1
    labels[labels != mainFeature] = 0
    # Fill holes (in 3D)
    brainmask3d = binary_fill_holes(labels == mainFeature).astype("float")

    brainmask3d = gaussian_filter(brainmask3d, 1.2)
    brainmask3d[brainmask3d > 0] = 1

    # Fill holes (in 2D)
    for x in range(brainmask3d.shape[0]):
        brainmask3d[x, :, :] = binary_fill_holes(brainmask3d[x, :, :])
    for y in range(brainmask3d.shape[1]):
        brainmask3d[:, y, :] = binary_fill_holes(brainmask3d[:, y, :])
    for z in range(brainmask3d.shape[2]):
        brainmask3d[:, :, z] = binary_fill_holes(brainmask3d[:, :, z])

    brainmask3d = binary_fill_holes(brainmask3d).astype("float")
    brainmask3d = filters.shrink(brainmask3d, (1.0 / shrinkFactor))
    if saveFlag == 1:
        affine = np.eye(4)
        brainmask3d = brainmask3d.astype("uint32")
        vol1Nifti = nib.Nifti1Image(brainmask3d, affine)
        nib.save(vol1Nifti, root + "_Mask.nii")
        masked = vol * brainmask3d
        vol1Nifti = nib.Nifti1Image(masked, affine)
        nib.save(vol1Nifti, root + "_Masked.nii")

    return brainmask3d

    # markers_3d[smaller_vol > threshold_otsu(smaller_vol)] = 2

    # seg_3d = watershed(edge_3d, markers_3d)


def slicer3d(
    vol,
    shrinkFactor=4,
    morphoEdgeKernelRadius=3,
    uselog=1,
    saveFlag=False,
    medianFilter=0,
    eqHist=False,
):
    """Segments a 3D volume reconstructed from the slicer data
    Parameters
    ----------
    vol : ndarray
        Volume to segment
    shrinkFactor : int
        Shrinking factor to use for the segmentation (to reduce computational resources)

    Returns
    -------
    ndarray
        Mask of the original volume

    """
    if isinstance(vol, str):

        fpath, fname = os.path.split(vol)
        root, ext = os.path.splitext(vol)
        saveFlag = 1
        im = nib.load(vol)
        vol = im.get_data()
        vol = np.squeeze(vol)

    if eqHist:
        vol = icorr.eqhist(vol)

    if medianFilter > 0:
        vol = median_filter(vol, medianFilter)

    if shrinkFactor > 1:
        newshape = list(np.ceil(np.array(vol.shape) / float(shrinkFactor)).astype(int))
        # smaller_vol = filters.shrink(vol, shrinkFactor)
        smaller_vol = xyzcorr.resampleITK(vol, newshape)
    else:
        smaller_vol = vol.astype(float)
    if uselog == 1:
        smaller_vol = smaller_vol + 1
        smaller_vol = np.log(smaller_vol)

    # Edge detection
    eroded_vol = erosion(smaller_vol, ball(morphoEdgeKernelRadius))
    dilated_vol = dilation(smaller_vol, ball(morphoEdgeKernelRadius))
    edge_3d = dilated_vol - eroded_vol

    markers_3d = np.zeros_like(smaller_vol)
    markers_3d[edge_3d < edge_3d.max() * 0.05] = 1
    markers_3d[smaller_vol > threshold_otsu(smaller_vol)] = 2

    seg_3d = watershed(edge_3d, markers_3d)

    # Fill holes (in 3D)
    brainmask3d = binary_fill_holes(seg_3d == 2)

    # Fill holes (in 2D)
    for x in range(brainmask3d.shape[0]):
        brainmask3d[x, :, :] = binary_fill_holes(brainmask3d[x, :, :])
    for y in range(brainmask3d.shape[1]):
        brainmask3d[:, y, :] = binary_fill_holes(brainmask3d[:, y, :])
    for z in range(brainmask3d.shape[2]):
        brainmask3d[:, :, z] = binary_fill_holes(brainmask3d[:, :, z])

    # Refill holes in 3D (in case some were missed)
    brainmask3d = binary_fill_holes(brainmask3d)

    # Label all connected regions in the mask
    labels = label(brainmask3d, connectivity=2).astype("float")
    hist = np.histogram(labels, int(np.max(labels)))

    # Removing all connected regions that are not part of the biggest region
    labels[labels != np.argmax(hist[0][1:]) + 1] = 0
    labels[labels == np.argmax(hist[0][1:]) + 1] = 1

    # mask = filters.resample(labels, vol.shape, order=0)
    mask = xyzcorr.resampleITK(labels.astype(bool), vol.shape)

    if saveFlag:
        affine = np.eye(4)
        mask = mask.astype("uint8")
        vol1Nifti = nib.Nifti1Image(mask, affine)
        nib.save(vol1Nifti, root + "_Mask.nii")
        masked = vol * mask
        masked = masked.astype("float32")
        vol1Nifti = nib.Nifti1Image(masked, affine)
        nib.save(vol1Nifti, root + "_Masked.nii")
    else:
        return mask


def get3dMask(filename, shrinkFactor=3):
    """Computes the volume mask in 3D.
    PARAMETERS
    * vol : ndarray to segment.
    * shrinkFactor (default=3) : Shrinking factor to use (to reduce memory usage)
    OUTPUT
    * mask : ndarray containing the volume mask.
    """

    fpath, fname = os.path.split(filename)
    root, ext = os.path.splitext(filename)
    im = nib.load(filename)
    vol = im.get_data()
    vol = np.squeeze(vol)
    # Reducing size of volume to limit resources
    smaller_vol = filters.shrink(vol, shrinkFactor)
    nx, ny, nz = smaller_vol.shape

    # Croping sides of volume (contain nothing)
    xy_mip = smaller_vol.mean(axis=2)
    edge_xy = dilation(xy_mip, disk(2)) - erosion(xy_mip, disk(2))

    x_profile = np.sum(edge_xy, axis=1)
    y_profile = np.sum(edge_xy, axis=0)

    # 2nd order derivative
    x_pp = np.gradient(np.gradient(x_profile))
    y_pp = np.gradient(np.gradient(y_profile))

    peaks_x = argrelmin(x_pp, order=10)[0]
    peaks_y = argrelmin(y_pp, order=10)[0]

    xmin = peaks_x[0] + 1
    xmax = peaks_x[-1] - 1
    ymin = peaks_y[0] + 1
    ymax = peaks_y[-1] - 1

    smaller_vol = smaller_vol[xmin:xmax, ymin:ymax, :]

    # Edge detection
    eroded_vol = erosion(smaller_vol, ball(2))
    dilated_vol = dilation(smaller_vol, ball(2))
    edge_3d = dilated_vol - eroded_vol

    markers_3d = np.zeros_like(smaller_vol)
    markers_3d[edge_3d < edge_3d.max() * 0.05] = 1
    markers_3d[smaller_vol > threshold_otsu(smaller_vol)] = 2

    seg_3d = watershed(edge_3d, markers_3d)

    # Fill holes (in 3D)
    brainmask3d = binary_fill_holes(seg_3d == 2)

    # Fill holes (in 2D)
    for x in range(brainmask3d.shape[0]):
        brainmask3d[x, :, :] = binary_fill_holes(brainmask3d[x, :, :])
    for y in range(brainmask3d.shape[1]):
        brainmask3d[:, y, :] = binary_fill_holes(brainmask3d[:, y, :])
    for z in range(brainmask3d.shape[2]):
        brainmask3d[:, :, z] = binary_fill_holes(brainmask3d[:, :, z])

    brainmask3d = binary_fill_holes(brainmask3d)

    # Only keep the biggest non-zero component
    mask_label, num_features = measurements.label(brainmask3d)
    hist = list()
    for i in range(num_features):
        hist.append(np.sum(mask_label == i))
    mainFeature = np.argmax(hist[1:]) + 1
    brainmask3d[mask_label != mainFeature] = 0

    # Reshape mask to original dimension.
    mask = np.zeros((nx, ny, nz), dtype=bool)
    mask = filters.resample(brainmask3d, vol.shape)
    affine = np.eye(4)
    mask = mask.astype("uint32")
    vol1Nifti = nib.Nifti1Image(mask, affine)
    nib.save(vol1Nifti, root + "_Mask.nii")
    masked = vol * mask
    vol1Nifti = nib.Nifti1Image(masked, affine)
    nib.save(vol1Nifti, root + "_Masked.nii")


def activeContourMask(filename, levelset=None, num_iters=100):
    from scipy.ndimage import filters

    fpath, fname = os.path.split(filename)
    root, ext = os.path.splitext(filename)
    im = nib.load(filename)
    vol = im.get_data()
    vol = np.squeeze(vol)
    value = threshold_otsu(vol)
    mask = vol < value
    mask = binary_fill_holes(mask == 0)
    eroded = erosion(mask, ball(2))
    from postproc import morphsnakes

    macwe = morphsnakes.MorphACWE(eroded, smoothing=1, lambda1=1, lambda2=1)
    macwe.levelset = morphsnakes.circle_levelset(
        np.shape(vol),
        (
            np.round(np.shape(vol)[0] / 2),
            np.round(np.shape(vol)[1] / 2),
            np.round(np.shape(vol)[2] / 2),
        ),
        round(np.max(np.shape(vol)) / 8),
    )
    if levelset is not None:
        macwe.levelset = levelset
    for i in range(num_iters):
        macwe.step()
        mask = macwe.levelset
        if float(float(i) / 10).is_integer() == 1:
            print(("Iteration %s/%s..." % (i, num_iters)))

    dilated = dilation(mask, ball(3))
    masked = vol * dilated
    affine = np.eye(4)
    vol1Nifti = nib.Nifti1Image(dilated, affine)
    nib.save(vol1Nifti, root + "_Mask.nii")
    vol1Nifti = nib.Nifti1Image(masked, affine)
    nib.save(vol1Nifti, root + "_Masked.nii")


def newMask(filename, shrinkFactor=1, morphsnake=1, levelset=None, num_iters=100):
    fpath, fname = os.path.split(filename)
    root, ext = os.path.splitext(filename)
    im = nib.load(filename)
    vol = im.get_data()
    vol = np.squeeze(vol).astype("float32")
    if shrinkFactor != 1:
        newshape = list(np.ceil(np.array(vol.shape) / float(shrinkFactor)).astype(int))
        smaller_vol = xyzcorr.resampleITK(vol, newshape)
    else:
        smaller_vol = filters.shrink(vol, 1)
    value = threshold_otsu(smaller_vol)
    mask = smaller_vol < (value * 1.2)
    mask = binary_fill_holes(mask == 0)
    temp = erosion(mask, ball(2))

    eroded = erosion(smaller_vol, ball(2))
    eroded[eroded < np.max(eroded) * 0.075] = 0  # 2000

    eroded_vol = erosion(eroded, ball(3))
    dilated_vol = dilation(eroded, ball(3))
    edge = dilated_vol - eroded_vol

    edge[edge > np.max(edge) / 10] = 0
    edge[edge != 0] = 1

    mask = temp + edge
    mask[mask > 0] = 1

    if morphsnake:
        from slicercode.postproc import morphsnakes

        macwe = morphsnakes.MorphACWE(mask, smoothing=1, lambda1=1, lambda2=1)
        macwe.levelset = morphsnakes.circle_levelset(
            np.shape(smaller_vol),
            (
                np.round(np.shape(smaller_vol)[0] / 2),
                np.round(np.shape(smaller_vol)[1] / 2),
                np.round(np.shape(smaller_vol)[2] / 2),
            ),
            round(np.max(np.shape(smaller_vol)) / 16),
        )
        if levelset is not None:
            macwe.levelset = levelset
        for i in range(num_iters):
            macwe.step()
            mask = macwe.levelset
            if float(float(i) / 10).is_integer() == 1:
                print(("Iteration %s/%s..." % (i, num_iters)))

    brainmask3d = binary_fill_holes(mask == 1)

    # Fill holes (in 2D)
    for x in range(brainmask3d.shape[0]):
        brainmask3d[x, :, :] = binary_fill_holes(brainmask3d[x, :, :])
    for y in range(brainmask3d.shape[1]):
        brainmask3d[:, y, :] = binary_fill_holes(brainmask3d[:, y, :])
    for z in range(brainmask3d.shape[2]):
        brainmask3d[:, :, z] = binary_fill_holes(brainmask3d[:, :, z])

    brainmask3d = binary_fill_holes(brainmask3d).astype("float32")

    labels = label(brainmask3d, connectivity=1).astype("float32")
    labels[labels != mode(labels[labels != 0], axis=None)[0]] = 0
    labels[labels != 0] = 1

    dilated = dilation(labels, ball(2))
    if shrinkFactor != 1:
        dilated = xyzcorr.resampleITK(dilated.astype("uint8"), vol.shape)
    masked = vol * dilated
    masked = masked.astype("float")
    affine = np.eye(4)
    vol1Nifti = nib.Nifti1Image(dilated, affine)
    nib.save(vol1Nifti, root + "_NewMask.nii")
    vol1Nifti = nib.Nifti1Image(masked, affine)
    nib.save(vol1Nifti, root + "_NewMasked.nii")
