# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 15:09:07 2014

@author: flesage
"""
import os
import sys
from datetime import datetime

import nibabel as nib
import numpy as np
import scipy.ndimage.filters
import tables
import tables as tb
from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d


def reduce_hd5slice_3d(
    hd5_file, shrink, vox_size, save_dir, new_dtype=np.uint8, returnOutput=False
):
    f = tb.open_file(hd5_file)

    path, fname = os.path.split(hd5_file)
    basename, ext = os.path.splitext(os.path.basename(fname))

    # First filter and find final size
    temp = scipy.ndimage.filters.gaussian_filter(f.root.x, shrink / 2.0)
    # Then reduce
    stack = scipy.ndimage.zoom(temp, 1.0 / shrink, order=1)

    # Normalisation
    oldMaxVal = np.iinfo(stack.dtype).max
    newMaxVal = np.iinfo(new_dtype).max
    stack = (stack * newMaxVal / oldMaxVal).astype(new_dtype)

    if returnOutput:
        return stack
    else:
        # Create affine matrix with the right spacing
        new_spacing = np.multiply(vox_size, shrink)
        afft = np.eye(4)
        afft[0][0] = new_spacing[0]
        afft[1][1] = new_spacing[1]
        afft[2][2] = new_spacing[2]

        # Save in same directory
        img = nib.Nifti1Image(stack, afft)
        shrinked_file = os.path.join(save_dir, basename + ".nii")
        img.to_filename(shrinked_file)
        return shrinked_file


def reduce_hd5slice_2d(hd5_file, shrink, vox_size):
    f = tb.open_file(hd5_file)

    fname, extension = os.path.splitext(hd5_file)
    # First filter and find final size
    temp = scipy.ndimage.filters.gaussian_filter(f.root.x[:, :, 0], shrink / 2.0)
    # Then reduce
    dummy = scipy.ndimage.zoom(temp, 1.0 / shrink, order=1)
    stack = np.zeros((dummy.shape[0], dummy.shape[1], f.root.x.shape[2]))
    stack[:, :, 0] = dummy

    # Loop over slices and shrink
    for i in range(f.root.x.shape[2] - 1):
        print(i)
        # First filter
        temp = scipy.ndimage.filters.gaussian_filter(
            f.root.x[:, :, i + 1], shrink / 2.0
        )
        # Then reduce
        stack[:, :, i + 1] = scipy.ndimage.zoom(temp, 1.0 / shrink, order=1)

    # Create affine matrix with the right spacing
    new_spacing = (1, 1, 1)
    for i in range(3):
        new_spacing[i] = vox_size * shrink
    afft = np.eye(4)
    afft[0][0] = new_spacing[0]
    afft[1][1] = new_spacing[1]
    afft[2][2] = new_spacing[2]
    # Save in same directory
    img = nib.Nifti1Image(stack, afft)

    img.to_filename(fname + ".nii")


def shrink(vol, factor=4):
    """Shrinks an isotropic volume by a given factor
    PARAMETERS
    * vol
    * factor (default=4)
    OUTPUT
    * smaller_vol
    """
    ndim = vol.ndim
    if ndim == 2:
        vol = np.reshape(vol, (vol.shape[0], vol.shape[1], 1))

    nx, ny, nz = vol.shape
    # XY shrinking

    for z in range(nz):
        smaller_slice = zoom(
            gaussian_filter(vol[:, :, z], sigma=(factor / 2.0, factor / 2.0)),
            zoom=(1.0 / factor, 1.0 / factor),
            order=1,
        )
        if z == 0:
            new_x, new_y = smaller_slice.shape
            xy_iso = np.zeros((new_x, new_y, nz), dtype=smaller_slice.dtype)
        xy_iso[:, :, z] = smaller_slice

    smaller_vol = zoom(
        gaussian_filter1d(xy_iso, factor / 2.0, axis=2),
        zoom=(1.0, 1.0, 1.0 / factor),
        order=1,
    )
    if ndim == 2:
        smaller_vol = np.squeeze(smaller_vol.mean(axis=2))

    return smaller_vol


def resample(vol, newshape, order=1):
    """Resamples a volume with new shape
    PARAMETERS
    * vol
    * newshape
    * order (default=1)
    OUTPUT
    * resampled_vol
    """
    ndim = vol.ndim
    if ndim == 2:
        vol = np.reshape(vol, (vol.shape[0], vol.shape[1], 1))
    nx, ny, nz = vol.shape
    factorx = nx / (1.0 * newshape[0])
    factory = ny / (1.0 * newshape[1])
    factorz = nz / (1.0 * newshape[2])

    # XY
    for z in range(nz):
        resampled_slice = zoom(
            vol[:, :, z], zoom=(1.0 / factorx, 1.0 / factory), order=order
        )
        if z == 0:
            new_x, new_y = resampled_slice.shape
            xy_iso = np.zeros((new_x, new_y, nz), dtype=resampled_slice.dtype)
        xy_iso[:, :, z] = resampled_slice

    # Z
    resampled_vol = zoom(xy_iso, zoom=(1.0, 1.0, 1.0 / factorz), order=order)

    if ndim == 2:
        resampled_vol = np.mean(resampled_vol, axis=2)
    return resampled_vol


def create_mosaicThumbnail(
    slice_files, thumbnail_file, shrink_factor=10, vox_size=(1, 1, 1)
):
    """
    Create a shrunken version of mosaic slices for visualization purposes.

    INPUTS
        * slice_files : list of slice mosaic h5 files
        * thumbnail_file : file name of the thumbnail to create (will be a nifti)
        * shrink_factor : (default=10)
        * vox_size : (default=(1,1,1)). Voxel shape in micron/pixel

    OUTPUTS
        * Status (0 if everything worked)

    """

    # Parameters initialization
    f = tb.open_file(slice_files[0])
    nx, ny, nz = f.root.x.shape
    save_dir = os.path.abspath(os.path.dirname(thumbnail_file))
    f.close()

    # Loop over slices and shrink them.
    slice_files.sort()
    nSlices = len(slice_files)

    z = 0
    for zSlice in slice_files:  # Loop over all slices
        print(("Processing file : %s" % (zSlice)))
        sys.stdout.flush()

        # Opening the slice file
        f = tb.open_file(zSlice)
        for iZ in range(f.root.x.shape[2]):  # Loop over all z in a single slice
            print(
                (
                    "DEBUG : iZ=%d (%s)"
                    % (iZ, datetime.now().strftime("%Y-%m-%d@%Hh:%M:%Ss"))
                )
            )
            sys.stdout.flush()
            # First filter and find final size, then reduce
            temp = scipy.ndimage.filters.gaussian_filter(
                f.root.x[:, :, iZ], shrink_factor / 2.0
            )

            if z == 0:  # Create a temporary pytable to contain the intermediate volume.
                dummy = scipy.ndimage.zoom(temp, 1.0 / shrink_factor, order=1)

                _path, fname = os.path.split(zSlice)
                basename, _ext = os.path.splitext(os.path.basename(fname))
                shrunkXY_file = os.path.join(
                    save_dir, "%s_srk%dXY.h5" % (basename, shrink_factor)
                )

                # Opening the pytable
                if nz == 1:
                    dt = tables.Float64Atom()
                else:
                    dt = tables.Float32Atom()
                sXY_h5f = tables.openFile(
                    shrunkXY_file, mode="w", title="Shrink_XY_%dx" % (shrink_factor)
                )
                if nz == 1:
                    stackXY = sXY_h5f.createCArray(
                        sXY_h5f.root,
                        "x",
                        dt,
                        shape=(dummy.shape[0], dummy.shape[1], nSlices),
                    )
                else:
                    stackXY = sXY_h5f.createCArray(
                        sXY_h5f.root,
                        "x",
                        dt,
                        shape=(dummy.shape[0], dummy.shape[1], nSlices * nz),
                    )
                stackXY[:, :, z] = dummy
                del dummy
            else:
                stackXY[:, :, z] = scipy.ndimage.zoom(
                    temp, 1.0 / shrink_factor, order=1
                )
            z += 1
    del temp

    # Reducing the Z resolution
    gauss1d = scipy.ndimage.filters.gaussian_filter1d
    zoom = scipy.ndimage.zoom
    mosaicThumbnail = zoom(
        gauss1d(stackXY, shrink_factor / 2.0, axis=2),
        (1, 1, 1.0 / shrink_factor),
        order=1,
    )

    # Converting to unsigned 8 bit format
    minVal = np.min(mosaicThumbnail)
    maxVal = np.max(mosaicThumbnail)
    mosaicThumbnail = np.uint8(255 * (mosaicThumbnail - minVal) / (maxVal - minVal))

    # Create affine matrix with the right spacing
    new_spacing = vox_size * shrink_factor
    afft = np.eye(4)
    afft[0, 0] = new_spacing[0]
    afft[1, 1] = new_spacing[1]
    afft[2, 2] = new_spacing[2]

    # Save the final mosaic thumbnail
    img = nib.Nifti1Image(mosaicThumbnail, afft)
    img.to_filename(thumbnail_file)
    sXY_h5f.close()

    print("Mosaic thumbnail done !")
    return 0


def join_thumbnailSlices(slice_files, mosaic_file):
    print("Joining slices")
    slice_files.sort()
    nSlices = len(slice_files)
    iZ = 0
    for filename in slice_files:
        print(("Processing slice : %s" % filename))
        img = nib.load(filename)
        vol = img.get_data()
        if iZ == 0:
            nx, ny, nz = vol.shape
            mosaic = np.zeros((nx, ny, nz * nSlices), dtype=vol.dtype)
        zmin = iZ * nz
        zmax = zmin + nz
        mosaic[:, :, zmin:zmax] = vol
        iZ += 1

    print("Saving the mosaic")
    img = nib.Nifti1Image(mosaic, img.get_affine())
    img.to_filename(mosaic_file)
    print("Done !")


def getSliceThumbnail(save_dir, z, shrink_factor=10, vox_size=(1, 1, 1)):
    """To get the thumbnail of a single h5 slice."""
    slice_file = [
        os.path.join(save_dir, "slice_z%d.h5" % (z))
    ]  # This needs to be in a list
    thumbnail_file = os.path.join(
        save_dir, "slice_z%d_shrk%dx.nii" % (z, shrink_factor)
    )
    create_mosaicThumbnail(slice_file, thumbnail_file, shrink_factor, vox_size)


def get_AIPThumbnail(mosaic_file, **kwargs):
    """Saves an Average intensity projection thumbnail as an image file
    PARAMETERS
    * mosaic_file

    ADDITIONAL KEYWORD ARGUMENTS (optional)
    * thumbnail_file : AIP image filename

    NOTE
    * Supported file formats : http://effbot.org/imagingbook/formats.htm
    """
    # Opening the slice
    f = tb.open_file(mosaic_file)

    # Computing Average Intensity projection along z axis
    aip = np.mean(f.root.x, axis=2)

    # Saving as jpg with save filename+.aipThumbnail.jpg
    if "thumbnail_file" in kwargs:
        thumbnail_file = kwargs["thumbnail_file"]
    else:
        file_path = os.path.dirname(mosaic_file)
        thumbnail_file = os.path.basename(mosaic_file)
        thumbnail_file = os.path.splitext(thumbnail_file)[0]
        thumbnail_file = os.path.join(file_path, thumbnail_file + ".aipThumbnail.jpg")

    imsave(thumbnail_file, aip)
    f.close()
