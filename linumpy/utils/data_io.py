#! /usr/bin/env python
# -*- coding: utf-8 -*-
""" This modules contains all methods related to I/O for the slicer data.

.. moduleauthor:: Joël Lefebvre <joel.lefebvre@polymtl.ca>

"""

import csv
import os
import re

import nibabel as nib
import numpy as np
import tables
from PIL import Image
from pathlib import Path


def listSlicesInDir(directory, extension=".nii", returnIndices=False):
    slice_list = list()
    content = os.listdir(directory)
    for elem in content:
        if elem.endswith(extension):
            slice_list.append(os.path.join(directory, elem))

    zlist = getSliceListIndices(slice_list)

    # Sort
    tmp = sorted(zip(zlist, slice_list))
    slice_list = [elem[1] for elem in tmp]
    zlist = [elem[0] for elem in tmp]
    if returnIndices:
        return slice_list, zlist
    else:
        return slice_list


def getSliceListIndices(slice_list):
    zList = list()
    for this_file in slice_list:
        filename_rx = re.compile(".*z(\d+).*")
        tmp = filename_rx.match(this_file)
        if tmp is not None:
            zList.append(int(tmp.groups()[0]))

    return zList


def load_volume(
    directory,
    pos,
    vol_shape,
    prefix="volume",
    extension=".bin",
    precision="float32",
    suffix="",
):
    """Load a volume, given its directory and its position in slicer coordinates.

    :param directory: (str) full path to the directory where the slicer volumes are stored.
    :param pos: (tuple 1x3) slicer coordinate of the volume (x,y,z).
    :param vol_shape: (tuple 1x3) Volume shape
    :param prefix: (str, default='volume') Prefix used at the beginning of each volume.
    :param extension: (str, default='.bin') extension name used for each volume
    :param precision: (str, default='float32') Data format precision.
    :param suffix: (str, default='') Suffix to use for each volume filename.

    :returns: ndarray containing the imported volume.

    .. note:: This method can load nifti files (*.nii* and *.nii.gz*) or binary files saved by matlab using the Fortran Order (*.bin*)

    """
    if len(vol_shape) == 2 or vol_shape[2] == 1:  # This is an image
        prefix = "image"  # FIXME: hardcoded
        precision = "float64"  # FIXME: hardcoded

    filename = os.path.join(
        directory,
        prefix
        + "_"
        + "x%02.0f" % (pos[0])
        + "_"
        + "y%02.0f" % (pos[1])
        + "_"
        + "z%02.0f" % (pos[2])
        + suffix
        + extension,
    )
    return load_volumeByFilename(filename, vol_shape, precision)


def load_slice(directory, z, prototype="slice_z%d", extension=".nii"):
    """Load a slice, given its directory and its position.

    Parameters
    ----------
    directory : str
        Full path to the directory where the slicer volumes are stored.
    z : int
        Slice index to open
    prototype : str
        Filename prototype with a %d field for the slice index.
    extension : str
        File extension. Available are : '.nii', '.h5'

    Returns
    -------
    ndarray or memmap


    """
    try:
        filename = os.path.join(directory, prototype % (z) + extension)
        return load_volumeByFilename(filename)
    except:
        print("Unable to create filename for this slice.")
        return -1


def load_volumeByFilename(filename: str, volshape: tuple=(512, 512, 120), precision: str="float32", convert2Bool: bool=True) -> np.ndarray:
    """Load a volume based on its filename.

    Parameters
    ----------
    filename : str
        Full path to file to load. Available formats are : '.nii', '.nii.gz', '.npy', '.bin' and '.h5'
    volshape : (3,) tuple
        Volume shape (only used for bin files)

    precision : str
        Pixel data format precision (only used for bin files)
    convert2Bool : bool
        Converts volumes containing only 2 unique values (e.g. 255 and 0) as bools.

    Returns
    -------
    ndarray
        An array containing the imported volume

    file handle
        If an h5 file is opened, the file handle is also returned as the second element of a list. This file must be closed by user.

    Notes
    -----
    * This method can load nifti files (*.nii* and *.nii.gz*) or binary files saved by matlab using the Fortran Order (*.bin*)
    * If an error occurs while trying to open the file, -1 is returned.
    * If the file format is '.h5', a memmap is returned.

    """
    available_formats = [".nii", ".nii.gz", ".npy", ".bin", ".h5"]
    extension = None
    for ext in available_formats:
        if filename.endswith(ext):
            extension = ext
    assert extension in available_formats, f"Supported formats are: {available_formats}"

    if extension in [".nii", ".nii.gz"]:
        img = nib.load(filename)
        volume = img.get_fdata()
        if len(np.unique(volume)) == 2 and convert2Bool:
            volume = volume.astype(bool)

    elif extension in [".bin"]:
        dt = precision  # big endian 32-bit floating-point number
        read_order = (
            "C"  # Matlab fwrite save the data in a column order (i.e. Fortran Order)
        )
        volume = np.fromfile(filename, dtype=dt)
        volume = np.reshape(volume, volshape, order=read_order)
        volume = np.swapaxes(volume, 0, 1)  # Matlab inverts the X and Y axis
        if len(np.unique(volume)) == 2 and convert2Bool:
            volume = volume.astype(bool)

    elif extension in [".h5"]:
        f = tables.open_file(filename)
        volume = f.root.x
        if len(np.unique(volume)) == 2 and convert2Bool:
            volume = volume.astype(bool)
    elif extension in [".npy"]:
        volume = np.load(filename)

    return volume


def save_nifti(
    fname, volume, pixDim=(1, 1, 1), pixelFormat=None, intent=1007, expand_dim=True
):
    """Save volume as a nifti format. The origin is assumed to be at the center of the volume.

    Parameters
    ----------
    fname : str
        Complete path to the volume. Extension should be *.nii.gz* of *.nii*
    volume : ndarray
        Volume to save
    pixDim : (3,) tuple
        3x1 array containing the pixel dimension in x,y,z in [micron/pixel].
    pixelFormat : str
        Pixel format to use (Default is the volume dtype)
    intent : int
        Default nifti intent code if ndim > 3 (1007 is vector). Otherwise set to none.
    expand_dim : bool
        Expand the volume dimension? (4th dimension is supposed to be time)

    """
    # Affine transformation matrix
    afft = np.eye(4)
    afft[0, 0] = pixDim[0]  # pixel x-size in mm
    afft[1, 1] = pixDim[1]  # pixel y-size in mm
    afft[2, 2] = pixDim[2]  # pixel z-size in mm

    # TODO: Should specify the main orientation (like SAR+).

    # Defining the volume origin
    if volume.ndim >= 3:
        nx, ny, nz = volume.shape[0:3]
    else:
        nx, ny = volume.shape
        nz = 1
    afft[3, 0] = -np.round(nx / 2) * pixDim[0]  # x origin
    afft[3, 1] = -np.round(ny / 2) * pixDim[1]  # y origin
    afft[3, 2] = -np.round(nz / 2) * pixDim[2]  # z origin

    if volume.dtype is np.dtype(bool):
        volume = 255 * volume.astype(np.uint8)
    elif len(np.unique(np.ravel(volume))) == 2:
        volume = 255 * volume.astype(np.uint8)

    # Create the nibabel img object and adjust header.
    if pixelFormat is None:
        pixelFormat = volume.dtype

    if volume.ndim > 3 and expand_dim:
        img = nib.Nifti1Image(
            np.expand_dims(volume.astype(pixelFormat), 3), afft
        )  # A nifti image
    else:
        img = nib.Nifti1Image(volume.astype(pixelFormat), afft)  # A nifti image
    header = img.header
    header.set_xyzt_units(xyz="micron")
    img.update_header()
    img.set_data_dtype(pixelFormat)

    # Set the nifti intent code if ndim > 3
    if volume.ndim > 3:
        img.header["intent_code"] = intent

    # Set the min and max display intensity
    img.header["cal_min"] = volume.min()
    img.header["cal_max"] = volume.max()

    # Saving image
    try:
        nib.save(img, fname)
    except:
        print("Unable to save the given image")
        raise


def save_rgbNifti(vol, filename):
    """Save volume as a RGB nifti. The origin is assumed to be at the center of the volume.

    :param vol: (ndarray) Volume to save (NxMx3 for a RGB image, NxMxOx3 for a RGB volume)
    :param filename: (str) Complete path to the volume. Extension should be *.nii.gz* of *.nii*

    """
    if vol.ndim == 3:  # This is an image
        nx, ny, nc = vol.shape
        nt = 1
        nz = 1
        vol_reshaped = np.zeros([nx, ny, nz, nt, nc])
        vol_reshaped[:, :, :, 0, 0] = np.reshape(vol[:, :, 0], [nx, ny, 1])
        vol_reshaped[:, :, :, 0, 1] = np.reshape(vol[:, :, 1], [nx, ny, 1])
        vol_reshaped[:, :, :, 0, 2] = np.reshape(vol[:, :, 2], [nx, ny, 1])

    elif vol.ndim == 4:  # This is a volume
        nx, ny, nz, nc = vol.shape
        nt = 1
        vol_reshaped = np.zeros([nx, ny, nz, nt, nc])
        vol_reshaped[:, :, :, 0, 0] = vol[:, :, :, 0]
        vol_reshaped[:, :, :, 0, 1] = vol[:, :, :, 1]
        vol_reshaped[:, :, :, 0, 2] = vol[:, :, :, 2]

    # Affine transformation matrix
    afft = np.eye(4, 4)

    # Saving image
    img = nib.Nifti1Image(vol_reshaped, afft)
    img.set_data_dtype(np.uint8)
    nib.save(img, filename)


def save_png(vol, filename):
    """Save image as a *.png* file.

    :param vol: ndarray to save
    :param filename: Complete path to the volume. Extension should be *.png*

    .. note:: Image intensity is normalized on a 2^8 intensity scale.
    """
    vol = np.squeeze(vol)
    if not (vol.ndim == 2):
        print("Dimension of array should be 2")
        raise

    # Rescale to 0-255 and convert to uint8
    rescaled = (255.0 / vol.max() * (vol - vol.min())).astype(np.uint8)

    im = Image.fromarray(rescaled)
    im.save(filename)


def load_acqinfo_from_csv(filename):
    """Import the acquisition information from a csv file"""
    with open(filename, "rb") as f:
        reader = csv.reader(f)
        info = dict()

        rownum = 0
        for row in reader:
            # Save header row.
            if rownum == 0:
                header = row
            else:
                colnum = 0
                for col in row:
                    if not len(header[colnum]) == 0:
                        info[header[colnum]] = _convert2num(col)
                        colnum += 1
            rownum += 1
        info["dx"] = info["fovX"] / info["nAlinesPerBframe"] * 1000.0
        info["dy"] = info["fovY"] / info["nBframes"] * 1000.0
        if "fovZ" in info:
            info["dz"] = info["fovZ"] / info["nBPixelZ"] * 1000.0
        else:
            info["dz"] = 6.5
        info["sx"] = 1.0
        info["sy"] = 1.0
        info["sz"] = 1.0

        return info


def _convert2num(s):
    """Convert string to number, unless it is a string"""
    a = s  # Default is str
    try:
        a = int(s)
    except ValueError:
        pass  # Not an int

    try:
        a = float(s)
    except ValueError:
        pass  # Not a float

    return a
