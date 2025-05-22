# -*- coding: utf-8 -*-
"""
This module contains all the functions used to find the tissue interface position
in Z and to correct it prior to shift and stitch operations

Created on Thu Nov 13 16:11:07 2014

@author: joel
"""
import os
import pickle as pcl
import sys

import numpy as np
import scipy.ndimage as ndimage
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter
from skimage.filters import threshold_otsu

import linumpy.utils.data_io as io
from linumpy.preproc import xyzcorr


def findInterface(vol, zPad=10, zMin=20, zMax=80, method="derivative"):
    """Finds the tissue interface using the meanvolume and save it in a numpy file

    PARAMETERS
        vol
        zPad (default=10) Number of z slices to join together when computing the discrete intensity difference.
        zMin
        zMax
        method = 'threshold','difference','derivative'

    OUTPUT
        none
    """
    nx, ny, nz = vol.shape
    # Filtering along the z dimension
    # vol = ndimage.gaussian_filter1d(vol,sigma=5,axis=2)
    vol = ndimage.gaussian_filter(vol, sigma=5)
    # vol = ndimage.median_filter(vol,size=5)

    if method == "difference":  # Similar to alex's original method
        zRange = list(range(zMin + zPad, zMax - zPad))
        M = np.zeros((nx, ny, len(zRange)), dtype=vol.dtype)
        xx, yy = np.meshgrid(list(range(nx)), list(range(ny)))
        for k in range(len(zRange)):
            z = zRange[k]
            shallowI = np.sum(vol[xx, yy, z - zPad : z], axis=2)
            deepI = np.sum(vol[xx, yy, z : z + zPad], axis=2)
            M[xx, yy, k] = deepI / shallowI
        interface = np.argmax(M, axis=2) + zRange[0]

    elif method == "derivative":
        subvol = vol
        # subvol = subSampling(vol,nGridX=30,nGridY=30)
        vol_cum = np.cumsum(subvol, axis=2)
        vp3_cum = ndimage.gaussian_filter1d(
            vol_cum, sigma=1, axis=2, order=3
        )  # 2st order derivative in z
        subVolInterface = np.zeros(subvol.shape[0:2])

        l3_thresh = 0.25
        depth_thresh = nz / 5
        for x in range(subvol.shape[0]):
            for y in range(subvol.shape[1]):

                # Finding all local maxima of the itensity rate of acceleration (I^4)
                lmax = local_maxima(np.arange(nz), vp3_cum[x, y, :])
                for l in lmax:
                    cond1 = vp3_cum[x, y, l] / np.max(vp3_cum[x, y, :]) > l3_thresh
                    cond2 = l > depth_thresh
                    if cond1 and cond2:
                        subVolInterface[x, y] = l
                        break

        # Remove outliers
        # outliers = (abs(subVolInterface - np.mean(subVolInterface)) > 2*np.std(subVolInterface))
        # subVolInterface[outliers] = np.mean(subVolInterface[~outliers])
        xRange = np.linspace(0, nx - 1, subvol.shape[0])
        yRange = np.linspace(0, ny - 1, subvol.shape[1])
        xx, yy = np.meshgrid(xRange, yRange)
        f = interpolate.RectBivariateSpline(xRange, yRange, subVolInterface)

        interface = f(list(range(nx)), list(range(ny))) + 5

    elif method == "threshold":
        threshold = threshold_otsu(vol[:, :, 0 : np.round(0.5 * nz)])
        tissuType = np.zeros(vol.shape)
        tissuType[vol > threshold] = 1

        cumulator = np.zeros([nx, ny])
        for k in range(zMin, zMax):
            cumulator += tissuType[:, :, k]
            interface[(cumulator == 1) * (interface == 0)] = k
            if not (np.any(interface == 0)):
                break
    else:
        print("Unknown method. Aborting.")
        sys.exit()

    # Interface Regularization
    #    interface = np.around(ndimage.gaussian_filter(interface, sigma=2))
    interface = ndimage.gaussian_filter(interface, sigma=2)

    #    return interface.astype(np.uint32)
    return interface.astype(np.float32)


def getCorrMatrices(dMap, vol_shape):
    """
    PARAMETERS
        @param dMap : 2d ndarray giving the tissue interface position.
        @param vol_shape

    RETURNS
        xCorr,yCorr, zCorr
    """
    nx, ny, nz = vol_shape[:]
    zRange = np.around(dMap.max() - dMap.min())

    xx, yy = np.meshgrid(list(range(nx)), list(range(ny)), indexing="ij")
    zz = (dMap - dMap.min()).astype(int)

    xCorr = np.zeros([nx, ny, nz - zRange], dtype=int)
    yCorr = np.zeros([nx, ny, nz - zRange], dtype=int)
    zCorr = np.zeros([nx, ny, nz - zRange], dtype=int)

    for z in range(nz - np.int(zRange)):
        xCorr[:, :, z] = xx
        yCorr[:, :, z] = yy
        zCorr[:, :, z] = zz[xx, yy]
        zz += 1

    return xCorr, yCorr, zCorr


def loadInterface(x, y, z, save_dir):
    """
    PARAMETERS
        x
        y
        z
        save_dir
    RETURNS
        interface
    """

    filename = os.path.join(
        save_dir, "interface_x%02.0f_y%02.0f_z%02.0f.npy" % (x, y, z)
    )
    interface = np.load(filename)
    return interface


def saveCorrMatrices(xCorr, yCorr, zCorr, save_dir):
    """To save the correction matrices in the 'save_dir' folder for later uses

    PARAMETERS
    xCorr
    yCorr
    zCorr

    RETURNS
    0 if it worked, -1 if something failed
    """
    # corrIdx = {'x':xCorr,'y':yCorr,'z':zCorr}
    xcorr_file = os.path.join(save_dir, "xcorr.npy")
    ycorr_file = os.path.join(save_dir, "ycorr.npy")
    zcorr_file = os.path.join(save_dir, "zcorr.npy")
    try:
        np.save(xcorr_file, xCorr)
        np.save(ycorr_file, yCorr)
        np.save(zcorr_file, zCorr)
        return 0
    except:
        print("Unable to save the correction matrices")
        return -1


def loadCorrMatrices(save_dir):
    """To load the correction matrices, if they exists

    PARAMETERS
    save_dir

    OUTPUT
    xCorr, yCorr, zCoor (if it works), -1 if it doesn't
    """
    xcorr_file = os.path.join(save_dir, "xcorr.npy")
    ycorr_file = os.path.join(save_dir, "ycorr.npy")
    zcorr_file = os.path.join(save_dir, "zcorr.npy")
    try:
        xCorr = np.load(xcorr_file)
        yCorr = np.load(ycorr_file)
        zCorr = np.load(zcorr_file)
        return xCorr, yCorr, zCorr
    except:
        print("Unable to load xCorr, yCorr or zCorr")
        return -1


def local_maxima(xval, yval):
    xval = np.asarray(xval)
    yval = np.asarray(yval)

    sort_idx = np.argsort(xval)
    yval = yval[sort_idx]
    gradient = np.diff(yval)
    maxima = np.diff((gradient > 0).view(np.int8))
    return np.concatenate(
        (([0],) if gradient[0] < 0 else ())
        + (np.where(maxima == -1)[0] + 1,)
        + (([len(yval) - 1],) if gradient[-1] > 0 else ())
    )


def subSampling(vol, nGridX=15, nGridY=15):
    nx, ny, nz = vol.shape
    subVol = np.zeros([nGridX, nGridY, nz], dtype=vol.dtype)
    xRange = np.round(np.linspace(0, nx - 1, nGridX))
    yRange = np.round(np.linspace(0, ny - 1, nGridY))
    yy, xx = np.meshgrid(xRange.astype(np.int), yRange.astype(np.int))
    subVol = vol[xx, yy, :]

    return subVol


def fixAllZ(data_dir, save_dir, frameX, frameY, frameZ, vol_shape, method="derivative"):
    """Loops over all volumes in data_dir and fix the Z tissue interface depth

    PARAMETERS
        data_dir
        save_dir
        frameX
        frameY
        frameZ

    RETURNS
        new_shape
    """

    # Loop over all volumes
    for x in range(1, frameX + 1):
        for y in range(1, frameY + 1):
            for z in range(1, frameZ + 1):
                # Load the original volume
                pos = [x, y, z]
                vol = io.load_volume(data_dir, pos, vol_shape)

                # Find the interface for this volume
                interface = findInterface(vol, method=method)

                # Correct the interface Z position (Size of vol is now nx,ny and 50)
                # crankedVol = cranckInterface(vol,interface)

                # Save the interface coord. in the result dir in a numpy file
                filename = os.path.join(
                    save_dir, "interface_x%02.0f_y%02.0f_z%02.0f.npy" % (x, y, z)
                )
                np.save(filename, interface)

    # new_shape = crankedVol.shape
    return 0


def get_meanvolume(
    vol_dir, save_dir, frameX, frameY, frameZ, vol_shape, parallel="None"
):
    """Compute the mean volume for normalization purposes

    PARAMETERS
        vol_dir
        save_dir
        frameX
        frameY
        frameZ
        vol_shape
        parallel='MPI',or 'qsub', or 'none'

    OUTPUT
        None (mean volume is saved in the same directory as meanvolume.npy)
    """

    if parallel == "mpi":
        # TODO implement the mpi mean volume method
        print("= MPI mean volume computation is not implemented yet")

    elif parallel == "qsub":
        # To disable numpy and scipy multithreading
        # os.environ["MKL_NUM_THREADS"] = "1"
        # os.environ["NUMEXPR_NUM_THREADS"] = "1"
        # os.environ["OMP_NUM_THREADS"] = "1"

        print("= Computing mean volume in parallel using qsub")
        param_files = list()
        job_scripts = list()
        jobIds = list()

        for z in range(1, frameZ + 1):
            # Saving parameters & jobscripts
            this_param = os.path.join(save_dir, "meanVparam_z%d.pcl" % (z))
            pythonscript = os.path.join(stitching.__path__[0], "qsub_meanVolume.py")
            this_jobname = "meanV_z%d" % (z)

            f = open(this_param, "w")
            pcl.dump((vol_dir, save_dir, frameX, frameY, z, vol_shape), f)
            f.close()

            # Launching mean volume jobs
            this_jobscript = qsub_utils.defaultJobscript(
                save_dir, this_jobname + ".sh", this_jobname, pythonscript, this_param
            )
            this_jobId = qsub_utils.submit(this_jobscript)

            param_files.append(this_param)
            job_scripts.append(this_jobscript)
            jobIds.append(this_jobId)

        # Waiting for all mean volume jobs to finish
        if len(jobIds) > 0:
            foo = qsub_utils.waitForQJobs(jobIds)

        meanVolume_file = list()

        # Joining all mean volumes and saving to file
        meanvolume = np.zeros(vol_shape, dtype=np.float32)
        nVol = 0
        for z in range(1, frameZ + 1):
            filename = os.path.join(save_dir, "meanvolume_z%d.npy" % (z))
            meanVolume_file.append(filename)
            vol = np.load(filename)
            if nVol == 0:
                meanvolume = vol
            else:
                meanvolume = (nVol * meanvolume + vol) / (nVol + 1)
            nVol += 1
        filename = os.path.join(save_dir, "meanvolume.npy")
        np.save(filename, meanvolume)

        # Removing the qsub jobscripts and parameters files, and zfiles
        for f in param_files:
            try:
                os.remove(f)
            except:
                print(("Unable to remove file : %s" % (f)))
        for f in job_scripts:
            try:
                os.remove(f)
            except:
                print(("Unable to remove file : %s" % (f)))
        for f in meanVolume_file:
            try:
                os.remove(f)
            except:
                print(("Unable to remove file : %s" % (f)))

    else:
        print("= Computing mean volume slice-by-slice")
        meanvolume = np.zeros(vol_shape, dtype=np.float32)
        nVol = 0
        for z in range(1, frameZ + 1):
            vol = get_meanvolumeZ(
                vol_dir, save_dir, frameX, frameY, z, vol_shape, save_output=False
            )
            if nVol == 0:
                meanvolume = vol
            else:
                meanvolume = (nVol * meanvolume + vol) / (nVol + 1)
            nVol += 1
        filename = os.path.join(save_dir, "meanvolume.npy")
        np.save(filename, meanvolume)


def get_meanvolumeZ(
    vol_dir, save_dir, frameX, frameY, z, vol_shape, save_output=True, **kwargs
):
    """Compute the mean volume for a given z slice

    PARAMETERS
        :param vol_dir: Path to the data directory
        :param save_dir: Path to the save directory
        :param frameX: Number of slicer frames in X
        :param frameY: Number of slicer frames in Y
        :param z: Slice to use
        :param vol_shape: Tuple giving the size of a single volume
        :param save_output: True (otherwise the output is returned)

    OPTIONAL PARAMETERS
        :param smoothSize : gaussian filter kernel size (pixel), (no smoothing otherwise)

    OUTPUT
        None -- (if save_output=True)
        meanvolume -- (if save_output=False)
    """

    meanvolume = np.zeros(vol_shape, dtype=np.float32)
    nVol = frameX * frameY
    nVol_i = 1.0 / nVol

    for y in range(1, frameY + 1):
        for x in range(1, frameX + 1):
            pos = (x, y, z)
            vol = io.load_volume(vol_dir, pos, vol_shape)
            if not (isinstance(vol, np.ndarray)):
                print(
                    (
                        "Unable to load volume (x,y,z) : (%d,%d,%d), ignoring it in the mean volume computation"
                        % (x, y, z)
                    )
                )
            else:
                if "smoothSize" in kwargs:  # Smoothing volume
                    meanvolume += nVol_i * gaussian_filter(
                        vol, sigma=kwargs["smoothSize"], order=[0, 1]
                    )
                else:
                    meanvolume += nVol_i * vol

    if save_output:
        try:
            meanv_file = os.path.join(save_dir, "meanvolume_z%d.npy" % (z))
            np.save(meanv_file, meanvolume)
        except:
            print(("Unable to save file %s" % (meanv_file)))
        return 0
    else:
        return meanvolume


def merge_meanVolumeZ(subject_file):
    """Merge slices mean volume into a single volume

    PARAMETERS
        :param subject_file

    OUTPUT
        None
    """

    f = open(subject_file, "r")
    subject = pcl.load(f)
    f.close()

    # Parameters and variables definition
    meanVolume_file = list()
    gridShape = subject.getSlicerGridShape()
    frameX, frameY, frameZ = gridShape[:]
    save_dir = subject.getResultDir()
    vol_shape = subject.getVolShape()
    zrange = list(range(1, frameZ + 1))

    # Joining all mean volumes and saving to file
    meanvolume = np.zeros(vol_shape, dtype=np.float32)
    nVol = 0
    for z in zrange:
        filename = os.path.join(save_dir, "meanvolume_z%d.npy" % (z))
        meanVolume_file.append(filename)
        vol = np.load(filename)
        if nVol == 0:
            meanvolume = vol
        else:
            meanvolume = (nVol * meanvolume + vol) / (nVol + 1)
        nVol += 1

    # Saving the final meanV file.
    filename = os.path.join(save_dir, "meanvolume.npy")
    np.save(filename, meanvolume)

    # Updating the subject information.
    subject.meanv_file = filename
    f = open(subject_file, "w")
    pcl.dump(subject, f)
    f.close()

    # Remove the individual files
    for f in meanVolume_file:
        try:
            os.remove(f)
        except:
            print(("Unable to remove file : %s" % (f)))


def get_z0(
    vol, sigma=1, iThresh=0.25, zThresh=0.2, method="derivative", zMin=20, zMax=100
):
    """Finds the cutting depth z0 for a given volume

    PARAMETERS
        vol : ndArray volume
        sigma (default=1) : standard deviation of the gaussian kernel
        iThresh (default=0.25) : intensity threshold (0 to 1)
        zThresh (default=0.2) : depth threshold (fraction of nz : 0 .. 1)

    OUTPUT
        z0 : cutting depth in pixel
    """
    _nx, _ny, nz = vol.shape
    iProfile = np.mean(vol, axis=(0, 1))  # Mean intensity per z
    z0 = 0

    if method == "derivative":
        iCum = np.cumsum(iProfile)  # Cumulative Intensity with depth
        ip3_cum = ndimage.gaussian_filter1d(
            iCum, sigma=sigma, order=3
        )  # 3rd order derivative in z (rate of intensity acceleration)

        # Finding all local maxima of the rate of intensity acceleration (dI^4/dz^4 = 0)
        lmax = local_maxima(np.arange(nz), ip3_cum)
        for l in lmax:
            cond1 = ip3_cum[l] / np.max(ip3_cum) > iThresh
            cond2 = l > zThresh * nz
            if cond1 and cond2:
                z0 = l
                break

        if z0 < zMin:
            z0 = zMin
        elif z0 > zMax:
            z0 = zMax

    elif method == "difference":  # Alex.
        zPad = 10  # Hardcoded parameter
        zRange = list(range(zMin + zPad, zMax - zPad))
        M = np.zeros(len(zRange), dtype=vol.dtype)
        for k in range(len(zRange)):
            z = zRange[k]
            shallowI = np.sum(iProfile[z - zPad : z])
            deepI = np.sum(iProfile[z : z + zPad])
            M[k] = deepI / shallowI
        z0 = np.argmax(M) + zRange[0]
    elif method == "edges":
        z0 = xyzcorr.findTissueDepth(vol)
    else:
        print("Unknown method, doing nothing")

    return z0


def get_sliceZ0Map(
    data_dir,
    save_dir,
    vol_shape,
    frameX,
    frameY,
    z,
    sigma=1,
    iThresh=0.25,
    zThresh=0.2,
    method="derivative",
    **kwargs
):
    """To compute the z0 map for a given slice."""
    if "z0map_file" in kwargs:
        z0map_file = kwargs["z0map_file"]
        saveOutput = True
    else:
        saveOutput = False

    if "meanV" in kwargs:
        useMeanV = True
        meanV = kwargs["meanV"]
        epsilon = 0.01 * meanV.max()  # To make sure there is no division by 0
        meanV += epsilon
    else:
        useMeanV = False

    z0map = np.zeros((frameX, frameY))
    # nFrames = frameX*frameY
    nCount = 0
    for x in range(frameX):
        for y in range(frameY):
            # print "%.2f%% - Processing vol(%d,%d,%d)" % (100.0*nCount/nFrames, x+1, y+1, z)
            pos = (x + 1, y + 1, z)
            vol = io.load_volume(data_dir, pos, vol_shape)
            if useMeanV:
                vol = vol / meanV

            nCount += 1
            z0map[x, y] = get_z0(vol, sigma, iThresh, zThresh, method)

    if saveOutput:
        try:
            np.save(z0map_file, z0map)
        except:
            print(("Unable to save file %s" % (z0map_file)))
            return -1
        return z0map_file
    else:
        return z0map


def merge_z0Maps(save_dir, frameZ, removeSlices=True):
    """To merge all z0 maps into a single file"""
    z0maps_file = os.path.join(save_dir, "z0map.npy")

    file_list = []
    for z in range(frameZ):
        this_slice = os.path.join(save_dir, "z0map_z%d.npy" % (z + 1))
        file_list.append(this_slice)
        this_z0map = np.load(this_slice)

        if z == 0:
            nx, ny = this_z0map.shape
            z0Maps = np.zeros((nx, ny, frameZ), dtype=this_z0map.dtype)
        z0Maps[:, :, z] = this_z0map

    try:
        np.save(z0maps_file, z0Maps)
    except:
        print(("Unable to save this file : %s" % (z0maps_file)))

    if removeSlices == True:
        for eachfile in file_list:
            try:
                os.remove(eachfile)
            except:
                print(("Unable to remove file : %s" % (eachfile)))

    return z0maps_file
