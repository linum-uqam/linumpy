# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 14:17:20 2014

@author: joel
"""

import itertools
import os
import pickle as pcl
import sys
import tempfile

import networkx
import nibabel as nib
import numpy as np

from scipy.ndimage.filters import gaussian_filter, median_filter
from scipy.ndimage.morphology import binary_erosion

import linumpy.registration
from linumpy.postproc import brainMask
from linumpy.preproc import fixZ, icorr, xyzcorr
from linumpy.stitching import phase_correlation, topology
from linumpy.stitching.stitch_utils import getOverlap
from linumpy.utils import data_io


def get_absPos2D(
        vol_dir, save_dir, frameX, frameY, z, vol_shape, method="dfs", overlap=0.4, **kwargs
):
    """Main function to get the absolution position of each volume in the mosaic, in 2D

    PARAMETERS
    * vol_dir
    * save_dir
    * frameX
    * frameY
    * z
    * vol_shape
    * method

    ADDITIONAL KEYWORDS ARGUMENTS (Optional)
    * tissue : a tissue map

    OUTPUT
    * absPos ndarray
    * topo : networkx graph object

    """

    # Initializing parameters & Variables
    nx, ny, nz = vol_shape[:]

    # Generate the mosaic topology and a path on it.
    topo = topology.generate_default(frameX, frameY)
    if "tissue" in kwargs:
        segmentMap = kwargs["tissue"]
        topo = topology.remove_agarose(topo, 1 * segmentMap)

    if "previousPos" in kwargs:
        previousPos = kwargs["previousPos"]
        usePreviousPos = True
    else:
        usePreviousPos = False

    if "maxDisplacement" in kwargs:
        maxDisplacement = kwargs["maxDisplacement"]
        useDisplacementThreshold = True
    else:
        useDisplacementThreshold = False

    # Choosing a root at the center of the graph
    centerNodes = networkx.center(topo)
    root_x = topo.node[centerNodes[0]]["x"]
    root_y = topo.node[centerNodes[0]]["y"]
    root = (root_x, root_y)

    if 0:
        sList, tList = topology.topoIterator(topo, root, method=method)

        # Loop over all sources and target
        absPos = np.zeros((frameX, frameY, 2), dtype=np.int)
        deltas = list()

        for source, target in zip(sList, tList):
            # Opening 2 volumes
            pos1 = (source[0], source[1], z)
            pos2 = (target[0], target[1], z)
            deltax, deltay, pcmax = get_deltaBetweenVols(
                vol_dir, vol_shape, pos1, pos2, overlap
            )
            deltas.append((deltax, deltay))

        # Computing absolute position from deltas
        absPos = get_absPosFromDeltas(deltas, sList, tList)
    else:
        for u, v in topo.edges_iter():
            # print "Processing edge (%d -> %d)" % (u, v)
            pos1 = (topo.node[u]["x"], topo.node[u]["y"], z)
            pos2 = (topo.node[v]["x"], topo.node[v]["y"], z)

            if usePreviousPos:
                lastpos = list()
                lastpos.append(previousPos[pos1[0] - 1, pos1[1] - 1, :])
                lastpos.append(previousPos[pos2[0] - 1, pos2[1] - 1, :])
                deltax, deltay, pcmax = get_deltaBetweenVols(
                    vol_dir, vol_shape, pos1, pos2, lastPos=lastpos
                )
            else:
                deltax, deltay, pcmax = get_deltaBetweenVols(
                    vol_dir, vol_shape, pos1, pos2, overlap
                )

            topo.edge[u][v]["reliability"] = 1 / pcmax
            topo.edge[u][v]["dx"] = deltax
            topo.edge[u][v]["dy"] = deltay
            topo.edge[u][v]["direction"] = (u, v)

            if usePreviousPos and useDisplacementThreshold:
                last_dx = (
                        previousPos[pos2[0] - 1, pos2[1] - 1, 0]
                        - previousPos[pos1[0] - 1, pos1[1] - 1, 0]
                )
                last_dy = (
                        previousPos[pos2[0] - 1, pos2[1] - 1, 1]
                        - previousPos[pos1[0] - 1, pos1[1] - 1, 1]
                )
                if (
                        np.sqrt((deltax - last_dx) ** 2 + (deltay - last_dy) ** 2)
                        > maxDisplacement
                ):
                    #                    topo.edge[u][v]['dx'] = last_dx
                    #                    topo.edge[u][v]['dy'] = last_dy
                    topo.edge[u][v]["reliability"] = 0

        absPos = get_absPosFromWeightedTopo(topo, root, frameX, frameY)

    return absPos, topo


def get_absPosFromWeightedTopo(topo, root, frameX, frameY):
    # Get root node
    rootNode = topology._pos2id(topo, root)

    # Get topo size for abspos matrix
    # nX, nY = topology.get_topoDim(topo)
    absPos = np.zeros((frameX, frameY, 2), dtype=np.int)
    visitedVol = np.zeros((frameX, frameY), dtype=np.bool)

    # Making sure we only keep the largest connected component
    topo_ccs = list(networkx.connected_component_subgraphs(topo))
    nNodes = list()
    for cc in topo_ccs:
        nNodes.append(cc.number_of_nodes())
    main_topo = topo_ccs[np.argmax(nNodes)]

    # Get shortest path from root to any node of the iterGraph
    try:
        paths = networkx.shortest_path(main_topo, source=rootNode, weight="reliability")

        for path in paths.values():
            dx = 0
            dy = 0

            for node in range(1, len(path)):
                u = path[node - 1]
                v = path[node]
                if topo.edge[u][v]["direction"] == (u, v):
                    dx += topo.edge[u][v]["dx"]
                    dy += topo.edge[u][v]["dy"]
                else:
                    dx -= topo.edge[u][v]["dx"]
                    dy -= topo.edge[u][v]["dy"]

            x = topo.node[path[-1]]["x"] - 1
            y = topo.node[path[-1]]["y"] - 1
            absPos[x, y, 0] = dx
            absPos[x, y, 1] = dy
            visitedVol[x, y] = True

        # Making sure all coordinates are positive

        absPos[:, :, 0] -= absPos[:, :, 0].min()
        absPos[:, :, 1] -= absPos[:, :, 1].min()
        absPos[~visitedVol] = -1
    except:
        print("Something is wrong with this topology. Continue")

    return absPos


def get_deltaBetweenVols(vol_dir, vol_shape, pos1, pos2, overlap=0.3, **kwargs):
    nx, ny, nz = vol_shape[:]
    vol1 = data_io.load_volume(vol_dir, pos1, vol_shape)
    vol2 = data_io.load_volume(vol_dir, pos2, vol_shape)

    vol1 = vol1[:, :, 40:80]
    vol2 = vol2[:, :, 40:80]

    direction = [pos2[0] - pos1[0], pos2[1] - pos1[1]]
    xlim1 = [0, -1]
    xlim2 = [0, -1]
    ylim1 = [0, -1]
    ylim2 = [0, -1]

    if "lastPos" in kwargs:
        lastPos = kwargs["lastPos"]
        im1, im2, o_pos1, o_pos2 = getOverlap(vol1, vol2, lastPos[0], lastPos[1])
        im1 = np.squeeze(np.max(im1, axis=2))
        im2 = np.squeeze(np.max(im2, axis=2))
        xlim1 = [o_pos1[0], o_pos1[2]]
        xlim2 = [o_pos2[0], o_pos2[2]]
        ylim1 = [o_pos1[1], o_pos1[3]]
        ylim2 = [o_pos2[1], o_pos2[3]]

        # Get origin
        xorg = 0
        yorg = 0
        if direction[0] == 1:
            xorg = xlim1[0]
        elif direction[0] == -1:
            xorg = -nx + xlim2[0]
        if direction[1] == 1:
            yorg = ylim1[0]
        elif direction[1] == -1:
            yorg = -ny + ylim2[0]

    else:
        # Using only the FOV
        if direction[0] == -1:
            xlim1 = [15, 15 + np.int(overlap * nx)]
            xlim2 = [-(xlim1[1] - xlim1[0]), -1]
        elif direction[0] == 1:
            xlim2 = [15, 15 + np.int(overlap * nx)]
            xlim1 = [-(xlim2[1] - xlim2[0]), -1]
        elif direction[1] == -1:
            ylim1 = [0, np.int(overlap * ny)]
            ylim2 = [-(ylim1[1] - ylim1[0]), -1]
        elif direction[1] == 1:
            ylim2 = [0, np.int(overlap * ny)]
            ylim1 = [-(ylim2[1] - ylim2[0]), -1]

        vol1 = xyzcorr.cropVolume(vol1, xlim1, ylim1)
        vol2 = xyzcorr.cropVolume(vol2, xlim2, ylim2)

        # Get Average Intensity projection along z-axis
        im1 = np.squeeze(np.max(vol1, axis=2))
        im2 = np.squeeze(np.max(vol2, axis=2))

        # Get origin
        xorg = 0
        yorg = 0
        if direction[0] == 1:
            xorg = nx + xlim1[0]
        elif direction[0] == -1:
            xorg = -nx - xlim2[0]
        if direction[1] == 1:
            yorg = ny + ylim1[0]
        elif direction[1] == -1:
            yorg = -ny - ylim2[0]

    # Computing delta between volumes
    deltax, deltay, pcmax = phase_correlation.extPhaseCorrelation2d(
        im1, im2, nDim=(2, 2)
    )
    # deltax, deltay, pcmax = phase_correlation.normalizedCrossCorrelation2d(im1, im2)

    return xorg + deltax, yorg + deltay, pcmax


def get_absPosFromDeltas(deltas, sList, tList):
    # Transform iterator into graph
    iterGraph = topology.generate_graphFromEdges(sList, tList)
    rootNode = "x%dy%d" % (sList[0][0], sList[0][1])

    # Create a node dict to link node name to position in array
    nodeKeys = list()
    nodeKeys.append(rootNode)
    nNodes = len(tList) + 1
    for iNode in range(nNodes - 1):
        this_key = "x%dy%d" % (tList[iNode][0], tList[iNode][1])
        nodeKeys.append(this_key)

    # Get shortest path from root to any node of the iterGraph
    paths = networkx.shortest_path(iterGraph, source=rootNode)

    # Transform dict into path matrix
    pathMatrix = np.zeros((nNodes, nNodes), dtype=np.int)
    for ii in range(nNodes):
        this_key = nodeKeys[ii]
        for node in paths[this_key]:
            jj = np.where(np.array(nodeKeys) == node)
            pathMatrix[ii, jj[0]] = 1

    # Computing absolute positions
    deltas_x = np.zeros((nNodes,))
    deltas_y = np.zeros((nNodes,))
    for ii in range(1, nNodes):
        deltas_x[ii] = deltas[ii - 1][0]
        deltas_y[ii] = deltas[ii - 1][1]
    absPos_x = np.dot(pathMatrix, deltas_x)
    absPos_y = np.dot(pathMatrix, deltas_y)

    # Reordering positions
    xx = np.zeros((nNodes,))
    yy = np.zeros((nNodes,))
    xx[0] = sList[0][0] - 1
    yy[0] = sList[0][1] - 1
    for ii in range(1, nNodes):
        xx[ii] = tList[ii - 1][0] - 1
        yy[ii] = tList[ii - 1][1] - 1
    nX = len(np.unique(xx))
    nY = len(np.unique(yy))
    absPos = np.zeros((nX, nY, 2), dtype=np.int)

    for ii in range(nNodes):
        absPos[xx[ii], yy[ii], 0] = absPos_x[ii]
        absPos[xx[ii], yy[ii], 1] = absPos_y[ii]

    # Making sure all coordinates are positive
    absPos[:, :, 0] -= absPos[:, :, 0].min()
    absPos[:, :, 1] -= absPos[:, :, 1].min()

    return absPos


def get_slicedeltas(
        vol_dir,
        save_dir,
        frameX,
        frameY,
        z,
        vol_shape,
        fovFactor=1,
        useFixedZ=False,
        use2D=True,
        returnOutput=False,
):
    """
    DESCRIPTION
        Compute deltas for a specific slice

    PARAMETERS
    - vol_dir : Path to the data folder
    - save_dir : Path to the output folder (where the deltas will be saved)
    - frameX :
    - frameY :
    - z :
    - vol_shape :
    - fovFactor (default=1) :
    - useFixedZ (default=False) :pcmax
    - use2D (default=True) :
    - returnOutput (default=False) :

    OUTPUT
    - None (if returnOutput=False) : Deltas will be saved in save_dir.
    - dx, dy, dz : deltas

    NOTE
    - If the delta files already exist in save_dir, they will be overwritten

    AUTHOR
        Python implementation by : Joel Lefebvre (joel.lefebvre<at>polymtl.ca)
    """

    # Creating the deltas matrices to contain the deltas
    slice_deltax = np.zeros([frameX, frameY])  # 2D array to contain the deltas
    slice_deltay = np.zeros([frameX, frameY])  # 2D array to contain the deltas
    slice_deltaz = np.zeros([frameX, frameY])  # 2D array to contain the deltas

    # Loading the interface correction matrices
    x_center = frameX / 2  # TODO : Make sure this is not agarose.

    # Defining crop indices
    xCrop = 20  # Begin at 20 to remove the galbo wrapping
    xlim2 = [xCrop, np.round(vol_shape[0] / fovFactor)]
    xlim1 = [-(xlim2[1] - xlim2[0]), -1]
    ylim1 = [np.round(-vol_shape[1] / fovFactor), -1]
    ylim2 = [0, np.round(vol_shape[1] / fovFactor)]

    if useFixedZ:
        xCorr, yCorr, zCorr = fixZ.loadCorrMatrices(save_dir)

    for x, y in itertools.product(list(range(1, frameX)), list(range(1, frameY + 1))):
        try:
            print(
                (
                        "Processing vol(%d, %d, %d) and vol(%d, %d, %d)"
                        % (x, y, z, x + 1, y, z)
                )
            )
            sys.stdout.flush()
            # Loading 2 adjacent volumes on an X line
            vol1 = data_io.load_volume(vol_dir, (x, y, z), vol_shape)
            vol2 = data_io.load_volume(vol_dir, (x + 1, y, z), vol_shape)

            if len(vol1) == 1 or len(vol2) == 1:
                print(
                    "One of the volume can't be loaded, assuming delta = 0 in each direction"
                )
                continue

            # Fix the interface deformation introduced by the galvos.
            if useFixedZ:
                vol1 = vol1[xCorr, yCorr, zCorr]
                vol2 = vol2[xCorr, yCorr, zCorr]

            # Limiting the FOV
            vol1 = xyzcorr.cropVolume(vol1, xlim=xlim1)
            vol2 = xyzcorr.cropVolume(vol2, xlim=xlim2)

            # Computing the shift
            if use2D:  # In 2d
                im1 = np.squeeze(np.mean(vol1, axis=2))
                im2 = np.squeeze(np.mean(vol2, axis=2))
                deltax, deltay, pcmax = phase_correlation.extPhaseCorrelation2d(
                    im1, im2, nDim=(2, 1)
                )
                deltaz = 0
            else:  # In 3D
                deltax, deltay, deltaz = phase_correlation.extPhaseCorrelation3d(
                    vol1, vol2, nDim=(2, 1, 1)
                )

            # Adding the shift value to the 2d array
            slice_deltax[x, y - 1] = deltax
            slice_deltay[x, y - 1] = deltay
            slice_deltaz[x, y - 1] = deltaz

            # Y computation to join the next Y line
            if (x == x_center) and (y != frameY):
                # Loading 2 adjacent volumes on an Y column
                vol1 = data_io.load_volume(vol_dir, (x, y, z), vol_shape)
                vol2 = data_io.load_volume(vol_dir, (x, y + 1, z), vol_shape)

                if len(vol1) == 1 or len(vol2) == 1:
                    print(
                        "One of the volume can't be loaded, assuming delta = 0 in each direction"
                    )
                    continue

                # Fix the interface deformation introduced by the galvos.
                if useFixedZ:
                    vol1 = vol1[xCorr, yCorr, zCorr]
                    vol2 = vol2[xCorr, yCorr, zCorr]

                # Limiting the FOV
                vol1 = xyzcorr.cropVolume(vol1, xlim=[xCrop, -1], ylim=ylim1)
                vol2 = xyzcorr.cropVolume(vol2, xlim=[xCrop, -1], ylim=ylim2)

                # Computing the shift
                if use2D:  # in 2d
                    im1 = np.squeeze(np.mean(vol1, axis=2))
                    im2 = np.squeeze(np.mean(vol2, axis=2))
                    deltax, deltay, pcmax = phase_correlation.extPhaseCorrelation2d(
                        im1, im2, nDim=(1, 2)
                    )
                    deltaz = 0
                else:  # in 3d
                    deltax, deltay, deltaz = phase_correlation.extPhaseCorrelation3d(
                        vol1, vol2, nDim=(1, 2, 1)
                    )  # We can have more dans nx/2 displacement in X

                # Adding the shift value to the 2d array
                slice_deltax[0, y] = deltax
                slice_deltay[0, y] = deltay
                slice_deltaz[0, y] = deltaz
        except:
            print("\tUnable to compute delta for this pair. Assuming 0 deltas")

    if returnOutput:
        return slice_deltax, slice_deltay, slice_deltaz
    else:
        # Saving deltas in a z-depedent file
        deltafile = os.path.join(save_dir, "deltas_z%d.pcl" % (z))
        print(("Saving output in : %s" % (deltafile)))
        deltas = {"x": slice_deltax, "y": slice_deltay, "z": slice_deltaz}
        try:
            f = open(deltafile, "w")
            pcl.dump(deltas, f)
            f.close()
        except:
            print(("Unable to save deltas in file : %s" % (deltafile)))
        return 0


def collect_sliceDeltas(subject_file):
    """To collect all shift files and join them in a single file"""
    # Loading the subject
    f = open(subject_file, "r")
    subject = pcl.load(f)
    f.close()

    # Parameters and variables definition
    shiftZ_file = list()
    gridShape = subject.getSlicerGridShape()
    frameX, frameY, frameZ = gridShape[:]
    save_dir = subject.getResultDir()
    vol_shape = subject.getVolShape()
    zrange = list(range(1, frameZ + 1))

    # Joining all deltas
    dx = np.zeros([frameX, frameY, frameZ])  # Contains the deltas for each frame
    dy = np.zeros([frameX, frameY, frameZ])
    dz = np.zeros([frameX, frameY, frameZ])

    for z in zrange:
        filename = os.path.join(save_dir, "deltas_z%d.pcl" % (z))
        shiftZ_file.append(filename)
        try:
            f = open(filename, "r")
            deltas = pcl.load(f)
            f.close()
            dx[:, :, z - 1] = deltas["x"]
            dy[:, :, z - 1] = deltas["y"]
            dz[:, :, z - 1] = deltas["z"]
        except:
            print(("Unable to read deltas in file : %s" % (filename)))

    # Clean the deltas
    dx, dy, dz = clean_deltas(dx, dy, dz, vol_shape)
    deltas = {"x": dx, "y": dy, "z": dz}
    deltas = correctForMiddleShift(deltas, frameX, frameY, frameZ)

    # OUTPUT
    deltafile = os.path.join(save_dir, "deltas.pickle")
    f = open(deltafile, "w")
    pcl.dump(deltas, f)
    f.close()

    # Updating the subject file
    subject.deltas_file = deltafile
    f = open(subject_file, "w")
    pcl.dump(subject, f)
    f.close()

    ## Removing the individual deltas
    for f in shiftZ_file:
        try:
            os.remove(f)
        except:
            print(("Unable to remove file : %s" % (f)))

    print("Shift file mergin is done")


def clean_deltas(dx, dy, dz, vol_shape, lThresh=0.15, hThresh=0.4):
    # Getting integer displacement
    dx = np.round(dx)
    dy = np.round(dy)
    dz = np.round(dz)

    # ==============================================================================
    #   Removing all deltas that are too large or too low
    # ==============================================================================
    nx, ny, nz = dx.shape
    for x, y, z in itertools.product(list(range(nx)), list(range(ny)), list(range(nz))):
        # Is delta_x within acceptable range ?
        test_x = not (x == 0)
        test_dx = (-vol_shape[0] * lThresh < dx[x, y, z]) or (
                -vol_shape[0] * hThresh > dx[x, y, z]
        )
        if test_x and test_dx:
            dx[x, y, z] = 0

        # Is delta_y within acceptable range (for the 1st column elements) ?
        test_x = x == 0
        test_y = not (y == 0)
        test_dy = (-vol_shape[1] * lThresh < dy[x, y, z]) or (
                -vol_shape[1] * hThresh > dy[x, y, z]
        )

        if test_x and test_y and test_dy:
            dy[x, y, z] = 0

        # Is |delta| bigger than the slice range ?
        test_d = (
                np.abs(dx[x, y, z]) + np.abs(dy[x, y, z]) + np.abs(dz[x, y, z])
                > vol_shape[0]
        )
        if test_d:
            dx[x, y, z] = 0
            dy[x, y, z] = 0
            dz[x, y, z] = 0

    # ==============================================================================
    #   Assigning the mean delta value to volumes that have no deltas.
    # ==============================================================================
    # Average delta along (for x==1 and not(y==1))
    a1 = sum(dx[0, 1:, :]) / np.count_nonzero(dx[0, 1:, :])
    a2 = sum(dy[0, 1:, :]) / np.count_nonzero(dy[0, 1:, :])
    a3 = sum(dz[0, 1:, :]) / np.count_nonzero(dz[0, 1:, :])

    # Average deltax (for not(x==1))
    b1 = sum(dx[1:, :, :]) / np.count_nonzero(dx[1:, :, :])
    b2 = sum(dy[1:, :, :]) / np.count_nonzero(dy[1:, :, :])
    b3 = sum(dz[1:, :, :]) / np.count_nonzero(dz[1:, :, :])

    for x, y, z in itertools.product(list(range(nx)), list(range(ny)), list(range(nz))):
        dp = np.abs(nx) + np.abs(ny) + np.abs(nz)
        # If |delta(x,y,z)| == 0, x==1 and not(y==1)
        if dp == 0 and x == 0 and not (y == 1):
            dx[x, y, z] = a1
            dy[x, y, z] = a2
            dz[x, y, z] = a3

        # if |delta(x,y,z)| == 0 and not(x==1)
        if dp == 0 and not (x == 1):
            dx[x, y, z] = b1
            dy[x, y, z] = b2
            dz[x, y, z] = b3

    dx = np.round(dx)
    dy = np.round(dy)
    dz = np.round(dz)

    return dx, dy, dz


def correctForMiddleShift(deltas, frameX, frameY, frameZ):
    # Shifts between 2 columns is determined at the middle (frameX/2). This loop is to determine shifts between the fisrt volumes of each row
    for z in range(0, frameZ):
        for y in range(1, frameY):
            deltas["x"][0, y, z] = deltas["x"][0, y, z] - (
                    sum(deltas["x"][1: np.round(frameX / 2), y, z])
                    - sum(deltas["x"][1: np.round(frameX / 2), y - 1, z])
            )
            deltas["y"][0, y, z] = deltas["y"][0, y, z] - (
                    sum(deltas["y"][1: np.round(frameX / 2), y, z])
                    - sum(deltas["y"][1: np.round(frameX / 2), y - 1, z])
            )
            deltas["z"][0, y, z] = deltas["z"][0, y, z] - (
                    sum(deltas["z"][1: np.round(frameX / 2), y, z])
                    - sum(deltas["z"][1: np.round(frameX / 2), y - 1, z])
            )

    return deltas


def get_vol_pos(
        vol_dir,
        save_dir,
        frameX,
        frameY,
        frameZ,
        vol_shape,
        parallel="None",
        save_output=True,
        slices=-1,
        useFixedZ=False,
):
    """Main function for measuring shifts between volumes

    PARAMETERS
        vol_dir
        save_dir
        frameX
        frameY
        frameZ
        vol_shape
        parallel (default='None'), others are 'mpi' and 'qsub'
        save_output (default='True')
        slices

    OUTPUT
        deltas (or none if the result_dir is specified)
    """
    vol_pos = dict()

    print("= Serial volume position computation")
    # Initialization
    px = np.zeros([frameX, frameY, frameZ])  # Contains the deltas for each frame
    py = np.zeros([frameX, frameY, frameZ])
    pz = np.zeros([frameX, frameY, frameZ])

    # z range to compute
    if slices == -1:
        zrange = list(range(1, frameZ + 1))
    else:
        zrange = slices

    # Loop over all slices
    for z in zrange:
        slice_posx, slice_posy, slice_posz = get_slice_vol_pos(
            vol_dir, save_dir, frameX, frameY, z, vol_shape, useFixedZ
        )  # Get this slice position
        px[:, :, z - 1] = slice_posx
        py[:, :, z - 1] = slice_posy
        pz[:, :, z - 1] = slice_posz

    # Clean the deltas
    #    dx,dy,dz = clean_deltas(dx,dy,dz,vol_shape)
    vol_pos = {"x": px, "y": py, "z": pz}
    #    deltas = correctForMiddleShift(deltas,frameX,frameY,frameZ)

    # OUTPUT
    if save_output:
        deltafile = os.path.join(save_dir, "vol_pos.pickle")
        f = open(deltafile, "w")
        pcl.dump(vol_pos, f)
        f.close()
        return 0
    else:
        return vol_pos


def get_slice_vol_pos(vol_dir, save_dir, frameX, frameY, z, vol_shape, useFixedZ=False):
    fovFactor = 4.0  # This will only consider an overlap region of size 1/fovfactor of the volume.

    """ Get the delta for a specific slice """
    slice_px = np.zeros([frameX, frameY])  # 2D array to contain the deltas
    slice_py = np.zeros([frameX, frameY])  # 2D array to contain the deltas
    slice_pz = np.zeros([frameX, frameY])  # 2D array to contain the deltas

    for y in range(1, frameY + 1):
        for x in range(1, frameX):
            # Loading 2 adjacent volumes on an X line
            #            print "Computing delta between v1 : x%dy%dz%d and v2 : x%dy%dz%d" %(x,y,z,x+1,y,z)

            vol1 = data_io.load_volume(vol_dir, (x, y, z), vol_shape)
            vol2 = data_io.load_volume(vol_dir, (x + 1, y, z), vol_shape)
            if useFixedZ:
                dMap1 = fixZ.loadInterface(x, y, z, save_dir)
                vol1 = fixZ.cranckInterface(vol1, dMap1)

                dMap2 = fixZ.loadInterface(x + 1, y, z, save_dir)
                vol2 = fixZ.cranckInterface(vol2, dMap2)

            if len(vol1) == 1 or len(vol2) == 1:  # One of the volume can't be loaded
                print(
                    "One of the volume can't be loaded, assuming position = -1 in each direction"
                )
                slice_px[x, y] = -1
                slice_py[x, y] = -1
                slice_pz[x, y] = -1
            else:

                # Limiting the FOV
                xlim1 = [np.round(-vol_shape[0] / fovFactor), -1]
                xlim2 = [0, np.round(vol_shape[0] / fovFactor)]
                vol1 = xyzcorr.cropVolume(vol1, xlim=xlim1)
                vol2 = xyzcorr.cropVolume(vol2, xlim=xlim2)

                # Computing the shift
                deltax, deltay, deltaz = phase_correlation.extPhaseCorrelation3d(
                    vol1, vol2
                )

                # Adding the shift value to the 2d array
                slice_px[x, y - 1] = (
                        x * vol_shape[0] + deltax - np.round(vol_shape[0] / fovFactor)
                )  # This is position x+1,y (due to python 0 indexing), and the rest is to adjust the delta according to the FOV size.
                slice_py[x, y - 1] = (y - 1) * vol_shape[1] + deltay
                slice_pz[x, y - 1] = deltaz

            # Y computation to join the next Y line
            if not y == frameY:
                # Loading 2 adjacent volumes on an X line
                #            print "Computing delta between v1 : x%dy%dz%d and v2 : x%dy%dz%d" %(1,y,z,1,y+1,z)

                vol1 = data_io.load_volume(
                    vol_dir, (1, y, z), vol_shape
                )  # In original code, x = round(frameZ/2)
                vol2 = data_io.load_volume(
                    vol_dir, (1, y + 1, z), vol_shape
                )  # In original code, x = round(frameZ/2)
                if useFixedZ:
                    dMap1 = fixZ.loadInterface(1, y, z, save_dir)
                    vol1 = fixZ.cranckInterface(vol1, dMap1)
                    dMap2 = fixZ.loadInterface(1, y + 1, z, save_dir)
                    vol2 = fixZ.cranckInterface(vol2, dMap2)

                if (
                        len(vol1) == 1 or len(vol2) == 1
                ):  # One of the volume can't be loaded
                    print(
                        "One of the volume can't be loaded, assuming delta = 0 in each direction"
                    )
                    slice_px[0, y] = -1
                    slice_py[0, y] = -1
                    slice_pz[0, y] = -1
                else:

                    # Limiting the FOV
                    ylim1 = [np.round(-vol_shape[1] / fovFactor), -1]
                    ylim2 = [0, np.round(vol_shape[1] / fovFactor)]
                    vol1 = xyzcorr.cropVolume(vol1, ylim=ylim1)
                    vol2 = xyzcorr.cropVolume(vol2, ylim=ylim2)

                    # Computing the shift
                    deltax, deltay, deltaz = phase_correlation.extPhaseCorrelation3d(
                        vol1, vol2
                    )

                    # Adding the shift value to the 2d array
                    slice_px[
                        0, y
                    ] = deltax  # This is position 1,y+1 (due to python 0 indexing)
                    slice_py[0, y] = (
                            y * vol_shape[1] + deltay - np.round(vol_shape[1] / fovFactor)
                    )
                    slice_pz[0, y] = deltaz

    return slice_px, slice_py, slice_pz


def optimize_deltas(
        topo, deltas, sList, tList, absPos, nStep, vol_dir, vol_shape, overlap, z
):
    """Algorithm to optimize deltas based on inter-volumes distances"""

    # Get intervolume distances
    stdDevList = np.zeros((nStep,))
    for iStep in range(nStep):
        print(("#%d" % (iStep)))
        oldPos = absPos
        iEdge = 0
        edgeList = topology.get_unvisitedEdges(topo, sList, tList)
        #        edgeList = topo.edges()
        nEdges = len(edgeList)
        sourceNodes = np.zeros((nEdges,), dtype="i4")
        targetNodes = np.zeros((nEdges,), dtype="i4")
        distList = np.zeros((nEdges,), dtype="f4")
        for iEdge in range(len(edgeList)):
            u = edgeList[iEdge][0]
            v = edgeList[iEdge][1]
            pos_u = (topo.node[u]["x"], topo.node[u]["y"])
            pos_v = (topo.node[v]["x"], topo.node[v]["y"])
            absPos_u = absPos[pos_u[0] - 1, pos_u[1] - 1, :]
            absPos_v = absPos[pos_v[0] - 1, pos_v[1] - 1, :]
            dist = np.sum((absPos_u - absPos_v) ** 2)
            sourceNodes[iEdge] = u
            targetNodes[iEdge] = v
            distList[iEdge] = dist

        distList /= distList.max()

        # Random sampling of an edge based on distance deviation
        meanDist = np.mean(distList)
        devDist = np.abs(distList - meanDist)
        stdDevList[iStep] = np.std(devDist)

        devDist_cs = np.cumsum(devDist)
        idx_rand = devDist_cs[-1] * np.random.rand(1)
        idx = np.argmax(devDist_cs > idx_rand)

        newEdge = (sourceNodes[idx], targetNodes[idx])

        # Update deltas and absolute positions
        try:
            sList, tList, deltas = update_pathAndDeltas(
                topo, sList, tList, deltas, newEdge, overlap, vol_dir, vol_shape, z
            )
            absPos = get_absPosFromDeltas(deltas, sList, tList)

        #            # Check if the new configuration gives a better mosaic
        #            edgeList = topology.get_unvisitedEdges(topo, new_sList, new_tList)
        #            #edgeList = topo.edges()
        #            nEdges = len(edgeList)
        #            sourceNodes = np.zeros((nEdges, ), dtype='i4')
        #            targetNodes = np.zeros((nEdges, ), dtype='i4')
        #            distList = np.zeros((nEdges, ), dtype='f4')
        #            for iEdge in range(len(edgeList)):
        #                u = edgeList[iEdge][0]
        #                v = edgeList[iEdge][1]
        #                pos_u = (topo.node[u]['x'], topo.node[u]['y'])
        #                pos_v = (topo.node[v]['x'], topo.node[v]['y'])
        #                absPos_u = new_absPos[pos_u[0]-1, pos_u[1]-1, :]
        #                absPos_v = new_absPos[pos_v[0]-1, pos_v[1]-1, :]
        #                dist = np.sum((absPos_u - absPos_v)**2)
        #                sourceNodes[iEdge] = u
        #                targetNodes[iEdge] = v
        #                distList[iEdge] = dist
        #
        #            distList /= distList.max()
        #            meanDist = np.mean(distList)
        #            devDist = np.abs(distList - meanDist)
        #            if (stdDevList[iStep] > np.std(devDist)) or np.random.choice([0,1]):
        #                sList = new_sList
        #                tList = new_tList
        #                deltas = new_deltas
        #                absPos = new_absPos
        #            else:
        #                print "This iteration doesn't improve the mosaic. Pass."
        #                pass
        except:
            print("Pass")
            pass

    return absPos, stdDevList, sList, tList, deltas


def update_pathAndDeltas(
        topo, sList, tList, deltas, newEdge, overlap, vol_dir, vol_shape, z
):
    path = topology.generate_graphFromEdges(sList, tList)

    for idx in [0, 1]:
        # Choose an edge to remove from the original path
        this_sList = list(sList)
        this_tList = list(tList)
        this_deltas = list(deltas)

        outNode = "x%dy%d" % (
            topo.node[newEdge[idx]]["x"],
            topo.node[newEdge[idx]]["y"],
        )
        inEdge = path.in_edges(outNode)
        inNode = inEdge[0][0]

        # find index of edge to remove
        inNodePos = (path.node[inNode]["x"], path.node[inNode]["y"])
        outNodePos = (path.node[outNode]["x"], path.node[outNode]["y"])
        idx_source = [i for i, x in enumerate(this_sList) if x == inNodePos]
        idx_target = [i for i, x in enumerate(this_tList) if x == outNodePos]
        idxToRemove = np.intersect1d(idx_source, idx_target)

        # Updating sList, tList and deltas
        this_sList.pop(idxToRemove)
        this_tList.pop(idxToRemove)
        this_deltas.pop(idxToRemove)

        # Computing delta between the volumes linked by this edge
        if idx:
            idx2 = 0
        else:
            idx2 = 1
        pos1 = (topo.node[newEdge[idx2]]["x"], topo.node[newEdge[idx2]]["y"], z)
        pos2 = (topo.node[newEdge[idx]]["x"], topo.node[newEdge[idx]]["y"], z)
        deltax, deltay = get_deltaBetweenVols(vol_dir, vol_shape, pos1, pos2, overlap)
        this_deltas.append((deltax, deltay))
        this_sList.append((pos1[0], pos1[1]))
        this_tList.append((pos2[0], pos2[1]))

        this_topo = topology.generate_graphFromEdges(this_sList, this_tList)
        this_topo = networkx.Graph(this_topo)
        nCC = networkx.number_connected_components(this_topo)
        if nCC == 1:
            sList = this_sList
            tList = this_tList
            deltas = this_deltas
        else:
            pass

    return sList, tList, deltas


def get_zTranslationBetweenSlices(
        vol1, vol2, sigma=5.0, maskInterface=False, factor=1, nzSize=10, compute2d=False
):
    """Computes the z translation between two volumes using the 2d image gradient and cross-correlation

    Parameters
    ----------
    vol1 : ndarray
        Fixed volume
    vol2 : ndarray
        Moving volume
    sigma : float > 0
        Median filter kernel size used to smooth the volumes before computing the gradient.
    maskInterface : bool
        If true, mask the data below the interface.
    factor : int
        Subsampling factor to accelerate the computation

    Returns
    -------
    int
        Estimated translation between the slices

    """
    # Mask data below interface
    if maskInterface:
        nx, ny, nz = vol1.shape
        # Computing interface from the data mask
        interface1 = xyzcorr.getInterfaceDepthFromMask(brainMask.slicer3d(vol1))
        interface2 = xyzcorr.getInterfaceDepthFromMask(brainMask.slicer3d(vol2))

        # Computing a mask based on the interface depth
        m1 = xyzcorr.maskUnderInterface(vol1, interface1, returnMask=True)
        m2 = xyzcorr.maskUnderInterface(vol2, interface2, returnMask=True)

        # Removing zero-depth interfaces (usually this doesn't contain tissue)
        tm1 = np.tile(np.reshape(interface1 > 0, (nx, ny, 1)), (1, 1, nz))
        tm2 = np.tile(np.reshape(interface2 > 0, (nx, ny, 1)), (1, 1, nz))
        m1 = m1 * tm1
        m2 = m2 * tm2
        del tm1, tm2
    else:
        m1 = np.ones_like(vol1).astype(bool)
        m2 = np.ones_like(vol2).astype(bool)

    if factor > 1:
        nx, ny, nz = vol1.shape
        newShape = np.round((nx / float(factor), ny / float(factor), nz)).astype(int)
        vol1 = xyzcorr.resampleITK(vol1, newShape)
        m1 = xyzcorr.resampleITK(255 * m1.astype(int), newShape).astype(bool)

        nx, ny, nz = vol2.shape
        newShape = np.round((nx / float(factor), ny / float(factor), nz)).astype(int)
        vol2 = xyzcorr.resampleITK(vol2, newShape)
        m2 = xyzcorr.resampleITK(255 * m2.astype(int), newShape).astype(bool)

    # Computing data mask
    mask1 = binary_erosion(vol1.mean(axis=2) > 0, iterations=5)
    mask2 = binary_erosion(vol2.mean(axis=2) > 0, iterations=5)

    # Computing 2D gradient magnitude for each slice
    nz = vol1.shape[2]
    e1 = np.zeros_like(vol1)
    for z in range(nz):
        im1 = median_filter(vol1[:, :, z], sigma)
        g1 = np.gradient(im1)
        e1[:, :, z] = (
                icorr.normalize(g1[0] ** 2.0 + g1[1] ** 2.0, highThresh=99.5) * mask1
        )

    nz = vol2.shape[2]
    e2 = np.zeros_like(vol2)
    for z in range(nz):
        im2 = median_filter(vol2[:, :, z], sigma)
        g2 = np.gradient(im2)
        e2[:, :, z] = (
                icorr.normalize(g2[0] ** 2.0 + g2[1] ** 2.0, highThresh=99.5) * mask2
        )

    # Compute cross-correlation
    corrList = list()
    zList = list()
    nz = vol1.shape[2]
    for z in range(nz - nzSize):
        a, b, _, _ = getOverlap(e1, e2, (0, 0, 0), (0, 0, z))
        c, d, _, _ = getOverlap(m1, m2, (0, 0, 0), (0, 0, z))

        im1 = a[:, :, -nzSize::]
        im2 = b[:, :, -nzSize::]
        m_im1 = c[:, :, -nzSize::]
        m_im2 = d[:, :, -nzSize::]

        if compute2d:
            im1 = vol1.mean(axis=0)
            im2 = vol2.mean(axis=0)
            mask_2d = np.mean(m_im1 * m_im2, axis=0).astype(np.bool)
            # corr = phase_correlation.mutualInformation(im1, im2, mask_2d)
            corr = linumpy.registration.crossCorrelation(im1, im2, mask_2d)
        else:
            # corr = phase_correlation.mutualInformation(vol1, vol2, m_im1*m_im2)
            corr = linumpy.registration.crossCorrelation(im1, im2, m_im1 * m_im2)
        corrList.append(corr)
        zList.append(z)

    # Keep the largest cross-correlation as the dz shift
    maxCorr = np.max(corrList)
    dz = np.where(corrList == maxCorr)[0][0]

    return dz


def compute_z_shift(
        vol1: np.ndarray,
        vol2: np.ndarray,
        dz: int = 1,
        sigma: float = 0.005,
        factor: int = 1,
        normalize: bool = True,
        z0: int = None,
        metric: str = "MI",
        useGradient: bool = True,
        align_xy: bool = False
) -> int:
    """
    Compute the z shift between a pair of 3D slices
    Parameters
    ----------
    vol1
        Fixed volume
    vol2
        Moving volume
    dz
        Number of AIP slices to use for the registration
    sigma
        Gaussian filter size, expressed as a fraction of the XY image size
    factor
        XY subsampling factor
    normalize
        To normalize the fixed and moving images before registration
    z0
        Minimum shift allowed.
    metric
        Similarity metric to use for registration. Available: 'CC', 'MI'
    useGradient
        Use the image gradient instead of intensity for simalarity computation
    align_xy
        Align the slice in the XY plane before z shift estimation

    Returns
    -------
    Estimated z shift (in pixel) between the two slices
    """
    assert metric in ['CC', 'MI']

    if factor > 1:
        nx, ny, nz = vol1.shape
        newShape = np.round((nx / float(factor), ny / float(factor), nz)).astype(int)

        vol1 = xyzcorr.resampleITK(vol1, newShape)
        vol2 = xyzcorr.resampleITK(vol2, newShape)

    nx, ny, nz = vol1.shape
    k = np.ceil(sigma * 0.5 * (nx + ny))

    aip1 = vol1[:, :, nz - dz: nz].mean(axis=2)
    if normalize:
        aip1 = icorr.normalize(aip1)
    aip1_g = gaussian_filter(aip1, k)
    if useGradient:
        gx, gy = np.gradient(aip1_g)
        aip1_g = gx ** 2.0 + gy ** 2.0

    aip1_g = np.round(
        (255 * (aip1_g - aip1_g.min()) / float(aip1_g.max() - aip1_g.min()))
    )
    m1 = aip1_g > 0

    corr_list = list()
    for z in range(vol2.shape[2] - dz):
        aip2 = vol2[:, :, z: z + dz].mean(axis=2)
        if normalize:
            aip2 = icorr.normalize(aip2)

        if align_xy:
            deltas = linumpy.registration.pairWisePhaseCorrelation(aip1, aip2)
            aip2 = phase_correlation.warp_rigid(aip2, deltas)

        aip2_g = gaussian_filter(aip2, k)
        if useGradient:
            gx, gy = np.gradient(aip2_g)
            aip2_g = gx ** 2.0 + gy ** 2.0
        aip2_g = np.round(
            (255 * (aip2_g - aip2_g.min()) / float(aip2_g.max() - aip2_g.min()))
        )
        m2 = aip2_g > 0

        if metric == "CC":
            corr_list.append(
                linumpy.registration.crossCorrelation(aip1_g, aip2_g, mask=m1 * m2)
            )
        elif metric == "MI":
            corr_list.append(
                phase_correlation.mutualInformation(aip1_g, aip2_g, mask=m1 * m2)
            )

    corr_list = np.array(corr_list)
    if z0 is not None:
        corr_list[0:z0] = 0

    dz = np.where(corr_list == corr_list.max())[0][0]
    return int(nz - dz - 1)


def getAffineTransform(
        slice1,
        slice2,
        delta_z,
        affine_prefix="moving",
        fpath=".",
        maxShrink=8,
        minShrink=1,
        maxConvSteps=1000,
        minConvSteps=100,
):
    """Compute the affine transform between two adjacent slices using ANTs

    Parameters
    ==========
    slice1 : str
        Filename of the bottom slice

    slice2 : str
        Filename of the top slice

    affine_prefix : str
        Prefix of the affine transform file

    fixed_dz : int
        Fixed dz shift between the slices (optional)


    Returns
    =======
    str
        Affine transform file to warp slice 2 onto slice1
    int
        Z shift between slice 2 and slice 1

    Notes
    =====
    * This function uses 'antsRegistration' to compute the affine transform matrix.

    """

    img1 = nib.load(slice1)
    img2 = nib.load(slice2)
    vol1 = img1.get_data()
    vol2 = img2.get_data()

    # Find two identical images using first image of slice2
    zlim_1 = (delta_z, vol1.shape[2])
    # print "Zlim 1 : ", zlim_1
    fixed_image = vol1[:, :, zlim_1[0]: zlim_1[1]]
    if fixed_image.ndim == 3:
        fixed_image = fixed_image.mean(axis=2)
    fixed_image = (
            255
            * (fixed_image - fixed_image.min())
            / float(fixed_image.max() - fixed_image.min())
    ).astype(np.uint8)

    zlim_2 = (0, zlim_1[1] - zlim_1[0] + 1)
    # print "Zlim 2 : ", zlim_2
    moving_image = vol2[:, :, zlim_2[0]: zlim_2[1]]
    if moving_image.ndim == 3:
        moving_image = moving_image.mean(axis=2)
    moving_image = (
            255
            * (moving_image - moving_image.min())
            / float(moving_image.max() - moving_image.min())
    ).astype(np.uint8)

    # Need to save those as nifti to be able to call antsRegistration
    fixed_file = tempfile.mkstemp(suffix="_fixed.nii", dir=fpath)[1]
    moving_file = tempfile.mkstemp(suffix="_moving.nii", dir=fpath)[1]
    afft_fixed = img1.get_affine()  # Affine transformation
    img = nib.Nifti1Image(fixed_image, afft_fixed)  # A nifti image
    nib.save(img, fixed_file)
    afft_moving = img2.get_affine()  # Affine transformation
    img = nib.Nifti1Image(moving_image, afft_moving)  # A nifti image
    nib.save(img, moving_file)

    # find transform matrix to pass from slice1 to slice2
    affine_transform = ants.register2DImages(
        fixed_file,
        moving_file,
        affine_prefix,
        minShrink=minShrink,
        maxShrink=maxShrink,
        minConvSteps=minConvSteps,
        maxConvSteps=maxConvSteps,
    )

    os.remove(fixed_file)
    os.remove(moving_file)
    return affine_transform
