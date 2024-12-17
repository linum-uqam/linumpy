#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module uses graph theory to describe and interact with the mosaic topology.
"""

import sys

import networkx as nx
import numpy as np
import SimpleITK as sitk


def generate_default(nX, nY):
    """
    Generates a default topology where all tiles in mosaic are nodes and all neighbor relation is an edge.

    Parameters
    ----------
    nX: int
        Number of tiles in X direction.
    nY: int
        Number of tiles in Y direction.

    Returns
    -------
    mosaicTopo : NetworkX graph object describing the mosaic topology

    Note
    ----
    Each node position (in tile reference) can be accessed as node attributes 'x' and 'y'.

    """
    # Creating graph
    mosaicTopo = nx.Graph()

    # Creating vertices
    xx, yy = np.meshgrid(list(range(nX)), list(range(nY)))
    vPos = {"x": xx.ravel(), "y": yy.ravel()}
    nPosX = dict()
    nPosY = dict()
    for ii in range(nX * nY):
        nPosX[ii] = vPos["x"][ii]
        nPosY[ii] = vPos["y"][ii]

    mosaicTopo.add_nodes_from(list(range(nX * nY)))
    # if float(nx.__version__) < 2.0:
    #    nx.set_node_attributes(mosaicTopo, 'x', nPosX)
    #    nx.set_node_attributes(mosaicTopo, 'y', nPosY)
    # else:
    #    nx.set_node_attributes(mosaicTopo, nPosX, 'x')
    #    nx.set_node_attributes(mosaicTopo, nPosY, 'y')
    nx.set_node_attributes(mosaicTopo, nPosX, "x")
    nx.set_node_attributes(mosaicTopo, nPosY, "y")

    # Creating edges (x)
    inX = np.tile(np.arange(nX - 1), (nY,))
    outX = np.tile(np.arange(1, nX), (nY,))
    y = np.zeros(inX.shape, dtype=np.int)
    for iY in range(nY):
        y[iY * (nX - 1) : iY * (nX - 1) + nX - 1] = iY
    inX += nX * y
    outX += nX * y

    edgeX = np.zeros((len(inX), 2), dtype=np.int)
    edgeX[:, 0] = inX
    edgeX[:, 1] = outX

    ex = tuple([tuple(row) for row in edgeX])
    mosaicTopo.add_edges_from(ex)

    # Creating edges (y)
    inY = np.arange(nX * (nY - 1))
    outY = np.arange(nX, nX * nY)
    edgeY = np.zeros((len(inY), 2), dtype=np.int)
    edgeY[:, 0] = inY
    edgeY[:, 1] = outY
    ey = tuple([tuple(row) for row in edgeY])
    mosaicTopo.add_edges_from(ey)

    return mosaicTopo


def generate_graphFromEdges(sources, targets):
    """
    Generates a graph for a list of source and target nodes

    Parameters
    ----------
    sources:
        List of source node position.
    targets:
        List of target node position.

    Returns
    -------
    topo :
        NetworkX graph object describing the mosaic topology

    """
    topo = nx.DiGraph()
    nSteps = len(sources)
    for iStep in range(nSteps):
        inNode = "x%dy%d" % (sources[iStep][0], sources[iStep][1])
        outNode = "x%dy%d" % (targets[iStep][0], targets[iStep][1])
        inAttr = {"x": sources[iStep][0], "y": sources[iStep][1]}
        outAttr = {"x": targets[iStep][0], "y": targets[iStep][1]}
        topo.add_node(inNode, inAttr)
        topo.add_node(outNode, outAttr)
        topo.add_edge(inNode, outNode)

    return topo


def remove_agarose(topo, tissueMask):
    """
    Remove agarose nodes from the mosaic topology

    Parameters
    ----------
    topo:
        NetworkX graph object describing the mosaic topology

    tissueMask:
        (m, n) bool ndarray of the tissue mask for this slice.

    Returns
    -------
    topo:
        Updated mosaic topology (without the agarose nodes)

    """
    agarosePos = np.where(tissueMask == 0)
    agaroseIds = list()
    for ii in range(len(agarosePos[0])):
        this_pos = (agarosePos[0][ii], agarosePos[1][ii])
        agaroseIds.append(_pos2id(topo, this_pos))

    topo.remove_nodes_from(agaroseIds)

    return topo


def topoIterator(topo, root=(1, 1), method="dfs"):
    """Generate a list of edges traversing the topology in a single pass.

    :param topo: NetworkX graph object describing the mosaic topology
    :param root: Root node position.
    :param method: Graph traversing method to use. Available methods are 'dfs' (default) and 'bfs'
    :returns: sourceList, targetList : Lists of source and target node positions.

    """
    sourceList = list()
    targetList = list()

    # Find the edge corresponding to position = root
    idx = _pos2id(topo, root)
    if method == "bfs":
        edgeList = nx.bfs_edges(topo, source=idx)

    elif method == "dfs":
        edgeList = nx.dfs_edges(topo, source=idx)

    xx = nx.get_node_attributes(topo, "x")
    yy = nx.get_node_attributes(topo, "y")
    for iEdge in edgeList:
        id_in = iEdge[0]
        id_out = iEdge[1]
        source = (xx[id_in], yy[id_in])
        target = (xx[id_out], yy[id_out])
        sourceList.append(source)
        targetList.append(target)

    return sourceList, targetList


def get_unvisitedEdges(topo, sList, tList):
    """Get list of unvisited edges by iterator.

    :param topo: NetworkX graph object describing the mosaic topology
    :param sList: List of source node position.
    :param tList: List of target node position.
    :returns: edgeList : List of unvisited edges in topology.
    """
    edgeList = topo.edges()
    for iEdge in range(len(sList)):
        this_edge1 = (_pos2id(topo, sList[iEdge]), _pos2id(topo, tList[iEdge]))
        this_edge2 = (this_edge1[1], this_edge1[0])

        try:
            edgeList.remove(this_edge1)
        except:
            try:
                edgeList.remove(this_edge2)
            except:
                continue

    return edgeList


def _pos2id(topo, pos):
    """Finds node in topology corresponding to a given position.

    Parameters
    ----------
    topo: NetworkX graph object describing the mosaic topology
    pos: (2,) tuple containing a node position

    Returns
    -------
    idx : Node id

    """
    # Find the edge corresponding to position
    nList = list()
    xx = list()
    yy = list()

    # Extracting the node x positions
    for this_node, this_x in list(nx.get_node_attributes(topo, "x").items()):
        nList.append(this_node)
        xx.append(this_x)

    # Extracting the node y positions
    for this_node, this_y in list(nx.get_node_attributes(topo, "y").items()):
        yy.append(this_y)

    # Detecting the corresponding node
    idx = np.intersect1d(np.where(xx == pos[0]), np.where(yy == pos[1]))
    idx = idx[0]
    idx = nList[idx]

    return idx


def get_topoDim(topo):
    """Compute topology dimension using nodes' position attributes.

    :param topo: NetworkX graph object describing the mosaic topology
    :returns: nX, nY : The mosaic topology dimension in X and Y direction

    """
    xx = np.array(list(nx.get_node_attributes(topo, "x").items()))
    yy = np.array(list(nx.get_node_attributes(topo, "y").items()))
    xx = xx[:, 1]
    yy = yy[:, 1]
    nX = np.max(xx)
    nY = np.max(yy)
    return nX, nY


def keepLargestCCInMask(mask):
    vol = sitk.GetImageFromArray(mask.astype(int))
    labels = sitk.ConnectedComponent(vol)
    lstats = sitk.LabelStatisticsImageFilter()
    lstats.Execute(vol, labels)
    nCC = lstats.GetNumberOfLabels()

    largestLabel = 0
    largestLabelSize = 0
    # Loop over all labels to find the largest connected component
    for iCC in range(nCC):
        if lstats.GetMean(iCC) > 0 and lstats.GetCount(iCC) > largestLabelSize:
            largestLabel = iCC
            largestLabelSize = lstats.GetCount(iCC)

    # Only keeping the laster connected component in this image
    output = sitk.LabelMapMask(
        sitk.LabelImageToLabelMap(labels), vol, label=largestLabel
    )

    return sitk.GetArrayFromImage(output)

    pass
