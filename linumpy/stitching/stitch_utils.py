#! /usr/bin/env python
# -*- coding: utf-8 -*-
""" All functions required by more than two other stitching module is kept here
"""
import numpy as np


def getOverlap(vol1, vol2, pos1, pos2):

    if len(pos1) == 2 or np.squeeze(vol1).ndim == 2:
        nx, ny = vol1.shape[0:2]

        if vol1.ndim == 2:
            nz = 1
            vol1 = np.reshape(vol1, (vol1.shape[0], vol1.shape[1], nz))
            vol2 = np.reshape(vol2, (vol2.shape[0], vol2.shape[1], nz))

        try:
            xmin = int(min([pos1[0], pos2[0]]))
            xmax = int(max([pos1[0], pos2[0]]) + nx)
            ymin = int(min([pos1[1], pos2[1]]))
            ymax = int(max([pos1[1], pos2[1]]) + ny)

            mosaic1 = np.zeros((xmax - xmin, ymax - ymin))
            mosaic2 = np.zeros((xmax - xmin, ymax - ymin))

            mosaic1[
                pos1[0] - xmin : pos1[0] - xmin + nx,
                pos1[1] - ymin : pos1[1] - ymin + ny,
            ] = (
                np.squeeze(vol1.mean(axis=2)) + 1
            )
            mosaic2[
                pos2[0] - xmin : pos2[0] - xmin + nx,
                pos2[1] - ymin : pos2[1] - ymin + ny,
            ] = (
                np.squeeze(vol2.mean(axis=2)) + 1
            )

            # Find intersection
            mask = mosaic1 * mosaic2 >= 1

            # Convert this into vol1 and vol2 coordinates
            x, y = np.where(mask)
            o_xmin = x.min()
            o_ymin = y.min()
            o_xmax = x.max()
            o_ymax = y.max()

            o_pos1 = (
                o_xmin - (pos1[0] - xmin),
                o_ymin - (pos1[1] - ymin),
                o_xmax - (pos1[0] - xmin),
                o_ymax - (pos1[1] - ymin),
            )
            o_pos2 = (
                o_xmin - (pos2[0] - xmin),
                o_ymin - (pos2[1] - ymin),
                o_xmax - (pos2[0] - xmin),
                o_ymax - (pos2[1] - ymin),
            )

            # Getting overlap
            overlap1 = vol1[o_pos1[0] : o_pos1[2], o_pos1[1] : o_pos1[3], :]
            overlap2 = vol2[o_pos2[0] : o_pos2[2], o_pos2[1] : o_pos2[3], :]

            if overlap1.shape[2] == 1:
                overlap1 = np.reshape(overlap1, overlap1.shape[:2])
                overlap2 = np.reshape(overlap2, overlap2.shape[:2])

            return overlap1, overlap2, o_pos1, o_pos2
        except:
            return None, None, None, None

    elif len(pos1) == 3:
        nx, ny, nz = vol1.shape

        if vol1.ndim == 2:
            nz = 1
            vol1 = np.reshape(vol1, (vol1.shape[0], vol1.shape[1], nz))
            vol2 = np.reshape(vol2, (vol2.shape[0], vol2.shape[1], nz))

        try:
            xmin = min([pos1[0], pos2[0]])
            xmax = max([pos1[0], pos2[0]]) + nx
            ymin = min([pos1[1], pos2[1]])
            ymax = max([pos1[1], pos2[1]]) + ny
            zmin = min([pos1[2], pos2[2]])
            zmax = max([pos1[2], pos2[2]]) + nz

            mosaic1 = np.zeros((xmax - xmin, ymax - ymin, zmax - zmin))
            mosaic2 = np.zeros((xmax - xmin, ymax - ymin, zmax - zmin))

            mosaic1[
                pos1[0] - xmin : pos1[0] - xmin + nx,
                pos1[1] - ymin : pos1[1] - ymin + ny,
                pos1[2] - zmin : pos1[2] - zmin + nz,
            ] = (
                vol1 + 1
            )
            mosaic2[
                pos2[0] - xmin : pos2[0] - xmin + nx,
                pos2[1] - ymin : pos2[1] - ymin + ny,
                pos2[2] - zmin : pos2[2] - zmin + nz,
            ] = (
                vol2 + 1
            )

            # Find intersection
            mask = mosaic1 * mosaic2 >= 1

            # Convert this into vol1 and vol2 coordinates
            x, y, z = np.where(mask)
            o_xmin = x.min()
            o_ymin = y.min()
            o_xmax = x.max()
            o_ymax = y.max()
            o_zmin = z.min()
            o_zmax = z.max()

            o_pos1 = (
                o_xmin - (pos1[0] - xmin),
                o_ymin - (pos1[1] - ymin),
                o_zmin - (pos1[2] - zmin),
                o_xmax - (pos1[0] - xmin),
                o_ymax - (pos1[1] - ymin),
                o_zmax - (pos1[2] - zmin),
            )
            o_pos2 = (
                o_xmin - (pos2[0] - xmin),
                o_ymin - (pos2[1] - ymin),
                o_zmin - (pos2[2] - zmin),
                o_xmax - (pos2[0] - xmin),
                o_ymax - (pos2[1] - ymin),
                o_zmax - (pos2[2] - zmin),
            )

            # Getting overlap
            overlap1 = vol1[
                o_pos1[0] : o_pos1[3], o_pos1[1] : o_pos1[4], o_pos1[2] : o_pos1[5]
            ]
            overlap2 = vol2[
                o_pos2[0] : o_pos2[3], o_pos2[1] : o_pos2[4], o_pos2[2] : o_pos2[5]
            ]

            if overlap1.shape[2] == 1:
                overlap1 = np.reshape(overlap1, overlap1.shape[:2])
                overlap2 = np.reshape(overlap2, overlap2.shape[:2])

            return overlap1, overlap2, o_pos1, o_pos2
        except:
            return None, None, None, None