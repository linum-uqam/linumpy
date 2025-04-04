#!/usr/bin/env python3
"""
Apply corrections from linum_estimate_slices_transforms_gui.py to volume.
"""
import argparse
import zarr
import numpy as np
from tqdm import tqdm

from linumpy.stitching.manual_registration import transform_and_rescale_slice


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_zarr',
                   help='Input zarr file to correct.')
    p.add_argument('in_corrections',
                   help='File (.npz) containing the correction parameters.')
    p.add_argument('out_zarr',
                   help='Output zarr file.')
    return p


def apply_transform(ty, tx, theta, coordinates):
    # Step 1. Rotate coordinates
    center_y = np.max(coordinates[:, :, 1]) / 2.0
    center_x = np.max(coordinates[:, :, 2]) / 2.0
    coordinates = coordinates - np.reshape([0, center_y, center_x], (1, 1, 3))
    rotated_y = np.atleast_2d(np.cos(theta)).T*coordinates[..., 1]\
        - np.atleast_2d(np.sin(theta)).T*coordinates[..., 2]
    rotated_x = np.atleast_2d(np.sin(theta)).T*coordinates[..., 1]\
        + np.atleast_2d(np.cos(theta)).T*coordinates[..., 2]
    coordinates[:, :, 1] = rotated_y + center_y
    coordinates[:, :, 2] = rotated_x + center_x

    # Step 2. Translate coordinates
    coordinates[:, :, 1] += np.atleast_2d(ty).T
    coordinates[:, :, 2] += np.atleast_2d(tx).T

    return coordinates


def apply_scaling(data, vmin, vmax):
    data = np.clip(data, vmin, vmax)
    data -= vmin
    if vmax - vmin > 0.0:
        data /= (vmax - vmin)
    # at this point the data is between [0, 1]
    return data


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_zarr = zarr.open(args.in_zarr, mode='r')
    imin, imax = np.min(in_zarr), np.max(in_zarr)

    checkpoint = np.load(args.in_corrections)
    custom_ranges = checkpoint['custom_ranges']
    transforms = checkpoint['transforms']

    out_zarr = zarr.open(args.out_zarr, mode='w',
                         shape=in_zarr.shape,
                         dtype=in_zarr.dtype)

    # process slices one at a time
    for z in tqdm(range(in_zarr.shape[0])):
        # estimate_slices_transforms_gui rescales intensities between (0, 1).
        data = (in_zarr[z] - imin) / (imax - imin)
        transform_z = transforms[z]
        ranges_z = custom_ranges[z]
        ty = transform_z[0]
        tx = transform_z[1]
        theta = transform_z[2]
        vmin = ranges_z[0]
        vmax = ranges_z[1]
        out_zarr[z] = transform_and_rescale_slice(data, ty, tx, theta, vmin, vmax)


if __name__ == '__main__':
    main()
