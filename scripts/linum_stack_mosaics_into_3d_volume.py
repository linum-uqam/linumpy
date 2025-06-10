#!/usr/bin/env python3
"""
Stack 3D mosaics into a single 3D volume saved as .ome.zarr file.
"""
import argparse

from linumpy.io.zarr import read_omezarr, save_omezarr
from linumpy.utils_images import apply_xy_shift
from linumpy.stitching.registration import register_consecutive_3d_mosaics

import zarr
import dask.array as da
import re

import numpy as np
import pandas as pd

from pathlib import Path

from tqdm import tqdm


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_mosaics_dir',
                    help='Path to the directory containing the mosaics.')
    p.add_argument('in_xy_shifts',
                   help='Path to the file containing the XY shifts.')
    p.add_argument('out_stack',
                   help='Path to the output stack.')
    p.add_argument('--out_offsets',
                   help='Optional output offsets file.')
    p.add_argument('--initial_search', type=int, default=20,
                   help='Initial depth for depth matching (in voxels). [%(default)s]')
    p.add_argument('--depth_offset', type=int, default=10,
                   help='Offset from interface for each volume. [%(default)s]')
    p.add_argument('--max_allowed_overlap', type=int, default=5,
                   help='Maximum allowed overlap for the alignment. [%(default)s]')
    return p


def compute_volume_shape(mosaics_files, mosaics_depth,
                         dx_list, dy_list):

    # Compute the volume shape
    xmin = []
    xmax = []
    ymin = []
    ymax = []

    for i, f in enumerate(mosaics_files):
        # Get this volume shape
        img, res = read_omezarr(f)
        shape = img.shape

        # Get the cumulative shift
        if i == 0:
            xmin.append(0)
            xmax.append(shape[-2])
            ymin.append(0)
            ymax.append(shape[-1])
        else:
            dx = np.cumsum(dx_list)[i - 1]
            dy = np.cumsum(dy_list)[i - 1]
            xmin.append(-dx)
            xmax.append(-dx + shape[-2])
            ymin.append(-dy)
            ymax.append(-dy + shape[-1])

    # Get the volume shape
    x0 = min(xmin)
    y0 = min(ymin)
    x1 = max(xmax)
    y1 = max(ymax)
    nx = int((x1 - x0))
    ny = int((y1 - y0))

    # TODO: Handle the case where resolution does not perfectly
    # divides the slicing interval
    # Important!!! The +1 is to make sure that the last mosaic
    # fits in the volume for the case where the best offset is always 1
    volume_shape = (mosaics_depth*len(mosaics_files) + 1, nx, ny)
    return volume_shape, x0, y0


def align_sitk_2d(prev_mosaic, img, max_allowed_overlap):
    best_offset = 0
    min_error = np.inf
    best_warp = np.zeros(img.shape)
    for i in range(1, max_allowed_overlap + 1):
        warped_img, error = register_consecutive_3d_mosaics(prev_mosaic[-i, :, :], img)
        if error < min_error:
            min_error = error
            best_offset = i
            best_warp[:] = warped_img
    return best_warp, best_offset


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # get all .ome.zarr files in in_mosaics_dir
    in_mosaics_dir = Path(args.in_mosaics_dir)
    mosaics_files = [p for p in in_mosaics_dir.glob('*.ome.zarr')]
    mosaics_files.sort()
    pattern = r".*z(\d+)_.*"
    slice_ids = []
    for f in mosaics_files:
        foo = re.match(pattern, f.name)
        slice_ids.append(int(foo.groups()[0]))
    slice_ids = np.array(slice_ids)

    # when indexing does not start from 0, we need to shift the slice ids
    slice_ids -= np.min(slice_ids)

    # Load cvs containing the shift values for each slice
    df = pd.read_csv(args.in_xy_shifts)

    # We load the shifts in mm, but we need to convert them to pixels
    dx_list = np.array(df["x_shift_mm"].tolist())
    dy_list = np.array(df["y_shift_mm"].tolist())

    # assume that the resolution is the same for all slices
    img, res = read_omezarr(mosaics_files[0])
    dx_list /= res[-2]
    dy_list /= res[-1]

    mosaics_depth = args.initial_search
    volume_shape, x0, y0 = compute_volume_shape(mosaics_files, mosaics_depth,
                                                dx_list, dy_list)
    print(f"Output volume shape: {volume_shape}")
    print(f"Mosaic depth: {mosaics_depth} voxels")

    start_index = args.depth_offset
    print(f"Interface index: {start_index}")

    mosaic_store = zarr.TempStore()
    mosaic = zarr.open(mosaic_store, mode="w", shape=volume_shape,
                       dtype=np.float32, chunks=(256, 256, 256))

    z_offsets = [0]

    # Loop over the slices
    for i in tqdm(range(len(mosaics_files)), unit="slice", desc="Stacking slices"):
        # Load the slice
        f = mosaics_files[i]
        current_z_offset = z_offsets[-1]

        img, res = read_omezarr(f)
        img = img[start_index:]

        # Get the shift values for the slice
        dx = x0
        dy = y0
        if i > 0:
            dx += np.cumsum(dx_list)[i - 1]
            dy += np.cumsum(dy_list)[i - 1]

        # Apply the shift as an initial alignment
        # dy and dx are inverted here to match the image coordinates
        img = apply_xy_shift(img, mosaic[:len(img), :, :], dy, dx)

        # Equalize intensities
        clip_ubound = np.percentile(img, 99.9, axis=(1, 2), keepdims=True)
        img = np.clip(img, a_min=None, a_max=clip_ubound)
        if img.max() - img.min() > 0.0:
            img /= np.max(img, axis=(1, 2), keepdims=True)
            img[np.isnan(img)] = 0.0
            img[np.isinf(img)] = 0.0

        if i > 0:
            prev_mosaic = mosaic[:current_z_offset]
            best_offset = 1
            img, best_offset = align_sitk_2d(prev_mosaic, img, args.max_allowed_overlap)

            # Add the overlapping slice to the volume
            zmax = min(len(img), mosaic.shape[0] - current_z_offset)
            mosaic[current_z_offset - best_offset:
                   current_z_offset + len(img) - best_offset] = img[:zmax]

            current_z_offset += mosaics_depth - best_offset

        else:  # only true at very first iteration
            mosaic[:len(img), :, :] = img
            current_z_offset += mosaics_depth

        z_offsets.append(current_z_offset)

    # save the z offsets in npz
    if args.out_offsets is not None:
        np.save(args.out_offsets, np.array(z_offsets, dtype=np.int32))
        print(f"Z offsets saved to {args.out_offsets}")

    dask_arr = da.from_zarr(mosaic)
    save_omezarr(dask_arr, args.out_stack, voxel_size=res,
                 chunks=(256, 256, 256), n_levels=3)

    print(f"Output volume saved to {args.out_stack}")


if __name__ == '__main__':
    main()
