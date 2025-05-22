#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""Slice Stitching Pipeline (to create a 3D volume from multiple 3D slices)"""

import argparse
import json
import dask.array as da
import numpy as np
from tqdm import tqdm
import zarr
from linumpy.stitching import stitch_oct
from linumpy.utils import data_io
from linumpy.io.zarr import read_omezarr, save_zarr

# TODO: move illumination compensation into a separate script
# TODO: move the mask shifts and other related computation in a separate script


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )

    p.add_argument("input", nargs="+", help="List of slice filenames to stitch")
    p.add_argument("output", help="Output volume filename")
    p.add_argument("z_shifts", help="Z shifts between slices (.json)")
    p.add_argument("-m", "--masks", default=None, nargs="+", help="Optional mask list")
    p.add_argument(
        "--blending_method",
        choices=("none", "average", "diffusion"),
        default="diffusion",
        help="Blending method (default=%(default)s)",
    )
    p.add_argument(
        "--blending_width",
        type=float,
        default=1.0,
        help="Blending width (in fraction of overlap size). (default=%(default)s)",
    )
    p.add_argument(
        "--blending_nsteps",
        type=int,
        default=5,
        help="Number of iterations (blending computation)",
    )
    p.add_argument(
        "--slice_indexes",
        default=None,
        type=str,
        help="Slice indexes to process (default: all slices)",
    )
    return p


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Processing the input files
    slice_list = args.input
    mask_list = args.masks

    # Convert z_shifts into a dictionary
    with open(args.z_shifts, "r", encoding="utf-8") as f:
        z_shifts = json.load(f)
    dz_dict = {}
    for pair in z_shifts:
        key = (int(pair["z1"]), int(pair["z2"]))
        value = pair["dz"]
        dz_dict[key] = value

    # Detect slice index
    if args.slice_indexes is not None:
        slice_ids = [int(i) for i in args.slice_indexes.split(",")]
    else:
        # Get the slice indices from the filenames
        slice_ids = data_io.getSliceListIndices(slice_list)

    slice_files = {}
    for i, f in zip(slice_ids, slice_list):
        slice_files[i] = f

    # Detect the mask slice index
    mask_files = {}
    if mask_list is not None:
        if args.slice_indexes is not None:
            mask_slice_z = [int(i) for i in args.slice_indexes.split(",")]
        else:
            # Get the slice indices from the filenames
            mask_slice_z = data_io.getSliceListIndices(mask_list)

        for i, f in zip(mask_slice_z, mask_list):
            mask_files[i] = f

    # Get slice dz_list
    dz_list = [0]
    for i in range(1, len(slice_ids)):
        key = (slice_ids[i - 1], slice_ids[i])
        dz = dz_dict[key]
        dz_list.append(dz)

    # Compute the depth of each slice
    slices_z = np.cumsum(dz_list)

    # Creating an empty volume.
    vol, res = read_omezarr(slice_list[0], level=0)
    # flip to z, x, y
    chunks = vol.chunks[1:3]
    vol = np.transpose(vol, (1, 2, 0))
    nx, ny, nz = vol.shape
    wholebrain = np.zeros((nx, ny, slices_z[-1] + nz), dtype=np.float32)

    blending_options = {
        "width": args.blending_width,
        "nSteps": args.blending_nsteps,
        "fill_belowMask": False,
    }
    for i in tqdm(range(len(slices_z)), desc="Stitching slices"):
        k = slice_ids[i]

        # Loading this slice
        vol, res = read_omezarr(slice_files[k], level=0)
        vol = np.transpose(vol, (1, 2, 0))
        f = mask_files.get(k, None)
        mask = None
        if f is not None:
            # Loading the mask
            mask, res = read_omezarr(f, level=0)
            mask = np.transpose(mask, (1, 2, 0))

        # Get the z position of this slice
        z = slices_z[i]
        if i == len(slices_z) - 1:
            blending_options["fill_belowMask"] = True

        target_shape = (nx, ny, nz)  # from the first slice
        current_shape = vol.shape

        # Check if the x-dimension differs (index 0)
        if current_shape[0] < target_shape[0]:
            pad_amount_x = target_shape[0] - current_shape[0]
            vol = np.pad(vol, ((0, pad_amount_x), (0, 0), (0, 0)), mode="constant")
            if mask is not None:
                mask = np.pad(
                    mask, ((0, pad_amount_x), (0, 0), (0, 0)), mode="constant"
                )
        elif current_shape[0] > target_shape[0]:
            vol = vol[: target_shape[0], :, :]
            if mask is not None:
                mask = mask[: target_shape[0], :, :]

        # Check if the y-dimension differs (index 1)
        if current_shape[1] < target_shape[1]:
            pad_amount_y = target_shape[1] - current_shape[1]
            vol = np.pad(vol, ((0, 0), (0, pad_amount_y), (0, 0)), mode="constant")
            if mask is not None:
                mask = np.pad(
                    mask, ((0, 0), (0, pad_amount_y), (0, 0)), mode="constant"
                )
        elif current_shape[1] > target_shape[1]:
            vol = vol[:, : target_shape[1], :]
            if mask is not None:
                mask = mask[:, : target_shape[1], :]

        max_z = wholebrain.shape[2]
        overlap_z = min(vol.shape[2], max_z - z)  # z is the starting depth
        if vol.shape[2] != overlap_z:
            vol = vol[:, :, :overlap_z]
            if mask is not None:
                mask = mask[:, :, :overlap_z]

        wholebrain = stitch_oct.addSliceToVolume(
            wholebrain, vol, z, maskMoving=mask, **blending_options
        )

    # Saving the stitched volume
    wholebrain = np.transpose(wholebrain, (2, 0, 1))
    zarr_store = zarr.TempStore(suffix=".zarr")
    volume = zarr.open(
        zarr_store, mode="w", shape=wholebrain.shape, dtype=np.float32, chunks=chunks
    )
    volume[:] = wholebrain
    out_dask = da.from_zarr(volume)
    save_zarr(out_dask, args.output, res[1:], chunks)


if __name__ == "__main__":
    main()
