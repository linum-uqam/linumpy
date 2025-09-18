#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crop a 3D OME-Zarr volume to a specified depth below the water/tissue interface.

This script loads a 3D OME-Zarr volume, detects the water/tissue interface per (Y,X) then crops the 
volume to a specified depth *below* the interface. The script can also crop the data before the
water/tissue interface. The cropped volume is saved as a new OME-Zarr file.
"""

import argparse
from pathlib import Path
import numpy as np
import dask.array as da
import zarr
from linumpy.io.zarr import read_omezarr, save_omezarr, create_tempstore
from scipy.ndimage import gaussian_filter1d, gaussian_filter


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_zarr",
                   help="Path to the input 3D OME-Zarr OCT volume")
    p.add_argument("output_zarr",
                   help="Path to the output 3D OME-Zarr *cropped* volume",)
    p.add_argument("--sigma_xy", type=float, default=3.0,
                   help="Gaussian smoothing sigma in X and Y before interface detection [%(default)s]")
    p.add_argument("--sigma_z", type=float, default=2.0,
                   help="Gaussian smoothing sigma in Z before interface detection [%(default)s]")
    p.add_argument("--use_log", action="store_true",
                   help="Apply log transform before gradient detection")
    p.add_argument("--depth", type=int, default=300,
                   help="Target depth in um [%(default)s]")
    p.add_argument("--crop_before_interface", action="store_true",
                   help='If set, also crop the volume before the interface.')
    p.add_argument("--pad_after", action='store_true',
                   help='If set, pad the volume such that its depth below interface'
                        ' is equal to `depth`.')
    p.add_argument("--resolution", type=float,
                   help="Resolution in um. If not provided, it will be read\n"
                        "from the input zarr metadata.")
    return p


def main():
    args = _build_arg_parser().parse_args()
    input_path = Path(args.input_zarr)
    output_path = Path(args.output_zarr)

    # Load volume
    vol, res = read_omezarr(input_path, level=0)
    print('Loaded volume shape:', vol.shape)
    if args.resolution is not None:
        resolution = args.resolution
    else:
        resolution = (
            res[0] * 1000
        )  # Extract the Z resolution in um from the zarr metadata
    # vol is (Z, X, Y); reorient to (X, Y, Z) for xyzcorr functions
    vol_f = np.abs(vol) if np.iscomplexobj(vol) else vol
    vol_f = np.transpose(vol_f, (1, 2, 0))

    # compute the derivative along z to find the average tissue depth
    pad_width = int(np.round(args.sigma_z*4))
    vol_padded = np.pad(vol_f, ((0, 0), (0, 0), (pad_width, 0)), mode='wrap')
    vol_padded = gaussian_filter(vol_padded, (args.sigma_xy, args.sigma_xy, 0))
    dz = gaussian_filter1d(vol_padded, sigma=args.sigma_z, axis=-1, order=1)
    avg_dz = np.sum(dz, axis=(0, 1))

    avg_iface = max(int(np.argmax(avg_dz)) - pad_width, 0)
    print(f"Average surface depth: {avg_iface} voxels")

    # Compute number of Z-slices for desired depth (um / um-per-voxel)
    depth_px = int(round(args.depth / resolution))
    print(f"Cropping depth: {depth_px} voxels ({args.depth} um)")

    # Compute end index for cropping
    surface_idx = max(0, min(avg_iface, vol.shape[0] - 1))
    end_idx = surface_idx + depth_px
    if end_idx > vol.shape[0]:
        if args.pad_after:
            out_shape = (end_idx, vol.shape[1], vol.shape[2])
        else:
            out_shape = vol.shape
        store = create_tempstore()
        out_vol = zarr.open(store, mode="w", shape=out_shape,
                            dtype=np.float32, chunks=vol.chunks)
        out_vol[:vol.shape[0]] = vol[:]
        vol = out_vol

    # Crop volume along Z axis
    start_idx = 0 if not args.crop_before_interface else surface_idx
    vol_crop = vol[start_idx:end_idx, :, :]

    crop_dask = da.from_array(vol_crop, chunks=vol.chunks)
    # Save cropped volume as OME-Zarr
    save_omezarr(crop_dask, output_path, voxel_size=res, chunks=vol.chunks)


if __name__ == "__main__":
    main()
