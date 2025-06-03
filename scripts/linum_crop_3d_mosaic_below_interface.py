#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crop a 3D OME-Zarr volume to a specified depth below the water/tissue interface.

This script loads a 3D OME-Zarr volume, detects the water/tissue interface per (Y,X) then crops the 
volume to a specified depth below the interface. The cropped volume is saved as a new OME-Zarr file.
"""

import argparse
from pathlib import Path
import numpy as np
import dask.array as da
from linumpy.io.zarr import read_omezarr, save_omezarr
from linumpy.preproc import xyzcorr

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "input_zarr",
        help="Path to the input 3D OME-Zarr OCT volume",
    )
    p.add_argument(
        "output_zarr",
        help="Path to the output 3D OME-Zarr *cropped* volume",
    )
    p.add_argument(
        "--sigma_xy",
        type=float,
        default=3.0,
        help="Gaussian smoothing sigma in X and Y before interface detection (default: 3)",
    )
    p.add_argument(
        "--sigma_z",
        type=float,
        default=2.0,
        help="Gaussian smoothing sigma in Z before interface detection (default: 2)",
    )
    p.add_argument(
        "--use_log",
        action="store_true",
        default=False,
        help="Apply log transform before gradient detection (default: False)",
    )
    p.add_argument("--depth", type=int, default=300, help="Target depth (default: 300)")
    p.add_argument(
        "--resolution",
        type=float,
        default=-1,
        help="Resolution in um. If not provided, it will be read from the input zarr metadata.",
    )
    return p


def main():
    args = _build_arg_parser().parse_args()
    input_path = Path(args.input_zarr)
    output_path = Path(args.output_zarr)

    # Load volume
    vol, res = read_omezarr(input_path, level=0)
    if args.resolution > 0:
        resolution = args.resolution
    else:
        resolution = (
            res[2] * 1000
        )  # Extract the Z resolution in µm from the zarr metadata
    # vol is (Z, X, Y); reorient to (X, Y, Z) for xyzcorr functions
    vol_f = np.abs(vol) if np.iscomplexobj(vol) else vol
    vol_f = np.transpose(vol_f, (1, 2, 0))

    # Detect interface
    interface = xyzcorr.findTissueInterface(
        vol_f, s_z=args.sigma_z, s_xy=args.sigma_xy, useLog=args.use_log
    )

    # Generate mask Under interface
    mask = xyzcorr.maskUnderInterface(vol_f, interface, returnMask=True)

    # Exclude out of bounds columns
    mask_all = mask.all(axis=2)  # True where mask is True for every voxel along the aline
    tissue_present = (
        ~mask_all
    )  # keep alines with at least one False (i.e., valid tissue interface)
    # Average interface only where tissue_present is True
    valid_ifaces = interface[tissue_present]
    avg_iface = int(round(valid_ifaces.mean())) if valid_ifaces.size > 0 else 0
    print(f"Average surface depth: {avg_iface} voxels")

    # Compute number of Z-slices for desired depth (µm / µm-per-voxel)
    depth_px = int(round(args.depth / resolution))
    print(f"Cropping depth: {depth_px} voxels ({args.depth} µm)")

    # Compute end index for cropping
    surface_idx = max(0, min(avg_iface, vol.shape[0] - 1))
    end_idx = min(vol.shape[0], surface_idx + depth_px)

    # Crop volume along Z axis
    vol_crop = vol[0:end_idx, :, :]

    crop_dask = da.from_array(vol_crop, chunks=vol.chunks)
    # Save cropped volume as OME-Zarr
    save_omezarr(
        crop_dask,
        output_path,
        scales=res,
        chunks=vol.chunks
    )


if __name__ == "__main__":
    main()
