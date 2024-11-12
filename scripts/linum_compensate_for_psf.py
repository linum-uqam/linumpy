#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
from linumpy.io.zarr import read_omezarr, save_zarr
from linumpy.psf.psf_estimator import extract_psfParametersFromMosaic, get_3dPSF


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_zarr',
                   help='Input stitched 3D slice (OME-zarr).')
    p.add_argument('out_zarr',
                   help='Output volume corrected for beam PSF (OME-zarr).')
    return p


def  main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # 1. load stitched tissue slice
    vol, res = read_omezarr(args.in_zarr, level=0)
    chunks = vol.chunks
    vol = np.moveaxis(vol, (0, 1, 2), (2, 1, 0))
    res = res[::-1]
    res_axial_microns = res[2] * 1000

    # 2. estimate psf
    zf, zr = extract_psfParametersFromMosaic(vol, res=res_axial_microns)
    psf3d = get_3dPSF(zf, zr, res_axial_microns, vol.shape)

    # Compensate by the PSF
    output = vol / psf3d

    # TODO: Use dask arrays
    output = np.moveaxis(output, (0, 1, 2), (2, 1, 0))
    res = res[::-1]

    save_zarr(output, args.out_zarr, scales=res, chunks=chunks)


if __name__ == '__main__':
    main()
