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
    p.add_argument('--out_psf',
                   help='Optional output PSF filename.')
    p.add_argument('--nz', type=int, default=25,
                   help='The "nz" first voxels belonging to background [%(default)s].')
    p.add_argument('--n_profiles', type=int, default=10,
                   help='Number of intensity profiles to use [%(default)s].')
    p.add_argument('--n_iterations', type=int, default=15,
                   help='Number of iterations [%(default)s].')
    p.add_argument('--smooth', type=float, default=0.01,
                   help='Smoothing factor as a fraction of volume depth [%(default)s].')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # 1. load stitched tissue slice
    vol, res = read_omezarr(args.in_zarr, level=0)
    chunks = vol.chunks
    vol = np.moveaxis(vol, (0, 1, 2), (2, 1, 0))
    res = res[::-1]
    res_axial_microns = res[2] * 1000

    # 2. estimate psf
    zf, zr = extract_psfParametersFromMosaic(vol, nProfiles=args.n_profiles,
                                             res=res_axial_microns, f=args.smooth,
                                             nIterations=args.n_iterations)
    psf_3d = get_3dPSF(zf, zr, res_axial_microns, vol.shape)

    # Compensate by the PSF
    background = np.mean(vol[..., :args.nz])
    output = (vol - background) / psf_3d + background

    # remove negative values
    output -= output.min()

    # TODO: Use dask arrays
    output = np.moveaxis(output, (0, 1, 2), (2, 1, 0))
    res = res[::-1]

    if args.out_psf:
        psf_3d = np.moveaxis(psf_3d, (0, 1, 2), (2, 1, 0))
        # when there are too many levels it'll break the viewer for some reason
        save_zarr(psf_3d.astype(np.float32), args.out_psf, voxel_size=res, chunks=chunks)

    save_zarr(output.astype(np.float32), args.out_zarr, voxel_size=res, chunks=chunks)


if __name__ == '__main__':
    main()
