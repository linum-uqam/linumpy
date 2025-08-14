#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of the FOD estimation method described in [1] based on
structure tensor analysis.
"""
import argparse
import numpy as np
import nibabel as nib
import dask.array as da

from linumpy.feature.structure_tensor import compute_structure_tensor, compute_principal_direction, compute_sh, StoreMode
from linumpy.io.zarr import read_omezarr, save_omezarr
from linumpy.utils.io import add_processes_arg, parse_processes_arg


EPILOG="""
[1] Schilling et al, "Comparison of 3D orientation distribution functions
    measured with confocal microscopy and diffusion MRI". Neuroimage. 2016
    April 1; 129: 185-197. doi:10.1016/j.neuroimage.2016.01.022
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_image',
                   help='Input image in .ome.zarr or nifti.')
    p.add_argument('out_sh',
                   help='Output SH image (.nii.gz).')
    p.add_argument('--out_coherence',
                   help='If provided, save the coherence image (.ome.zarr).')
    p.add_argument('--out_peak',
                   help='If provided, save the peak directions image (.ome.zarr).')
    p.add_argument('--level', type=int, default=0,
                   help='Level of pyramid of image to process. [%(default)s]')
    p.add_argument('--sigma', default=0.010, type=float,
                   help='Standard deviation of derivative of Gaussian (mm). [%(default)s]')
    p.add_argument('--rho', default=0.025, type=float,
                   help='Standard deviation of Gaussian window for '
                        'structure tensor estimation (mm). [%(default)s]')
    p.add_argument('--new_voxel_size', default=0.1, type=float,
                   help='Size of voxels for histological-FOD (mm). [%(default)s]')
    p.add_argument('--damp', default=1.0, type=float,
                   help='Dampening factor for weighting structure tensor directions '
                        'based on the certainty measure `c_p`.\nEach tensor direction'
                        ' will be weighted by `c_p**damp` prior to binning. [%(default)s]')
    p.add_argument('--reg', type=float, default=0.0,
                   help='Regularize by OCT intensity. [%(default)s]')
    p.add_argument('--sh_order', type=int, default=8,
                   help='Spherical harmonics maximum order [%(default)s].')
    p.add_argument('--store_mode', default='TEMPSTORE',
                   choices=[member.name for member in StoreMode],
                   help='Store mode for the temporary files used '
                        'during the computation. [%(default)s]')
    add_processes_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if '.ome.zarr' in args.in_image:
        data, res = read_omezarr(args.in_image, level=args.level)
    else:
        parser.error("Input image must be .ome.zarr.")

    n_jobs = parse_processes_arg(args.n_processes)
    store_mode = StoreMode[args.store_mode]

    # Axis order is (z, y, x)
    sigma_to_vox = np.array([args.sigma]) / np.asarray(res)
    rho_to_vox = np.array([args.rho]) / np.asarray(res)

    # Compute the structure tensor
    # /!\ disk space required will be 6 times the size of the input image
    st_xx, st_xy, st_xz, st_yy, st_yz, st_zz = compute_structure_tensor(
        data, sigma_to_vox, rho_to_vox, n_jobs=n_jobs, store_mode=store_mode)

    # Estimate principal direction and coherence from
    # structure tensor eigenvalues and eigenvectors
    # /!\ disk space will be 3 times the size of the input image (peaks)
    # and 1 time the size of the input image (coherence)
    peak, coherence = compute_principal_direction(
        st_xx, st_xy, st_xz, st_yy, st_yz, st_zz,
        data, args.damp, n_jobs=n_jobs, store_mode=store_mode)

    # optionally save the coherence and peak images
    if args.out_coherence is not None:
        save_omezarr(da.from_zarr(coherence), args.out_coherence, voxel_size=res, chunks=data.chunks)
    if args.out_peak is not None:
        save_omezarr(da.from_zarr(peak), args.out_peak, voxel_size=(1,) + res, chunks=(3,) + data.chunks)

    # Compute SH inside bigger voxels
    # TODO: Add warning if the new voxel size does not divide the original voxel size
    new_voxel_size_to_vox = (np.array([args.new_voxel_size]) / np.asarray(res)).astype(int)
    sh = compute_sh(data, peak, coherence, new_voxel_size_to_vox, args.sh_order, args.reg)

    nib.save(nib.Nifti1Image(sh, np.diag(np.append([args.new_voxel_size]*3, [1.0]))), args.out_sh)


if __name__ == '__main__':
    main()
