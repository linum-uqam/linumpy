#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of the FOD estimation method described in [1] based on
structure tensor analysis.
"""
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import convolve
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.core.sphere import HemiSphere

from linumpy.preproc.icorr import normalize
from linumpy.io.zarr import read_omezarr
from linumpy.feature.orientation import\
    _make_xfilter, _make_yfilter, _make_zfilter


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
    p.add_argument('--level', type=int, default=0,
                   help='Level of pyramid of image to process.')
    p.add_argument('--sigma', default=0.010, type=float,
                   help='Standard deviation of derivative of Gaussian (mm).')
    p.add_argument('--rho', default=0.025, type=float,
                   help='Standard deviation of Gaussian window for '
                        'structure tensor estimation (mm).')
    p.add_argument('--new_voxel_size', default=0.1, type=float,
                   help='Size of voxels for histological-FOD (mm).')
    p.add_argument('--damp', default=1.0, type=float,
                   help='Dampening factor for weighting structure tensor directions '
                        'based on the certainty measure `c_p`. Each tensor direction'
                        ' will be weighted by `c_p**damp` prior to binning.')
    p.add_argument('--reg', type=float, default=0.0,
                   help='Regularize by OCT intensity. [%(default)s]')
    return p


def samples_from_sigma(sigma):
    return np.arange(-int(np.ceil(sigma * 3)), int(np.ceil(sigma * 3)) + 1)


def gaussian(sigma):
    r = samples_from_sigma(sigma)
    ret = 1.0 / np.sqrt(2.0 * np.pi * sigma**2) * np.exp(-r**2 / 2.0 / sigma**2)
    ret = 1.0 / np.sqrt(np.sum(ret**2)) * ret
    return ret


def gaussian_derivative(sigma):
    r = samples_from_sigma(sigma)
    ret = -r / sigma**2 * gaussian(sigma)
    ret = 1.0 / np.sqrt(np.sum(ret**2)) * ret
    return ret


def convolve_zarr(volume, kernel, output):
    """
    Convolve a zarr array with a kernel.
    """
    chunk_size = volume.chunks

    padding = np.array(kernel.shape) // 2
    n_chunks_z = int(np.ceil(volume.shape[0] / chunk_size[0]))
    n_chunks_x = int(np.ceil(volume.shape[1] / chunk_size[1]))
    n_chunks_y = int(np.ceil(volume.shape[2] / chunk_size[2]))
    for i in range(n_chunks_z):
        for i in range(n_chunks_x):
            for j in range(n_chunks_y):
                x_lb = i * chunk_size[1]
                x_ub = min((i + 1) * chunk_size[1], volume.shape[1])
                y_lb = j * chunk_size[2]
                y_ub = min((j + 1) * chunk_size[2], volume.shape[2])
                chunk = volume[:, x_lb:x_ub, y_lb:y_ub]
                filtered_chunk = convolve(chunk, kernel, mode='wrap')
                output[:, x_lb:x_ub, y_lb:y_ub] = filtered_chunk


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if '.ome.zarr' in args.in_image:
        data, res = read_omezarr(args.in_image, level=args.level)
        data = np.moveaxis(data, (0, 1, 2), (2, 1, 0))
        data = np.swapaxes(data, 0, 1)
    elif '.nii' in args.in_image:
        nifti_img = nib.load(args.in_image)
        data = nifti_img.get_fdata()
        res = nifti_img.header.get_zooms()[:3]
    else:
        raise ValueError("Input image must be in .ome.zarr or .nii format.")

    data = normalize(data, 1, 99.99)

    sigma_to_vox = np.array([args.sigma]) / np.asarray(res)[::-1]
    rho_to_vox = np.array([args.rho]) / np.asarray(res)[::-1]

    # 1. Estimate derivatives f_x, f_y, f_z
    # For estimating derivatives (function of sigma)
    dx = _make_xfilter(gaussian_derivative(sigma_to_vox[0]))
    dy = _make_yfilter(gaussian_derivative(sigma_to_vox[1]))
    dz = _make_zfilter(gaussian_derivative(sigma_to_vox[2]))

    # Windowing function (function of rho)
    gaussian_filters = []
    gaussian_filters.append(_make_xfilter(gaussian(rho_to_vox[0])))
    gaussian_filters.append(_make_yfilter(gaussian(rho_to_vox[1])))
    gaussian_filters.append(_make_zfilter(gaussian(rho_to_vox[2])))

    derivatives = []
    # TODO: Use zarr to store derivatives
    # TODO: Use dask for computing derivatives
    derivatives.append(convolve(data, dx, mode='wrap'))
    derivatives.append(convolve(data, dy, mode='wrap'))
    derivatives.append(convolve(data, dz, mode='wrap'))

    # 2. Build structure tensor
    ST = np.zeros(data.shape + (3, 3))
    for i in range(3):
        for j in np.arange(i, 3):
            derivative = derivatives[i] * derivatives[j]
            for g_filter in gaussian_filters:
                derivative = convolve(derivative, g_filter)
            ST[..., i, j] = derivative
            ST[..., j, i] = derivative

    evals, evecs = np.linalg.eigh(ST)
    peaks = np.swapaxes(evecs, -2, -1)
    p = peaks[..., 0, :][..., None, :]

    # at the difference of Schilling et al (2016) here we use
    # the certainty measure to weight each peak direction instead
    # of thresholding the peaks based on this value
    c_p = ((evals[..., 1] - evals[..., 0]))
    lambda0 = evals[..., 2]
    c_p[lambda0 > 0] /= lambda0[lambda0 > 0]
    c_p = c_p**args.damp

    weight = c_p + args.reg*data

    new_voxel_size_to_vox = (np.array([args.new_voxel_size]) / np.asarray(res)[::-1]).astype(int)
    n_chunks_per_axis = np.ceil(np.asarray(data.shape) / np.asarray(new_voxel_size_to_vox)).astype(int)

    # 4. Create histogram for each new voxel
    sphere = HemiSphere.from_sphere(get_sphere(name='repulsion100'))
    b_mat, b_inv = sh_to_sf_matrix(sphere, sh_order_max=8)
    sh = np.zeros(np.append(n_chunks_per_axis, b_mat.shape[0]))
    resampled_oct = np.zeros(n_chunks_per_axis)
    print("Creating hist fod here.")
    for chunk_x in range(n_chunks_per_axis[0]):
        for chunk_y in range(n_chunks_per_axis[1]):
            for chunk_z in range(n_chunks_per_axis[2]):
                chunk = p[chunk_x * new_voxel_size_to_vox[0]:
                          (chunk_x + 1) * new_voxel_size_to_vox[0],
                          chunk_y * new_voxel_size_to_vox[1]:
                          (chunk_y + 1) * new_voxel_size_to_vox[1],
                          chunk_z * new_voxel_size_to_vox[2]:
                          (chunk_z + 1) * new_voxel_size_to_vox[2], :]
                chunk_oct = data[chunk_x * new_voxel_size_to_vox[0]:
                                 (chunk_x + 1) * new_voxel_size_to_vox[0],
                                 chunk_y * new_voxel_size_to_vox[1]:
                                 (chunk_y + 1) * new_voxel_size_to_vox[1],
                                 chunk_z * new_voxel_size_to_vox[2]:
                                 (chunk_z + 1) * new_voxel_size_to_vox[2]]
                chunk_certainty = weight[chunk_x * new_voxel_size_to_vox[0]:
                                         (chunk_x + 1) * new_voxel_size_to_vox[0],
                                         chunk_y * new_voxel_size_to_vox[1]:
                                         (chunk_y + 1) * new_voxel_size_to_vox[1],
                                         chunk_z * new_voxel_size_to_vox[2]:
                                         (chunk_z + 1) * new_voxel_size_to_vox[2]]
                score = np.abs(chunk.dot(sphere.vertices.T))
                ind = np.argmax(score, axis=-1).flatten()
                sf = np.zeros(sphere.vertices.shape[0])
                sf[ind] += chunk_certainty.flatten()
                sh[chunk_x, chunk_y, chunk_z, :] = sf.dot(b_inv)
                resampled_oct[chunk_x, chunk_y, chunk_z] = chunk_oct.mean()

    # we save the voxel size in microns for nifti images
    nib.save(nib.Nifti1Image(sh, np.diag(np.append([args.new_voxel_size]*3, [1.0]))),
             args.out_sh)
    nib.save(nib.Nifti1Image(c_p, np.diag((res[0], res[1], res[2], 1.0))),
             args.out_sh.replace('.nii.gz', '_confidence.nii.gz'))
    nib.save(nib.Nifti1Image(weight, np.diag((res[0], res[1], res[2], 1.0))),
             args.out_sh.replace('.nii.gz', '_weight.nii.gz'))
    # save the resampled oct image (might be useful for tracking mask)
    nib.save(nib.Nifti1Image(resampled_oct, np.diag(np.append([args.new_voxel_size]*3, [1.0]))),
             args.out_sh.replace('.nii.gz', '_oct.nii.gz'))


if __name__ == '__main__':
    main()
