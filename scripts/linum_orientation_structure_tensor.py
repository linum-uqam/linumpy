#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of the FOD estimation method described in [1] based on
structure tensor analysis.
"""
from multiprocessing.pool import ThreadPool
from pqdm.processes import pqdm
import dask

import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import convolve
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.core.sphere import HemiSphere

from linumpy.io.zarr import read_omezarr, save_omezarr
from linumpy.feature.orientation import\
    _make_xfilter, _make_yfilter, _make_zfilter

from tqdm import tqdm
import zarr
import dask.array as da
import itertools


EPILOG="""
[1] Schilling et al, "Comparison of 3D orientation distribution functions
    measured with confocal microscopy and diffusion MRI". Neuroimage. 2016
    April 1; 129: 185-197. doi:10.1016/j.neuroimage.2016.01.022
"""
N_NEW_VOXELS_PER_CHUNK = 10  # TODO: make this a parameter


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
    p.add_argument('--sh_order', type=int, default=8,
                   help='Spherical harmonics maximum order [%(default)s].')
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


def dask_to_zarr(array, shape, chunks, dtype='float32'):
    """
    Save a dask array to a zarr store.
    """
    zarr_store = zarr.TempStore()
    zarr_array = zarr.open(zarr_store, mode='w', shape=shape, chunks=chunks, dtype=dtype)
    da.to_zarr(array, zarr_array, shape=shape, chunks=chunks, dtype=dtype)
    return zarr_array


def compute_sh_inside_voxel(chunk_id, new_voxel_size_to_vox,
                            n_new_voxels_per_chunk, data, peak,
                            coherence, reg, sphere):
    chunk_x, chunk_y, chunk_z = chunk_id
    xmin = chunk_x * new_voxel_size_to_vox[0] * n_new_voxels_per_chunk
    xmax = min(data.shape[0], (chunk_x + 1) * new_voxel_size_to_vox[0] * n_new_voxels_per_chunk)
    ymin = chunk_y * new_voxel_size_to_vox[1] * n_new_voxels_per_chunk
    ymax = min(data.shape[1], (chunk_y + 1) * new_voxel_size_to_vox[1] * n_new_voxels_per_chunk)
    zmin = chunk_z * new_voxel_size_to_vox[2] * n_new_voxels_per_chunk
    zmax = min(data.shape[2], (chunk_z + 1) * new_voxel_size_to_vox[2] * n_new_voxels_per_chunk)

    slice_x = slice(xmin, xmax)
    slice_y = slice(ymin, ymax)
    slice_z = slice(zmin, zmax)

    data_chunk = np.asarray(data[slice_x, slice_y, slice_z])
    peak_chunk = np.moveaxis(np.asarray(peak[:, slice_x, slice_y, slice_z]), 0, -1)[..., None, :]
    coherence_chunk = np.asarray(coherence[slice_x, slice_y, slice_z])

    # shape (nx, ny, nz)
    weight = coherence_chunk + reg*data_chunk

    # shape (nx, ny, nz, n_dirs)
    score = np.abs(peak_chunk.dot(sphere.vertices.T))

    # for each voxel we compute the peak index corresponding to the direction
    ind = np.argmax(score, axis=-1)[..., 0]
    grid_indices = np.indices(weight.shape)
    new_grid_x = (grid_indices[0] / new_voxel_size_to_vox[0]).astype(int)
    new_grid_y = (grid_indices[1] / new_voxel_size_to_vox[1]).astype(int)
    new_grid_z = (grid_indices[2] / new_voxel_size_to_vox[2]).astype(int)
    sf = np.zeros((new_grid_x.max() + 1, new_grid_y.max() + 1,
                   new_grid_z.max() + 1, sphere.vertices.shape[0]))
    sf[new_grid_x, new_grid_y, new_grid_z, ind] += weight
    return sf


def compute_peak_direction_and_coherence_for_chunk(chunk, coherence, peak, chunk_shape,
                                                   st_xx, st_yy, st_zz, st_xy, st_xz, st_yz,
                                                   damp):
    chunk_x, chunk_y, chunk_z = chunk
    xmin = chunk_x * chunk_shape[0]
    xmax = min(coherence.shape[0], (chunk_x + 1) * chunk_shape[0])
    ymin = chunk_y * chunk_shape[1]
    ymax = min(coherence.shape[1], (chunk_y + 1) * chunk_shape[1])
    zmin = chunk_z * chunk_shape[2]
    zmax = min(coherence.shape[2], (chunk_z + 1) * chunk_shape[2])
    st = np.zeros((xmax-xmin, ymax-ymin, zmax-zmin, 3, 3))
    slice_x = slice(xmin, xmax)
    slice_y = slice(ymin, ymax)
    slice_z = slice(zmin, zmax)
    st[..., 0, 0] = st_xx[slice_x, slice_y, slice_z]
    st[..., 1, 1] = st_yy[slice_x, slice_y, slice_z]
    st[..., 2, 2] = st_zz[slice_x, slice_y, slice_z]
    st[..., 0, 1] = st[..., 1, 0] = st_xy[slice_x, slice_y, slice_z]
    st[..., 0, 2] = st[..., 2, 0] = st_xz[slice_x, slice_y, slice_z]
    st[..., 1, 2] = st[..., 2, 1] = st_yz[slice_x, slice_y, slice_z]

    evals, evecs = np.linalg.eigh(st)
    p0 = np.swapaxes(evecs, -2, -1)[..., 0, :]
    p0 = np.moveaxis(p0, -1, 0)  # Move to (3, x, y, z)

    c_p = ((evals[..., 1] - evals[..., 0]))
    lambda0 = evals[..., 2]
    c_p[lambda0 > 0] /= lambda0[lambda0 > 0]
    c_p = c_p**damp
    coherence[slice_x, slice_y, slice_z] = c_p
    peak[:, slice_x, slice_y, slice_z] = p0


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if '.ome.zarr' in args.in_image:
        data, res = read_omezarr(args.in_image, level=args.level)
    else:
        parser.error("Input image must be .ome.zarr.")

    dask.config.set(pool=ThreadPool(12))

    # note that nothing has been flipped yet -> axis order is (z, y, x)
    sigma_to_vox = np.array([args.sigma]) / np.asarray(res)
    rho_to_vox = np.array([args.rho]) / np.asarray(res)

    # 1. Estimate derivatives f_x, f_y, f_z
    # For estimating derivatives (function of sigma)
    dx_filter = _make_xfilter(gaussian_derivative(sigma_to_vox[0]))
    dy_filter = _make_yfilter(gaussian_derivative(sigma_to_vox[1]))
    dz_filter = _make_zfilter(gaussian_derivative(sigma_to_vox[2]))

    dx = da.from_zarr(data, chunks=(128, 128, 128))
    dy = da.from_zarr(data, chunks=(128, 128, 128))
    dz = da.from_zarr(data, chunks=(128, 128, 128))
    dx = dx.map_overlap(lambda x: convolve(x, dx_filter),
                        depth=dx_filter.size//2)
    dy = dy.map_overlap(lambda x: convolve(x, dy_filter),
                        depth=dy_filter.size//2)
    dz = dz.map_overlap(lambda x: convolve(x, dz_filter),
                        depth=dz_filter.size//2)

    # Windowing function (function of rho)
    xfilter = _make_xfilter(gaussian(rho_to_vox[0]))
    yfilter = _make_yfilter(gaussian(rho_to_vox[1]))
    zfilter = _make_zfilter(gaussian(rho_to_vox[2]))

    # 2. Build structure tensor
    st_xx = dx * dx
    st_xx = st_xx.map_overlap(lambda x: convolve(convolve(convolve(x, xfilter), yfilter), zfilter),
                              depth=xfilter.size//2)
    st_xx = dask_to_zarr(st_xx, data.shape, data.chunks)

    st_xy = dx * dy
    st_xy = st_xy.map_overlap(lambda x: convolve(convolve(convolve(x, xfilter), yfilter), zfilter),
                              depth=xfilter.size//2)
    st_xy = dask_to_zarr(st_xy, data.shape, data.chunks)

    st_xz = dx * dz
    st_xz = st_xz.map_overlap(lambda x: convolve(convolve(convolve(x, xfilter), yfilter), zfilter),
                              depth=xfilter.size//2)
    st_xz = dask_to_zarr(st_xz, data.shape, data.chunks)

    st_yy = dy * dy
    st_yy = st_yy.map_overlap(lambda x: convolve(convolve(convolve(x, xfilter), yfilter), zfilter),
                              depth=xfilter.size//2)
    st_yy = dask_to_zarr(st_yy, data.shape, data.chunks)

    st_yz = dy * dz
    st_yz = st_yz.map_overlap(lambda x: convolve(convolve(convolve(x, xfilter), yfilter), zfilter),
                              depth=xfilter.size//2)
    st_yz = dask_to_zarr(st_yz, data.shape, data.chunks)

    st_zz = dz * dz
    st_zz = st_zz.map_overlap(lambda x: convolve(convolve(convolve(x, xfilter), yfilter), zfilter),
                              depth=xfilter.size//2)
    st_zz = dask_to_zarr(st_zz, data.shape, data.chunks)

    coherence = zarr.open(zarr.TempStore(), mode='w', shape=data.shape,
                          chunks=data.chunks, dtype=data.dtype)
    peak = zarr.open(zarr.TempStore(), mode='w',
                     shape=(3,) + data.shape,
                     chunks=(3,) + data.chunks,
                     dtype=data.dtype)

    n_chunks_per_axis = [np.ceil(data.shape[i] / float(data.chunks[i])).astype(int)
                         for i in range(3)]
    chunks = list(itertools.product(*[range(n) for n in n_chunks_per_axis]))
    chunk_shape = data.chunks

    params = zip(chunks,
                 itertools.repeat(coherence),
                 itertools.repeat(peak),
                 itertools.repeat(chunk_shape),
                 itertools.repeat(st_xx),
                 itertools.repeat(st_yy),
                 itertools.repeat(st_zz),
                 itertools.repeat(st_xy),
                 itertools.repeat(st_xz),
                 itertools.repeat(st_yz),
                 itertools.repeat(args.damp))
    pqdm(params, compute_peak_direction_and_coherence_for_chunk, n_jobs=12, argument_type='args')
    save_omezarr(da.from_zarr(coherence), 'coherence.ome.zarr', voxel_size=res, chunks=data.chunks)
    save_omezarr(da.from_zarr(peak), 'peak.ome.zarr', voxel_size=(1,) + res, chunks=(3,) + data.chunks)

    # Merge the result into bigger voxels
    # TODO: Add warning if the new voxel size does not divide the original voxel size
    new_voxel_size_to_vox = (np.array([args.new_voxel_size]) / np.asarray(res)).astype(int)
    new_shape = np.ceil(np.asarray(data.shape) / new_voxel_size_to_vox).astype(int)
    n_chunks_per_axis = np.ceil(new_shape /  N_NEW_VOXELS_PER_CHUNK).astype(int)

    sphere = HemiSphere.from_sphere(get_sphere(name='repulsion100'))
    b_mat, b_inv = sh_to_sf_matrix(sphere, sh_order_max=args.sh_order)

    chunks = list(itertools.product(*[range(n_chunks_per_axis[i]) for i in range(len(n_chunks_per_axis))]))
    sh = np.zeros(np.append(new_shape, b_mat.shape[0]), dtype=np.float32)
    for chunk in tqdm(chunks, desc='Computing SH inside voxel'):
        sf = compute_sh_inside_voxel(chunk, new_voxel_size_to_vox,
                                     N_NEW_VOXELS_PER_CHUNK, data,
                                     peak, coherence, args.reg, sphere)
        sh_chunk = sf.dot(b_inv)
        sh[chunk[0]*N_NEW_VOXELS_PER_CHUNK:(chunk[0]+1)*N_NEW_VOXELS_PER_CHUNK,
           chunk[1]*N_NEW_VOXELS_PER_CHUNK:(chunk[1]+1)*N_NEW_VOXELS_PER_CHUNK,
           chunk[2]*N_NEW_VOXELS_PER_CHUNK:(chunk[2]+1)*N_NEW_VOXELS_PER_CHUNK, :] = sh_chunk

    nib.save(nib.Nifti1Image(sh, np.diag(np.append([args.new_voxel_size]*3, [1.0]))), args.out_sh)


if __name__ == '__main__':
    main()
