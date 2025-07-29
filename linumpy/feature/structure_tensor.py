# -*- coding:utf-8 -*-
from linumpy.feature.kernel import make_xfilter, make_yfilter, make_zfilter, gaussian, gaussian_derivative

import numpy as np
import zarr
import dask.array as da
import dask
from tqdm import tqdm
import itertools

from dipy.core.sphere import HemiSphere
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.data import get_sphere
from scipy.ndimage import convolve

from multiprocessing.pool import ThreadPool
from pqdm.processes import pqdm


def dask_to_zarr(array, shape, chunks, dtype='float32'):
    """
    Save a dask array to a temporary zarr store. Any delayed dask task
    will be computed and saved to the zarr store.

    Parameters
    ----------
    array: dask.array
        Dask array to store to zarr.
    shape: tuple
        Shape of array.
    chunks: tuple
        Shape of chunks in zarr array.
    dtype: str or numpy.dtype
        Datatype for stored data.

    Returns
    -------
    zarr_array: zarr.core.array
        Zarr array.
    """
    zarr_store = zarr.TempStore()
    zarr_array = zarr.open(zarr_store, mode='w', shape=shape, chunks=chunks, dtype=dtype)
    da.to_zarr(array, zarr_array, shape=shape, chunks=chunks, dtype=dtype)
    return zarr_array


def compute_sh(data, peak, coherence, new_voxel_size_to_vox, sh_order_max, reg_factor):
    """
    Compute the spherical harmonics (SH) representation of the input data
    using the peak directions and coherence from the structure tensor.

    Parameters
    ----------
    data: zarr.core.Array
        Input data array (3D).
    peak: zarr.core.Array
        Peak directions array (3D, shape (3, z, y, x)).
    coherence: zarr.core.Array
        Coherence array (3D).
    new_voxel_size_to_vox: np.ndarray
        New voxel size in voxels (shape (3,)).
    sh_order_max: int
        Maximum order of spherical harmonics to compute.
    reg_factor: float
        Regularization factor to apply to the coherence.

    Returns
    -------
    sh: np.ndarray
        Spherical harmonics representation of the input data (shape (z, y, x, n_sh)).
        Where n_sh is the number of spherical harmonics coefficients.
    """
    new_shape = np.ceil(np.asarray(data.shape) / new_voxel_size_to_vox).astype(int)
    sh_chunk_shape = (np.asarray(data.chunks) / new_voxel_size_to_vox).astype(int)
    data_chunk_shape = (sh_chunk_shape * new_voxel_size_to_vox).astype(int)
    n_chunks_per_axis = np.ceil(new_shape / sh_chunk_shape).astype(int)

    sphere = HemiSphere.from_sphere(get_sphere(name='repulsion100'))
    b_mat, b_inv = sh_to_sf_matrix(sphere, sh_order_max=sh_order_max)

    chunks = list(itertools.product(*[range(n_chunks_per_axis[i])
                                      for i in range(len(n_chunks_per_axis))]))
    sh = np.zeros(np.append(new_shape, b_mat.shape[0]), dtype=np.float32)

    for chunk in tqdm(chunks, desc='Computing SH inside voxel'):
        sf = compute_sf_for_chunk(chunk, new_voxel_size_to_vox,
                                  data_chunk_shape, data, peak, coherence,
                                  reg_factor, sphere)
        sh_chunk = sf.dot(b_inv)
        sh[chunk[0]*sh_chunk_shape[0]:(chunk[0]+1)*sh_chunk_shape[0],
           chunk[1]*sh_chunk_shape[1]:(chunk[1]+1)*sh_chunk_shape[1],
           chunk[2]*sh_chunk_shape[2]:(chunk[2]+1)*sh_chunk_shape[2], :] = sh_chunk

    return sh


def compute_sf_for_chunk(chunk_id, new_voxel_size_to_vox, chunk_shape,
                         data, peak, coherence, reg, sphere):
    """
    Compute the spherical function representation for a chunk of data.

    Parameters
    ----------
    chunk_id: tuple
        Tuple of integers representing the chunk coordinates.
    new_voxel_size_to_vox: np.ndarray
        New voxel size in voxels (shape (3,)).
    chunk_shape: tuple
        Shape of the chunk in voxels.
    data: zarr.core.Array
        Input data array (3D).
    peak: zarr.core.Array
        Peak directions array (4D, shape (3, z, y, x)).
    coherence: zarr.core.Array
        Coherence array (3D).
    reg: float
        Regularization factor to apply to the coherence.
    sphere: dipy.core.sphere.HemiSphere
        Sphere object containing the sphere vertices.

    Returns
    -------
    sf: np.ndarray
        Spherical harmonics representation for the chunk (shape (nx, ny, nz, n_dirs)).
    """
    chunk_x, chunk_y, chunk_z = chunk_id
    xmin = chunk_x * chunk_shape[0]
    xmax = min(data.shape[0], (chunk_x + 1) * chunk_shape[0])
    ymin = chunk_y * chunk_shape[1]
    ymax = min(data.shape[1], (chunk_y + 1) * chunk_shape[1])
    zmin = chunk_z * chunk_shape[2]
    zmax = min(data.shape[2], (chunk_z + 1) * chunk_shape[2])

    slice_x = slice(xmin, xmax)
    slice_y = slice(ymin, ymax)
    slice_z = slice(zmin, zmax)

    data_chunk = np.asarray(data[slice_x, slice_y, slice_z])
    peak_chunk = np.moveaxis(np.asarray(peak[:, slice_x, slice_y, slice_z]), 0, -1)[..., None, :]
    coherence_chunk = np.asarray(coherence[slice_x, slice_y, slice_z])

    # shape (nx, ny, nz)
    weight = coherence_chunk + reg * data_chunk

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


def compute_principal_direction(st_xx, st_xy, st_xz, st_yy, st_yz, st_zz,
                                data, damp_factor, n_jobs=12):
    """
    Compute the principal direction and coherence from the structure tensor components.

    Parameters
    ----------
    st_xx: zarr.core.Array
        Structure tensor component xx (3D).
    st_xy: zarr.core.Array
        Structure tensor component xy (3D).
    st_xz: zarr.core.Array
        Structure tensor component xz (3D).
    st_yy: zarr.core.Array
        Structure tensor component yy (3D).
    st_yz: zarr.core.Array
        Structure tensor component yz (3D).
    st_zz: zarr.core.Array
        Structure tensor component zz (3D).
    data: zarr.core.Array
        Input data array (3D).
    damp_factor: float
        Dampening factor for the coherence computation.
    n_jobs: int
        Number of parallel jobs to use for computation.

    Returns
    -------
    peak: zarr.core.Array
        Peak directions array (4D, shape (3, z, y, x)).
    coherence: zarr.core.Array
        Coherence array (3D).
    """
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
                 itertools.repeat(damp_factor))
    pqdm(params, compute_peak_direction_and_coherence_for_chunk,
         n_jobs=n_jobs, argument_type='args')

    return peak, coherence


def compute_peak_direction_and_coherence_for_chunk(chunk, coherence, peak, chunk_shape,
                                                   st_xx, st_yy, st_zz, st_xy, st_xz, st_yz,
                                                   damp):
    """
    Compute the principal direction and coherence from the structure tensor components.

    Parameters
    ----------
    chunk: tuple
        Tuple of integers representing the chunk coordinates.
    coherence: zarr.core.Array
        Coherence array (3D). Filled inplace.
    peak: zarr.core.Array
        Peak directions array (4D, shape (3, z, y, x)). Filled inplace.
    chunk_shape: tuple
        Shape of the chunk in voxels.
    st_xx: zarr.core.Array
        Structure tensor component xx (3D).
    st_xy: zarr.core.Array
        Structure tensor component xy (3D).
    st_xz: zarr.core.Array
        Structure tensor component xz (3D).
    st_yy: zarr.core.Array
        Structure tensor component yy (3D).
    st_yz: zarr.core.Array
        Structure tensor component yz (3D).
    st_zz: zarr.core.Array
        Structure tensor component zz (3D).
    data: zarr.core.Array
        Input data array (3D).
    damp_factor: float
        Dampening factor for the coherence computation.
    n_jobs: int
        Number of parallel jobs to use for computation.
    """
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
    p0 = np.moveaxis(p0, -1, 0)  # Move to (3, z, y ,x)

    c_p = ((evals[..., 1] - evals[..., 0]))
    lambda0 = evals[..., 2]
    c_p[lambda0 > 0] /= lambda0[lambda0 > 0]
    c_p = c_p**damp
    coherence[slice_x, slice_y, slice_z] = c_p
    peak[:, slice_x, slice_y, slice_z] = p0


def compute_structure_tensor(data, sigma_to_vox, rho_to_vox, n_threads=12):
    """
    Compute the structure tensor components from the input data.
    
    Parameters
    ----------
    data: zarr.core.Array
        Input data array (3D).
    sigma_to_vox: np.ndarray
        Standard deviation of the Gaussian derivative in voxels (shape (3,)).
    rho_to_vox: np.ndarray
        Standard deviation of the Gaussian window in voxels (shape (3,)).
    n_threads: int
        Number of threads to use for parallel computation.

    Returns
    -------
    st_xx, st_xy, st_xz, st_yy, st_yz, st_zz: zarr.core.Array
        Structure tensor components (3D zarr arrays).
    """
    # Set number of threads for dask
    dask.config.set(pool=ThreadPool(n_threads))

    # Derivatives (function of sigma)
    dx_filter = make_xfilter(gaussian_derivative(sigma_to_vox[0]))
    dy_filter = make_yfilter(gaussian_derivative(sigma_to_vox[1]))
    dz_filter = make_zfilter(gaussian_derivative(sigma_to_vox[2]))

    dx = da.from_zarr(data, chunks=data.chunks)
    dy = da.from_zarr(data, chunks=data.chunks)
    dz = da.from_zarr(data, chunks=data.chunks)
    dx = dx.map_overlap(lambda x: convolve(x, dx_filter), depth=dx_filter.size//2)
    dy = dy.map_overlap(lambda x: convolve(x, dy_filter), depth=dy_filter.size//2)
    dz = dz.map_overlap(lambda x: convolve(x, dz_filter), depth=dz_filter.size//2)

    # Windowing function (function of rho)
    xfilter = make_xfilter(gaussian(rho_to_vox[0]))
    yfilter = make_yfilter(gaussian(rho_to_vox[1]))
    zfilter = make_zfilter(gaussian(rho_to_vox[2]))

    # Structure tensor components
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

    return st_xx, st_xy, st_xz, st_yy, st_yz, st_zz
