"""
GPU-accelerated interpolation and resampling operations for linumpy.

Provides GPU versions of image resampling, affine transforms, and
coordinate mapping operations.
"""

import numpy as np

from . import GPU_AVAILABLE, to_cpu


def affine_transform(image, matrix, output_shape=None, order=1, use_gpu=True):
    """
    GPU-accelerated affine transformation.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (2D or 3D)
    matrix : np.ndarray
        Affine transformation matrix
    output_shape : tuple, optional
        Shape of output image. If None, uses input shape.
    order : int
        Interpolation order (0=nearest, 1=linear, 3=cubic)
    use_gpu : bool
        Whether to use GPU acceleration
        
    Returns
    -------
    np.ndarray
        Transformed image
    """
    if output_shape is None:
        output_shape = image.shape

    if use_gpu and GPU_AVAILABLE:
        return _affine_transform_gpu(image, matrix, output_shape, order)
    else:
        return _affine_transform_cpu(image, matrix, output_shape, order)


def _affine_transform_gpu(image, matrix, output_shape, order):
    """GPU implementation of affine transform."""
    import cupy as cp
    from cupyx.scipy.ndimage import affine_transform as cp_affine

    img_gpu = cp.asarray(image.astype(np.float32))
    matrix_gpu = cp.asarray(matrix.astype(np.float32))

    result = cp_affine(img_gpu, matrix_gpu, output_shape=output_shape, order=order)

    return to_cpu(result)


def _affine_transform_cpu(image, matrix, output_shape, order):
    """CPU fallback for affine transform."""
    from scipy.ndimage import affine_transform as scipy_affine
    return scipy_affine(image, matrix, output_shape=output_shape, order=order)


def map_coordinates(image, coordinates, order=1, use_gpu=True):
    """
    GPU-accelerated coordinate mapping (general interpolation).
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    coordinates : np.ndarray
        Coordinates to sample at, shape (ndim, ...)
    order : int
        Interpolation order
    use_gpu : bool
        Whether to use GPU
        
    Returns
    -------
    np.ndarray
        Interpolated values
    """
    if use_gpu and GPU_AVAILABLE:
        return _map_coordinates_gpu(image, coordinates, order)
    else:
        return _map_coordinates_cpu(image, coordinates, order)


def _map_coordinates_gpu(image, coordinates, order):
    """GPU implementation of map_coordinates."""
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates as cp_map

    img_gpu = cp.asarray(image.astype(np.float32))
    coords_gpu = cp.asarray(coordinates.astype(np.float32))

    result = cp_map(img_gpu, coords_gpu, order=order)

    return to_cpu(result)


def _map_coordinates_cpu(image, coordinates, order):
    """CPU fallback for map_coordinates."""
    from scipy.ndimage import map_coordinates as scipy_map
    return scipy_map(image, coordinates, order=order)


def resize(image, output_shape, order=1, anti_aliasing=True, use_gpu=True):
    """
    GPU-accelerated image resize.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    output_shape : tuple
        Desired output shape
    order : int
        Interpolation order
    anti_aliasing : bool
        Whether to apply anti-aliasing filter before downsampling
    use_gpu : bool
        Whether to use GPU
        
    Returns
    -------
    np.ndarray
        Resized image
    """
    if use_gpu and GPU_AVAILABLE:
        return _resize_gpu(image, output_shape, order, anti_aliasing)
    else:
        return _resize_cpu(image, output_shape, order, anti_aliasing)


def _resize_gpu(image, output_shape, order, anti_aliasing):
    """GPU implementation of resize using zoom."""
    import cupy as cp
    from cupyx.scipy.ndimage import zoom as cp_zoom
    from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian

    img_gpu = cp.asarray(image if image.dtype == np.float32 else image.astype(np.float32))

    # Scale factors: input/output for Gaussian sigma, output/input for zoom.
    scale_factors = tuple(i / o for i, o in zip(image.shape, output_shape))
    zoom_factors = tuple(o / i for i, o in zip(image.shape, output_shape))

    # Anti-aliasing: single fused Gaussian call with per-axis sigma vector,
    # replacing N sequential per-axis kernel launches.
    if anti_aliasing:
        sigmas = [(f - 1) / 2 if f > 1 else 0.0 for f in scale_factors]
        if any(s > 0 for s in sigmas):
            img_gpu = cp_gaussian(img_gpu, sigma=sigmas)

    result = cp_zoom(img_gpu, zoom_factors, order=order)

    return to_cpu(result)


def _resize_cpu(image, output_shape, order, anti_aliasing):
    """CPU fallback for resize using zoom."""
    from scipy.ndimage import zoom as scipy_zoom
    from scipy.ndimage import gaussian_filter as scipy_gaussian

    img = image if image.dtype == np.float32 else image.astype(np.float32)

    scale_factors = tuple(i / o for i, o in zip(image.shape, output_shape))
    zoom_factors = tuple(o / i for i, o in zip(image.shape, output_shape))

    if anti_aliasing:
        sigmas = [(f - 1) / 2 if f > 1 else 0.0 for f in scale_factors]
        if any(s > 0 for s in sigmas):
            img = scipy_gaussian(img, sigma=sigmas)

    return scipy_zoom(img, zoom_factors, order=order)


def apply_displacement_field(image, displacement_field, use_gpu=True):
    """
    Apply a displacement field to warp an image.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (2D or 3D)
    displacement_field : np.ndarray
        Displacement field with shape (ndim, *image.shape)
    use_gpu : bool
        Whether to use GPU
        
    Returns
    -------
    np.ndarray
        Warped image
    """
    ndim = image.ndim

    # Create coordinate grid
    coords = np.meshgrid(*[np.arange(s) for s in image.shape], indexing='ij')
    coords = np.array(coords)

    # Add displacement
    new_coords = coords + displacement_field

    return map_coordinates(image, new_coords, order=1, use_gpu=use_gpu)


def resample_volume(volume, current_spacing, target_spacing, order=1, use_gpu=True):
    """
    Resample a volume to a new spacing.
    
    Parameters
    ----------
    volume : np.ndarray
        Input volume
    current_spacing : tuple
        Current voxel spacing
    target_spacing : tuple
        Target voxel spacing
    order : int
        Interpolation order
    use_gpu : bool
        Whether to use GPU
        
    Returns
    -------
    np.ndarray
        Resampled volume
    """
    # Compute new shape
    scale_factors = tuple(c / t for c, t in zip(current_spacing, target_spacing))
    new_shape = tuple(int(s * f) for s, f in zip(volume.shape, scale_factors))

    return resize(volume, new_shape, order=order, anti_aliasing=True, use_gpu=use_gpu)
