"""
GPU-accelerated morphological operations for linumpy.

Provides GPU versions of binary morphology, mask creation,
and connected component operations.
"""

import numpy as np

from . import GPU_AVAILABLE, to_cpu


def binary_closing(mask, iterations=1, structure=None, use_gpu=True):
    """
    GPU-accelerated binary closing.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask
    iterations : int
        Number of iterations
    structure : np.ndarray, optional
        Structuring element
    use_gpu : bool
        Whether to use GPU
        
    Returns
    -------
    np.ndarray
        Closed mask
    """
    if use_gpu and GPU_AVAILABLE:
        import cupy as cp
        from cupyx.scipy.ndimage import binary_closing as cp_closing
        from cupyx.scipy.ndimage import generate_binary_structure

        mask_gpu = cp.asarray(mask.astype(np.bool_))

        if structure is None:
            structure = generate_binary_structure(mask.ndim, 1)
        else:
            structure = cp.asarray(structure)

        result = cp_closing(mask_gpu, structure=structure, iterations=iterations, brute_force=True)

        output = to_cpu(result)
        # Free GPU memory
        del mask_gpu, result
        cp.get_default_memory_pool().free_all_blocks()
        return output
    else:
        from scipy.ndimage import binary_closing as scipy_closing
        from scipy.ndimage import generate_binary_structure

        if structure is None:
            structure = generate_binary_structure(mask.ndim, 1)

        return scipy_closing(mask, structure=structure, iterations=iterations)


def binary_opening(mask, iterations=1, structure=None, use_gpu=True):
    """
    GPU-accelerated binary opening.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask
    iterations : int
        Number of iterations
    structure : np.ndarray, optional
        Structuring element
    use_gpu : bool
        Whether to use GPU
        
    Returns
    -------
    np.ndarray
        Opened mask
    """
    if use_gpu and GPU_AVAILABLE:
        import cupy as cp
        from cupyx.scipy.ndimage import binary_opening as cp_opening
        from cupyx.scipy.ndimage import generate_binary_structure

        mask_gpu = cp.asarray(mask.astype(np.bool_))

        if structure is None:
            structure = generate_binary_structure(mask.ndim, 1)
        else:
            structure = cp.asarray(structure)

        result = cp_opening(mask_gpu, structure=structure, iterations=iterations, brute_force=True)

        output = to_cpu(result)
        # Free GPU memory
        del mask_gpu, result
        cp.get_default_memory_pool().free_all_blocks()
        return output
    else:
        from scipy.ndimage import binary_opening as scipy_opening
        from scipy.ndimage import generate_binary_structure

        if structure is None:
            structure = generate_binary_structure(mask.ndim, 1)

        return scipy_opening(mask, structure=structure, iterations=iterations)


def binary_dilation(mask, iterations=1, structure=None, use_gpu=True):
    """
    GPU-accelerated binary dilation.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask
    iterations : int
        Number of iterations
    structure : np.ndarray, optional
        Structuring element
    use_gpu : bool
        Whether to use GPU
        
    Returns
    -------
    np.ndarray
        Dilated mask
    """
    if use_gpu and GPU_AVAILABLE:
        import cupy as cp
        from cupyx.scipy.ndimage import binary_dilation as cp_dilation
        from cupyx.scipy.ndimage import generate_binary_structure

        mask_gpu = cp.asarray(mask.astype(np.bool_))

        if structure is None:
            structure = generate_binary_structure(mask.ndim, 1)
        else:
            structure = cp.asarray(structure)

        result = cp_dilation(mask_gpu, structure=structure, iterations=iterations, brute_force=True)

        output = to_cpu(result)
        # Free GPU memory
        del mask_gpu, result
        cp.get_default_memory_pool().free_all_blocks()
        return output
    else:
        from scipy.ndimage import binary_dilation as scipy_dilation
        from scipy.ndimage import generate_binary_structure

        if structure is None:
            structure = generate_binary_structure(mask.ndim, 1)

        return scipy_dilation(mask, structure=structure, iterations=iterations)


def binary_erosion(mask, iterations=1, structure=None, use_gpu=True):
    """
    GPU-accelerated binary erosion.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask
    iterations : int
        Number of iterations
    structure : np.ndarray, optional
        Structuring element
    use_gpu : bool
        Whether to use GPU
        
    Returns
    -------
    np.ndarray
        Eroded mask
    """
    if use_gpu and GPU_AVAILABLE:
        import cupy as cp
        from cupyx.scipy.ndimage import binary_erosion as cp_erosion
        from cupyx.scipy.ndimage import generate_binary_structure

        mask_gpu = cp.asarray(mask.astype(np.bool_))

        if structure is None:
            structure = generate_binary_structure(mask.ndim, 1)
        else:
            structure = cp.asarray(structure)

        result = cp_erosion(mask_gpu, structure=structure, iterations=iterations, brute_force=True)

        output = to_cpu(result)
        # Free GPU memory
        del mask_gpu, result
        cp.get_default_memory_pool().free_all_blocks()
        return output
    else:
        from scipy.ndimage import binary_erosion as scipy_erosion
        from scipy.ndimage import generate_binary_structure

        if structure is None:
            structure = generate_binary_structure(mask.ndim, 1)

        return scipy_erosion(mask, structure=structure, iterations=iterations)


def binary_fill_holes(mask, use_gpu=True):
    """
    GPU-accelerated binary hole filling.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask
    use_gpu : bool
        Whether to use GPU
        
    Returns
    -------
    np.ndarray
        Mask with holes filled
    """
    if use_gpu and GPU_AVAILABLE:
        import cupy as cp
        from cupyx.scipy.ndimage import binary_fill_holes as cp_fill

        mask_gpu = cp.asarray(mask.astype(np.bool_))
        result = cp_fill(mask_gpu)

        output = to_cpu(result)
        # Free GPU memory
        del mask_gpu, result
        cp.get_default_memory_pool().free_all_blocks()
        return output
    else:
        from scipy.ndimage import binary_fill_holes as scipy_fill
        return scipy_fill(mask)


def gaussian_filter(image, sigma, use_gpu=True):
    """
    GPU-accelerated Gaussian filter.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    sigma : float or sequence
        Standard deviation for Gaussian kernel
    use_gpu : bool
        Whether to use GPU
        
    Returns
    -------
    np.ndarray
        Filtered image
    """
    if use_gpu and GPU_AVAILABLE:
        import cupy as cp
        from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian

        img_gpu = cp.asarray(image.astype(np.float32))
        result = cp_gaussian(img_gpu, sigma=sigma)

        output = to_cpu(result)
        # Free GPU memory
        del img_gpu, result
        cp.get_default_memory_pool().free_all_blocks()
        return output
    else:
        from scipy.ndimage import gaussian_filter as scipy_gaussian
        return scipy_gaussian(image, sigma=sigma)


def median_filter(image, size, use_gpu=True):
    """
    GPU-accelerated median filter.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    size : int or sequence
        Filter size
    use_gpu : bool
        Whether to use GPU
        
    Returns
    -------
    np.ndarray
        Filtered image
    """
    if use_gpu and GPU_AVAILABLE:
        import cupy as cp
        from cupyx.scipy.ndimage import median_filter as cp_median

        img_gpu = cp.asarray(image)
        result = cp_median(img_gpu, size=size)

        output = to_cpu(result)
        # Free GPU memory
        del img_gpu, result
        cp.get_default_memory_pool().free_all_blocks()
        return output
    else:
        from scipy.ndimage import median_filter as scipy_median
        return scipy_median(image, size=size)


def create_tissue_mask(image, sigma=2, threshold=None, fill_holes=True,
                       min_opening=1, use_gpu=True):
    """
    GPU-accelerated tissue mask creation.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    sigma : float
        Gaussian smoothing sigma
    threshold : float, optional
        Threshold value. If None, uses Otsu
    fill_holes : bool
        Whether to fill holes
    min_opening : int
        Opening iterations for noise removal
    use_gpu : bool
        Whether to use GPU
        
    Returns
    -------
    np.ndarray
        Binary tissue mask
    """
    from .array_ops import threshold_otsu

    # Smooth
    smoothed = gaussian_filter(image, sigma, use_gpu=use_gpu)

    # Threshold
    if threshold is None:
        threshold = threshold_otsu(smoothed, use_gpu=use_gpu)

    if use_gpu and GPU_AVAILABLE:
        import cupy as cp
        smoothed_gpu = cp.asarray(smoothed)
        mask = smoothed_gpu > threshold
        mask = to_cpu(mask)
    else:
        mask = smoothed > threshold

    # Clean up
    if min_opening > 0:
        mask = binary_opening(mask, iterations=min_opening, use_gpu=use_gpu)

    if fill_holes:
        mask = binary_fill_holes(mask, use_gpu=use_gpu)

    return mask


def label_connected_components(mask, use_gpu=True):
    """
    Label connected components in a binary mask.
    
    Note: CuPy's connected components is limited. Falls back to CPU
    for complex cases.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask
    use_gpu : bool
        Whether to attempt GPU (may fall back to CPU)
        
    Returns
    -------
    np.ndarray
        Labeled array
    int
        Number of labels
    """
    # CuPy's label function is limited, use CPU for reliability
    from scipy.ndimage import label as scipy_label
    return scipy_label(mask)


def get_largest_component(mask, use_gpu=True):
    """
    Get the largest connected component from a mask.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask
    use_gpu : bool
        Whether to use GPU for histogram
        
    Returns
    -------
    np.ndarray
        Binary mask of largest component
    """
    labeled, n_labels = label_connected_components(mask, use_gpu=False)

    if n_labels == 0:
        return mask

    if use_gpu and GPU_AVAILABLE:
        import cupy as cp
        labeled_gpu = cp.asarray(labeled)

        # Find largest component (excluding background 0)
        counts = cp.bincount(labeled_gpu.ravel())
        counts[0] = 0  # Ignore background
        largest_label = int(cp.argmax(counts).get())

        result = labeled_gpu == largest_label
        return to_cpu(result)
    else:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        largest_label = np.argmax(counts)
        return labeled == largest_label
