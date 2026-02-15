"""GPU-accelerated array operations for linumpy.

Provides GPU versions of normalization, clipping, and thresholding.
Note: Simple reductions (mean, max) should use numpy directly - GPU offers no benefit.
"""

import numpy as np

from . import GPU_AVAILABLE, to_cpu


def normalize_percentile(image, p_low=1, p_high=99, use_gpu=True):
    """
    GPU-accelerated percentile-based normalization.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    p_low : float
        Lower percentile for normalization (0-100)
    p_high : float
        Upper percentile for normalization (0-100)
    use_gpu : bool
        Whether to use GPU
        
    Returns
    -------
    np.ndarray
        Normalized image in [0, 1] range
    """
    if use_gpu and GPU_AVAILABLE:
        import cupy as cp
        img_gpu = cp.asarray(image.astype(np.float32))

        low, high = cp.percentile(img_gpu, [p_low, p_high])

        if high - low < 1e-10:
            return to_cpu(cp.zeros_like(img_gpu))

        normalized = (img_gpu - low) / (high - low)
        normalized = cp.clip(normalized, 0, 1)

        return to_cpu(normalized)
    else:
        low, high = np.percentile(image, [p_low, p_high])
        if high - low < 1e-10:
            return np.zeros_like(image, dtype=np.float32)
        normalized = (image - low) / (high - low)
        return np.clip(normalized, 0, 1).astype(np.float32)


def normalize_minmax(image, use_gpu=True):
    """
    GPU-accelerated min-max normalization.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    use_gpu : bool
        Whether to use GPU
        
    Returns
    -------
    np.ndarray
        Normalized image in [0, 1] range
    """
    if use_gpu and GPU_AVAILABLE:
        import cupy as cp
        img_gpu = cp.asarray(image.astype(np.float32))

        vmin, vmax = cp.min(img_gpu), cp.max(img_gpu)

        if vmax - vmin < 1e-10:
            return to_cpu(cp.zeros_like(img_gpu))

        normalized = (img_gpu - vmin) / (vmax - vmin)

        return to_cpu(normalized)
    else:
        vmin, vmax = np.min(image), np.max(image)
        if vmax - vmin < 1e-10:
            return np.zeros_like(image, dtype=np.float32)
        return ((image - vmin) / (vmax - vmin)).astype(np.float32)


def clip_percentile(image, p_low=0.5, p_high=99.5, use_gpu=True):
    """
    GPU-accelerated percentile clipping.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    p_low : float
        Lower percentile to clip
    p_high : float
        Upper percentile to clip
    use_gpu : bool
        Whether to use GPU
        
    Returns
    -------
    np.ndarray
        Clipped image
    """
    if use_gpu and GPU_AVAILABLE:
        import cupy as cp
        img_gpu = cp.asarray(image)

        low, high = cp.percentile(img_gpu, [p_low, p_high])
        clipped = cp.clip(img_gpu, low, high)

        return to_cpu(clipped)
    else:
        low, high = np.percentile(image, [p_low, p_high])
        return np.clip(image, low, high)


def compute_percentiles_memory_efficient(image: np.ndarray, percentiles: list,
                                         use_gpu: bool = True,
                                         max_samples: int = 10_000_000) -> list:
    """
    Compute percentiles using subsampling to reduce memory usage.

    For large arrays, computing exact percentiles requires sorting the entire array,
    which can cause memory issues. This function uses random subsampling to estimate
    percentiles with minimal memory overhead.

    Parameters
    ----------
    image : np.ndarray
        Input image
    percentiles : list
        List of percentiles to compute (0-100)
    use_gpu : bool
        Whether to use GPU
    max_samples : int
        Maximum number of samples to use for percentile estimation

    Returns
    -------
    list
        Computed percentile values
    """
    flat = image.ravel()

    # Subsample if the array is too large
    if flat.size > max_samples:
        # Use random sampling for memory efficiency
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        indices = rng.choice(flat.size, size=max_samples, replace=False)
        sample = flat[indices]
    else:
        sample = flat

    if use_gpu and GPU_AVAILABLE:
        import cupy as cp
        try:
            sample_gpu = cp.asarray(sample)
            result = [float(cp.percentile(sample_gpu, p).get()) for p in percentiles]
            del sample_gpu
            cp.get_default_memory_pool().free_all_blocks()
            return result
        except cp.cuda.memory.OutOfMemoryError:
            # Fall back to CPU if GPU runs out of memory
            pass

    return [float(np.percentile(sample, p)) for p in percentiles]


def compute_nonzero_percentile_memory_efficient(image: np.ndarray, percentile: float,
                                                use_gpu: bool = True,
                                                max_samples: int = 10_000_000) -> float:
    """
    Compute percentile of non-zero values using subsampling.

    Parameters
    ----------
    image : np.ndarray
        Input image
    percentile : float
        Percentile to compute (0-100)
    use_gpu : bool
        Whether to use GPU
    max_samples : int
        Maximum number of samples to use

    Returns
    -------
    float
        Computed percentile value
    """
    flat = image.ravel()
    nonzero_mask = flat > 0
    nonzero_vals = flat[nonzero_mask]

    if nonzero_vals.size == 0:
        return 0.0

    # Subsample if too large
    if nonzero_vals.size > max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(nonzero_vals.size, size=max_samples, replace=False)
        sample = nonzero_vals[indices]
    else:
        sample = nonzero_vals

    if use_gpu and GPU_AVAILABLE:
        import cupy as cp
        try:
            sample_gpu = cp.asarray(sample)
            result = float(cp.percentile(sample_gpu, percentile).get())
            del sample_gpu
            cp.get_default_memory_pool().free_all_blocks()
            return result
        except cp.cuda.memory.OutOfMemoryError:
            pass

    return float(np.percentile(sample, percentile))


def apply_flatfield_correction(image, flatfield, darkfield=None, use_gpu=True):
    """
    GPU-accelerated flatfield correction.
    
    Corrected = (Image - Darkfield) / (Flatfield - Darkfield)
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    flatfield : np.ndarray
        Flatfield image
    darkfield : np.ndarray, optional
        Darkfield image
    use_gpu : bool
        Whether to use GPU
        
    Returns
    -------
    np.ndarray
        Corrected image
    """
    if use_gpu and GPU_AVAILABLE:
        import cupy as cp
        img_gpu = cp.asarray(image.astype(np.float32))
        flat_gpu = cp.asarray(flatfield.astype(np.float32))

        if darkfield is not None:
            dark_gpu = cp.asarray(darkfield.astype(np.float32))
            numerator = img_gpu - dark_gpu
            denominator = flat_gpu - dark_gpu
        else:
            numerator = img_gpu
            denominator = flat_gpu

        # Avoid division by zero
        denominator = cp.where(cp.abs(denominator) < 1e-10, 1.0, denominator)
        corrected = numerator / denominator

        return to_cpu(corrected)
    else:
        if darkfield is not None:
            numerator = image.astype(np.float32) - darkfield
            denominator = flatfield.astype(np.float32) - darkfield
        else:
            numerator = image.astype(np.float32)
            denominator = flatfield.astype(np.float32)

        denominator = np.where(np.abs(denominator) < 1e-10, 1.0, denominator)
        return numerator / denominator


def compute_std_projection(volume, axis=0, use_gpu=True):
    """
    GPU-accelerated standard deviation projection.
    
    Parameters
    ----------
    volume : np.ndarray
        Input volume
    axis : int
        Axis along which to compute std
    use_gpu : bool
        Whether to use GPU
        
    Returns
    -------
    np.ndarray
        Standard deviation projection
    """
    if use_gpu and GPU_AVAILABLE:
        import cupy as cp
        vol_gpu = cp.asarray(volume)
        result = cp.std(vol_gpu, axis=axis)
        return to_cpu(result)
    else:
        return np.std(volume, axis=axis)


def threshold_otsu(image, use_gpu=True):
    """
    GPU-accelerated Otsu thresholding.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    use_gpu : bool
        Whether to use GPU
        
    Returns
    -------
    float
        Otsu threshold value
    """
    if use_gpu and GPU_AVAILABLE:
        import cupy as cp
        img_gpu = cp.asarray(image.astype(np.float32))

        # Compute histogram
        hist, bin_edges = cp.histogram(img_gpu.ravel(), bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        hist = hist.astype(cp.float64)
        hist_norm = hist / hist.sum()

        # Cumulative sums
        weight1 = cp.cumsum(hist_norm)
        weight2 = cp.cumsum(hist_norm[::-1])[::-1]

        # Cumulative means
        mean1 = cp.cumsum(hist_norm * bin_centers) / weight1
        mean2 = (cp.cumsum((hist_norm * bin_centers)[::-1]) / weight2[::-1])[::-1]

        # Between-class variance
        variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

        # Find maximum
        idx = cp.argmax(variance)
        threshold = float(bin_centers[idx].get())

        # Free GPU memory
        del img_gpu, hist, bin_edges, bin_centers, hist_norm, weight1, weight2, mean1, mean2, variance
        cp.get_default_memory_pool().free_all_blocks()

        return threshold
    else:
        from skimage.filters import threshold_otsu as sk_otsu
        return sk_otsu(image)


def apply_xy_shift(image, reference, dy, dx, use_gpu=True):
    """
    GPU-accelerated XY shift application.
    
    Parameters
    ----------
    image : np.ndarray
        Image to shift
    reference : np.ndarray
        Reference image (determines output shape)
    dy : float
        Y shift in pixels
    dx : float
        X shift in pixels
    use_gpu : bool
        Whether to use GPU
        
    Returns
    -------
    np.ndarray
        Shifted image
    """
    # Get a representative non-zero value for out-of-bounds fill
    nonzero_vals = image[image > 0]
    if len(nonzero_vals) > 0:
        cval = float(np.percentile(nonzero_vals, 1))
    else:
        cval = 0.0

    if use_gpu and GPU_AVAILABLE:
        import cupy as cp
        from cupyx.scipy.ndimage import shift as cp_shift

        img_gpu = cp.asarray(image.astype(np.float32))

        # Apply shift with edge value fill to avoid black dots
        if image.ndim == 2:
            shifted = cp_shift(img_gpu, [dy, dx], order=1, cval=cval)
        else:  # 3D
            shifted = cp_shift(img_gpu, [0, dy, dx], order=1, cval=cval)

        return to_cpu(shifted)
    else:
        from scipy.ndimage import shift as scipy_shift
        if image.ndim == 2:
            return scipy_shift(image, [dy, dx], order=1, cval=cval)
        else:
            return scipy_shift(image, [0, dy, dx], order=1, cval=cval)
