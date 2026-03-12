#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU-accelerated image quality assessment functions.

This module provides CuPy-accelerated versions of quality assessment functions.
All functions automatically fall back to CPU if GPU is not available.

Usage:
    from linumpy.gpu.image_quality import (
        compute_ssim_2d_gpu,
        compute_ssim_3d_gpu,
        compute_edge_score_gpu,
        assess_slice_quality_gpu,
    )

    # All functions accept numpy arrays and return numpy scalars
    ssim = compute_ssim_3d_gpu(vol1, vol2)
"""

from typing import Optional, Tuple, Dict, Any

import numpy as np

from linumpy.gpu import GPU_AVAILABLE, CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp
    from cupyx.scipy.ndimage import sobel as cupy_sobel
    from cupyx.scipy.ndimage import uniform_filter as cupy_uniform_filter
else:
    cp = None
    cupy_sobel = None
    cupy_uniform_filter = None


def _to_gpu(arr: np.ndarray) -> "cp.ndarray":
    """Transfer numpy array to GPU."""
    return cp.asarray(arr, dtype=cp.float32)


def _to_cpu(arr) -> np.ndarray:
    """Transfer GPU array to CPU."""
    if hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)


def normalize_image_gpu(img: "cp.ndarray") -> "cp.ndarray":
    """
    Normalize image to [0, 1] range on GPU.

    Parameters
    ----------
    img : cp.ndarray
        Input image on GPU.

    Returns
    -------
    cp.ndarray
        Normalized image.
    """
    img_min = cp.min(img)
    img_max = cp.max(img)
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    return img


def compute_ssim_2d_gpu(img1: np.ndarray, img2: np.ndarray,
                        win_size: int = 7) -> float:
    """
    Compute SSIM between two 2D images using GPU.

    Falls back to CPU if GPU is not available.

    Parameters
    ----------
    img1, img2 : np.ndarray
        Input images (2D).
    win_size : int
        Window size for SSIM computation.

    Returns
    -------
    float
        SSIM score (0 to 1, higher is better).
    """
    if not GPU_AVAILABLE or cp is None:
        from linumpy.utils.image_quality import compute_ssim_2d
        return compute_ssim_2d(img1, img2, win_size)

    if img1.shape != img2.shape:
        min_y = min(img1.shape[0], img2.shape[0])
        min_x = min(img1.shape[1], img2.shape[1])
        img1 = img1[:min_y, :min_x]
        img2 = img2[:min_y, :min_x]

    try:
        # Transfer to GPU
        i1 = _to_gpu(img1)
        i2 = _to_gpu(img2)

        # Normalize
        i1 = normalize_image_gpu(i1)
        i2 = normalize_image_gpu(i2)

        # SSIM constants
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Compute local means using uniform filter
        mu1 = cupy_uniform_filter(i1, size=win_size)
        mu2 = cupy_uniform_filter(i2, size=win_size)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cupy_uniform_filter(i1 * i1, size=win_size) - mu1_sq
        sigma2_sq = cupy_uniform_filter(i2 * i2, size=win_size) - mu2_sq
        sigma12 = cupy_uniform_filter(i1 * i2, size=win_size) - mu1_mu2

        # SSIM formula
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim_map = numerator / denominator

        return float(cp.mean(ssim_map))
    except Exception:
        # Fall back to CPU
        from linumpy.utils.image_quality import compute_ssim_2d
        return compute_ssim_2d(img1, img2, win_size)


def compute_ssim_3d_gpu(vol1: np.ndarray, vol2: np.ndarray,
                        win_size: int = 7, sample_depth: int = 0) -> float:
    """
    Compute mean SSIM between two 3D volumes using GPU.

    Parameters
    ----------
    vol1, vol2 : np.ndarray
        Input volumes (Z, Y, X).
    win_size : int
        Window size for SSIM computation.
    sample_depth : int
        Number of z-planes to sample. 0 = all planes.

    Returns
    -------
    float
        Mean SSIM score (0 to 1, higher is better).
    """
    if not GPU_AVAILABLE:
        from linumpy.utils.image_quality import compute_ssim_3d
        return compute_ssim_3d(vol1, vol2, win_size, sample_depth)

    if vol1.shape != vol2.shape:
        min_z = min(vol1.shape[0], vol2.shape[0])
        min_y = min(vol1.shape[1], vol2.shape[1])
        min_x = min(vol1.shape[2], vol2.shape[2])
        vol1 = vol1[:min_z, :min_y, :min_x]
        vol2 = vol2[:min_z, :min_y, :min_x]

    # Sample z-planes if requested
    if sample_depth > 0 and vol1.shape[0] > sample_depth:
        indices = np.linspace(0, vol1.shape[0] - 1, sample_depth, dtype=int)
    else:
        indices = np.arange(vol1.shape[0])

    ssim_scores = []
    for z in indices:
        score = compute_ssim_2d_gpu(vol1[z], vol2[z], win_size)
        ssim_scores.append(score)

    return float(np.mean(ssim_scores))


def compute_edge_score_gpu(vol: np.ndarray, reference: np.ndarray,
                           sample_z: Optional[int] = None) -> float:
    """
    Compute edge preservation score using GPU.

    Parameters
    ----------
    vol : np.ndarray
        Input volume (Z, Y, X) or 2D image.
    reference : np.ndarray
        Reference volume or image.
    sample_z : int, optional
        Z-index to sample for 3D volumes.

    Returns
    -------
    float
        Edge preservation score (0 to 1, higher is better).
    """
    if not GPU_AVAILABLE or cp is None:
        from linumpy.utils.image_quality import compute_edge_score
        return compute_edge_score(vol, reference, sample_z)

    try:
        # Handle 3D volumes
        if vol.ndim == 3:
            if sample_z is None:
                sample_z = vol.shape[0] // 2
            v_cpu = vol[sample_z]
            r_cpu = reference[sample_z] if reference.ndim == 3 else reference
        else:
            v_cpu = vol
            r_cpu = reference

        if v_cpu.shape != r_cpu.shape:
            min_y = min(v_cpu.shape[0], r_cpu.shape[0])
            min_x = min(v_cpu.shape[1], r_cpu.shape[1])
            v_cpu = v_cpu[:min_y, :min_x]
            r_cpu = r_cpu[:min_y, :min_x]

        # Transfer to GPU and normalize
        v = normalize_image_gpu(_to_gpu(v_cpu))
        r = normalize_image_gpu(_to_gpu(r_cpu))

        # Compute edges using Sobel
        edges_v = cp.sqrt(cupy_sobel(v, axis=0) ** 2 + cupy_sobel(v, axis=1) ** 2)
        edges_r = cp.sqrt(cupy_sobel(r, axis=0) ** 2 + cupy_sobel(r, axis=1) ** 2)

        # Normalize edges
        if cp.max(edges_v) > 0:
            edges_v = edges_v / cp.max(edges_v)
        if cp.max(edges_r) > 0:
            edges_r = edges_r / cp.max(edges_r)

        # Compute correlation on GPU
        flat_v = edges_v.flatten()
        flat_r = edges_r.flatten()

        mean_v = cp.mean(flat_v)
        mean_r = cp.mean(flat_r)

        num = cp.sum((flat_v - mean_v) * (flat_r - mean_r))
        den = cp.sqrt(cp.sum((flat_v - mean_v) ** 2) * cp.sum((flat_r - mean_r) ** 2))

        if den > 0:
            corr = float(num / den)
            return max(0.0, corr) if not np.isnan(corr) else 0.0
        return 0.0
    except Exception:
        from linumpy.utils.image_quality import compute_edge_score
        return compute_edge_score(vol, reference, sample_z)


def compute_variance_score_gpu(vol: np.ndarray, reference: np.ndarray) -> float:
    """
    Compute variance score using GPU.

    Parameters
    ----------
    vol : np.ndarray
        Input volume.
    reference : np.ndarray
        Reference volume.

    Returns
    -------
    float
        Variance score (0 to 1).
    """
    if not GPU_AVAILABLE or cp is None:
        from linumpy.utils.image_quality import compute_variance_score
        return compute_variance_score(vol, reference)

    try:
        v = _to_gpu(vol)
        r = _to_gpu(reference)

        var_v = float(cp.var(v))
        var_r = float(cp.var(r))

        if var_r == 0:
            return 0.0

        ratio = var_v / var_r
        score = 2.0 / (1.0 + abs(np.log(ratio + 1e-10)))

        return float(min(1.0, max(0.0, score)))
    except Exception:
        from linumpy.utils.image_quality import compute_variance_score
        return compute_variance_score(vol, reference)


def assess_slice_quality_gpu(vol: np.ndarray,
                             vol_before: Optional[np.ndarray],
                             vol_after: Optional[np.ndarray],
                             sample_depth: int = 5,
                             weights: Optional[Dict[str, float]] = None) -> Tuple[float, Dict[str, Any]]:
    """
    Assess overall quality of a slice volume using GPU acceleration.

    Parameters
    ----------
    vol : np.ndarray
        The slice volume (Z, Y, X).
    vol_before : np.ndarray or None
        The previous slice volume.
    vol_after : np.ndarray or None
        The next slice volume.
    sample_depth : int
        Number of z-planes to sample for SSIM.
    weights : dict, optional
        Custom weights for metrics.

    Returns
    -------
    float
        Overall quality score (0 to 1).
    dict
        Individual metric values.
    """
    if not GPU_AVAILABLE:
        from linumpy.utils.image_quality import assess_slice_quality
        return assess_slice_quality(vol, vol_before, vol_after, sample_depth, weights)

    if weights is None:
        weights = {'ssim': 0.5, 'edge': 0.3, 'variance': 0.2}

    depth = vol.shape[0] if vol.ndim == 3 else 1
    metrics: Dict[str, Any] = {
        'ssim_before': 0.0,
        'ssim_after': 0.0,
        'ssim_mean': 0.0,
        'edge_score': 0.0,
        'variance_score': 0.0,
        'depth': depth,
        'has_data': True,
    }

    # Check if slice has meaningful data by sampling a single centre z-plane.
    # zarr.Array supports integer indexing (returns numpy), so no full-volume I/O.
    z_check = depth // 2 if vol.ndim == 3 else 0
    check_plane = np.asarray(vol[z_check])
    if check_plane.max() == check_plane.min() or np.std(check_plane) < 1e-6:
        metrics['has_data'] = False
        metrics['overall'] = 0.0
        return 0.0, metrics

    # Compute SSIM with neighbours.
    # compute_ssim_3d_gpu internally accesses vol[z] one plane at a time, so
    # zarr arrays are handled without loading the whole volume.
    ssim_scores = []
    if vol_before is not None:
        metrics['ssim_before'] = compute_ssim_3d_gpu(vol, vol_before, sample_depth=sample_depth)
        ssim_scores.append(metrics['ssim_before'])
    if vol_after is not None:
        metrics['ssim_after'] = compute_ssim_3d_gpu(vol, vol_after, sample_depth=sample_depth)
        ssim_scores.append(metrics['ssim_after'])

    if ssim_scores:
        metrics['ssim_mean'] = float(np.mean(ssim_scores))

    # Build sampled numpy arrays for edge and variance scores.
    # Read only sample_depth z-planes via zarr integer indexing to avoid loading
    # the full volume (compute_variance_score_gpu would otherwise call
    # cp.asarray on the whole array).
    n_planes = max(1, min(sample_depth, depth) if sample_depth > 0 else depth)
    z_indices = np.linspace(0, depth - 1, n_planes, dtype=int)
    vol_s = np.stack([np.asarray(vol[int(z)], dtype=np.float32) for z in z_indices])

    ref_s = None
    if vol_before is not None and vol_after is not None:
        min_y = min(vol_before.shape[1], vol_after.shape[1])
        min_x = min(vol_before.shape[2], vol_after.shape[2])
        ref_s = (
            0.5 * np.stack([np.asarray(vol_before[int(z)], dtype=np.float32)[:min_y, :min_x] for z in z_indices])
            + 0.5 * np.stack([np.asarray(vol_after[int(z)], dtype=np.float32)[:min_y, :min_x] for z in z_indices])
        )
    elif vol_before is not None:
        ref_s = np.stack([np.asarray(vol_before[int(z)], dtype=np.float32) for z in z_indices])
    elif vol_after is not None:
        ref_s = np.stack([np.asarray(vol_after[int(z)], dtype=np.float32) for z in z_indices])

    # Compute edge preservation score
    if ref_s is not None:
        metrics['edge_score'] = compute_edge_score_gpu(vol_s, ref_s)

    # Compute variance consistency
    if ref_s is not None:
        metrics['variance_score'] = compute_variance_score_gpu(vol_s, ref_s)

    # Compute overall score
    overall = (
            weights['ssim'] * metrics['ssim_mean'] +
            weights['edge'] * metrics['edge_score'] +
            weights['variance'] * metrics['variance_score']
    )
    metrics['overall'] = float(overall)

    return float(overall), metrics


def clear_gpu_memory():
    """Clear GPU memory pools."""
    if GPU_AVAILABLE and cp is not None:
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
