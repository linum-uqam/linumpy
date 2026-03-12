#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image quality assessment functions for slice analysis.

This module provides CPU-based functions for assessing image quality in 3D volumes,
including:
- Structural Similarity Index (SSIM)
- Edge preservation scoring
- Variance consistency analysis
- Overall slice quality assessment

For GPU-accelerated versions, see `linumpy.gpu.image_quality`.

Usage:
    from linumpy.utils.image_quality import (
        compute_ssim_2d,
        compute_ssim_3d,
        compute_edge_score,
        compute_variance_score,
        assess_slice_quality,
    )

    # Compare two volumes
    ssim = compute_ssim_3d(vol1, vol2)

    # Assess overall slice quality
    quality, metrics = assess_slice_quality(vol, vol_before, vol_after)
"""

from typing import Optional, Tuple, Dict, List, Any

import numpy as np


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range.

    Parameters
    ----------
    img : np.ndarray
        Input image.

    Returns
    -------
    np.ndarray
        Normalized image as float32.
    """
    result = img.astype(np.float32)
    img_min, img_max = result.min(), result.max()
    if img_max > img_min:
        result = (result - img_min) / (img_max - img_min)
    return result


def compute_ssim_2d(img1: np.ndarray, img2: np.ndarray, win_size: int = 7) -> float:
    """
    Compute SSIM between two 2D images.

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
    if img1.shape != img2.shape:
        min_y = min(img1.shape[0], img2.shape[0])
        min_x = min(img1.shape[1], img2.shape[1])
        img1 = img1[:min_y, :min_x]
        img2 = img2[:min_y, :min_x]

    try:
        from skimage.metrics import structural_similarity as ssim

        # Normalize images
        i1 = normalize_image(img1)
        i2 = normalize_image(img2)

        # Adjust window size for image dimensions
        actual_win_size = min(win_size, min(i1.shape) - 1)
        if actual_win_size % 2 == 0:
            actual_win_size -= 1
        if actual_win_size < 3:
            actual_win_size = 3

        return float(ssim(i1, i2, win_size=actual_win_size, data_range=1.0))
    except Exception:
        # Fallback to normalized cross-correlation
        i1 = normalize_image(img1)
        i2 = normalize_image(img2)
        corr = np.corrcoef(i1.flatten(), i2.flatten())[0, 1]
        return float(max(0.0, corr)) if not np.isnan(corr) else 0.0


def compute_ssim_3d(vol1: np.ndarray, vol2: np.ndarray,
                    win_size: int = 7, sample_depth: int = 0) -> float:
    """
    Compute mean SSIM between two 3D volumes.

    Computes SSIM for each z-slice and returns the mean.

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
        score = compute_ssim_2d(vol1[z], vol2[z], win_size)
        ssim_scores.append(score)

    return float(np.mean(ssim_scores))


def compute_edge_score(vol: np.ndarray, reference: np.ndarray,
                       sample_z: Optional[int] = None) -> float:
    """
    Compute edge preservation score between volume and reference.

    Uses Sobel edge detection to compare edge structures.

    Parameters
    ----------
    vol : np.ndarray
        Input volume (Z, Y, X) or 2D image.
    reference : np.ndarray
        Reference volume or image.
    sample_z : int, optional
        Z-index to sample for 3D volumes. If None, uses middle slice.

    Returns
    -------
    float
        Edge preservation score (0 to 1, higher is better).
    """
    from scipy.ndimage import sobel

    # Handle 3D volumes
    if vol.ndim == 3:
        if sample_z is None:
            sample_z = vol.shape[0] // 2
        v = normalize_image(vol[sample_z])
        r = normalize_image(reference[sample_z] if reference.ndim == 3 else reference)
    else:
        v = normalize_image(vol)
        r = normalize_image(reference)

    if v.shape != r.shape:
        min_y = min(v.shape[0], r.shape[0])
        min_x = min(v.shape[1], r.shape[1])
        v = v[:min_y, :min_x]
        r = r[:min_y, :min_x]

    # Compute edges using Sobel
    edges_v = np.sqrt(sobel(v, axis=0) ** 2 + sobel(v, axis=1) ** 2)
    edges_r = np.sqrt(sobel(r, axis=0) ** 2 + sobel(r, axis=1) ** 2)

    # Normalize edges
    if edges_v.max() > 0:
        edges_v = edges_v / edges_v.max()
    if edges_r.max() > 0:
        edges_r = edges_r / edges_r.max()

    # Compute correlation
    correlation = np.corrcoef(edges_v.flatten(), edges_r.flatten())[0, 1]

    if np.isnan(correlation):
        return 0.0

    return float(max(0.0, correlation))


def compute_variance_score(vol: np.ndarray, reference: np.ndarray) -> float:
    """
    Compute variance consistency score between volume and reference.

    Low variance may indicate data loss or corruption.

    Parameters
    ----------
    vol : np.ndarray
        Input volume.
    reference : np.ndarray
        Reference volume.

    Returns
    -------
    float
        Variance score (0 to 1, higher means more similar variance).
    """
    var_vol = float(np.var(vol))
    var_ref = float(np.var(reference))

    if var_ref == 0:
        return 0.0

    ratio = var_vol / var_ref

    # Score is 1 when variances are equal, decreases as they diverge
    score = 2.0 / (1.0 + abs(np.log(ratio + 1e-10)))

    return float(min(1.0, max(0.0, score)))


def assess_slice_quality(vol: np.ndarray,
                         vol_before: Optional[np.ndarray],
                         vol_after: Optional[np.ndarray],
                         sample_depth: int = 5,
                         weights: Optional[Dict[str, float]] = None) -> Tuple[float, Dict[str, Any]]:
    """
    Assess overall quality of a slice volume.

    Uses multiple metrics to determine slice quality:
    - SSIM with neighboring slices (50%)
    - Edge preservation compared to expected structure (30%)
    - Variance consistency (20%)

    Parameters
    ----------
    vol : np.ndarray
        The slice volume (Z, Y, X).
    vol_before : np.ndarray or None
        The previous slice volume.
    vol_after : np.ndarray or None
        The next slice volume.
    sample_depth : int
        Number of z-planes to sample for SSIM. 0 = all.
    weights : dict, optional
        Custom weights for metrics. Keys: 'ssim', 'edge', 'variance'.

    Returns
    -------
    float
        Overall quality score (0 to 1).
    dict
        Individual metric values.
    """
    if weights is None:
        weights = {'ssim': 0.5, 'edge': 0.3, 'variance': 0.2}

    metrics: Dict[str, Any] = {
        'ssim_before': 0.0,
        'ssim_after': 0.0,
        'ssim_mean': 0.0,
        'edge_score': 0.0,
        'variance_score': 0.0,
        'depth': vol.shape[0] if vol.ndim == 3 else 1,
        'has_data': True,
    }

    # Check if slice has meaningful data
    if vol.max() == vol.min() or np.std(vol) < 1e-6:
        metrics['has_data'] = False
        metrics['overall'] = 0.0
        return 0.0, metrics

    # Compute SSIM with neighbors
    ssim_scores = []
    if vol_before is not None:
        metrics['ssim_before'] = compute_ssim_3d(vol, vol_before, sample_depth=sample_depth)
        ssim_scores.append(metrics['ssim_before'])
    if vol_after is not None:
        metrics['ssim_after'] = compute_ssim_3d(vol, vol_after, sample_depth=sample_depth)
        ssim_scores.append(metrics['ssim_after'])

    if ssim_scores:
        metrics['ssim_mean'] = float(np.mean(ssim_scores))

    # Create reference from neighbors
    if vol_before is not None and vol_after is not None:
        min_z = min(vol.shape[0], vol_before.shape[0], vol_after.shape[0])
        min_y = min(vol.shape[1], vol_before.shape[1], vol_after.shape[1])
        min_x = min(vol.shape[2], vol_before.shape[2], vol_after.shape[2])
        ref = (0.5 * vol_before[:min_z, :min_y, :min_x].astype(np.float32)
               + 0.5 * vol_after[:min_z, :min_y, :min_x].astype(np.float32))
    elif vol_before is not None:
        ref = vol_before.astype(np.float32)
    elif vol_after is not None:
        ref = vol_after.astype(np.float32)
    else:
        ref = None

    # Compute edge preservation score
    if ref is not None:
        metrics['edge_score'] = compute_edge_score(vol, ref)

    # Compute variance consistency
    if ref is not None:
        metrics['variance_score'] = compute_variance_score(vol, ref)

    # Compute overall score
    overall = (
            weights['ssim'] * metrics['ssim_mean'] +
            weights['edge'] * metrics['edge_score'] +
            weights['variance'] * metrics['variance_score']
    )
    metrics['overall'] = float(overall)

    return float(overall), metrics


def detect_calibration_slice(volumes: Dict[int, np.ndarray],
                             thickness_ratio: float = 1.5) -> List[int]:
    """
    Detect calibration slices by their different thickness.

    Calibration slices are typically thicker than regular slices.

    Parameters
    ----------
    volumes : dict
        Mapping from slice_id to volume array.
    thickness_ratio : float
        Slices with depth > median * ratio are flagged.

    Returns
    -------
    list
        List of slice IDs identified as calibration slices.
    """
    if not volumes:
        return []

    slice_ids = sorted(volumes.keys())
    depths = {sid: vol.shape[0] for sid, vol in volumes.items()}

    valid_depths = [d for d in depths.values() if d > 0]
    if not valid_depths:
        return []

    median_depth = float(np.median(valid_depths))

    # Check first few slices for unusual thickness
    calibration = []
    for sid in slice_ids[:3]:
        if sid in depths and depths[sid] > 0:
            ratio = depths[sid] / median_depth
            if ratio > thickness_ratio:
                calibration.append(sid)

    return calibration


def compute_quality_report(slice_qualities: Dict[int, Dict[str, Any]],
                           min_quality: float = 0.0) -> Dict[str, Any]:
    """
    Generate a quality report from slice quality assessments.

    Parameters
    ----------
    slice_qualities : dict
        Mapping from slice_id to quality metrics dict.
    min_quality : float
        Minimum quality threshold for flagging.

    Returns
    -------
    dict
        Summary report with statistics and flagged slices.
    """
    if not slice_qualities:
        return {'error': 'No slices to analyze'}

    overall_scores = [q.get('overall', 0.0) for q in slice_qualities.values()]

    report = {
        'n_slices': len(slice_qualities),
        'mean_quality': float(np.mean(overall_scores)),
        'std_quality': float(np.std(overall_scores)),
        'min_quality': float(np.min(overall_scores)),
        'max_quality': float(np.max(overall_scores)),
        'low_quality_slices': [],
        'no_data_slices': [],
    }

    for sid, metrics in slice_qualities.items():
        if not metrics.get('has_data', True):
            report['no_data_slices'].append(sid)
        elif metrics.get('overall', 0.0) < min_quality:
            report['low_quality_slices'].append(sid)

    return report
