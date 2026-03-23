#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intensity normalization functions for OCT volumes.

This module provides functions for normalizing OCT volume intensities
based on agarose background detection.
"""

from typing import Tuple

import numpy as np


def normalize_volume(vol: np.ndarray,
                     agarose_mask: np.ndarray,
                     percentile_max: float = 99.9,
                     min_contrast_fraction: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize volume intensities based on agarose background.

    Intensities for each z-slice are rescaled between the minimum value
    inside agarose and the value defined by the percentile_max argument.

    Parameters
    ----------
    vol : np.ndarray
        Input volume with shape (Z, X, Y).
    agarose_mask : np.ndarray
        2D binary mask indicating agarose regions (shape X, Y).
    percentile_max : float
        Values above this percentile will be clipped. Default 99.9.
    min_contrast_fraction : float
        Minimum contrast (max-min) as a fraction of the global max.
        Slices with lower contrast will use this threshold to avoid
        over-amplification of noise in weak/bad slices. Default 0.1.

    Returns
    -------
    tuple
        (normalized_volume, background_thresholds)
        - normalized_volume: The normalized volume
        - background_thresholds: Array of background threshold per slice
    """
    # Clip to percentile max per slice
    pmax = np.percentile(vol, percentile_max, axis=(1, 2))
    vol = np.clip(vol, None, pmax[:, None, None])

    # Compute background threshold per slice from agarose regions
    background_thresholds = []
    for curr_slice in vol:
        agarose = curr_slice[agarose_mask]
        bg_median = np.median(agarose)
        background_thresholds.append(bg_median)

    background_thresholds = np.array(background_thresholds)
    vol = np.clip(vol, background_thresholds[:, None, None], None)

    # Rescale to [0, 1]
    vol = vol - np.min(vol, axis=(1, 2), keepdims=True)
    vmax = np.max(vol, axis=(1, 2))

    # Compute minimum acceptable contrast based on global statistics
    # This prevents over-amplification of slices with very weak signal
    global_max = np.max(vmax)
    min_contrast = global_max * min_contrast_fraction

    # For slices with sufficient contrast, normalize normally
    # For weak slices, use the minimum contrast threshold to avoid over-amplification
    effective_max = np.maximum(vmax, min_contrast)

    # Apply normalization
    for i in range(vol.shape[0]):
        if effective_max[i] > 0:
            vol[i] = vol[i] / effective_max[i]

    return vol, background_thresholds


def get_agarose_mask(vol: np.ndarray, smoothing_sigma: float = 1.0):
    """Compute agarose mask using Otsu thresholding on a mean projection.

    The agarose is the low-intensity background surrounding the tissue.
    Uses a Gaussian-smoothed mean projection through Z to get a robust
    2D estimate, then thresholds with Otsu.

    Parameters
    ----------
    vol : np.ndarray
        3D volume with shape (Z, Y, X).
    smoothing_sigma : float
        Gaussian smoothing sigma applied before Otsu thresholding.

    Returns
    -------
    agarose_mask : np.ndarray
        2D boolean mask (Y, X) — True where agarose is present.
    threshold : float
        The Otsu threshold used.
    """
    from scipy.ndimage import gaussian_filter
    from skimage.filters import threshold_otsu

    reference = np.mean(vol, axis=0)
    reference_smooth = gaussian_filter(reference, sigma=smoothing_sigma)
    threshold = threshold_otsu(reference_smooth[reference > 0])
    agarose_mask = np.logical_and(reference_smooth < threshold, reference > 0)
    return agarose_mask, threshold


def _robust_percentile(chunk: np.ndarray, percentile: float) -> float:
    """Return Nth percentile of non-zero voxels; 0 for nearly-empty chunks."""
    flat = chunk.ravel()
    nonzero = flat[flat > 0]
    if nonzero.size < 500:
        return 0.0
    return float(np.percentile(nonzero, percentile))


def _smooth_weighted(values: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth an array that may contain zeros (missing data).

    Uses weighted convolution so zeros do not bias the smoothed curve.
    """
    from scipy.ndimage import gaussian_filter1d

    weights = (values > 0).astype(np.float64)
    smoothed_v = gaussian_filter1d(values * weights, sigma=sigma, mode='reflect')
    smoothed_w = gaussian_filter1d(weights, sigma=sigma, mode='reflect')
    out = np.where(smoothed_w > 1e-6, smoothed_v / smoothed_w, 0.0)
    return out


def _chunk_boundaries(n_z: int, n_serial_slices):
    """Return list of (start, end) Z-index pairs, one per chunk."""
    if n_serial_slices is not None:
        chunk_size = n_z / n_serial_slices
        starts = [int(round(i * chunk_size)) for i in range(n_serial_slices)]
        ends = [int(round(i * chunk_size)) for i in range(1, n_serial_slices + 1)]
    else:
        starts = list(range(n_z))
        ends = list(range(1, n_z + 1))
    return list(zip(starts, ends))


def compute_scale_factors(vol: np.ndarray,
                          n_serial_slices,
                          smooth_sigma: float,
                          percentile: float,
                          min_scale: float,
                          max_scale: float):
    """Compute per-Z-plane linear scale factors for percentile-based normalization.

    Corrects slow acquisition drift (focus changes, laser power) between
    serial sections while preserving genuine anatomical intensity differences.

    Parameters
    ----------
    vol : np.ndarray
        Input volume (Z, Y, X) in [0, 1].
    n_serial_slices : int or None
        Number of serial sections. None = operate at individual Z-plane level.
    smooth_sigma : float
        Gaussian smoothing sigma in serial-section units.
    percentile : float
        Percentile of non-zero voxels used as intensity reference per chunk.
    min_scale, max_scale : float
        Clamping range for scale factors.

    Returns
    -------
    scale_factors : np.ndarray, shape (n_z,)
    raw_metrics : np.ndarray
    smoothed : np.ndarray
    boundaries : list of int
    """
    n_z = vol.shape[0]
    bounds = _chunk_boundaries(n_z, n_serial_slices)
    n_chunks = len(bounds)

    raw_metrics = np.array([
        _robust_percentile(vol[s:e], percentile)
        for s, e in bounds
    ])

    smoothed = _smooth_weighted(raw_metrics, sigma=smooth_sigma)

    valid = smoothed > 0
    global_ref = float(np.median(smoothed[valid])) if valid.any() else 1.0

    scale_per_chunk = np.ones(n_chunks)
    scale_per_chunk[valid] = global_ref / smoothed[valid]
    scale_per_chunk = np.clip(scale_per_chunk, min_scale, max_scale)

    scale_factors = np.ones(n_z, dtype=np.float32)
    for i, (s, e) in enumerate(bounds):
        scale_factors[s:e] = scale_per_chunk[i]

    boundaries = [s for s, _ in bounds]
    return scale_factors, raw_metrics, smoothed, boundaries


def _build_cdf(values: np.ndarray, n_bins: int):
    """Build a cumulative distribution function from an array of values.

    Parameters
    ----------
    values : np.ndarray, 1-D, in [0, 1]
    n_bins : int

    Returns
    -------
    bin_centers : np.ndarray
    cdf : np.ndarray, normalized to [0, 1]
    """
    hist, edges = np.histogram(values, bins=n_bins, range=(0.0, 1.0))
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    cdf = np.cumsum(hist).astype(np.float64)
    if cdf[-1] > 0:
        cdf /= cdf[-1]
    return bin_centers, cdf


def _match_chunk_to_reference(chunk: np.ndarray,
                               ref_bins: np.ndarray,
                               ref_cdf: np.ndarray,
                               n_bins: int,
                               tissue_threshold: float = 0.0) -> np.ndarray:
    """Map chunk intensities to match the reference CDF.

    Only voxels above tissue_threshold are mapped; background stays unchanged.
    """
    flat = chunk.ravel().astype(np.float32)
    tissue_mask = flat > tissue_threshold
    if tissue_mask.sum() < 500:
        return chunk

    tissue = flat[tissue_mask]
    src_bins, src_cdf = _build_cdf(tissue, n_bins)
    src_percentiles = np.interp(tissue, src_bins, src_cdf)
    matched = np.interp(src_percentiles, ref_cdf, ref_bins)

    result = flat.copy()
    result[tissue_mask] = matched
    return result.reshape(chunk.shape)


def apply_histogram_matching(vol: np.ndarray,
                             n_serial_slices,
                             n_bins: int,
                             tissue_threshold: float = 0.0) -> np.ndarray:
    """Apply per-section histogram matching to a global reference distribution.

    Corrects section-to-section intensity drift while preserving relative contrast
    within each section. Voxels at or below tissue_threshold are left unchanged.

    Parameters
    ----------
    vol : np.ndarray
        Input volume (Z, Y, X).
    n_serial_slices : int or None
        Number of serial sections. None = per Z-plane.
    n_bins : int
        Number of histogram bins.
    tissue_threshold : float
        Minimum intensity to classify as tissue (default 0.0).

    Returns
    -------
    np.ndarray
        Histogram-matched volume.
    """
    flat_all = vol.ravel()
    tissue_all = flat_all[flat_all > tissue_threshold]
    if tissue_all.size < 500:
        return vol

    ref_bins, ref_cdf = _build_cdf(tissue_all.astype(np.float64), n_bins)

    bounds = _chunk_boundaries(vol.shape[0], n_serial_slices)
    n_chunks = len(bounds)

    out = np.empty_like(vol)
    for i, (s, e) in enumerate(bounds):
        chunk = vol[s:e]
        out[s:e] = _match_chunk_to_reference(chunk, ref_bins, ref_cdf, n_bins, tissue_threshold)

    return out
