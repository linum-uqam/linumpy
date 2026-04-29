#!/usr/bin/env python3
"""
Intensity normalization functions for OCT volumes.

This module provides functions for normalizing OCT volume intensities
based on agarose background detection.
"""

import numpy as np


def normalize_volume(
    vol: np.ndarray,
    agarose_mask: np.ndarray,
    percentile_max: float = 99.9,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize volume intensities based on agarose background.

    Each z-slice is clipped at its per-slice percentile cap and agarose-median
    floor, then the agarose floor is subtracted per slice (so background goes
    to exactly 0).  The entire volume is then divided by a single global
    divisor (the maximum per-slice tissue span across all slices), so relative
    inter-section brightness is preserved.

    Parameters
    ----------
    vol : np.ndarray
        Input volume with shape (Z, Y, X).
    agarose_mask : np.ndarray
        2D binary mask indicating agarose regions (shape Y, X).
    percentile_max : float
        Values above this percentile will be clipped per slice. Default 99.9.

    Returns
    -------
    tuple
        (normalized_volume, background_thresholds)
        - normalized_volume: float32 volume in [0, 1] with agarose at 0.
        - background_thresholds: Array of agarose-median per slice.
    """
    vol = vol.astype(np.float32, copy=False)

    # Per-slice percentile cap
    pmax = np.percentile(vol, percentile_max, axis=(1, 2))
    vol = np.clip(vol, None, pmax[:, None, None])

    # Per-slice agarose-median floor
    background_thresholds = np.array([np.median(s[agarose_mask]) for s in vol])
    vol = np.clip(vol, background_thresholds[:, None, None], None)

    # Subtract per-slice agarose floor so background voxels become exactly 0
    vol = vol - background_thresholds[:, None, None]

    # Single global divisor: preserves relative inter-section brightness
    global_max = float((pmax - background_thresholds).max())
    if global_max > 0:
        vol = vol / global_max

    return vol, background_thresholds


def get_agarose_mask(vol: np.ndarray, smoothing_sigma: float = 1.0) -> tuple[np.ndarray, float]:
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
    smoothed_v = gaussian_filter1d(values * weights, sigma=sigma, mode="reflect")
    smoothed_w = gaussian_filter1d(weights, sigma=sigma, mode="reflect")
    out = np.where(smoothed_w > 1e-6, smoothed_v / smoothed_w, 0.0)
    return out


def _chunk_boundaries(n_z: int, n_serial_slices: int | None) -> list[tuple[int, int]]:
    """Return list of (start, end) Z-index pairs, one per chunk."""
    if n_serial_slices is not None:
        chunk_size = n_z / n_serial_slices
        starts = [round(i * chunk_size) for i in range(n_serial_slices)]
        ends = [round(i * chunk_size) for i in range(1, n_serial_slices + 1)]
    else:
        starts = list(range(n_z))
        ends = list(range(1, n_z + 1))
    return list(zip(starts, ends, strict=False))


def compute_scale_factors(
    vol: np.ndarray, n_serial_slices: int | None, smooth_sigma: float, percentile: float, min_scale: float, max_scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
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

    raw_metrics = np.array([_robust_percentile(vol[s:e], percentile) for s, e in bounds])

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


def _build_cdf(values: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a cumulative distribution function from an array of values.

    Parameters
    ----------
    values : np.ndarray
        1-D array in [0, 1].
    n_bins : int
        Number of histogram bins.

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


def _build_tissue_cdf(flat_values: np.ndarray, n_bins: int, tissue_threshold: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Build a CDF of tissue voxels (strictly above tissue_threshold).

    Unlike ``_build_cdf``, this avoids materialising a tissue-only copy of the
    input array by using ``np.histogram``'s ``range`` parameter with a small
    positive epsilon to exclude the background. For large volumes this saves
    an allocation on the order of the volume itself.

    Parameters
    ----------
    flat_values : np.ndarray
        1-D array in [0, 1] containing both tissue and background voxels.
    n_bins : int
        Number of histogram bins.
    tissue_threshold : float
        Voxels strictly greater than this are considered tissue.

    Returns
    -------
    bin_centers : np.ndarray
    cdf : np.ndarray, normalized to [0, 1]
    tissue_count : int
    """
    # Choose a lower edge that excludes background voxels (value == threshold).
    # For threshold == 0 this reliably drops exact zeros; for small positive
    # thresholds it drops <= threshold. Bin centers remain within [0, 1].
    lo = tissue_threshold + max(1e-6, tissue_threshold * 1e-6)
    lo = min(lo, 1.0)
    hist, edges = np.histogram(flat_values, bins=n_bins, range=(lo, 1.0))
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    total = int(hist.sum())
    cdf = np.cumsum(hist).astype(np.float64)
    if cdf[-1] > 0:
        cdf /= cdf[-1]
    return bin_centers, cdf, total


def _match_chunk_to_reference(
    chunk: np.ndarray, ref_bins: np.ndarray, ref_cdf: np.ndarray, n_bins: int, tissue_threshold: float = 0.0
) -> np.ndarray:
    """Map chunk intensities to match the reference CDF.

    Only voxels above tissue_threshold are mapped; background stays unchanged.

    Implementation note: uses a small (n_bins-sized) ``src_bin -> matched``
    lookup table so that the per-voxel work collapses from two large
    ``np.interp`` calls to a single one plus a ``np.where``.
    """
    # Avoid an unnecessary copy when the input is already float32 (the main
    # driver casts the whole volume up front).
    flat = np.ascontiguousarray(chunk, dtype=np.float32).ravel()

    src_bins, src_cdf, tissue_count = _build_tissue_cdf(flat, n_bins, tissue_threshold)
    if tissue_count < 500:
        return chunk

    # LUT on bin centers: src intensity percentile -> matched reference intensity.
    matched_lut = np.interp(src_cdf, ref_cdf, ref_bins)

    mapped = np.interp(flat, src_bins, matched_lut).astype(np.float32, copy=False)
    result = np.where(flat > tissue_threshold, mapped, flat)
    return result.reshape(chunk.shape)


def apply_histogram_matching(
    vol: np.ndarray,
    n_serial_slices: int | None,
    n_bins: int,
    tissue_threshold: float = 0.0,
    use_gpu: bool = False,
) -> np.ndarray:
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
    use_gpu : bool
        If True, run the per-chunk matching loop on GPU via CuPy. Falls back
        to CPU silently if CuPy is unavailable. The volume itself is moved to
        GPU one chunk at a time, so memory usage stays bounded.

    Returns
    -------
    np.ndarray
        Histogram-matched volume.
    """
    flat_all = vol.ravel()
    ref_bins, ref_cdf, tissue_count = _build_tissue_cdf(flat_all, n_bins, tissue_threshold)
    if tissue_count < 500:
        return vol

    bounds = _chunk_boundaries(vol.shape[0], n_serial_slices)

    if use_gpu:
        try:
            return _apply_histogram_matching_gpu(vol, bounds, ref_bins, ref_cdf, n_bins, tissue_threshold)
        except ImportError:
            pass

    out = np.empty_like(vol)
    for s, e in bounds:
        chunk = vol[s:e]
        out[s:e] = _match_chunk_to_reference(chunk, ref_bins, ref_cdf, n_bins, tissue_threshold)

    return out


def _apply_histogram_matching_gpu(
    vol: np.ndarray,
    bounds: list[tuple[int, int]],
    ref_bins: np.ndarray,
    ref_cdf: np.ndarray,
    n_bins: int,
    tissue_threshold: float,
) -> np.ndarray:
    """GPU implementation of the per-chunk histogram-matching loop.

    Each chunk is moved to GPU, has its tissue CDF computed, an
    ``n_bins``-sized LUT built, and the per-voxel mapping applied.
    Result is moved back to CPU per chunk so the host array fills
    incrementally without holding the whole volume on GPU.
    """
    import cupy as cp

    ref_bins_g = cp.asarray(ref_bins, dtype=cp.float32)
    ref_cdf_g = cp.asarray(ref_cdf, dtype=cp.float32)

    lo = tissue_threshold + max(1e-6, tissue_threshold * 1e-6)
    lo = min(lo, 1.0)

    out = np.empty_like(vol)
    for s, e in bounds:
        chunk_g = cp.asarray(vol[s:e], dtype=cp.float32)
        flat = chunk_g.ravel()

        hist = cp.histogram(flat, bins=n_bins, range=(lo, 1.0))[0]
        tissue_count = int(hist.sum().item())
        if tissue_count < 500:
            out[s:e] = vol[s:e]
            continue

        edges = cp.linspace(lo, 1.0, n_bins + 1, dtype=cp.float32)
        src_bins = 0.5 * (edges[:-1] + edges[1:])
        src_cdf = cp.cumsum(hist).astype(cp.float32)
        src_cdf /= src_cdf[-1]

        matched_lut = cp.interp(src_cdf, ref_cdf_g, ref_bins_g)
        mapped = cp.interp(flat, src_bins, matched_lut).astype(cp.float32, copy=False)
        result = cp.where(flat > tissue_threshold, mapped, flat).reshape(chunk_g.shape)

        out[s:e] = cp.asnumpy(result)

    return out


def apply_zprofile_smoothing(
    vol: np.ndarray,
    mask: np.ndarray,
    sigma: float,
    min_tissue_voxels: int = 100,
) -> np.ndarray:
    """Remove residual per-Z-plane intensity jitter via a smoothed scalar gain.

    For each Z-plane, computes the tissue mean (over `mask`), smooths the
    Z-mean profile with a Gaussian (sigma in Z-plane units), then applies a
    per-Z multiplicative gain `target / observed` to align each plane's tissue
    mean to the smoothed trend.  Background voxels (~mask) are left unchanged.

    The correction is bounded in magnitude by the smoothed-vs-observed ratio
    and acts only on the high-frequency component of the Z-profile, so the
    smooth depth attenuation and large-scale anatomical variation are
    preserved.  Best applied after `apply_histogram_matching` to clean up the
    residual ~1-2% inter-slice step that HM cannot remove.

    Parameters
    ----------
    vol : np.ndarray
        Input volume (Z, Y, X).
    mask : np.ndarray
        Tissue mask (Z, Y, X), bool.
    sigma : float
        Gaussian smoothing sigma in Z-plane units.  Larger = preserves more
        depth structure but removes less jitter.  2.0-4.0 works well in practice.
    min_tissue_voxels : int
        Z-planes with fewer tissue voxels are left unchanged (no reliable mean).

    Returns
    -------
    np.ndarray
        Volume with per-Z gain applied to tissue voxels.
    """
    from scipy.ndimage import gaussian_filter1d

    if sigma <= 0:
        return vol
    n_z = vol.shape[0]
    z_means = np.full(n_z, np.nan, dtype=np.float64)
    for z in range(n_z):
        m = mask[z]
        if m.sum() >= min_tissue_voxels:
            z_means[z] = vol[z][m].mean()
    valid = ~np.isnan(z_means)
    if valid.sum() < 3:
        return vol
    target = z_means.copy()
    target[valid] = gaussian_filter1d(z_means[valid], sigma=sigma)
    gains = np.where(valid, target / np.clip(z_means, 1e-6, None), 1.0).astype(np.float32)

    out = vol.astype(np.float32, copy=True)
    out *= gains[:, None, None]
    out[~mask] = vol[~mask]  # restore background
    return out
