"""GPU-accelerated Z-intensity normalization for serial OCT stacks.

Provides CuPy-based versions of :func:`linumpy.preproc.normalization.compute_scale_factors`
and :func:`linumpy.preproc.normalization.apply_histogram_matching`.

Each helper falls back to the CPU implementation when CuPy or a CUDA device
is unavailable, so callers can pass ``use_gpu=True`` unconditionally.
"""

from __future__ import annotations

import numpy as np

from linumpy.preproc.normalization import (
    _chunk_boundaries,
    _smooth_weighted,
    apply_histogram_matching,
    compute_scale_factors,
)

from . import GPU_AVAILABLE


def _robust_percentile_gpu(chunk, percentile: float) -> float:
    """GPU version of ``_robust_percentile``: Nth percentile of non-zero voxels."""
    import cupy as cp

    flat = chunk.ravel()
    nonzero = flat[flat > 0]
    if int(nonzero.size) < 500:
        return 0.0
    return float(cp.percentile(nonzero, percentile))


def compute_scale_factors_gpu(
    vol: np.ndarray,
    n_serial_slices,
    smooth_sigma: float,
    percentile: float,
    min_scale: float,
    max_scale: float,
    use_gpu: bool = True,
):
    """GPU-accelerated per-Z-plane linear scale factors for percentile normalization.

    Mirrors :func:`linumpy.preproc.normalization.compute_scale_factors` but runs
    the expensive per-chunk percentile on the GPU. The small 1-D Gaussian
    smoothing is done on the CPU (SciPy) as it is negligible in cost.
    """
    if not (use_gpu and GPU_AVAILABLE):
        return compute_scale_factors(
            vol,
            n_serial_slices=n_serial_slices,
            smooth_sigma=smooth_sigma,
            percentile=percentile,
            min_scale=min_scale,
            max_scale=max_scale,
        )

    import cupy as cp

    n_z = vol.shape[0]
    bounds = _chunk_boundaries(n_z, n_serial_slices)
    n_chunks = len(bounds)

    vol_gpu = cp.asarray(vol, dtype=cp.float32)
    raw_metrics = np.zeros(n_chunks, dtype=np.float64)
    for i, (s, e) in enumerate(bounds):
        raw_metrics[i] = _robust_percentile_gpu(vol_gpu[s:e], percentile)

    del vol_gpu
    cp.get_default_memory_pool().free_all_blocks()

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


def _build_tissue_cdf_gpu(flat_values, n_bins: int, tissue_threshold: float):
    """GPU version of ``_build_tissue_cdf``. Operates on a CuPy flat array."""
    import cupy as cp

    lo = tissue_threshold + max(1e-6, tissue_threshold * 1e-6)
    lo = min(lo, 1.0)
    hist, edges = cp.histogram(flat_values, bins=n_bins, range=(lo, 1.0))
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    total = int(hist.sum())
    cdf = cp.cumsum(hist).astype(cp.float64)
    if float(cdf[-1]) > 0:
        cdf /= cdf[-1]
    return bin_centers, cdf, total


def apply_histogram_matching_gpu(
    vol: np.ndarray,
    n_serial_slices,
    n_bins: int,
    tissue_threshold: float = 0.0,
    use_gpu: bool = True,
) -> np.ndarray:
    """GPU-accelerated per-section histogram matching to a global reference.

    Mirrors :func:`linumpy.preproc.normalization.apply_histogram_matching` and
    returns identical output (up to floating-point rounding). Falls back to the
    CPU implementation when no GPU is available.
    """
    if not (use_gpu and GPU_AVAILABLE):
        return apply_histogram_matching(
            vol,
            n_serial_slices=n_serial_slices,
            n_bins=n_bins,
            tissue_threshold=tissue_threshold,
        )

    import cupy as cp

    vol_gpu = cp.asarray(vol, dtype=cp.float32)
    flat_all = vol_gpu.ravel()
    ref_bins, ref_cdf, tissue_count = _build_tissue_cdf_gpu(flat_all, n_bins, tissue_threshold)
    if tissue_count < 500:
        del vol_gpu
        cp.get_default_memory_pool().free_all_blocks()
        return vol

    bounds = _chunk_boundaries(vol.shape[0], n_serial_slices)
    out_gpu = cp.empty_like(vol_gpu)
    for s, e in bounds:
        chunk = vol_gpu[s:e]
        flat = chunk.ravel()
        src_bins, src_cdf, tissue_count_chunk = _build_tissue_cdf_gpu(flat, n_bins, tissue_threshold)
        if tissue_count_chunk < 500:
            out_gpu[s:e] = chunk
            continue

        matched_lut = cp.interp(src_cdf, ref_cdf, ref_bins)
        mapped = cp.interp(flat, src_bins, matched_lut).astype(cp.float32, copy=False)
        result = cp.where(flat > tissue_threshold, mapped, flat)
        out_gpu[s:e] = result.reshape(chunk.shape)

    out = cp.asnumpy(out_gpu)
    del vol_gpu, out_gpu
    cp.get_default_memory_pool().free_all_blocks()
    return out


__all__ = [
    "apply_histogram_matching_gpu",
    "compute_scale_factors_gpu",
]
