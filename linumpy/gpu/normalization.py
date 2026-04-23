"""GPU-accelerated Z-intensity normalization for serial OCT stacks.

Provides CuPy-based versions of :func:`linumpy.preproc.normalization.compute_scale_factors`
and :func:`linumpy.preproc.normalization.apply_histogram_matching`.

Each helper falls back to the CPU implementation when CuPy or a CUDA device
is unavailable, so callers can pass ``use_gpu=True`` unconditionally.

The GPU helpers stream the volume to the device chunk-by-chunk. The full
volume is **never** materialised on the GPU; this keeps peak device usage
bounded by the largest serial-section chunk regardless of total volume size.
"""

from __future__ import annotations

import numpy as np

from linumpy.intensity.normalization import (
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
    the expensive per-chunk percentile on the GPU. Each serial-section chunk is
    streamed to the device individually; the full volume is never resident on
    the GPU. The small 1-D Gaussian smoothing is done on the CPU (SciPy) as it
    is negligible in cost.
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

    raw_metrics = np.zeros(n_chunks, dtype=np.float64)
    mempool = cp.get_default_memory_pool()
    for i, (s, e) in enumerate(bounds):
        chunk_gpu = cp.asarray(vol[s:e], dtype=cp.float32)
        raw_metrics[i] = _robust_percentile_gpu(chunk_gpu, percentile)
        del chunk_gpu
        mempool.free_all_blocks()

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


def _histogram_edges_gpu(n_bins: int, tissue_threshold: float):
    """Reproducible bin edges shared across all chunks (matches CPU CDF range)."""
    import cupy as cp

    lo = tissue_threshold + max(1e-6, tissue_threshold * 1e-6)
    lo = min(lo, 1.0)
    return cp.linspace(lo, 1.0, n_bins + 1, dtype=cp.float64)


def apply_histogram_matching_gpu(
    vol: np.ndarray,
    n_serial_slices,
    n_bins: int,
    tissue_threshold: float = 0.0,
    use_gpu: bool = True,
) -> np.ndarray:
    """GPU-accelerated per-section histogram matching to a global reference.

    Mirrors :func:`linumpy.preproc.normalization.apply_histogram_matching` and
    returns identical output (up to floating-point rounding). The volume is
    streamed chunk-by-chunk to the device in two passes — one to accumulate the
    global reference histogram and one to apply the per-section LUT — so peak
    GPU memory stays on the order of a single chunk regardless of volume size.
    Falls back to the CPU implementation when no GPU is available.
    """
    if not (use_gpu and GPU_AVAILABLE):
        return apply_histogram_matching(
            vol,
            n_serial_slices=n_serial_slices,
            n_bins=n_bins,
            tissue_threshold=tissue_threshold,
        )

    import cupy as cp

    bounds = _chunk_boundaries(vol.shape[0], n_serial_slices)
    edges = _histogram_edges_gpu(n_bins, tissue_threshold)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    mempool = cp.get_default_memory_pool()

    # Pass 1: accumulate global reference histogram by streaming chunks.
    ref_hist = cp.zeros(n_bins, dtype=cp.int64)
    for s, e in bounds:
        chunk_gpu = cp.asarray(vol[s:e], dtype=cp.float32)
        h, _ = cp.histogram(chunk_gpu.ravel(), bins=edges)
        ref_hist += h
        del chunk_gpu, h
        mempool.free_all_blocks()

    ref_total = int(ref_hist.sum())
    if ref_total < 500:
        del ref_hist
        mempool.free_all_blocks()
        return vol

    ref_cdf = cp.cumsum(ref_hist).astype(cp.float64)
    if float(ref_cdf[-1]) > 0:
        ref_cdf /= ref_cdf[-1]
    del ref_hist

    # Pass 2: per-section histogram match, streaming. The output is assembled
    # in host RAM (matching the CPU path), so the GPU only ever holds one
    # chunk plus its working buffers.
    out = np.empty_like(vol)
    for s, e in bounds:
        chunk_gpu = cp.asarray(vol[s:e], dtype=cp.float32)
        flat = chunk_gpu.ravel()
        src_hist, _ = cp.histogram(flat, bins=edges)
        src_total = int(src_hist.sum())
        if src_total < 500:
            out[s:e] = vol[s:e]
            del chunk_gpu, flat, src_hist
            mempool.free_all_blocks()
            continue

        src_cdf = cp.cumsum(src_hist).astype(cp.float64)
        if float(src_cdf[-1]) > 0:
            src_cdf /= src_cdf[-1]
        matched_lut = cp.interp(src_cdf, ref_cdf, bin_centers)
        mapped = cp.interp(flat, bin_centers, matched_lut).astype(cp.float32, copy=False)
        result = cp.where(flat > tissue_threshold, mapped, flat)
        out[s:e] = cp.asnumpy(result.reshape(chunk_gpu.shape))
        del chunk_gpu, flat, src_hist, src_cdf, matched_lut, mapped, result
        mempool.free_all_blocks()

    del ref_cdf, bin_centers, edges
    mempool.free_all_blocks()
    return out


__all__ = [
    "apply_histogram_matching_gpu",
    "compute_scale_factors_gpu",
]
