#!/usr/bin/env python3
# Configure thread limits before numpy/scipy imports
import linumpy._thread_config  # noqa: F401

# -*- coding: utf-8 -*-
"""
Normalize intensity drift across serial sections in a stacked 3D OCT volume.

Corrects slow acquisition drift (focus changes, water level variations,
laser power changes) between serial sections while preserving genuine
anatomical intensity differences (e.g. white matter vs grey matter).

Two normalization modes are available (--mode):

  percentile (default)
    1. Compute the Nth percentile of non-zero voxels per serial section.
    2. Smooth the per-section curve with a Gaussian (sigma controls
       how aggressively drift vs anatomy is corrected).
    3. Scale each section by (global_ref / smoothed_local_ref).
    4. Clamp scale factors to [min_scale, max_scale].

    Limitation: uniformly scales each section, so intrinsically bright
    sections (e.g. white-matter-rich) get darkened and dark sections
    get brightened.

  histogram
    Per-section histogram matching to a global reference distribution.
    1. Build a reference CDF from all non-zero voxels in the volume.
    2. For each serial section, compute its own CDF and derive a
       monotonic intensity-mapping that transforms its histogram to
       match the global reference.
    3. Apply the mapping only to non-zero voxels; background stays zero.

    Advantage: preserves relative contrast *within* each section
    (white matter stays brighter than grey matter) while correcting
    section-to-section drift without uniformly darkening bright sections.

Usage examples
--------------
# Histogram matching (recommended for OCT serial sections)
linum_normalize_z_intensity.py input.ome.zarr output.ome.zarr --mode histogram

# Percentile mode with 10-slice smoothing
linum_normalize_z_intensity.py input.ome.zarr output.ome.zarr \\
    --n_serial_slices 64 --mode percentile --smooth_sigma 10.0

# Save a diagnostic plot
linum_normalize_z_intensity.py input.ome.zarr output.ome.zarr \\
    --n_serial_slices 64 --plot correction_curve.png
"""

import argparse
from typing import Optional

import numpy as np
import dask.array as da
from scipy.ndimage import gaussian_filter1d

from linumpy.io.zarr import read_omezarr, save_omezarr


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument('in_zarr',
                   help='Input stacked 3D OME-Zarr volume.')
    p.add_argument('out_zarr',
                   help='Output intensity-normalised OME-Zarr volume.')
    p.add_argument('--mode', choices=['percentile', 'histogram'], default='percentile',
                   help='Normalization mode.\n'
                        '  percentile : linear scaling to match smoothed percentile curve\n'
                        '  histogram  : per-section histogram matching to global reference\n'
                        '               Preserves relative contrast; does not uniformly\n'
                        '               darken bright (white-matter) sections. [%(default)s]')
    p.add_argument('--n_serial_slices', type=int, default=None,
                   help='Number of serial sections in the stacked volume.\n'
                        'Used to compute one metric per section.\n'
                        'If omitted, operates at individual Z-plane level.')
    p.add_argument('--smooth_sigma', type=float, default=10.0,
                   help='(percentile mode) Gaussian smoothing sigma in serial-section units.\n'
                        'Larger values correct only slower drift. [%(default)s]')
    p.add_argument('--percentile', type=float, default=80.0,
                   help='(percentile mode) Percentile of non-zero voxels used as the\n'
                        'intensity reference per chunk. [%(default)s]')
    p.add_argument('--max_scale', type=float, default=2.0,
                   help='(percentile mode) Maximum allowed scale factor per section. [%(default)s]')
    p.add_argument('--min_scale', type=float, default=0.5,
                   help='(percentile mode) Minimum allowed scale factor per section. [%(default)s]')
    p.add_argument('--n_bins', type=int, default=512,
                   help='(histogram mode) Number of histogram bins. [%(default)s]')
    p.add_argument('--tissue_threshold', type=float, default=0.0,
                   help='(histogram mode) Minimum intensity to classify as tissue.\n'
                        'Voxels at or below this value are treated as background and\n'
                        'left unchanged by histogram matching.  Use a small positive\n'
                        'value (e.g. 0.02) to exclude near-zero background noise from\n'
                        'the histogram mapping and prevent background brightening.\n'
                        '0.0 classifies everything above zero as tissue. [%(default)s]')
    p.add_argument('--strength', type=float, default=1.0,
                   help='Mixing strength of the correction (0.0–1.0).\n'
                        '  1.0 = full normalization (default)\n'
                        '  0.5 = blend 50%% normalized + 50%% original\n'
                        '  0.0 = no correction (passthrough)\n'
                        'Use values < 1 to apply a gentler correction when the\n'
                        'default is too aggressive. [%(default)s]')
    p.add_argument('--plot', type=str, default=None,
                   help='Optional path to save a diagnostic PNG.')
    return p


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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
    weights = (values > 0).astype(np.float64)
    smoothed_v = gaussian_filter1d(values * weights, sigma=sigma, mode='reflect')
    smoothed_w = gaussian_filter1d(weights, sigma=sigma, mode='reflect')
    out = np.where(smoothed_w > 1e-6, smoothed_v / smoothed_w, 0.0)
    return out


def _chunk_boundaries(n_z: int, n_serial_slices: Optional[int]):
    """Return list of (start, end) Z-index pairs, one per chunk."""
    if n_serial_slices is not None:
        chunk_size = n_z / n_serial_slices
        starts = [int(round(i * chunk_size)) for i in range(n_serial_slices)]
        ends = [int(round(i * chunk_size)) for i in range(1, n_serial_slices + 1)]
    else:
        starts = list(range(n_z))
        ends = list(range(1, n_z + 1))
    return list(zip(starts, ends))


# ---------------------------------------------------------------------------
# Percentile mode
# ---------------------------------------------------------------------------

def compute_scale_factors(vol: np.ndarray,
                          n_serial_slices: Optional[int],
                          smooth_sigma: float,
                          percentile: float,
                          min_scale: float,
                          max_scale: float):
    """Compute per-Z-plane linear scale factors (percentile mode).

    Returns
    -------
    scale_factors : np.ndarray, shape (n_z,)
    raw_metrics   : np.ndarray  – per-chunk metric before smoothing
    smoothed      : np.ndarray  – smoothed reference curve
    boundaries    : list of int – Z-plane start indices of each chunk
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


# ---------------------------------------------------------------------------
# Histogram matching mode
# ---------------------------------------------------------------------------

def _build_cdf(values: np.ndarray, n_bins: int):
    """Build a cumulative distribution function from an array of values.

    Parameters
    ----------
    values : np.ndarray, 1-D, in [0, 1]
    n_bins : int

    Returns
    -------
    bin_centers : np.ndarray, shape (n_bins,)
    cdf         : np.ndarray, shape (n_bins,), normalised to [0, 1]
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
    """Map chunk intensities so that its histogram matches the reference CDF.

    Only voxels above ``tissue_threshold`` are mapped; everything at or below
    that value (background / near-zero noise) remains unchanged.
    The mapping is monotonic (preserves relative contrast within the chunk).
    """
    flat = chunk.ravel().astype(np.float32)
    tissue_mask = flat > tissue_threshold
    if tissue_mask.sum() < 500:
        return chunk  # Too few tissue voxels – leave unchanged

    tissue = flat[tissue_mask]

    # Source CDF
    src_bins, src_cdf = _build_cdf(tissue, n_bins)

    # Mapping: source_value -> source_cdf_value -> reference_value
    # Step 1: for each tissue voxel, find its percentile in source CDF
    src_percentiles = np.interp(tissue, src_bins, src_cdf)
    # Step 2: find the reference value at that percentile (inverse ref CDF)
    matched = np.interp(src_percentiles, ref_cdf, ref_bins)

    result = flat.copy()
    result[tissue_mask] = matched
    return result.reshape(chunk.shape)


def apply_histogram_matching(vol: np.ndarray,
                             n_serial_slices: Optional[int],
                             n_bins: int,
                             tissue_threshold: float = 0.0) -> np.ndarray:
    """Apply per-section histogram matching to a global reference distribution.

    The global reference is the CDF of all voxels above ``tissue_threshold``
    across the entire volume.  Each serial section's tissue histogram is
    independently mapped to that reference, correcting section-to-section
    intensity drift while preserving relative contrast within each section.
    Voxels at or below ``tissue_threshold`` are left completely unchanged.
    """
    # Build global reference CDF from tissue voxels only
    flat_all = vol.ravel()
    tissue_all = flat_all[flat_all > tissue_threshold]
    if tissue_all.size < 500:
        print("WARNING: too few tissue voxels for histogram matching; returning input unchanged.")
        return vol

    print(f"Building global reference CDF from {tissue_all.size:,} tissue voxels "
          f"(threshold={tissue_threshold}) ...")
    ref_bins, ref_cdf = _build_cdf(tissue_all.astype(np.float64), n_bins)

    bounds = _chunk_boundaries(vol.shape[0], n_serial_slices)
    n_chunks = len(bounds)

    out = np.empty_like(vol)
    for i, (s, e) in enumerate(bounds):
        chunk = vol[s:e]
        out[s:e] = _match_chunk_to_reference(chunk, ref_bins, ref_cdf, n_bins, tissue_threshold)
        if (i + 1) % max(1, n_chunks // 10) == 0 or i == n_chunks - 1:
            print(f"  Matched {i+1}/{n_chunks} sections ...")

    return out


# ---------------------------------------------------------------------------
# Diagnostic plot
# ---------------------------------------------------------------------------

def save_plot(raw_metrics, smoothed, scale_factors, n_serial_slices, plot_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    x = np.arange(len(raw_metrics))
    axes[0].plot(x, raw_metrics, 'o-', ms=3, lw=1, label='Raw per-slice metric', alpha=0.8)
    axes[0].plot(x, smoothed, '-', lw=2, label='Smoothed reference', color='red')
    axes[0].set_ylabel('Intensity metric (Nth percentile)')
    axes[0].legend()
    axes[0].set_title('Inter-slice intensity drift')
    axes[0].grid(True, alpha=0.3)

    expanded_scale = np.array([
        scale_factors[int(round(i * len(scale_factors) / len(raw_metrics)))]
        for i in range(len(raw_metrics))
    ]) if len(scale_factors) != len(raw_metrics) else scale_factors

    axes[1].plot(x, expanded_scale, 'o-', ms=3, lw=1, color='green')
    axes[1].axhline(1.0, color='gray', lw=1, ls='--')
    axes[1].set_ylabel('Scale factor applied')
    axes[1].set_xlabel('Serial section index' if n_serial_slices else 'Z-plane index')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Correction scale factors')

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Diagnostic plot saved to {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    print(f"Loading {args.in_zarr} ...")
    vol_da, res = read_omezarr(args.in_zarr, level=0)
    vol = vol_da[:].astype(np.float32)
    print(f"Volume shape: Z={vol.shape[0]}, X={vol.shape[1]}, Y={vol.shape[2]}")

    if args.mode == 'histogram':
        print(f"Mode: histogram matching "
              f"({'serial-slice mode, n=' + str(args.n_serial_slices) if args.n_serial_slices else 'Z-plane mode'}), "
              f"n_bins={args.n_bins}, strength={args.strength} ...")
        vol_matched = apply_histogram_matching(
            vol, args.n_serial_slices, args.n_bins, args.tissue_threshold)
        vol_matched = np.clip(vol_matched, 0.0, 1.0)

        if args.strength < 1.0:
            print(f"Blending: {args.strength:.2f} * matched + {1.0 - args.strength:.2f} * original ...")
            vol = args.strength * vol_matched + (1.0 - args.strength) * vol
        else:
            vol = vol_matched

        if args.plot:
            print("NOTE: diagnostic plot is only available in percentile mode; skipping.")

    else:  # percentile (default)
        print(f"Mode: percentile scaling "
              f"({'serial-slice mode, n=' + str(args.n_serial_slices) if args.n_serial_slices else 'Z-plane mode'}), "
              f"sigma={args.smooth_sigma}, percentile={args.percentile}, strength={args.strength} ...")
        scale_factors, raw_metrics, smoothed, boundaries = compute_scale_factors(
            vol,
            n_serial_slices=args.n_serial_slices,
            smooth_sigma=args.smooth_sigma,
            percentile=args.percentile,
            min_scale=args.min_scale,
            max_scale=args.max_scale,
        )

        if args.strength < 1.0:
            # Blend scale factors toward 1.0 (no correction) by the strength factor
            scale_factors = 1.0 + args.strength * (scale_factors - 1.0)
            print(f"Adjusted scale factor range after strength={args.strength}: "
                  f"{scale_factors.min():.3f} – {scale_factors.max():.3f}")
        else:
            print(f"Scale factor range: {scale_factors.min():.3f} – {scale_factors.max():.3f}  "
                  f"(mean={scale_factors.mean():.3f})")

        if args.plot:
            save_plot(raw_metrics, smoothed, scale_factors, args.n_serial_slices, args.plot)

        print("Applying scale factors ...")
        vol = vol * scale_factors[:, None, None]
        vol = np.clip(vol, 0.0, 1.0)

    print(f"Saving to {args.out_zarr} ...")
    save_omezarr(da.from_array(vol), args.out_zarr, res)
    print("Done.")


if __name__ == '__main__':
    main()
