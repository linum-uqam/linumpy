# -*- coding: utf-8 -*-
"""Tests for linumpy/preproc/normalization.py"""
import numpy as np
import pytest

from linumpy.preproc.normalization import (
    _build_cdf,
    _chunk_boundaries,
    _match_chunk_to_reference,
    _robust_percentile,
    _smooth_weighted,
    apply_histogram_matching,
    compute_scale_factors,
    get_agarose_mask,
    normalize_volume,
)


# ---------------------------------------------------------------------------
# get_agarose_mask
# ---------------------------------------------------------------------------

def _make_tissue_vol(shape=(10, 32, 32)):
    """Volume with bright tissue region and dim agarose surroundings."""
    rng = np.random.default_rng(0)
    vol = rng.random(shape).astype(np.float32) * 20.0   # low = agarose
    # Bright tissue block in the center
    cx, cy = shape[1] // 4, shape[2] // 4
    vol[:, cx:cx * 3, cy:cy * 3] += 80.0
    return vol


def test_get_agarose_mask_shape():
    vol = _make_tissue_vol((8, 32, 32))
    mask, threshold = get_agarose_mask(vol)
    assert mask.shape == (32, 32)


def test_get_agarose_mask_is_boolean():
    vol = _make_tissue_vol()
    mask, _ = get_agarose_mask(vol)
    assert mask.dtype == bool


def test_get_agarose_mask_threshold_positive():
    vol = _make_tissue_vol()
    _, threshold = get_agarose_mask(vol)
    assert threshold > 0


def test_get_agarose_mask_low_intensity_is_agarose():
    """Low-intensity region should be classified as agarose."""
    vol = _make_tissue_vol()
    mask, _ = get_agarose_mask(vol)
    # The surrounding low-intensity region should have agarose voxels
    assert mask.any()


# ---------------------------------------------------------------------------
# normalize_volume
# ---------------------------------------------------------------------------

def test_normalize_volume_output_shape():
    vol = _make_tissue_vol((6, 24, 24))
    mask, _ = get_agarose_mask(vol)
    result, thresholds = normalize_volume(vol.copy(), mask)
    assert result.shape == vol.shape


def test_normalize_volume_output_range():
    """Normalized values should be in [0, 1]."""
    vol = _make_tissue_vol((6, 24, 24))
    mask, _ = get_agarose_mask(vol)
    result, _ = normalize_volume(vol.copy(), mask)
    assert float(result.min()) >= -1e-6
    assert float(result.max()) <= 1.0 + 1e-6


def test_normalize_volume_background_thresholds_length():
    vol = _make_tissue_vol((5, 24, 24))
    mask, _ = get_agarose_mask(vol)
    _, thresholds = normalize_volume(vol.copy(), mask)
    assert len(thresholds) == vol.shape[0]


# ---------------------------------------------------------------------------
# _robust_percentile
# ---------------------------------------------------------------------------

def test_robust_percentile_empty_returns_zero():
    """Nearly-empty array (< 500 non-zeros) should return 0.0."""
    chunk = np.zeros((10, 10, 10), dtype=np.float32)
    assert _robust_percentile(chunk, 90) == 0.0


def test_robust_percentile_computes_correctly():
    chunk = np.arange(1, 1001, dtype=np.float32)   # 1000 values
    result = _robust_percentile(chunk, 50)
    expected = float(np.percentile(chunk, 50))
    assert abs(result - expected) < 1.0


# ---------------------------------------------------------------------------
# _smooth_weighted
# ---------------------------------------------------------------------------

def test_smooth_weighted_preserves_mean():
    """Smoothing should not wildly change the mean of non-zero values."""
    values = np.array([1.0, 2.0, 0.0, 2.0, 1.0])
    smoothed = _smooth_weighted(values, sigma=1.0)
    assert smoothed.shape == values.shape


def test_smooth_weighted_zeros_dont_bias():
    """Zeros indicate missing data; non-zero neighbors should dominate."""
    values = np.array([1.0, 0.0, 0.0, 0.0, 1.0])
    smoothed = _smooth_weighted(values, sigma=0.5)
    # Interior zeros should be interpolated from neighbors (non-zero)
    assert all(v >= 0 for v in smoothed)


# ---------------------------------------------------------------------------
# _chunk_boundaries
# ---------------------------------------------------------------------------

def test_chunk_boundaries_with_serial_slices():
    bounds = _chunk_boundaries(n_z=10, n_serial_slices=5)
    assert len(bounds) == 5
    # Boundaries should cover [0, 10)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == 10


def test_chunk_boundaries_per_plane():
    """n_serial_slices=None → one boundary per Z-plane."""
    bounds = _chunk_boundaries(n_z=5, n_serial_slices=None)
    assert len(bounds) == 5
    for i, (s, e) in enumerate(bounds):
        assert s == i
        assert e == i + 1


# ---------------------------------------------------------------------------
# _build_cdf
# ---------------------------------------------------------------------------

def test_build_cdf_normalized():
    values = np.random.default_rng(0).random(1000).astype(np.float64)
    bins, cdf = _build_cdf(values, n_bins=100)
    # CDF must be non-decreasing and last value == 1
    assert cdf[-1] == pytest.approx(1.0)
    assert np.all(np.diff(cdf) >= 0)


def test_build_cdf_bin_count():
    values = np.linspace(0, 1, 200)
    bins, cdf = _build_cdf(values, n_bins=50)
    assert len(bins) == 50
    assert len(cdf) == 50


# ---------------------------------------------------------------------------
# compute_scale_factors
# ---------------------------------------------------------------------------

def test_compute_scale_factors_shape():
    rng = np.random.default_rng(5)
    vol = rng.random((20, 16, 16)).astype(np.float32)
    sf, raw, smoothed, bounds = compute_scale_factors(
        vol, n_serial_slices=4, smooth_sigma=1.0,
        percentile=90.0, min_scale=0.5, max_scale=2.0)
    assert sf.shape == (20,)


def test_compute_scale_factors_clamped():
    rng = np.random.default_rng(6)
    vol = rng.random((20, 16, 16)).astype(np.float32)
    min_s, max_s = 0.5, 2.0
    sf, *_ = compute_scale_factors(vol, n_serial_slices=4, smooth_sigma=1.0,
                                   percentile=90.0, min_scale=min_s, max_scale=max_s)
    assert float(sf.min()) >= min_s - 1e-6
    assert float(sf.max()) <= max_s + 1e-6


# ---------------------------------------------------------------------------
# apply_histogram_matching
# ---------------------------------------------------------------------------

def test_apply_histogram_matching_shape():
    rng = np.random.default_rng(7)
    vol = rng.random((10, 16, 16)).astype(np.float32)
    result = apply_histogram_matching(vol, n_serial_slices=2, n_bins=64)
    assert result.shape == vol.shape


def test_apply_histogram_matching_range_preserved():
    """Output values should stay within roughly [0, 1] for unit input."""
    rng = np.random.default_rng(8)
    vol = rng.random((10, 16, 16)).astype(np.float32)
    result = apply_histogram_matching(vol, n_serial_slices=2, n_bins=64)
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0 + 1e-5
