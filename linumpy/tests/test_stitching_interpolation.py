# -*- coding: utf-8 -*-
"""Tests for linumpy/stitching/interpolation.py"""
import numpy as np
import pytest

from linumpy.stitching.interpolation import (
    assess_degraded_slice_quality,
    blend_with_degraded,
    interpolate_average,
    interpolate_weighted,
)


def _vol(shape=(8, 16, 16), seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random(shape) * 100.0).astype(np.float32)


# ---------------------------------------------------------------------------
# interpolate_average
# ---------------------------------------------------------------------------

def test_interpolate_average_shape():
    before = _vol()
    after = _vol(seed=1)
    result = interpolate_average(before, after)
    assert result.shape == before.shape


def test_interpolate_average_midpoint():
    """Result should be the exact arithmetic mean."""
    before = np.zeros((4, 8, 8), dtype=np.float32)
    after = np.full((4, 8, 8), 2.0, dtype=np.float32)
    result = interpolate_average(before, after)
    np.testing.assert_allclose(result, 1.0)


def test_interpolate_average_identical():
    """Averaging a volume with itself preserves values."""
    vol = _vol()
    result = interpolate_average(vol, vol)
    np.testing.assert_allclose(result, vol, rtol=1e-5)


def test_interpolate_average_dtype_float32():
    """Output dtype should be float32."""
    before = np.ones((4, 8, 8), dtype=np.uint8)
    after = np.ones((4, 8, 8), dtype=np.uint8)
    result = interpolate_average(before, after)
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# interpolate_weighted
# ---------------------------------------------------------------------------

def test_interpolate_weighted_shape():
    before = _vol()
    after = _vol(seed=1)
    result = interpolate_weighted(before, after, sigma=1.0)
    assert result.shape == before.shape


def test_interpolate_weighted_same_result_as_average_when_sigma_zero():
    """With sigma ≈ 0, weighted should be close to simple average."""
    before = np.zeros((6, 8, 8), dtype=np.float32)
    after = np.full((6, 8, 8), 4.0, dtype=np.float32)
    avg = interpolate_average(before, after)
    weighted = interpolate_weighted(before, after, sigma=0.01)
    np.testing.assert_allclose(weighted, avg, rtol=0.05)


def test_interpolate_weighted_smoothing_reduces_variance():
    """Larger sigma should produce smoother output (lower std dev along Z)."""
    rng = np.random.default_rng(42)
    before = rng.random((20, 8, 8)).astype(np.float32)
    after = rng.random((20, 8, 8)).astype(np.float32)
    std_low_sigma = interpolate_weighted(before, after, sigma=0.1).std()
    std_high_sigma = interpolate_weighted(before, after, sigma=3.0).std()
    assert std_high_sigma < std_low_sigma


# ---------------------------------------------------------------------------
# blend_with_degraded
# ---------------------------------------------------------------------------

def test_blend_with_degraded_pure_interpolated():
    """quality_weight=0 → output equals interpolated."""
    interp = np.ones((4, 8, 8), dtype=np.float32)
    degraded = np.full((4, 8, 8), 10.0, dtype=np.float32)
    result = blend_with_degraded(interp, degraded, quality_weight=0.0)
    np.testing.assert_allclose(result, interp)


def test_blend_with_degraded_pure_degraded():
    """quality_weight=1 → output equals degraded."""
    interp = np.ones((4, 8, 8), dtype=np.float32)
    degraded = np.full((4, 8, 8), 10.0, dtype=np.float32)
    result = blend_with_degraded(interp, degraded, quality_weight=1.0)
    np.testing.assert_allclose(result, degraded)


def test_blend_with_degraded_half_weight():
    """quality_weight=0.5 → average of interpolated and degraded."""
    interp = np.zeros((4, 8, 8), dtype=np.float32)
    degraded = np.full((4, 8, 8), 4.0, dtype=np.float32)
    result = blend_with_degraded(interp, degraded, quality_weight=0.5)
    np.testing.assert_allclose(result, 2.0)


def test_blend_with_degraded_shape_preserved():
    interp = _vol()
    degraded = _vol(seed=1)
    result = blend_with_degraded(interp, degraded, quality_weight=0.3)
    assert result.shape == interp.shape


# ---------------------------------------------------------------------------
# assess_degraded_slice_quality
# ---------------------------------------------------------------------------

def test_assess_degraded_slice_quality_perfect_quality():
    """If degraded == reference, quality score should be near 1."""
    rng = np.random.default_rng(10)
    vol = (rng.random((8, 16, 16)) * 100.0).astype(np.float32)
    score, metrics = assess_degraded_slice_quality(vol, vol, vol)
    assert 0.0 <= score <= 1.0
    # Perfect match → quality near 1
    assert score > 0.8


def test_assess_degraded_slice_quality_zeros_degrade_score():
    """Zero-filled degraded slice should have low quality score."""
    rng = np.random.default_rng(11)
    before = (rng.random((8, 16, 16)) * 100.0 + 1.0).astype(np.float32)
    after = (rng.random((8, 16, 16)) * 100.0 + 1.0).astype(np.float32)
    degraded = np.zeros_like(before)
    score, metrics = assess_degraded_slice_quality(degraded, before, after)
    assert 0.0 <= score <= 1.0
    assert score < 0.5


def test_assess_degraded_slice_quality_returns_metrics_dict():
    rng = np.random.default_rng(12)
    vol = rng.random((6, 12, 12)).astype(np.float32)
    _, metrics = assess_degraded_slice_quality(vol, vol, vol)
    expected_keys = {'ssim_before', 'ssim_after', 'ssim_mean',
                     'edge_preservation', 'variance_ratio', 'overall'}
    assert expected_keys.issubset(set(metrics.keys()))
