# -*- coding: utf-8 -*-
"""Tests for linumpy/stitching/stacking.py"""
import numpy as np
import pytest

from linumpy.stitching.stacking import (
    apply_xy_shift,
    blend_overlap_xy,
    blend_overlap_z,
    find_z_overlap,
)


def _make_vol(shape=(10, 32, 32), fill=1.0):
    return (np.ones(shape) * fill).astype(np.float32)


# ---------------------------------------------------------------------------
# find_z_overlap
# ---------------------------------------------------------------------------

def test_find_z_overlap_returns_tuple():
    fixed = _make_vol((20, 16, 16))
    moving = _make_vol((20, 16, 16))
    overlap, corr = find_z_overlap(fixed, moving,
                                   slicing_interval_mm=0.1,
                                   search_range_mm=0.05,
                                   resolution_um=5.0)
    assert isinstance(overlap, int)
    assert isinstance(corr, (float, np.floating))


def test_find_z_overlap_identical_volumes():
    """Identical volumes have perfect correlation at some overlap."""
    rng = np.random.default_rng(0)
    vol = rng.random((20, 16, 16)).astype(np.float32)
    overlap, corr = find_z_overlap(vol, vol,
                                   slicing_interval_mm=0.05,
                                   search_range_mm=0.1,
                                   resolution_um=5.0)
    # Correlation should be high (>= 0.0 at minimum)
    assert corr >= 0.0
    assert 1 <= overlap <= 20


def test_find_z_overlap_min_max_degenerate():
    """When search range collapses, falls back to expected overlap."""
    fixed = _make_vol((10, 8, 8))
    moving = _make_vol((10, 8, 8))
    # Very large interval → expected overlap < 0 → min >= max edge case
    overlap, corr = find_z_overlap(fixed, moving,
                                   slicing_interval_mm=10.0,
                                   search_range_mm=0.0,
                                   resolution_um=5.0)
    assert isinstance(overlap, int)


# ---------------------------------------------------------------------------
# apply_xy_shift
# ---------------------------------------------------------------------------

def test_apply_xy_shift_zero_shift():
    vol = _make_vol((4, 10, 10))
    cropped, dst = apply_xy_shift(vol, 0.0, 0.0, output_shape=(10, 10))
    assert cropped is not None
    assert dst == (0, 10, 0, 10)


def test_apply_xy_shift_positive():
    vol = _make_vol((4, 8, 8))
    cropped, dst = apply_xy_shift(vol, 2.0, 3.0, output_shape=(12, 12))
    # dest starts at (dy=3, dx=2) in (y_start, y_end, x_start, x_end)
    assert dst[0] == 3                # y_start
    assert dst[2] == 2                # x_start
    assert cropped.shape[1] == 8
    assert cropped.shape[2] == 8


def test_apply_xy_shift_negative_clips_src():
    vol = _make_vol((4, 10, 10))
    # Shift by -2 in both dims: source crops 2 from start
    cropped, dst = apply_xy_shift(vol, -2.0, -2.0, output_shape=(10, 10))
    assert cropped is not None
    assert dst[0] == 0   # clamped to canvas start
    assert cropped.shape[1] == 8   # 2 rows clipped


def test_apply_xy_shift_fully_outside_canvas():
    vol = _make_vol((4, 8, 8))
    cropped, dst = apply_xy_shift(vol, 100.0, 100.0, output_shape=(10, 10))
    assert cropped is None
    assert dst is None


# ---------------------------------------------------------------------------
# blend_overlap_z
# ---------------------------------------------------------------------------

def test_blend_overlap_z_output_shape():
    fixed = _make_vol((5, 8, 8), fill=1.0)
    moving = _make_vol((5, 8, 8), fill=2.0)
    result = blend_overlap_z(fixed, moving)
    assert result.shape == fixed.shape


def test_blend_overlap_z_both_valid():
    """With both regions non-zero, result is between fixed and moving."""
    fixed = np.ones((6, 8, 8), dtype=np.float32)
    moving = np.full((6, 8, 8), 3.0, dtype=np.float32)
    result = blend_overlap_z(fixed, moving)
    assert float(result.min()) >= 1.0
    assert float(result.max()) <= 3.0


def test_blend_overlap_z_one_sided_fixed_only():
    """When moving is zero, fixed values are preserved."""
    fixed = np.ones((6, 8, 8), dtype=np.float32)
    moving = np.zeros((6, 8, 8), dtype=np.float32)
    result = blend_overlap_z(fixed, moving)
    np.testing.assert_allclose(result[fixed > 0], 1.0)


def test_blend_overlap_z_one_sided_moving_only():
    """When fixed is zero, moving values are preserved."""
    fixed = np.zeros((6, 8, 8), dtype=np.float32)
    moving = np.ones((6, 8, 8), dtype=np.float32)
    result = blend_overlap_z(fixed, moving)
    np.testing.assert_allclose(result[moving > 0], 1.0)


def test_blend_overlap_z_single_slice():
    """Single z-slice edge case: picks the region with more non-zero voxels."""
    fixed = np.ones((1, 8, 8), dtype=np.float32)
    moving = np.zeros((1, 8, 8), dtype=np.float32)
    result = blend_overlap_z(fixed, moving)
    assert result.shape == (1, 8, 8)


# ---------------------------------------------------------------------------
# blend_overlap_xy
# ---------------------------------------------------------------------------

def test_blend_overlap_xy_none_overwrites():
    existing = np.ones((4, 8, 8), dtype=np.float32)
    new_data = np.full((4, 8, 8), 5.0, dtype=np.float32)
    result = blend_overlap_xy(existing.copy(), new_data, method='none')
    np.testing.assert_allclose(result, 5.0)


def test_blend_overlap_xy_average():
    existing = np.ones((4, 8, 8), dtype=np.float32)
    new_data = np.full((4, 8, 8), 3.0, dtype=np.float32)
    result = blend_overlap_xy(existing.copy(), new_data, method='average')
    np.testing.assert_allclose(result, 2.0)


def test_blend_overlap_xy_max():
    existing = np.ones((4, 8, 8), dtype=np.float32)
    new_data = np.full((4, 8, 8), 3.0, dtype=np.float32)
    result = blend_overlap_xy(existing.copy(), new_data, method='max')
    np.testing.assert_allclose(result, 3.0)


def test_blend_overlap_xy_average_respects_zeros():
    """Pixels zero in existing (no data) should take new_data value."""
    existing = np.zeros((4, 8, 8), dtype=np.float32)
    new_data = np.full((4, 8, 8), 2.0, dtype=np.float32)
    result = blend_overlap_xy(existing.copy(), new_data, method='average')
    np.testing.assert_allclose(result, 2.0)
