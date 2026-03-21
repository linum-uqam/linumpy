# -*- coding: utf-8 -*-
"""Tests for linumpy/utils/shifts.py"""
import io

import numpy as np
import pandas as pd
import pytest

from linumpy.utils.shifts import (
    build_cumulative_shifts,
    center_shifts,
    convert_shifts_to_pixels,
    correct_tile_offset_shifts,
    detect_shift_units,
    filter_outlier_shifts,
    filter_step_outliers,
    load_shifts_csv,
)


# ---------------------------------------------------------------------------
# detect_shift_units
# ---------------------------------------------------------------------------

def test_detect_shift_units_mm():
    """Values < 1 are treated as mm and converted to µm."""
    res_x, res_y = detect_shift_units([0.006, 0.01, 0.01])
    assert abs(res_x - 10.0) < 1e-9
    assert abs(res_y - 10.0) < 1e-9


def test_detect_shift_units_um():
    """Values >= 1 are treated as µm and returned as-is."""
    res_x, res_y = detect_shift_units([6.5, 10.0, 10.0])
    assert abs(res_x - 10.0) < 1e-9
    assert abs(res_y - 10.0) < 1e-9


def test_detect_shift_units_single_element():
    """With a single resolution value both x and y are equal."""
    res_x, res_y = detect_shift_units([10.0])
    assert res_x == res_y == 10.0


# ---------------------------------------------------------------------------
# convert_shifts_to_pixels
# ---------------------------------------------------------------------------

def test_convert_shifts_to_pixels_basic():
    cumsum_mm = {0: (0.0, 0.0), 1: (0.01, 0.02), 2: (0.03, 0.04)}
    px = convert_shifts_to_pixels(cumsum_mm, resolution_um=10.0)
    assert abs(px[1][0] - 1.0) < 1e-6    # 0.01 mm / (10 µm/px) = 1 px
    assert abs(px[1][1] - 2.0) < 1e-6
    assert abs(px[2][0] - 3.0) < 1e-6


# ---------------------------------------------------------------------------
# center_shifts
# ---------------------------------------------------------------------------

def test_center_shifts_middle_is_zero():
    cumsum_px = {0: (0.0, 0.0), 1: (10.0, 5.0), 2: (20.0, 10.0)}
    slice_ids = [0, 1, 2]
    centered = center_shifts(cumsum_px, slice_ids)
    # Middle slice (index 1) must be (0, 0)
    assert centered[1] == (0.0, 0.0)
    assert centered[0] == (-10.0, -5.0)
    assert centered[2] == (10.0, 5.0)


def test_center_shifts_empty():
    assert center_shifts({}, []) == {}


# ---------------------------------------------------------------------------
# filter_outlier_shifts
# ---------------------------------------------------------------------------

def _make_shifts_df(x_shifts, y_shifts):
    n = len(x_shifts)
    return pd.DataFrame({
        'fixed_id': list(range(n)),
        'moving_id': list(range(1, n + 1)),
        'x_shift_mm': x_shifts,
        'y_shift_mm': y_shifts,
    })


def test_filter_outlier_shifts_no_outliers():
    df = _make_shifts_df([0.01] * 10, [0.01] * 10)
    result = filter_outlier_shifts(df, max_shift_mm=0.5, method='clamp')
    pd.testing.assert_frame_equal(result, df)


def test_filter_outlier_shifts_clamp():
    x = [0.01] * 9 + [1.0]   # last entry is huge
    y = [0.01] * 9 + [1.0]
    df = _make_shifts_df(x, y)
    result = filter_outlier_shifts(df, max_shift_mm=0.1, method='clamp')
    mag = np.sqrt(result['x_shift_mm'] ** 2 + result['y_shift_mm'] ** 2)
    assert float(mag.iloc[-1]) <= 0.1 + 1e-6


def test_filter_outlier_shifts_zero():
    x = [0.01] * 9 + [1.0]
    y = [0.01] * 9 + [1.0]
    df = _make_shifts_df(x, y)
    result = filter_outlier_shifts(df, max_shift_mm=0.1, method='zero')
    assert result['x_shift_mm'].iloc[-1] == 0.0
    assert result['y_shift_mm'].iloc[-1] == 0.0


def test_filter_outlier_shifts_median():
    x = [0.01] * 9 + [1.0]
    y = [0.01] * 9 + [1.0]
    df = _make_shifts_df(x, y)
    result = filter_outlier_shifts(df, max_shift_mm=0.1, method='median')
    assert abs(result['x_shift_mm'].iloc[-1] - 0.01) < 1e-6


def test_filter_outlier_shifts_iqr():
    x = [0.01] * 9 + [1.0]
    y = [0.01] * 9 + [1.0]
    df = _make_shifts_df(x, y)
    result = filter_outlier_shifts(df, max_shift_mm=0.5, method='iqr')
    # The spike at 1.0 should be replaced by local median ≈ 0.01
    mag = np.sqrt(result['x_shift_mm'].iloc[-1] ** 2 + result['y_shift_mm'].iloc[-1] ** 2)
    assert mag < 0.5


# ---------------------------------------------------------------------------
# filter_step_outliers
# ---------------------------------------------------------------------------

def test_filter_step_outliers_local_mad_detects_spike():
    x = [0.01] * 5 + [0.5] + [0.01] * 5
    y = [0.0] * 11
    df = _make_shifts_df(x, y)
    result = filter_step_outliers(df, method='local_mad', mad_threshold=3.0)
    # Spike at index 5 should have been replaced
    assert result['x_shift_mm'].iloc[5] < 0.5


def test_filter_step_outliers_no_outliers():
    x = [0.01] * 10
    y = [0.01] * 10
    df = _make_shifts_df(x, y)
    result = filter_step_outliers(df, method='local_mad', mad_threshold=3.0)
    pd.testing.assert_frame_equal(result, df)


def test_filter_step_outliers_clamp():
    x = [0.01] * 5 + [1.0] + [0.01] * 5
    y = [0.0] * 11
    df = _make_shifts_df(x, y)
    result = filter_step_outliers(df, max_step_mm=0.1, method='clamp')
    mag = abs(result['x_shift_mm'].iloc[5])
    assert float(mag) <= 0.1 + 1e-6


def test_filter_step_outliers_zero_max_step_returns_unchanged():
    """max_step_mm=0 disables filtering for non-mad methods."""
    x = [0.01] * 5 + [1.0] + [0.01] * 5
    y = [0.0] * 11
    df = _make_shifts_df(x, y)
    result = filter_step_outliers(df, max_step_mm=0, method='clamp')
    # Should be unchanged when max_step is 0
    assert result['x_shift_mm'].iloc[5] == 1.0


# ---------------------------------------------------------------------------
# load_shifts_csv & build_cumulative_shifts
# ---------------------------------------------------------------------------

def test_load_shifts_csv(tmp_path):
    csv_content = "fixed_id,moving_id,x_shift_mm,y_shift_mm\n0,1,0.01,0.02\n1,2,0.01,0.02\n"
    csv_path = tmp_path / "shifts.csv"
    csv_path.write_text(csv_content)
    cumsum, all_ids = load_shifts_csv(str(csv_path))
    assert all_ids == [0, 1, 2]
    assert cumsum[0] == (0.0, 0.0)
    assert abs(cumsum[1][0] - 0.01) < 1e-9
    assert abs(cumsum[2][0] - 0.02) < 1e-9


def test_build_cumulative_shifts(tmp_path):
    df = pd.DataFrame({
        'fixed_id': [0, 1, 2],
        'moving_id': [1, 2, 3],
        'x_shift_mm': [0.01, 0.01, 0.01],
        'y_shift_mm': [0.0, 0.0, 0.0],
    })
    # resolution: 10 µm → 1 mm = 100 px
    result = build_cumulative_shifts(df, [0, 1, 2, 3], resolution=[0.01, 0.01, 0.01],
                                     center_drift=False)
    assert abs(result[0][0] - 0.0) < 1e-6      # first slice: 0 px
    assert abs(result[1][0] - 1.0) < 1e-6      # 0.01 mm × 100 px/mm = 1 px
    assert abs(result[3][0] - 3.0) < 1e-6


def test_build_cumulative_shifts_center_drift():
    df = pd.DataFrame({
        'fixed_id': [0, 1, 2, 3],
        'moving_id': [1, 2, 3, 4],
        'x_shift_mm': [0.01, 0.01, 0.01, 0.01],
        'y_shift_mm': [0.0, 0.0, 0.0, 0.0],
    })
    slice_ids = [0, 1, 2, 3, 4]
    result = build_cumulative_shifts(df, slice_ids, resolution=[0.01, 0.01, 0.01],
                                     center_drift=True)
    # Middle slice (index 2) should be (0, 0)
    assert result[2][0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# correct_tile_offset_shifts
# ---------------------------------------------------------------------------

def test_correct_tile_offset_single_step():
    """A step of exactly 1 tile-width is subtracted to recover near-zero drift."""
    tile_fov = 0.875
    x = [0.02] * 5 + [tile_fov + 0.01] + [0.02] * 5   # one exact-tile step
    y = [0.0] * 11
    df = _make_shifts_df(x, y)
    result, corrected = correct_tile_offset_shifts(df, tile_fov_x_mm=tile_fov)
    assert 5 in corrected
    # Corrected value should be the residual only (~0.01 mm)
    assert abs(result['x_shift_mm'].iloc[5] - 0.01) < 1e-6
    # Other rows unchanged
    assert all(abs(result['x_shift_mm'].iloc[i] - 0.02) < 1e-9 for i in range(5))


def test_correct_tile_offset_negative_multiple():
    """A step of -2 tile-widths is corrected by adding 2 × tile_fov."""
    tile_fov = 0.875
    x = [0.02] * 5 + [-2 * tile_fov + 0.005] + [0.02] * 5
    y = [0.0] * 11
    df = _make_shifts_df(x, y)
    result, corrected = correct_tile_offset_shifts(df, tile_fov_x_mm=tile_fov)
    assert 5 in corrected
    assert abs(result['x_shift_mm'].iloc[5] - 0.005) < 1e-5


def test_correct_tile_offset_consecutive_steps():
    """Three consecutive +1-tile steps are each individually corrected."""
    tile_fov = 0.875
    # Simulates sub-22: slices 21-22, 22-23, 23-24 all have +tile_fov step
    x = [0.02] * 3 + [tile_fov + 0.015, tile_fov + 0.02, tile_fov + 0.01] + [0.02] * 3
    y = [0.0] * 9
    df = _make_shifts_df(x, y)
    result, corrected = correct_tile_offset_shifts(df, tile_fov_x_mm=tile_fov)
    assert sorted(corrected) == [3, 4, 5]
    assert abs(result['x_shift_mm'].iloc[3] - 0.015) < 1e-5
    assert abs(result['x_shift_mm'].iloc[4] - 0.020) < 1e-5
    assert abs(result['x_shift_mm'].iloc[5] - 0.010) < 1e-5


def test_correct_tile_offset_no_correction_for_normal_step():
    """Steps well below tile_fov are not modified."""
    tile_fov = 0.875
    x = [0.3, 0.1, 0.4, 0.05]
    y = [0.0] * 4
    df = _make_shifts_df(x, y)
    result, corrected = correct_tile_offset_shifts(df, tile_fov_x_mm=tile_fov)
    assert corrected == []
    pd.testing.assert_frame_equal(result, df)


def test_correct_tile_offset_updates_pixel_column():
    """The pixel-unit x_shift column is updated in proportion."""
    tile_fov = 0.875
    px_per_mm = 170.0
    x_mm = [tile_fov + 0.02]   # 1 tile + small drift
    x_px = [x_mm[0] * px_per_mm]
    df = pd.DataFrame({
        'fixed_id': [0], 'moving_id': [1],
        'x_shift_mm': x_mm, 'y_shift_mm': [0.0],
        'x_shift': x_px, 'y_shift': [0.0],
    })
    result, corrected = correct_tile_offset_shifts(df, tile_fov_x_mm=tile_fov)
    assert 0 in corrected
    expected_mm = 0.02
    expected_px = 0.02 * px_per_mm
    assert abs(result['x_shift_mm'].iloc[0] - expected_mm) < 1e-5
    assert abs(result['x_shift'].iloc[0] - expected_px) < 1e-2
