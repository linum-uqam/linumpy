# -*- coding: utf-8 -*-
"""Tests for linumpy/stitching/motor.py"""
import json

import numpy as np
import pytest

from linumpy.stitching.motor import (
    apply_blend_shift_refinement,
    compare_motor_vs_registration,
    compute_motor_positions,
)


# ---------------------------------------------------------------------------
# compute_motor_positions
# ---------------------------------------------------------------------------

def test_compute_motor_positions_count():
    positions, step_y, step_x = compute_motor_positions(
        nx=3, ny=4, tile_shape=(10, 64, 64), overlap_fraction=0.1)
    assert len(positions) == 12   # 3 × 4


def test_compute_motor_positions_step_sizes():
    tile_shape = (10, 100, 80)
    overlap = 0.2
    positions, step_y, step_x = compute_motor_positions(
        nx=2, ny=2, tile_shape=tile_shape, overlap_fraction=overlap)
    expected_step_y = int(100 * (1 - overlap))  # 80
    expected_step_x = int(80 * (1 - overlap))   # 64
    assert step_y == expected_step_y
    assert step_x == expected_step_x


def test_compute_motor_positions_first_is_origin():
    positions, _, _ = compute_motor_positions(
        nx=2, ny=3, tile_shape=(5, 50, 50), overlap_fraction=0.1)
    first = positions[0]
    assert first[0] == 0
    assert first[1] == 0


def test_compute_motor_positions_scale_factor():
    tile_shape = (10, 100, 100)
    positions_1x, step_y_1x, _ = compute_motor_positions(
        nx=2, ny=1, tile_shape=tile_shape, overlap_fraction=0.0, scale_factor=1.0)
    positions_2x, step_y_2x, _ = compute_motor_positions(
        nx=2, ny=1, tile_shape=tile_shape, overlap_fraction=0.0, scale_factor=2.0)
    assert step_y_2x == 2 * step_y_1x


# ---------------------------------------------------------------------------
# apply_blend_shift_refinement
# ---------------------------------------------------------------------------

def test_apply_blend_shift_refinement_empty_refinements():
    """No refinements → tile returned unchanged."""
    tile = np.ones((5, 16, 16), dtype=np.float32)
    result = apply_blend_shift_refinement(tile, [], overlap_fraction=0.1)
    np.testing.assert_array_equal(result, tile)


def test_apply_blend_shift_refinement_negligible_shift():
    """Sub-threshold shifts (< 0.1 px) → tile returned unchanged."""
    tile = np.ones((5, 16, 16), dtype=np.float32)
    refinements = [{'dx': 0.05, 'dy': 0.05}]
    result = apply_blend_shift_refinement(tile, refinements, overlap_fraction=0.1)
    np.testing.assert_array_equal(result, tile)


def test_apply_blend_shift_refinement_applies_shift():
    """Large shift is applied, changing the tile data."""
    rng = np.random.default_rng(7)
    tile = (rng.random((5, 32, 32)) * 100.0).astype(np.float32)
    refinements = [{'dx': 3.0, 'dy': 3.0}]
    result = apply_blend_shift_refinement(tile, refinements, overlap_fraction=0.2)
    # Shape must be preserved
    assert result.shape == tile.shape
    # Content must have changed
    assert not np.array_equal(result, tile)


def test_apply_blend_shift_refinement_averages_multiple():
    """Multiple refinements are averaged before application."""
    tile = (np.ones((5, 32, 32)) * 50.0).astype(np.float32)
    # Two opposite shifts → average ≈ 0 → no change (may not be exact due to shift)
    refinements = [{'dx': 0.0, 'dy': 4.0}, {'dx': 0.0, 'dy': -4.0}]
    result = apply_blend_shift_refinement(tile, refinements, overlap_fraction=0.1)
    # Average dy = 0 / 2 / 2 = 0 → negligible → should be unchanged
    np.testing.assert_array_equal(result, tile)


# ---------------------------------------------------------------------------
# compare_motor_vs_registration
# ---------------------------------------------------------------------------

def test_compare_motor_vs_registration_basic():
    motor = [(0, 0), (10, 0), (0, 10), (10, 10)]
    reg = [(1, 1), (11, 1), (1, 11), (11, 11)]
    result = compare_motor_vs_registration(motor, reg)
    assert result['n_tiles'] == 4
    assert abs(result['mean_diff_y'] - 1.0) < 1e-9
    assert abs(result['mean_diff_x'] - 1.0) < 1e-9
    assert result['systematic_offset'] is False   # only 1 px offset, threshold is 5


def test_compare_motor_vs_registration_systematic_offset():
    motor = [(0, 0)] * 5
    reg = [(10, 10)] * 5     # 10 px systematic offset
    result = compare_motor_vs_registration(motor, reg)
    assert result['systematic_offset'] is True
    assert 'offset_warning' in result


def test_compare_motor_vs_registration_writes_json(tmp_path):
    motor = [(0, 0), (10, 0)]
    reg = [(1, 0), (11, 0)]
    out_path = str(tmp_path / "comparison.json")
    result = compare_motor_vs_registration(motor, reg, output_path=out_path)
    with open(out_path) as f:
        loaded = json.load(f)
    assert loaded['n_tiles'] == 2
    assert abs(loaded['mean_diff_y'] - 1.0) < 1e-9


def test_compare_motor_vs_registration_no_dilation_flag():
    """Fewer than 10 tiles: no dilation_indicator key."""
    motor = [(0, 0), (10, 0)]
    reg = [(0, 0), (10, 0)]
    result = compare_motor_vs_registration(motor, reg)
    assert 'dilation_indicator' not in result
