#!/usr/bin/env python3
"""Characterization tests for ``scripts/diagnostics/linum_analyze_acquisition_rotation.py``.

The script is loaded via :mod:`importlib` so we can test its pure-Python
helper functions without relying on the console entry point (see
``scripts/tests/stitching/test_align_to_ras.py`` for the established pattern).

These tests lock the CURRENT rotation-estimation math (shift angle,
angular velocity, cumulative rotation, pattern detection) before the core
logic is extracted into ``linumpy.diagnostics.acquisition_rotation`` (D-85).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "diagnostics" / "linum_analyze_acquisition_rotation.py"


@pytest.fixture(scope="module")
def mod():
    """Load ``linum-analyze-acquisition-rotation`` as a module."""
    spec = importlib.util.spec_from_file_location("linum_analyze_acquisition_rotation", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def shifts_df():
    """Synthetic shifts with a consistent ~45 degree direction (no rotation).

    Uses more rows than the default ``window_size=5`` smoothing window so
    ``np.convolve(..., mode="same")`` returns an array matching the input
    length (current script behavior when len(df) < window_size returns a
    longer array, which is not exercised here).
    """
    return pd.DataFrame(
        {
            "fixed_id": [0, 1, 2, 3, 4, 5],
            "moving_id": [1, 2, 3, 4, 5, 6],
            "x_shift_mm": [0.01, 0.011, 0.0105, 0.0102, 0.0108, 0.0099],
            "y_shift_mm": [0.01, 0.0095, 0.0103, 0.0099, 0.0101, 0.0097],
        }
    )


# ---------------------------------------------------------------------------
# CLI help
# ---------------------------------------------------------------------------


def test_help(script_runner):
    ret = script_runner.run(["linum-analyze-acquisition-rotation", "--help"])
    assert ret.success


# ---------------------------------------------------------------------------
# load_shifts
# ---------------------------------------------------------------------------


def test_load_shifts_requires_columns(mod, tmp_path):
    csv_path = tmp_path / "shifts_xy.csv"
    csv_path.write_text("fixed_id,moving_id,x_shift_mm\n0,1,0.01\n")
    with pytest.raises(ValueError, match="Missing required column"):
        mod.load_shifts(csv_path)


def test_load_shifts_ok(mod, tmp_path, shifts_df):
    csv_path = tmp_path / "shifts_xy.csv"
    shifts_df.to_csv(csv_path, index=False)
    df = mod.load_shifts(csv_path)
    assert len(df) == len(shifts_df)


# ---------------------------------------------------------------------------
# compute_shift_angles
# ---------------------------------------------------------------------------


def test_compute_shift_angles_45_degrees(mod):
    df = pd.DataFrame({"x_shift_mm": [1.0], "y_shift_mm": [1.0]})
    angles = mod.compute_shift_angles(df)
    np.testing.assert_allclose(angles.to_numpy(), [45.0], atol=1e-9)


# ---------------------------------------------------------------------------
# compute_angular_velocity
# ---------------------------------------------------------------------------


def test_compute_angular_velocity_constant_angle_is_zero(mod):
    angles = pd.Series([45.0, 45.0, 45.0, 45.0])
    raw, smoothed = mod.compute_angular_velocity(angles, window_size=1)
    np.testing.assert_allclose(raw, np.zeros(4), atol=1e-9)
    np.testing.assert_allclose(smoothed, np.zeros(4), atol=1e-9)


# ---------------------------------------------------------------------------
# compute_cumulative_rotation
# ---------------------------------------------------------------------------


def test_compute_cumulative_rotation_from_first_angle(mod):
    angles = pd.Series([10.0, 20.0, 30.0])
    cumulative = mod.compute_cumulative_rotation(angles)
    np.testing.assert_allclose(cumulative.to_numpy(), [0.0, 10.0, 20.0], atol=1e-9)


# ---------------------------------------------------------------------------
# detect_rotation_patterns
# ---------------------------------------------------------------------------


def test_detect_rotation_patterns_systematic_drift(mod):
    angular_velocity = np.full(10, 1.0)
    patterns = mod.detect_rotation_patterns(None, angular_velocity)
    assert patterns["systematic_drift"] is True
    assert patterns["drift_rate"] == pytest.approx(1.0)
    assert patterns["oscillation"] is False
    assert patterns["sudden_jumps"] == []


def test_detect_rotation_patterns_no_pattern(mod):
    angular_velocity = np.zeros(10)
    patterns = mod.detect_rotation_patterns(None, angular_velocity)
    assert patterns["systematic_drift"] is False
    assert patterns["oscillation"] is False
    assert patterns["sudden_jumps"] == []


def test_detect_rotation_patterns_sudden_jumps(mod):
    angular_velocity = np.zeros(10)
    angular_velocity[3] = 10.0
    patterns = mod.detect_rotation_patterns(None, angular_velocity)
    assert patterns["sudden_jumps"] == [3]


# ---------------------------------------------------------------------------
# analyze_acquisition_rotation (integration of pure helpers)
# ---------------------------------------------------------------------------


def test_analyze_acquisition_rotation_shape(mod, shifts_df):
    analysis, angles, angular_velocity, cumulative_rotation = mod.analyze_acquisition_rotation(shifts_df)
    assert analysis["n_shifts"] == len(shifts_df)
    assert len(angles) == len(shifts_df)
    assert len(angular_velocity) == len(shifts_df)
    assert len(cumulative_rotation) == len(shifts_df)
    assert "angle_stats" in analysis
    assert "magnitude_stats" in analysis
    assert "cumulative_rotation" in analysis
    assert "patterns" in analysis
    assert "interpretation" in analysis


def test_analyze_acquisition_rotation_expected_angle(mod, shifts_df):
    analysis, *_ = mod.analyze_acquisition_rotation(shifts_df, expected_angle=45.0)
    assert analysis["expected_angle"] == 45.0
    assert "mean_deviation_from_expected" in analysis


# ---------------------------------------------------------------------------
# compare_with_registration
# ---------------------------------------------------------------------------


def test_compare_with_registration_none_when_no_reg_df(mod, shifts_df):
    cumulative = mod.compute_cumulative_rotation(mod.compute_shift_angles(shifts_df))
    result = mod.compare_with_registration(cumulative, None, shifts_df["moving_id"].to_numpy())
    assert result is None
