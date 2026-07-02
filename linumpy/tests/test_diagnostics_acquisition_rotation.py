"""Tests for :mod:`linumpy.diagnostics.acquisition_rotation` (library functions, direct import)."""

import numpy as np
import pandas as pd
import pytest

from linumpy.diagnostics.acquisition_rotation import (
    compare_with_registration,
    compute_angular_velocity,
    compute_cumulative_rotation,
    compute_shift_angles,
    detect_rotation_patterns,
    load_shifts,
)


@pytest.fixture
def shifts_df():
    return pd.DataFrame(
        {
            "fixed_id": [0, 1, 2, 3, 4, 5],
            "moving_id": [1, 2, 3, 4, 5, 6],
            "x_shift_mm": [0.01, 0.011, 0.0105, 0.0102, 0.0108, 0.0099],
            "y_shift_mm": [0.01, 0.0095, 0.0103, 0.0099, 0.0101, 0.0097],
        }
    )


def test_load_shifts_requires_columns(tmp_path):
    csv_path = tmp_path / "shifts_xy.csv"
    csv_path.write_text("fixed_id,moving_id,x_shift_mm\n0,1,0.01\n")
    with pytest.raises(ValueError, match="Missing required column"):
        load_shifts(csv_path)


def test_compute_shift_angles_45_degrees():
    df = pd.DataFrame({"x_shift_mm": [1.0], "y_shift_mm": [1.0]})
    angles = compute_shift_angles(df)
    np.testing.assert_allclose(angles.to_numpy(), [45.0], atol=1e-9)


def test_compute_angular_velocity_constant_angle_is_zero():
    angles = pd.Series([45.0, 45.0, 45.0, 45.0])
    raw, smoothed = compute_angular_velocity(angles, window_size=1)
    np.testing.assert_allclose(raw, np.zeros(4), atol=1e-9)
    np.testing.assert_allclose(smoothed, np.zeros(4), atol=1e-9)


def test_compute_cumulative_rotation_from_first_angle():
    angles = pd.Series([10.0, 20.0, 30.0])
    cumulative = compute_cumulative_rotation(angles)
    np.testing.assert_allclose(cumulative.to_numpy(), [0.0, 10.0, 20.0], atol=1e-9)


def test_detect_rotation_patterns_systematic_drift():
    angular_velocity = np.full(10, 1.0)
    patterns = detect_rotation_patterns(None, angular_velocity)
    assert patterns["systematic_drift"] is True
    assert patterns["drift_rate"] == pytest.approx(1.0)


def test_detect_rotation_patterns_sudden_jumps():
    angular_velocity = np.zeros(10)
    angular_velocity[3] = 10.0
    patterns = detect_rotation_patterns(None, angular_velocity)
    assert patterns["sudden_jumps"] == [3]


def test_compare_with_registration_none_when_no_reg_df(shifts_df):
    cumulative = compute_cumulative_rotation(compute_shift_angles(shifts_df))
    result = compare_with_registration(cumulative, None, shifts_df["moving_id"].to_numpy())
    assert result is None
