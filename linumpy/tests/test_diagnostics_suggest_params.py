"""Tests for :mod:`linumpy.diagnostics.suggest_params` (library functions, direct import)."""

import argparse
import json

import numpy as np
import pandas as pd
import pytest

from linumpy.diagnostics.suggest_params import (
    OCT_AXIAL_RES_UM,
    analyze_metadata,
    analyze_shifts,
    build_config_snippet,
    build_report,
    ceil_to,
    detect_rehoming,
    detect_slice_gaps,
    load_shifts,
    suggest_target_resolution,
)


@pytest.fixture
def shift_stats():
    """Minimal synthetic shift-analysis result (no gaps, no re-homing)."""
    df = pd.DataFrame(
        {
            "fixed_id": [0, 1, 2],
            "moving_id": [1, 2, 3],
            "x_shift_mm": [0.01, 0.02, 0.015],
            "y_shift_mm": [0.01, 0.015, 0.012],
        }
    )
    return analyze_shifts(df)


def test_oct_axial_res_um_default():
    assert OCT_AXIAL_RES_UM == 3.5


def test_load_shifts_missing_columns(tmp_path):
    csv_path = tmp_path / "shifts_xy.csv"
    csv_path.write_text("fixed_id,moving_id,x_shift_mm\n0,1,0.01\n")
    with pytest.raises(ValueError, match="Missing columns"):
        load_shifts(csv_path)


def test_detect_rehoming_mad_outlier():
    mag = np.array([0.1, 0.12, 0.11, 2.0])
    flags = detect_rehoming(mag)
    assert flags[-1]
    assert not flags[:-1].any()


def test_detect_slice_gaps():
    df = pd.DataFrame({"fixed_id": [0, 2], "moving_id": [1, 3]})
    gaps = detect_slice_gaps(df)
    assert gaps == []
    df_gap = pd.DataFrame({"fixed_id": [0], "moving_id": [3]})
    gaps = detect_slice_gaps(df_gap)
    assert gaps == [{"fixed_id": 0, "moving_id": 3, "n_missing": 2}]


def test_analyze_shifts_shape(shift_stats):
    assert shift_stats["has_rehoming"] is False
    assert shift_stats["has_gaps"] is False
    assert shift_stats["max_shift_mm"] >= 0


def test_analyze_metadata_overlap(tmp_path):
    (tmp_path / "state.json").write_text(json.dumps({"overlap_fraction": 0.25}))
    result = analyze_metadata(tmp_path, axial_res_um=3.5)
    assert result["ok"] is True
    assert result["overlap_fraction"] == 0.25


def test_analyze_metadata_malformed_json(tmp_path):
    (tmp_path / "state.json").write_text("{not valid json")
    result = analyze_metadata(tmp_path, axial_res_um=3.5)
    assert result["ok"] is False
    assert "error" in result


def test_ceil_to():
    assert ceil_to(0.31, 0.05) == pytest.approx(0.35)


def test_suggest_target_resolution():
    assert suggest_target_resolution(4.0) == 5
    assert suggest_target_resolution(60.0) == 25


def test_build_config_snippet_overlap(shift_stats):
    acq = {"ok": True, "overlap_fraction": 0.18}
    args = argparse.Namespace(resolution_um=None)
    snippet = build_config_snippet(shift_stats, acq, args)
    assert "stitch_overlap_fraction = 0.18" in snippet
    assert "initial_overlap = 0.18" in snippet


def test_build_report_no_rehoming_no_gaps(shift_stats):
    report = build_report(shift_stats, {}, "shifts_xy.csv")
    assert "NO RE-HOMING EVENTS DETECTED" in report
    assert "NO MISSING SLICES DETECTED" in report
