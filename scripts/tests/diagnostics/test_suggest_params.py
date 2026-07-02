#!/usr/bin/env python3
"""Characterization tests for ``scripts/diagnostics/linum_suggest_params.py``.

The script is loaded via :mod:`importlib` so we can test its pure-Python
helper functions without relying on the console entry point (see
``scripts/tests/stitching/test_align_to_ras.py`` for the established pattern).

These tests lock the CURRENT metadata-reading and config-snippet-emission
behavior (D-79/D-82) before the 2.5D ``initial_overlap`` emission and the
malformed-metadata hardening are added. Tests for behavior that does not
exist yet are marked ``xfail(strict=True)`` and are expected to flip to
passing once that behavior is implemented.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "diagnostics" / "linum_suggest_params.py"


@pytest.fixture(scope="module")
def mod():
    """Load ``linum-suggest-params`` as a module."""
    spec = importlib.util.spec_from_file_location("linum_suggest_params", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def shift_stats(mod):
    """Minimal synthetic shift-analysis result (no gaps, no re-homing)."""
    df = pd.DataFrame(
        {
            "fixed_id": [0, 1, 2],
            "moving_id": [1, 2, 3],
            "x_shift_mm": [0.01, 0.02, 0.015],
            "y_shift_mm": [0.01, 0.015, 0.012],
        }
    )
    return mod.analyze_shifts(df)


# ---------------------------------------------------------------------------
# CLI help
# ---------------------------------------------------------------------------


def test_help(script_runner):
    ret = script_runner.run(["linum-suggest-params", "--help"])
    assert ret.success


# ---------------------------------------------------------------------------
# detect_rehoming (MAD outlier rejection, D-85 precursor)
# ---------------------------------------------------------------------------


def test_detect_rehoming_mad_outlier(mod):
    mag = np.array([0.1, 0.12, 0.11, 2.0])
    flags = mod.detect_rehoming(mag)
    assert flags[-1]
    assert not flags[:-1].any()


# ---------------------------------------------------------------------------
# analyze_metadata: overlap_fraction read from state.json (D-79)
# ---------------------------------------------------------------------------


def test_analyze_metadata_overlap(mod, tmp_path):
    (tmp_path / "state.json").write_text(json.dumps({"overlap_fraction": 0.25}))
    result = mod.analyze_metadata(tmp_path, axial_res_um=3.5)
    assert result["ok"] is True
    assert result["overlap_fraction"] == 0.25


# ---------------------------------------------------------------------------
# build_config_snippet: 3D stitch_overlap_fraction emission (D-82, existing)
# ---------------------------------------------------------------------------


def test_build_config_snippet_overlap(mod, shift_stats):
    acq = {"ok": True, "overlap_fraction": 0.18}
    args = argparse.Namespace(resolution_um=None)
    snippet = mod.build_config_snippet(shift_stats, acq, args)
    assert "stitch_overlap_fraction = 0.18" in snippet
    assert "from overlap_fraction in metadata" in snippet


# ---------------------------------------------------------------------------
# build_config_snippet: 2.5D initial_overlap emission (D-82)
# ---------------------------------------------------------------------------


def test_build_config_snippet_initial_overlap(mod, shift_stats):
    acq = {"ok": True, "overlap_fraction": 0.18}
    args = argparse.Namespace(resolution_um=None)
    snippet = mod.build_config_snippet(shift_stats, acq, args)
    assert "initial_overlap = 0.18" in snippet


# ---------------------------------------------------------------------------
# analyze_metadata: malformed state.json handled with actionable error
# (Security DoS mitigation)
# ---------------------------------------------------------------------------


def test_malformed_metadata_actionable(mod, tmp_path):
    (tmp_path / "state.json").write_text("{not valid json")
    result = mod.analyze_metadata(tmp_path, axial_res_um=3.5)
    assert result["ok"] is False
    assert "error" in result
