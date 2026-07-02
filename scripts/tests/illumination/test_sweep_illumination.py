#!/usr/bin/env python3
"""Tests for ``scripts/illumination/linum_sweep_illumination.py``.

The script is loaded via :mod:`importlib` so we can test its pure-Python
helper functions (no ``zarr``/matplotlib I/O) without relying on the console
entry point. ``parse_float_none_list``/``parse_bool_list`` now live in
``linumpy/intensity/sweep.py`` (D-85) and are re-exported into the script's
namespace via the thin-CLI import; ``linumpy/tests/test_intensity_sweep.py``
exercises the library module directly.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "illumination" / "linum_sweep_illumination.py"


@pytest.fixture(scope="module")
def sweep_module():
    """Load ``linum-sweep-illumination`` as a module."""
    spec = importlib.util.spec_from_file_location("linum_sweep_illumination", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# CLI help
# ---------------------------------------------------------------------------


def test_help(script_runner):
    ret = script_runner.run(["linum-sweep-illumination", "--help"])
    assert ret.success


# ---------------------------------------------------------------------------
# parse_float_none_list (re-exported from linumpy.intensity.sweep)
# ---------------------------------------------------------------------------


class TestParseFloatNoneList:
    def test_parses_mixed_none_and_floats(self, sweep_module):
        result = sweep_module.parse_float_none_list("none,99.0,99.5")
        assert result == [None, 99.0, 99.5]

    def test_strips_whitespace(self, sweep_module):
        result = sweep_module.parse_float_none_list(" 1.0 , None , 2.5 ")
        assert result == [1.0, None, 2.5]

    def test_single_value(self, sweep_module):
        assert sweep_module.parse_float_none_list("0.1") == [0.1]


# ---------------------------------------------------------------------------
# parse_bool_list (re-exported from linumpy.intensity.sweep)
# ---------------------------------------------------------------------------


class TestParseBoolList:
    def test_parses_true_false(self, sweep_module):
        assert sweep_module.parse_bool_list("true,false") == [True, False]

    def test_accepts_numeric_and_word_aliases(self, sweep_module):
        assert sweep_module.parse_bool_list("1,0,yes,no") == [True, False, True, False]

    def test_invalid_token_raises(self, sweep_module):
        with pytest.raises(ValueError, match="Cannot parse bool value"):
            sweep_module.parse_bool_list("maybe")


# ---------------------------------------------------------------------------
# _display_vmax
# ---------------------------------------------------------------------------


class TestDisplayVmax:
    def test_uses_percentile_of_positive_values(self, sweep_module):
        arr = np.array([-5.0, 0.0, 1.0, 2.0, 3.0, 4.0])
        result = sweep_module._display_vmax(arr, p=100.0)
        assert result == pytest.approx(4.0)

    def test_all_nonpositive_returns_one(self, sweep_module):
        arr = np.array([-2.0, -1.0, 0.0])
        assert sweep_module._display_vmax(arr) == 1.0
