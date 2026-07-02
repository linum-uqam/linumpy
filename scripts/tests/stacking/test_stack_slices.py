#!/usr/bin/env python3
"""Tests for ``scripts/stacking/linum_stack_slices_motor.py``.

The motor-stacking script is loaded via :mod:`importlib` so its pure helper
functions (canvas sizing, transform loading) can be characterized on small
synthetic inputs, without depending on real OME-Zarr slice volumes.
"""

import importlib.util
import json
from pathlib import Path

import SimpleITK as sitk

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "stacking" / "linum_stack_slices_motor.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("linum_stack_slices_motor", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_help(script_runner):
    ret = script_runner.run(["linum-stack-slices-2d", "--help"])
    assert ret.success


def test_stack_slices_motor_help(script_runner):
    ret = script_runner.run(["linum-stack-slices-motor", "--help"])
    assert ret.success


# ---------------------------------------------------------------------------
# compute_output_shape (canvas sizing)
# ---------------------------------------------------------------------------


class TestComputeOutputShape:
    def test_single_slice_no_shift(self):
        mod = _load_module()
        first_vol_shape = (10, 20, 30)  # Z, Y, X
        ny, nx, x0, y0 = mod.compute_output_shape({0: "slice0"}, {0: (0.0, 0.0)}, first_vol_shape)
        assert (ny, nx) == (20, 30)
        assert (x0, y0) == (0, 0)

    def test_two_slices_offset_expands_canvas(self):
        mod = _load_module()
        first_vol_shape = (10, 20, 30)
        cumsum_px = {0: (0.0, 0.0), 1: (5.0, -3.0)}
        ny, nx, x0, y0 = mod.compute_output_shape({0: "slice0", 1: "slice1"}, cumsum_px, first_vol_shape)
        assert nx == 35
        assert ny == 23
        assert x0 == 0
        assert y0 == -3


# ---------------------------------------------------------------------------
# load_registration_transforms
# ---------------------------------------------------------------------------


class TestLoadRegistrationTransforms:
    def _write_transform_dir(self, tmp_path: Path, metrics: dict | None) -> None:
        transform_dir = tmp_path / "slice_z01"
        transform_dir.mkdir()
        sitk.WriteTransform(sitk.Euler2DTransform(), str(transform_dir / "transform.tfm"))
        (transform_dir / "offsets.txt").write_text("5\n10\n")
        if metrics is not None:
            (transform_dir / "pairwise_registration_metrics.json").write_text(json.dumps(metrics))

    def test_loads_transform_offsets_and_confidence(self, tmp_path: Path):
        mod = _load_module()
        metrics = {
            "overall_status": "ok",
            "metrics": {
                "registration_confidence": {"value": 0.9},
                "translation_x": {"value": 1.5},
                "translation_y": {"value": -2.0},
                "z_correlation": {"value": 0.8},
                "rotation": {"value": 0.1},
            },
        }
        self._write_transform_dir(tmp_path, metrics)

        transforms, pairwise = mod.load_registration_transforms(tmp_path, [0, 1])

        assert transforms[1] is not None
        _tfm, fixed_z, moving_z, confidence = transforms[1]
        assert fixed_z == 5
        assert moving_z == 10
        assert abs(confidence - 0.9) < 1e-9
        assert pairwise[1] == (1.5, -2.0, 0.8)

    def test_missing_transform_dir_returns_none(self, tmp_path: Path):
        mod = _load_module()
        transforms, pairwise = mod.load_registration_transforms(tmp_path, [0, 1])
        assert transforms[1] is None
        assert pairwise == {}

    def test_skip_error_status_discards_transform(self, tmp_path: Path):
        mod = _load_module()
        self._write_transform_dir(tmp_path, {"overall_status": "error", "metrics": {}})

        transforms, _pairwise = mod.load_registration_transforms(tmp_path, [0, 1], skip_error_status=True)

        assert transforms[1] is None
