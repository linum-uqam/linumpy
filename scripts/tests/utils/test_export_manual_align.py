#!/usr/bin/env python3
"""Tests for ``scripts/utils/linum_export_manual_align.py``.

The script is loaded via :mod:`importlib` so we can test its pure-Python
helper functions (AIP save/load, discovery, offsets parsing) without relying
on the console entry point. Locks behavior before extraction to
``linumpy/registration/manual.py`` (D-85).
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "utils" / "linum_export_manual_align.py"


@pytest.fixture(scope="module")
def export_module():
    """Load ``linum-export-manual-align`` as a module."""
    spec = importlib.util.spec_from_file_location("linum_export_manual_align", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# CLI help
# ---------------------------------------------------------------------------


def test_help(script_runner):
    ret = script_runner.run(["linum-export-manual-align", "--help"])
    assert ret.success


# ---------------------------------------------------------------------------
# _is_interpolated
# ---------------------------------------------------------------------------


def test_is_interpolated_detects_suffix(export_module):
    assert export_module._is_interpolated(Path("slice_z01_interpolated.ome.zarr"))
    assert not export_module._is_interpolated(Path("slice_z01.ome.zarr"))


# ---------------------------------------------------------------------------
# _discover_slices / _discover_transforms
# ---------------------------------------------------------------------------


def test_discover_slices_matches_pattern(export_module, tmp_path):
    (tmp_path / "slice_z00.ome.zarr").mkdir()
    (tmp_path / "slice_z01_interpolated.ome.zarr").mkdir()
    (tmp_path / "not_a_slice.txt").touch()

    slices = export_module._discover_slices(tmp_path)

    assert set(slices.keys()) == {0, 1}
    assert slices[0].name == "slice_z00.ome.zarr"
    assert slices[1].name == "slice_z01_interpolated.ome.zarr"


def test_discover_transforms_matches_directories(export_module, tmp_path):
    (tmp_path / "slice_z01").mkdir()
    (tmp_path / "slice_z02").mkdir()
    (tmp_path / "not_a_dir_marker.txt").touch()

    transforms = export_module._discover_transforms(tmp_path)

    assert set(transforms.keys()) == {1, 2}


# ---------------------------------------------------------------------------
# _read_overlap_z_offsets
# ---------------------------------------------------------------------------


def test_read_overlap_z_offsets_missing_file_returns_zeros(export_module, tmp_path):
    assert export_module._read_overlap_z_offsets(tmp_path / "missing.txt") == (0, 0)


def test_read_overlap_z_offsets_reads_values(export_module, tmp_path):
    offsets_file = tmp_path / "offsets.txt"
    offsets_file.write_text("3 7\n")
    assert export_module._read_overlap_z_offsets(offsets_file) == (3, 7)


def test_read_overlap_z_offsets_invalid_content_returns_zeros(export_module, tmp_path):
    offsets_file = tmp_path / "offsets.txt"
    offsets_file.write_text("not,a,number\n")
    assert export_module._read_overlap_z_offsets(offsets_file) == (0, 0)


# ---------------------------------------------------------------------------
# End-to-end smoke: minimal slices_dir/transforms_dir/output_dir run
# ---------------------------------------------------------------------------


def test_execution_writes_metadata(script_runner, tmp_path):
    import dask.array as da

    from linumpy.io.zarr import save_omezarr

    shape = (4, 16, 16)
    resolution = (0.005, 0.01, 0.01)
    vol0 = np.random.rand(*shape).astype(np.float32)
    vol1 = np.random.rand(*shape).astype(np.float32)

    slices_dir = tmp_path / "slices"
    slices_dir.mkdir()
    save_omezarr(da.from_array(vol0), slices_dir / "slice_z00.ome.zarr", resolution)
    save_omezarr(da.from_array(vol1), slices_dir / "slice_z01.ome.zarr", resolution)

    transforms_dir = tmp_path / "transforms"
    transforms_dir.mkdir()

    output_dir = tmp_path / "output"

    ret = script_runner.run(
        [
            "linum-export-manual-align",
            str(slices_dir),
            str(transforms_dir),
            str(output_dir),
        ]
    )
    assert ret.success, ret.stderr

    metadata_path = output_dir / "manual_align_metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["n_slices"] == 2
    assert metadata["slice_ids"] == [0, 1]
