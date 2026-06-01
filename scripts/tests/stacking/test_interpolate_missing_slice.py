#!/usr/bin/env python3
"""Script-level tests for linum_interpolate_missing_slice.py."""

import json

import dask.array as da
import numpy as np


def _make_structured_slice(shape, drift_px, seed):
    """Create a synthetic structured slice with a controlled XY drift."""
    nz, ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    rng = np.random.default_rng(seed)
    vol = np.zeros(shape, dtype=np.float32)
    for z in range(nz):
        depth = z / max(nz - 1, 1)
        cy = ny * 0.5 + drift_px
        cx = nx * 0.5
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * (ny / 5.0) ** 2))
        vol[z] = np.clip(
            0.1 + 0.6 * blob + 0.1 * np.sin((xx + 4.0 * depth) * 0.35) + rng.normal(0.0, 0.01, size=(ny, nx)), 0.0, None
        ).astype(np.float32)
    return vol


def _save_pair(tmp_path, shape=(8, 48, 48), drift_px=1.0):
    """Save synthetic slice_z00.ome.zarr / slice_z02.ome.zarr to tmp_path."""
    from linumpy.io.zarr import save_omezarr

    resolution = (0.005, 0.005, 0.005)
    before = _make_structured_slice(shape, -drift_px, seed=1)
    after = _make_structured_slice(shape, +drift_px, seed=2)
    slice_before = tmp_path / "slice_z00.ome.zarr"
    slice_after = tmp_path / "slice_z02.ome.zarr"
    save_omezarr(da.from_array(before), slice_before, resolution)
    save_omezarr(da.from_array(after), slice_after, resolution)
    return slice_before, slice_after


def test_help(script_runner):
    ret = script_runner.run(["linum_interpolate_missing_slice.py", "--help"])
    assert ret.success


def test_average_method(script_runner, tmp_path):
    """Test basic interpolation with average method using synthetic data."""
    from linumpy.io.zarr import save_omezarr

    shape = (10, 32, 32)
    resolution = (0.001, 0.01, 0.01)
    vol_before = np.random.rand(*shape).astype(np.float32) * 100
    vol_after = np.random.rand(*shape).astype(np.float32) * 100

    slice_before = tmp_path / "slice_z00.ome.zarr"
    slice_after = tmp_path / "slice_z02.ome.zarr"
    output = tmp_path / "slice_z01_interpolated.ome.zarr"

    save_omezarr(da.from_array(vol_before), slice_before, resolution)
    save_omezarr(da.from_array(vol_after), slice_after, resolution)

    ret = script_runner.run(
        ["linum_interpolate_missing_slice.py", str(slice_before), str(slice_after), str(output), "--method", "average"]
    )

    assert ret.success
    assert output.exists()


def test_zmorph_method_with_diagnostics_and_manifest(script_runner, tmp_path):
    """Run --method zmorph and assert diagnostics / manifest files are emitted."""
    slice_before, slice_after = _save_pair(tmp_path)
    output = tmp_path / "slice_z01_interpolated.ome.zarr"
    diagnostics = tmp_path / "diagnostics.json"
    manifest = tmp_path / "manifest.csv"

    ret = script_runner.run(
        [
            "linum_interpolate_missing_slice.py",
            str(slice_before),
            str(slice_after),
            str(output),
            "--method",
            "zmorph",
            "--max_iterations",
            "50",
            "--min_overlap_correlation",
            "0.0",
            "--min_ncc_improvement",
            "-10.0",
            "--slice_id",
            "01",
            "--diagnostics",
            str(diagnostics),
            "--manifest_entry",
            str(manifest),
        ]
    )

    assert ret.success, ret.stderr
    assert output.exists()
    assert diagnostics.exists()
    assert manifest.exists()

    with diagnostics.open() as fh:
        diag = json.load(fh)
    assert diag.get("method") == "zmorph"
    assert "pre_reg_ncc" in diag
    assert diag.get("slice_id") == "01"

    lines = manifest.read_text().strip().splitlines()
    assert len(lines) == 2
    header = lines[0].split(",")
    assert "slice_id" in header
    assert "method_used" in header
    assert "fallback_reason" in header


def test_zmorph_hard_skips_on_unrelated_volumes(script_runner, tmp_path):
    """Unrelated random volumes trigger a hard skip: no zarr, manifest flags failure."""
    from linumpy.io.zarr import save_omezarr

    shape = (6, 48, 48)
    resolution = (0.005, 0.005, 0.005)
    rng = np.random.default_rng(123)
    before = rng.random(shape).astype(np.float32)
    after = rng.random(shape).astype(np.float32)
    slice_before = tmp_path / "slice_z00.ome.zarr"
    slice_after = tmp_path / "slice_z02.ome.zarr"
    save_omezarr(da.from_array(before), slice_before, resolution)
    save_omezarr(da.from_array(after), slice_after, resolution)

    output = tmp_path / "slice_z01_interpolated.ome.zarr"
    diagnostics = tmp_path / "diag.json"
    manifest = tmp_path / "manifest.csv"

    ret = script_runner.run(
        [
            "linum_interpolate_missing_slice.py",
            str(slice_before),
            str(slice_after),
            str(output),
            "--method",
            "zmorph",
            "--max_iterations",
            "20",
            "--min_overlap_correlation",
            "0.99",
            "--min_ncc_improvement",
            "0.5",
            "--slice_id",
            "01",
            "--diagnostics",
            str(diagnostics),
            "--manifest_entry",
            str(manifest),
        ]
    )
    assert ret.success, ret.stderr

    with diagnostics.open() as fh:
        diag = json.load(fh)
    assert diag["interpolation_failed"] is True
    assert diag["method_used"] is None
    assert diag["fallback_reason"] in {
        "low_overlap_ncc",
        "no_foreground_planes",
        "reg_did_not_improve",
        "registration_exception",
        "affine_determinant_non_positive",
    }
    assert diag["output_path"] is None
    assert not output.exists(), "No zarr must be produced when zmorph gates fail"

    manifest_rows = manifest.read_text().strip().splitlines()
    assert len(manifest_rows) == 2
    header = manifest_rows[0].split(",")
    values = manifest_rows[1].split(",")
    row = dict(zip(header, values, strict=False))
    assert row["interpolation_failed"] == "true"
    assert row["method_used"] == ""
    assert row["output_path"] == ""


def test_finalise_merges_fragments_into_slice_config(script_runner, tmp_path):
    """--finalise stamps interpolated=true + method_used from fragment CSVs."""
    slice_config_in = tmp_path / "slice_config.csv"
    slice_config_in.write_text(
        "slice_id,use,quality_score\n00,true,0.9\n01,false,0.1\n02,true,0.8\n",
        encoding="utf-8",
    )

    fragments = tmp_path / "fragments"
    fragments.mkdir()
    (fragments / "slice_z01_manifest.csv").write_text(
        "slice_id,method,method_used,fallback_reason\n01,zmorph,zmorph,\n",
        encoding="utf-8",
    )

    slice_config_out = tmp_path / "slice_config_out.csv"

    ret = script_runner.run(
        [
            "linum_interpolate_missing_slice.py",
            "--finalise",
            "--slice_config_in",
            str(slice_config_in),
            "--slice_config_out",
            str(slice_config_out),
            "--fragments",
            str(fragments),
        ]
    )
    assert ret.success, ret.stderr
    assert slice_config_out.exists()

    from linumpy.io import slice_config as slice_config_io

    rows = slice_config_io.read(slice_config_out)
    assert rows["01"]["interpolated"] == "true"
    assert rows["01"]["interpolation_method_used"] == "zmorph"
    assert rows["00"].get("interpolated", "") == ""
    assert rows["02"].get("interpolated", "") == ""


def test_finalise_stamps_interpolation_failed(script_runner, tmp_path):
    """A hard-skip fragment stamps interpolation_failed=true, interpolated=false."""
    slice_config_in = tmp_path / "slice_config.csv"
    slice_config_in.write_text(
        "slice_id,use,quality_score\n00,true,0.9\n01,false,0.1\n02,true,0.8\n",
        encoding="utf-8",
    )

    fragments = tmp_path / "fragments"
    fragments.mkdir()
    (fragments / "slice_z01_manifest.csv").write_text(
        "slice_id,method,method_used,fallback_reason,interpolation_failed,output_path\n01,zmorph,,low_overlap_ncc,true,\n",
        encoding="utf-8",
    )

    slice_config_out = tmp_path / "slice_config_out.csv"

    ret = script_runner.run(
        [
            "linum_interpolate_missing_slice.py",
            "--finalise",
            "--slice_config_in",
            str(slice_config_in),
            "--slice_config_out",
            str(slice_config_out),
            "--fragments",
            str(fragments),
        ]
    )
    assert ret.success, ret.stderr

    from linumpy.io import slice_config as slice_config_io

    rows = slice_config_io.read(slice_config_out)
    assert rows["01"]["interpolated"] == "false"
    assert rows["01"]["interpolation_failed"] == "true"
    assert rows["01"]["interpolation_fallback_reason"] == "low_overlap_ncc"
    assert rows["01"]["interpolation_method_used"] == ""
    # Un-touched rows remain untouched
    assert rows["00"].get("interpolated", "") == ""
    assert rows["00"].get("interpolation_failed", "") == ""


def test_finalise_no_fragments_copies_slice_config_unchanged(script_runner, tmp_path):
    """Empty fragments directory is a valid (no-op) finalise."""
    slice_config_in = tmp_path / "slice_config.csv"
    slice_config_in.write_text(
        "slice_id,use\n00,true\n01,true\n",
        encoding="utf-8",
    )
    fragments = tmp_path / "fragments"
    fragments.mkdir()
    slice_config_out = tmp_path / "slice_config_out.csv"

    ret = script_runner.run(
        [
            "linum_interpolate_missing_slice.py",
            "--finalise",
            "--slice_config_in",
            str(slice_config_in),
            "--slice_config_out",
            str(slice_config_out),
            "--fragments",
            str(fragments),
        ]
    )
    assert ret.success, ret.stderr
    from linumpy.io import slice_config as slice_config_io

    rows = slice_config_io.read(slice_config_out)
    assert set(rows) == {"00", "01"}
    assert rows["00"].get("interpolated", "") == ""
