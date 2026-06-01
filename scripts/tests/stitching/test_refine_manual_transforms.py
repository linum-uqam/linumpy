#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import zarr.storage


def _load_script_module():
    """Import scripts/stitching/linum_refine_manual_transforms.py as a module."""
    script_path = Path(__file__).resolve().parents[2] / "stitching" / "linum_refine_manual_transforms.py"
    spec = importlib.util.spec_from_file_location("linum_refine_manual_transforms", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_zarr_slice(path, shape=(10, 32, 32)):
    """Write a tiny OME-Zarr volume filled with random data."""
    store = zarr.storage.LocalStore(str(path))
    root = zarr.open_group(store, mode="w")
    data = (np.random.rand(*shape) * 255).astype(np.uint16)
    arr = root.create_array("0", shape=shape, chunks=shape, dtype=np.uint16)
    arr[:] = data
    root.attrs["multiscales"] = [
        {
            "axes": [
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [10.0, 10.0, 10.0]}]}],
            "version": "0.4",
        }
    ]


def _make_transform(path, tx=0.0, ty=0.0, rot_deg=0.0, cx=16.0, cy=16.0):
    """Write a trivial Euler3DTransform .tfm file."""
    tfm = sitk.Euler3DTransform()
    tfm.SetFixedParameters([cx, cy, 0.0, 0.0])
    tfm.SetParameters([0.0, 0.0, float(np.radians(rot_deg)), tx, ty, 0.0])
    sitk.WriteTransform(tfm, str(path))


def test_help(script_runner):
    ret = script_runner.run(["linum_refine_manual_transforms.py", "--help"])
    assert ret.success


def test_run_no_manual_transforms(tmp_path, script_runner):
    """Without any manual transforms the pair is copied unchanged."""
    fixed_zarr = tmp_path / "slice_z04.ome.zarr"
    moving_zarr = tmp_path / "slice_z05.ome.zarr"
    auto_dir = tmp_path / "auto_transforms"
    manual_dir = tmp_path / "manual"
    out_dir = tmp_path / "out"

    _make_zarr_slice(fixed_zarr)
    _make_zarr_slice(moving_zarr)
    manual_dir.mkdir()

    auto_dir.mkdir()
    _make_transform(auto_dir / "transform.tfm")
    np.savetxt(str(auto_dir / "offsets.txt"), [8, 2], fmt="%d")
    (auto_dir / "pairwise_registration_metrics.json").write_text(json.dumps({"source": "auto"}))

    ret = script_runner.run(
        [
            "linum_refine_manual_transforms.py",
            str(fixed_zarr),
            str(moving_zarr),
            str(auto_dir),
            str(out_dir),
            "--manual_transforms_dir",
            str(manual_dir),
        ]
    )
    assert ret.success, ret.stderr
    assert (out_dir / "transform.tfm").exists()


def test_run_with_manual_transform(tmp_path, script_runner):
    """With a manual transform the pair is refined and output written."""
    fixed_zarr = tmp_path / "slice_z04.ome.zarr"
    moving_zarr = tmp_path / "slice_z05.ome.zarr"
    auto_dir = tmp_path / "auto_transforms"
    manual_dir = tmp_path / "manual"
    out_dir = tmp_path / "out"

    _make_zarr_slice(fixed_zarr)
    _make_zarr_slice(moving_zarr)

    auto_dir.mkdir()
    _make_transform(auto_dir / "transform.tfm")
    np.savetxt(str(auto_dir / "offsets.txt"), [8, 2], fmt="%d")

    manual_pair = manual_dir / "slice_z05"
    manual_pair.mkdir(parents=True)
    _make_transform(manual_pair / "transform.tfm", tx=1.0, ty=0.5)

    ret = script_runner.run(
        [
            "linum_refine_manual_transforms.py",
            str(fixed_zarr),
            str(moving_zarr),
            str(auto_dir),
            str(out_dir),
            "--manual_transforms_dir",
            str(manual_dir),
        ]
    )
    assert ret.success, ret.stderr
    assert (out_dir / "transform.tfm").exists()
    metrics = json.loads((out_dir / "pairwise_registration_metrics.json").read_text())
    assert metrics["source"] == "manual_refined"


def test_overwrite_guard(tmp_path, script_runner):
    """Running twice without -f should fail; with -f should succeed."""
    fixed_zarr = tmp_path / "slice_z04.ome.zarr"
    moving_zarr = tmp_path / "slice_z05.ome.zarr"
    auto_dir = tmp_path / "auto_transforms"
    manual_dir = tmp_path / "manual"
    out_dir = tmp_path / "out"
    out_dir.mkdir()  # pre-create to trigger guard

    _make_zarr_slice(fixed_zarr)
    _make_zarr_slice(moving_zarr)
    manual_dir.mkdir()
    auto_dir.mkdir()
    _make_transform(auto_dir / "transform.tfm")

    base_args = [
        "linum_refine_manual_transforms.py",
        str(fixed_zarr),
        str(moving_zarr),
        str(auto_dir),
        str(out_dir),
        "--manual_transforms_dir",
        str(manual_dir),
    ]

    ret = script_runner.run(base_args)
    assert not ret.success, "should fail without -f when out_dir exists"

    ret = script_runner.run([*base_args, "-f"])
    assert ret.success, ret.stderr


def _apply_rigid_2d(tx, ty, rot_deg, cx, cy, point):
    """Evaluate a 2D rigid transform T(p) = R (p - c) + c + t."""
    theta = np.radians(rot_deg)
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    c = np.array([cx, cy])
    t = np.array([tx, ty])
    return r @ (np.asarray(point) - c) + c + t


def test_compose_rigid_2d_matches_point_evaluation():
    """Closed-form composition must match explicit per-point evaluation.

    This is the regression for the old additive composition
    ``final = man + delta`` which is only valid when the manual rotation
    centre coincides with the image centre. Here the manual centre is
    deliberately off-centre so the additive formula would disagree with the
    explicit composition at every corner.
    """
    module = _load_script_module()

    # Image: 200 (W) x 160 (H); manual rotation centre at (W/4, H/4).
    w, h = 200, 160
    final_cx, final_cy = w / 2.0, h / 2.0
    man_tx, man_ty, man_rot = 3.5, -2.0, 1.5
    man_cx, man_cy = w / 4.0, h / 4.0
    delta_tx, delta_ty, delta_rot = 0.2, 0.1, 0.05

    tx, ty, rot = module._compose_rigid_2d(
        man_tx, man_ty, man_rot, man_cx, man_cy, delta_tx, delta_ty, delta_rot, final_cx, final_cy
    )

    # θ_final = θ_manual + θ_delta for 2D planar rotations.
    assert rot == pytest_approx(man_rot + delta_rot)

    # Evaluate at each image corner and compare against explicit
    # T_delta(T_manual(p)).
    corners = [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)]
    for p in corners:
        p_manual = _apply_rigid_2d(man_tx, man_ty, man_rot, man_cx, man_cy, p)
        expected = _apply_rigid_2d(delta_tx, delta_ty, delta_rot, final_cx, final_cy, p_manual)
        got = _apply_rigid_2d(tx, ty, rot, final_cx, final_cy, p)
        assert np.allclose(got, expected, atol=1e-6), f"mismatch at {p}: got={got}, expected={expected}"


def test_compose_rigid_2d_reduces_to_sum_when_centres_match():
    """When all centres equal c, the composition collapses to additive params."""
    module = _load_script_module()
    c = (50.0, 50.0)
    man_tx, man_ty, man_rot = 1.0, -0.5, 2.0
    delta_tx, delta_ty, delta_rot = -0.3, 0.8, 0.25
    tx, ty, rot = module._compose_rigid_2d(
        man_tx,
        man_ty,
        man_rot,
        c[0],
        c[1],
        delta_tx,
        delta_ty,
        delta_rot,
        c[0],
        c[1],
    )
    # Rotate the manual translation by the delta rotation, then add delta.
    theta = np.radians(delta_rot)
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    expected_t = r @ np.array([man_tx, man_ty]) + np.array([delta_tx, delta_ty])
    assert np.allclose((tx, ty), expected_t, atol=1e-6)
    assert rot == pytest_approx(man_rot + delta_rot)


# Local approx helper to avoid importing pytest.approx at module scope.
def pytest_approx(expected, rel=1e-6, abs_=1e-6):
    import pytest

    return pytest.approx(expected, rel=rel, abs=abs_)
