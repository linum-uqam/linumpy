#!/usr/bin/env python3
import pytest

from linumpy.geometry.resampling import resolution_is_mm


def test_help(script_runner):
    ret = script_runner.run(["linum-crop-3d-mosaic-below-interface", "--help"])
    assert ret.success


@pytest.mark.parametrize(
    ("resolution", "expected_mm"),
    [
        ((0.0035, 0.0035, 0.0035), True),  # stored as mm (3.5 µm)
        ((3.5, 3.5, 3.5), False),  # stored as µm
        ((10.0, 10.0, 10.0), False),
        ((1e-3, 1e-3, 1e-3), True),
    ],
)
def test_resolution_is_mm_heuristic(resolution, expected_mm):
    """Sub-micron voxels are impossible in practice, so <1 ⇒ mm, ≥1 ⇒ µm."""
    assert resolution_is_mm(resolution) is expected_mm


def test_crop_depth_voxels_respects_um_resolution():
    """Regression for the crop depth calculation when resolution is in µm.

    The script historically assumed ``res[0]`` was in mm, which inflated
    ``resolution_um`` by 1000x for legacy mosaics that still stored µm in
    their NGFF metadata -- effectively asking for ``depth_um/1000`` voxels
    and returning a single-voxel crop regardless of the requested depth.
    """
    depth_um = 400.0

    res_mm = (0.0035, 0.0035, 0.0035)
    resolution_um_from_mm = res_mm[0] * 1000 if resolution_is_mm(res_mm) else float(res_mm[0])
    depth_px_from_mm = round(depth_um / resolution_um_from_mm)

    res_um = (3.5, 3.5, 3.5)
    resolution_um_from_um = res_um[0] * 1000 if resolution_is_mm(res_um) else float(res_um[0])
    depth_px_from_um = round(depth_um / resolution_um_from_um)

    assert depth_px_from_mm == depth_px_from_um
    assert depth_px_from_mm == round(depth_um / 3.5)
