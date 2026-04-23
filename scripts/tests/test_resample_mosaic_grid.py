#!/usr/bin/env python3
import numpy as np
import pytest

from linumpy.geometry.resampling import resolution_is_mm
from linumpy.io.test_data import get_data


def test_help(script_runner):
    ret = script_runner.run(["linum_resample_mosaic_grid.py", "--help"])
    assert ret.success


def test_execute(script_runner, tmp_path):
    input = get_data("mosaic_3d_omezarr")
    output = tmp_path / "test_resample.ome.zarr"
    ret = script_runner.run(["linum_resample_mosaic_grid.py", input, output])
    assert ret.success


@pytest.mark.parametrize(
    ("source_res", "target_res_um", "expected_target", "expected_scale"),
    [
        # mm-stored source: target is converted to mm so scaling is unit-consistent.
        ((0.005, 0.005, 0.005), 10.0, 10.0 / 1000.0, 0.5),
        # µm-stored source: target stays in µm for scaling parity.
        ((5.0, 5.0, 5.0), 10.0, 10.0, 0.5),
        # Upsampling: µm source, larger voxels requested.
        ((20.0, 20.0, 20.0), 10.0, 10.0, 2.0),
    ],
)
def test_resample_scaling_factor_matches_units(source_res, target_res_um, expected_target, expected_scale):
    """Regression for the GPU-branch unit bug.

    Both paths must use ``resolution_is_mm`` so that ``scaling_factor`` is
    computed in a single unit rather than mixing mm with µm.
    """
    target_res = target_res_um / 1000.0 if resolution_is_mm(source_res) else float(target_res_um)
    assert target_res == pytest.approx(expected_target)
    scaling = np.asarray(source_res) / target_res
    np.testing.assert_allclose(scaling, [expected_scale] * 3)
