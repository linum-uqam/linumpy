#!/usr/bin/env python3
import pytest

from linumpy.io.test_data import get_data


def test_help(script_runner):
    ret = script_runner.run(["linum_detect_focal_curvature.py", "--help"])
    assert ret.success


# BaSiC uses JAX/XLA which initializes a native thread pool that conflicts
# with threads left by other inprocess tests (e.g. dask from mosaic grid).
# Run in a subprocess to ensure a clean process-level state.
@pytest.mark.script_launch_mode("subprocess")
def test_execute(script_runner, tmp_path):
    input = get_data("mosaic_3d_omezarr")
    output = tmp_path / "fix_focal.ome.zarr"
    ret = script_runner.run(["linum_detect_focal_curvature.py", input, output])
    assert ret.success
