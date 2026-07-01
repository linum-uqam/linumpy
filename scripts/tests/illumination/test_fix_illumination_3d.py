#!/usr/bin/env python3
import pytest

from linumpy.io.test_data import get_data


def test_help(script_runner):
    ret = script_runner.run(["linum-fix-illumination-3d", "--help"])
    assert ret.success


# BaSiCPy initializes a native thread pool at import time that conflicts
# with threads left by other in-process tests (e.g. process pools from
# estimate_xy_shift or mosaic grid creation). Run in subprocess for isolation.
@pytest.mark.script_launch_mode("subprocess")
def test_execute(script_runner, tmp_path):
    input = get_data("mosaic_3d_omezarr")
    output = tmp_path / "fix_illumination.ome.zarr"
    ret = script_runner.run(["linum-fix-illumination-3d", input, output, "--max_iterations", "40"])
    assert ret.success
