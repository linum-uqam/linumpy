#!/usr/bin/env python3
import pytest

from linumpy.io.test_data import get_data


def test_help(script_runner):
    ret = script_runner.run(["linum-fix-illumination-basic", "--help"])
    assert ret.success


# Run in subprocess for isolation from native thread pools left by other tests.
@pytest.mark.script_launch_mode("subprocess")
def test_execute(script_runner, tmp_path):
    pytest.importorskip("linum_basic")
    input = get_data("mosaic_3d_omezarr")
    output = tmp_path / "fix_illumination_basic.ome.zarr"
    ret = script_runner.run(["linum-fix-illumination-basic", input, output, "--max_iterations", "5"])
    assert ret.success
    assert output.exists()
