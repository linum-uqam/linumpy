#!/usr/bin/env python3
import pytest

from linumpy.io.test_data import get_data


def test_help(script_runner):
    ret = script_runner.run(["linum-basic-preview", "--help"])
    assert ret.success


@pytest.mark.script_launch_mode("subprocess")
def test_execute(script_runner, tmp_path):
    pytest.importorskip("linum_basic")
    input = get_data("mosaic_3d_omezarr")
    output = tmp_path / "preview.png"
    ret = script_runner.run(["linum-basic-preview", input, output])
    assert ret.success
    assert output.exists()
    assert output.stat().st_size > 0
