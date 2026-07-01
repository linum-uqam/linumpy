#!/usr/bin/env python3
from linumpy.io.test_data import get_data


def test_help(script_runner):
    ret = script_runner.run(["linum-create-mosaic-grid-2d", "--help"])
    assert ret.success


def test_execution(script_runner, tmp_path):
    input = get_data("raw_tiles")
    output = tmp_path / "output.zarr"

    ret = script_runner.run(["linum-create-mosaic-grid-2d", input, output, "--n_cpus", 1])
    assert ret.success
