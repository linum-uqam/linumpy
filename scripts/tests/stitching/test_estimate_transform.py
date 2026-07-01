#!/usr/bin/env python3
from linumpy.io.test_data import get_data


def test_help(script_runner):
    ret = script_runner.run(["linum-estimate-transform", "--help"])
    assert ret.success


def test_execute(script_runner, tmp_path):
    input = get_data("aip")
    output = tmp_path / "transform.npy"
    ret = script_runner.run(["linum-estimate-transform", input, output])
    assert ret.success
