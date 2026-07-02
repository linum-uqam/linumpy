#!/usr/bin/env python3
from linumpy.io.test_data import get_data


def test_help(script_runner):
    ret = script_runner.run(["linum-estimate-xy-shift-from-metadata", "--help"])
    assert ret.success


def test_execute(script_runner, tmp_path):
    input = get_data("raw_tiles")
    output = tmp_path / "xy_shifts.csv"
    ret = script_runner.run(["linum-estimate-xy-shift-from-metadata", input, output])
    assert ret.success
