#!/usr/bin/env python3


def test_help(script_runner):
    ret = script_runner.run(["linum-create-all-mosaic-grids-2d", "--help"])
    assert ret.success
