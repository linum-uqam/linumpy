#!/usr/bin/env python3


def test_help(script_runner):
    ret = script_runner.run(["linum_crop_tiles.py", "--help"])
    assert ret.success
