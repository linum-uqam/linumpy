#!/usr/bin/env python3


def test_help(script_runner):
    ret = script_runner.run(["linum-crop-tiles", "--help"])
    assert ret.success
