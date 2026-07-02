#!/usr/bin/env python3


def test_help(script_runner):
    ret = script_runner.run(["linum-estimate-slices-transforms-gui", "--help"])
    assert ret.success
