#!/usr/bin/env python3


def test_help(script_runner):
    ret = script_runner.run(["linum-axis-xyz-to-zyx", "--help"])
    assert ret.success
