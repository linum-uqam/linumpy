#!/usr/bin/env python3


def test_help(script_runner):
    ret = script_runner.run(["linum-intensity-normalization", "--help"])
    assert ret.success
