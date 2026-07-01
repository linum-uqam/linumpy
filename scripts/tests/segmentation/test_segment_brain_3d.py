#!/usr/bin/env python3


def test_help(script_runner):
    ret = script_runner.run(["linum-segment-brain-3d", "--help"])
    assert ret.success
