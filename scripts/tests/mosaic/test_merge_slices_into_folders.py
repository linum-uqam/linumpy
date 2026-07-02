#!/usr/bin/env python3


def test_help(script_runner):
    ret = script_runner.run(["linum-merge-slices-into-folders", "--help"])
    assert ret.success
