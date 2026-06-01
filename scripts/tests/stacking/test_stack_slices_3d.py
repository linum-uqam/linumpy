#!/usr/bin/env python3


def test_help(script_runner):
    ret = script_runner.run(["linum_stack_slices_3d.py", "--help"])
    assert ret.success
