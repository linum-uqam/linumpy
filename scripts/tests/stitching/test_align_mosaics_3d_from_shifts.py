#!/usr/bin/env python3


def test_help(script_runner):
    ret = script_runner.run(["linum-align-mosaics-3d-from-shifts", "--help"])
    assert ret.success
