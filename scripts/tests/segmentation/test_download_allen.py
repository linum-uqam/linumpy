#!/usr/bin/env python3


def test_help(script_runner):
    ret = script_runner.run(["linum-download-allen", "--help"])
    assert ret.success
