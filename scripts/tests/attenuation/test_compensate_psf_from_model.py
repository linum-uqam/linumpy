#!/usr/bin/env python3


def test_help(script_runner):
    ret = script_runner.run(["linum-compensate-psf-from-model", "--help"])
    assert ret.success
