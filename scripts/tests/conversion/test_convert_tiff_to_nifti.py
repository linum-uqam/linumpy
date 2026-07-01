#!/usr/bin/env python3


def test_help(script_runner):
    ret = script_runner.run(["linum-convert-tiff-to-nifti", "--help"])
    assert ret.success
