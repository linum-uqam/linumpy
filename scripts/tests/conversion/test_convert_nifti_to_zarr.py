#!/usr/bin/env python3


def test_help(script_runner):
    ret = script_runner.run(["linum-convert-nifti-to-zarr", "--help"])
    assert ret.success
