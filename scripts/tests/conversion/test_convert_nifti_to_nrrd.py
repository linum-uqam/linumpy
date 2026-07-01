#!/usr/bin/env python3

from linumpy.io.test_data import get_data


def test_help(script_runner):
    ret = script_runner.run(["linum-convert-nifti-to-nrrd", "--help"])
    assert ret.success


def test_execution(script_runner, tmp_path):
    input = get_data("mosaic_3d_nifti")
    output = tmp_path / "output.nrrd"

    ret = script_runner.run(["linum-convert-nifti-to-nrrd", input, output])
    assert ret.success
