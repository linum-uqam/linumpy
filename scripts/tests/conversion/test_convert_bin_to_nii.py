#!/usr/bin/env python3
from pathlib import Path

from linumpy.io.test_data import get_data


def test_help(script_runner):
    ret = script_runner.run(["linum-convert-bin-to-nii", "--help"])
    assert ret.success


def test_execution(script_runner):
    input = get_data("raw_tiles")
    ret = script_runner.run(["linum-convert-bin-to-nii", str(Path(input) / "tile_x00_y00_z00"), "output.nii.gz"])
    assert ret.success
