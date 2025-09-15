#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from linumpy.io.test_data import get_data


def test_help(script_runner):
    ret = script_runner.run(['linum_fix_illumination_3d.py', '--help'])
    assert ret.success


def test_execute(script_runner, tmp_path):
    input = get_data('mosaic_3d_omezarr')
    output = tmp_path / 'fix_illumination.ome.zarr'
    ret = script_runner.run(['linum_fix_illumination_3d.py', input, output,
                             '--max_iterations', 10])
    assert ret.success
