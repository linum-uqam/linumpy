#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from linumpy.io.test_data import get_data


def test_help(script_runner):
    ret = script_runner.run(['linum_create_mosaic_grid.py', '--help'])
    assert ret.success


def test_execution(script_runner, tmp_path):
    input = get_data('raw_tiles')
    output = tmp_path / 'output.zarr'

    ret = script_runner.run(['linum_create_mosaic_grid.py',
                             input, output, '--n_cpus', 1])
    assert ret.success
