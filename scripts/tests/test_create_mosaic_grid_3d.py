#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from linumpy.io.test_data import get_data
import os


def test_help(script_runner):
    ret = script_runner.run(['linum_create_mosaic_grid_3d.py', '--help'])
    assert ret.success


def test_execution_from_directory(script_runner, tmp_path):
    input = get_data('raw_tiles')
    output = tmp_path / 'output.ome.zarr'
    ret = script_runner.run(['linum_create_mosaic_grid_3d.py', output,
                             '--from_root_directory', input, '-z', 0, '-r', 2])
    assert ret.success


def test_execution_from_list(script_runner, tmp_path):
    input = get_data('raw_tiles')
    output = tmp_path / 'output.ome.zarr'
    ret = script_runner.run(['linum_create_mosaic_grid_3d.py', output,
                             '--from_tiles_list',
                             os.path.join(input, 'tile_x00_y00_z01'),
                             os.path.join(input, 'tile_x01_y00_z01'),
                             os.path.join(input, 'tile_x00_y01_z01'),
                             os.path.join(input, 'tile_x01_y01_z01'),
                             '-r', 2])
    assert ret.success
