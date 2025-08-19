#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from linumpy.io.test_data import get_data
import os


def test_help(script_runner):
    ret = script_runner.run(['linum_stack_mosaics_into_3d_volume.py', '--help'])
    assert ret.success


def test_execute(script_runner, tmp_path):
    input = get_data('raw_tiles')
    stack_dir = tmp_path / 'stack'
    z0 = stack_dir / 'slice_z00_r2.ome.zarr'
    z1 = stack_dir / 'slice_z01_r2.ome.zarr'
    xy_shifts = tmp_path / 'xy_shifts.csv'
    os.makedirs(stack_dir, exist_ok=True)

    # TODO: Precompute data and cache it in .linumpy to avoid
    # running these scripts every time.
    script_runner.run(['linum_create_mosaic_grid_3d.py', z0,
                       '--from_root_directory', input, '-z', 0, '-r', 2])
    script_runner.run(['linum_create_mosaic_grid_3d.py', z1,
                       '--from_root_directory', input, '-z', 1, '-r', 2])
    script_runner.run(['linum_estimate_xy_shift_from_metadata.py', input, xy_shifts])

    output = tmp_path / 'output.ome.zarr'
    ret = script_runner.run(['linum_stack_mosaics_into_3d_volume.py', stack_dir, xy_shifts, output])
    assert ret.success
