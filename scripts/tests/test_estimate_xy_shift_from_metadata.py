#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from linumpy.io.test_data import get_data


def test_help(script_runner):
    ret = script_runner.run(['linum_estimate_xy_shift_from_metadata.py', '--help'])
    assert ret.success


def test_execute(script_runner, tmp_path):
    input = get_data('raw_tiles')
    output = tmp_path / 'xy_shifts.csv'
    ret = script_runner.run(['linum_estimate_xy_shift_from_metadata.py', input,
                             output])
    assert ret.success
