#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from linumpy.io.test_data import get_data


def test_help(script_runner):
    ret = script_runner.run(['linum_compensate_psf_model_free.py', '--help'])
    assert ret.success

def test_execute(script_runner, tmp_path):
    input = get_data('mosaic_3d_omezarr')
    output = tmp_path / 'compensated.ome.zarr'
    ret = script_runner.run(['linum_compensate_psf_model_free.py', input, output])
    assert ret.success
