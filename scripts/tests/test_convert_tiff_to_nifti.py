#!/usr/bin/env python3
# -*- coding:utf-8 -*-


def test_help(script_runner):
    ret = script_runner.run(['linum_convert_tiff_to_nifti.py', '--help'])
    assert ret.success
