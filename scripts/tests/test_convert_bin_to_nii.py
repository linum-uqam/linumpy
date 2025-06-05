#!/usr/bin/env python3
# -*- coding:utf-8 -*-


def test_help(script_runner):
    ret = script_runner.run(['linum_convert_bin_to_nii.py', '--help'])
    assert ret.success
