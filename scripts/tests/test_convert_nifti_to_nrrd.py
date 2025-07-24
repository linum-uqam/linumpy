#!/usr/bin/env python3
# -*- coding:utf-8 -*-


def test_help(script_runner):
    ret = script_runner.run(['linum_convert_nifti_to_nrrd.py', '--help'])
    assert ret.success
