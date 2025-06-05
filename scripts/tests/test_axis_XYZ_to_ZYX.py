#!/usr/bin/env python3
# -*- coding:utf-8 -*-


def test_help(script_runner):
    ret = script_runner.run(['linum_axis_XYZ_to_ZYX.py', '--help'])
    assert ret.success
