#!/usr/bin/env python3
# -*- coding:utf-8 -*-


def test_help(script_runner):
    ret = script_runner.run(['linum_intensity_normalization.py', '--help'])
    assert ret.success
