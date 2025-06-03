#!/usr/bin/env python3
# -*- coding:utf-8 -*-


def test_help(script_runner):
    ret = script_runner.run(['linum_create_mosaic_grid.py', '--help'])
    assert ret.success
