#!/usr/bin/env python3
# -*- coding:utf-8 -*-


def test_help(script_runner):
    ret = script_runner.run(['linum_create_all_mosaic_grids_2d.py', '--help'])
    assert ret.success
