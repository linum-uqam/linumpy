#!/usr/bin/env python3
# -*- coding:utf-8 -*-


def test_help(script_runner):
    ret = script_runner.run(['linum_compensate_psf_from_model.py', '--help'])
    assert ret.success
