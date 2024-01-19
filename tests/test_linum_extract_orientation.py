#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def test_help_option(script_runner):
    ret = script_runner.run("linum_extract_orientation.py", "--help")
    assert ret.success

def test_no_input(script_runner):
    ret = script_runner.run("linum_extract_orientation.py")
    assert ret.returncode != 0