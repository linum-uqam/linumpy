#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from linumpy.io.zarr import save_omezarr
import dask.array as da
import numpy as np

OMEZARR_PATH = "data.ome.zarr"


def test_help(script_runner):
    ret = script_runner.run('linum_aip.py', '--help')
    assert ret.success


def test_execution(script_runner, tmp_path):
    d = tmp_path / "input"
    d.mkdir()
    in_file = str(d / OMEZARR_PATH)

    data = np.random.randn(400, 100, 100)
    dask_array = da.from_array(data)
    save_omezarr(dask_array, in_file, chunks=(400, 100, 100))
