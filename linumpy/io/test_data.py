# -*- coding:utf-8 -*-
from linumpy import LINUMPY_HOME
from linumpy.io.zarr import save_omezarr
import os
import logging
import numpy as np
import dask.array as da


def get_data(name):
    data = {
        'mosaic_3d_omezarr': _get_mosaic_3d_omezarr()
    }
    if name not in data.keys():
        raise ValueError(f'Unknown key for data: {name}')
    return data[name]


def _create_linumpy_home_if_not_exists():
    if not os.path.exists(LINUMPY_HOME):
        os.mkdir(LINUMPY_HOME)


def _get_mosaic_3d_omezarr():
    _create_linumpy_home_if_not_exists()
    filename = os.path.join(LINUMPY_HOME, 'mosaic_3d.ome.zarr')
    if not os.path.exists(filename):
        # create test data
        data = np.random.randn(128, 128, 128)
        dask_array = da.from_array(data)
        save_omezarr(dask_array, filename, chunks=(16, 16, 16),
                     n_levels=5, overwrite=False)

    return filename
