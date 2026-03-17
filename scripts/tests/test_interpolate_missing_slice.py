#!/usr/bin/env python3
# -*- coding:utf-8 -*-


def test_help(script_runner):
    ret = script_runner.run(['linum_interpolate_missing_slice.py', '--help'])
    assert ret.success


def test_average_method(script_runner, tmp_path):
    """Test basic interpolation with average method using synthetic data."""
    import numpy as np
    from linumpy.io.zarr import save_omezarr
    import dask.array as da
    
    # Create synthetic test volumes
    shape = (10, 32, 32)
    resolution = (0.001, 0.01, 0.01)  # mm/pixel
    
    # Create two slightly different volumes
    vol_before = np.random.rand(*shape).astype(np.float32) * 100
    vol_after = np.random.rand(*shape).astype(np.float32) * 100
    
    # Save as ome.zarr
    slice_before = tmp_path / 'slice_z00.ome.zarr'
    slice_after = tmp_path / 'slice_z02.ome.zarr'
    output = tmp_path / 'slice_z01_interpolated.ome.zarr'
    
    save_omezarr(da.from_array(vol_before), str(slice_before), resolution)
    save_omezarr(da.from_array(vol_after), str(slice_after), resolution)
    
    # Run interpolation with average method (faster for testing)
    ret = script_runner.run([
        'linum_interpolate_missing_slice.py',
        str(slice_before),
        str(slice_after),
        str(output),
        '--method', 'average'
    ])
    
    assert ret.success
    assert output.exists()

