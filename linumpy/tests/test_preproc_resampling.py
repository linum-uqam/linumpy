# -*- coding: utf-8 -*-
"""Tests for linumpy/preproc/resampling.py"""
import numpy as np
import pytest
import zarr

from linumpy.preproc.resampling import resample_mosaic_grid


def _make_zarr_mosaic(tmp_path, n_tiles_x=2, n_tiles_y=2,
                      tile_shape=(4, 8, 8), fill=1.0, dtype=np.float32):
    """
    Create a zarr array mosaic grid.

    zarr's .chunks returns a plain tuple of ints (e.g. (4, 8, 8)), which is
    what resample_mosaic_grid expects — unlike dask's .chunks which returns
    tuples of tuples.
    """
    nz, th, tw = tile_shape
    shape = (nz, n_tiles_x * th, n_tiles_y * tw)
    arr = zarr.open(str(tmp_path / "mosaic.zarr"), mode='w', shape=shape,
                    chunks=tile_shape, dtype=dtype)
    arr[:] = fill
    return arr


# ---------------------------------------------------------------------------
# resample_mosaic_grid — validation
# ---------------------------------------------------------------------------

def test_resample_mosaic_grid_raises_without_chunks():
    """Plain ndarray without 'chunks' attribute must raise ValueError."""
    arr = np.ones((10, 20, 20), dtype=np.float32)
    with pytest.raises(ValueError, match="chunks"):
        resample_mosaic_grid(arr, source_res=(0.01, 0.01, 0.01), target_res_um=10.0)


# ---------------------------------------------------------------------------
# resample_mosaic_grid — source resolution in mm (< 1)
# ---------------------------------------------------------------------------

def test_resample_mosaic_grid_returns_array_when_no_outpath(tmp_path):
    """Returns an ndarray when out_path is not provided."""
    vol = _make_zarr_mosaic(tmp_path, n_tiles_x=1, n_tiles_y=1,
                            tile_shape=(4, 8, 8))
    # source 0.01 mm = 10 µm, target 20 µm → half resolution
    result = resample_mosaic_grid(vol, source_res=(0.01, 0.01, 0.01),
                                  target_res_um=20.0)
    assert isinstance(result, np.ndarray)


def test_resample_mosaic_grid_output_is_smaller_for_downscale(tmp_path):
    """Down-sampling (target > source) must produce a smaller volume."""
    vol = _make_zarr_mosaic(tmp_path, n_tiles_x=2, n_tiles_y=2,
                            tile_shape=(8, 16, 16))
    # source 0.005 mm = 5 µm, target 20 µm → factor 0.25
    result = resample_mosaic_grid(vol, source_res=(0.005, 0.005, 0.005),
                                  target_res_um=20.0)
    assert (result.shape[1] < vol.shape[1] or result.shape[0] < vol.shape[0])


def test_resample_mosaic_grid_output_is_larger_for_upscale(tmp_path):
    """Up-sampling (target < source) must produce a larger volume."""
    vol = _make_zarr_mosaic(tmp_path, n_tiles_x=1, n_tiles_y=1,
                            tile_shape=(4, 8, 8))
    # source 0.050 mm = 50 µm, target 10 µm → scale ×5
    result = resample_mosaic_grid(vol, source_res=(0.05, 0.05, 0.05),
                                  target_res_um=10.0)
    assert result.shape[0] > vol.shape[0]


def test_resample_mosaic_grid_um_source_resolution(tmp_path):
    """source_res >= 1 is treated as µm (not mm)."""
    vol = _make_zarr_mosaic(tmp_path, n_tiles_x=1, n_tiles_y=1,
                            tile_shape=(4, 8, 8))
    # source 10 µm, target 20 µm → factor 0.5
    result = resample_mosaic_grid(vol, source_res=(10.0, 10.0, 10.0),
                                  target_res_um=20.0)
    assert isinstance(result, np.ndarray)
    assert result.shape[1] <= vol.shape[1]


def test_resample_mosaic_grid_to_file(tmp_path):
    """With out_path, the function writes to disk and returns None."""
    vol = _make_zarr_mosaic(tmp_path, n_tiles_x=1, n_tiles_y=1,
                            tile_shape=(4, 8, 8))
    out = str(tmp_path / "resampled.ome.zarr")
    result = resample_mosaic_grid(vol, source_res=(0.01, 0.01, 0.01),
                                  target_res_um=20.0,
                                  n_levels=1, out_path=out)
    assert result is None
    ds = zarr.open(out, mode='r')
    assert ds is not None


def test_resample_mosaic_grid_multi_tile_consistency(tmp_path):
    """2×2 tiles produces ≈2× the per-tile output size compared to 1×1."""
    tile_shape = (4, 8, 8)
    tmp1 = tmp_path / "a"
    tmp2 = tmp_path / "b"
    tmp1.mkdir()
    tmp2.mkdir()
    vol_1x1 = _make_zarr_mosaic(tmp1, 1, 1, tile_shape=tile_shape, fill=1.0)
    vol_2x2 = _make_zarr_mosaic(tmp2, 2, 2, tile_shape=tile_shape, fill=1.0)
    res_1x1 = resample_mosaic_grid(vol_1x1, (0.01, 0.01, 0.01), 20.0)
    res_2x2 = resample_mosaic_grid(vol_2x2, (0.01, 0.01, 0.01), 20.0)
    ts = res_1x1.shape
    assert res_2x2.shape[1] == pytest.approx(ts[1] * 2, abs=2)
    assert res_2x2.shape[2] == pytest.approx(ts[2] * 2, abs=2)
