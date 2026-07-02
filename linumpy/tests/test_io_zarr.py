"""Tests for linumpy/io/zarr.py OME-Zarr metadata round-trip."""

from pathlib import Path

import dask.array as da
import numpy as np

from linumpy.io.zarr import read_omezarr, save_omezarr

VOXEL_SIZE_MM = (0.01, 0.01, 0.01)
TILE_FOV_MM = 0.875


def _small_volume() -> da.Array:
    return da.from_array(np.zeros((2, 10, 10), dtype=np.float32))


def test_save_omezarr_preserves_voxel_size_n_levels_0(tmp_path: Path) -> None:
    out = tmp_path / "volume.ome.zarr"
    save_omezarr(
        _small_volume(),
        out,
        voxel_size=VOXEL_SIZE_MM,
        chunks=(2, 10, 10),
        n_levels=0,
    )
    _, scale = read_omezarr(out, 0)
    assert scale == list(VOXEL_SIZE_MM)


def test_save_omezarr_preserves_voxel_size_pyramid(tmp_path: Path) -> None:
    out = tmp_path / "pyramid.ome.zarr"
    save_omezarr(
        _small_volume(),
        out,
        voxel_size=VOXEL_SIZE_MM,
        chunks=(2, 10, 10),
        n_levels=1,
    )
    _, scale_l0 = read_omezarr(out, 0)
    _, scale_l1 = read_omezarr(out, 1)
    assert scale_l0 == list(VOXEL_SIZE_MM)
    assert scale_l1 == [v * 2 for v in VOXEL_SIZE_MM]


def test_tile_size_px_from_resolution(tmp_path: Path) -> None:
    out = tmp_path / "tile_sizing.ome.zarr"
    save_omezarr(
        _small_volume(),
        out,
        voxel_size=VOXEL_SIZE_MM,
        chunks=(2, 10, 10),
        n_levels=0,
    )
    _, res = read_omezarr(out, 0)
    tile_px = round(TILE_FOV_MM / float(res[1]))
    assert tile_px == 88
