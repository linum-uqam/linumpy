"""Unit tests for linumpy.microscope.oct.OCT loader contract (RECON-02)."""

import numpy as np
import pytest

from linumpy.io.test_data import write_synthetic_oct_tile
from linumpy.microscope.oct import OCT


def test_load_image_axis_order(tmp_path):
    """D-75: load_image returns (Z, Y, X) with distinct nx != ny != nz."""
    nx, ny, nz = 16, 8, 8
    tile_dir = write_synthetic_oct_tile(tmp_path, nx=nx, ny=ny, nz=nz)

    oct = OCT(tile_dir)
    vol = oct.load_image(crop=True, fix_galvo_shift=False, fix_camera_shift=False)

    assert vol.shape == (nz, ny, nx)


def test_n_repeat_averaging(tmp_path):
    """D-76: repeated frames are averaged before crop; shape matches n_repeat=1."""
    nx, ny, nz, n_repeat = 4, 2, 4, 2
    n_alines_per_bscan = nx
    n_frames = ny * n_repeat
    raw = np.zeros((nz, n_alines_per_bscan, n_frames), dtype=np.float32)
    for y in range(ny):
        raw[:, :, y * n_repeat] = 0.0
        raw[:, :, y * n_repeat + 1] = 10.0

    vol_zyx = np.full((nz, ny, nx), 5.0, dtype=np.float32)
    tile_dir = write_synthetic_oct_tile(
        tmp_path,
        nx=nx,
        ny=ny,
        nz=nz,
        n_repeat=n_repeat,
        vol_zyx=vol_zyx,
    )
    # Overwrite bin with explicit 0/10 repeat pattern (fixture writer duplicates frames).
    raw.reshape(-1, order="F").tofile(tile_dir / "image_00000.bin")

    oct = OCT(tile_dir)
    vol = oct.load_image(crop=True, fix_galvo_shift=False, fix_camera_shift=False)

    assert vol.shape == (nz, ny, nx)
    np.testing.assert_allclose(vol, 5.0, rtol=0, atol=1e-5)


def test_axial_res_from_info(tmp_path):
    """D-77: axial resolution from info.txt; fallback to constructor default."""
    custom_res = 4.2
    tile_with = write_synthetic_oct_tile(tmp_path / "with_key", axial_res=custom_res)
    oct_with = OCT(tile_with)
    assert oct_with.rz == pytest.approx(custom_res)

    tile_without = write_synthetic_oct_tile(tmp_path / "without_key")
    oct_without = OCT(tile_without, axial_res=3.5)
    assert oct_without.rz == pytest.approx(3.5)
