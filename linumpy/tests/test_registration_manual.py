"""Tests for the manual-alignment export helpers in linumpy/registration/manual.py"""

import numpy as np
import pytest

from linumpy.registration.manual import (
    _brightest_index,
    _discover_slices,
    _discover_transforms,
    _is_interpolated,
    _read_overlap_z_offsets,
    _save_aip_npz,
    _save_axis_views,
    _save_axis_views_for_pair,
    _save_xy_aips_for_pair,
    _tissue_centroid,
)

# ---------------------------------------------------------------------------
# _brightest_index
# ---------------------------------------------------------------------------


def test_brightest_index_finds_peak_plane():
    vol = np.zeros((4, 6, 6), dtype=np.float32)
    vol[2] = 1.0  # brightest along axis 0 at index 2
    assert _brightest_index(vol, axis=0) == 2


# ---------------------------------------------------------------------------
# _tissue_centroid
# ---------------------------------------------------------------------------


def test_tissue_centroid_of_flat_profile_is_midpoint():
    profile = np.zeros(10, dtype=np.float32)
    centroid = _tissue_centroid(profile)
    assert centroid == pytest.approx(5.0)


def test_tissue_centroid_weighted_toward_peak():
    profile = np.zeros(10, dtype=np.float32)
    profile[8] = 1.0
    centroid = _tissue_centroid(profile)
    assert centroid == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# _save_aip_npz
# ---------------------------------------------------------------------------


def test_save_aip_npz_round_trip(tmp_path):
    aip = np.random.rand(8, 8).astype(np.float32)
    scale = np.array([1.0, 2.0], dtype=float)
    out_path = tmp_path / "aip.npz"

    _save_aip_npz(aip, scale, out_path, center_pos=3)

    assert out_path.exists()
    loaded = np.load(out_path)
    np.testing.assert_allclose(loaded["aip"], aip)
    np.testing.assert_allclose(loaded["scale"], scale)
    assert int(loaded["center_pos"]) == 3


def test_save_aip_npz_without_center_pos(tmp_path):
    aip = np.zeros((4, 4), dtype=np.float32)
    scale = np.array([1.0, 1.0], dtype=float)
    out_path = tmp_path / "aip_no_center.npz"

    _save_aip_npz(aip, scale, out_path)

    loaded = np.load(out_path)
    assert "center_pos" not in loaded.files


# ---------------------------------------------------------------------------
# _save_axis_views
# ---------------------------------------------------------------------------


def test_save_axis_views_writes_xz_yz_files(tmp_path):
    vol = np.random.rand(4, 8, 8).astype(np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=float)
    aips_xz_dir = tmp_path / "aips_xz"
    aips_yz_dir = tmp_path / "aips_yz"
    aips_xz_dir.mkdir()
    aips_yz_dir.mkdir()

    _save_axis_views(vol, scale, sid=1, aips_xz_dir=aips_xz_dir, aips_yz_dir=aips_yz_dir)

    assert (aips_xz_dir / "slice_z01.npz").exists()
    assert (aips_yz_dir / "slice_z01.npz").exists()


# ---------------------------------------------------------------------------
# _save_xy_aips_for_pair / _save_axis_views_for_pair
# ---------------------------------------------------------------------------


def test_save_xy_aips_for_pair_writes_fixed_and_moving(tmp_path):
    fixed = np.random.rand(6, 8, 8).astype(np.float32)
    moving = np.random.rand(6, 8, 8).astype(np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=float)

    _save_xy_aips_for_pair(fixed, moving, scale, scale, overlap_px=2, fid=0, mid=1, aips_dir=tmp_path)

    assert (tmp_path / "pair_z00_z01_fixed.npz").exists()
    assert (tmp_path / "pair_z00_z01_moving.npz").exists()


def test_save_axis_views_for_pair_writes_xz_yz_files(tmp_path):
    fixed = np.random.rand(6, 8, 8).astype(np.float32)
    moving = np.random.rand(6, 8, 8).astype(np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=float)
    aips_xz_dir = tmp_path / "aips_xz"
    aips_yz_dir = tmp_path / "aips_yz"
    aips_xz_dir.mkdir()
    aips_yz_dir.mkdir()

    _save_axis_views_for_pair(
        fixed, moving, scale, scale, fixed_z=5, moving_z=0, fid=0, mid=1, aips_xz_dir=aips_xz_dir, aips_yz_dir=aips_yz_dir
    )

    assert (aips_xz_dir / "pair_z00_z01_fixed.npz").exists()
    assert (aips_xz_dir / "pair_z00_z01_moving.npz").exists()
    assert (aips_yz_dir / "pair_z00_z01_fixed.npz").exists()
    assert (aips_yz_dir / "pair_z00_z01_moving.npz").exists()


# ---------------------------------------------------------------------------
# _is_interpolated
# ---------------------------------------------------------------------------


def test_is_interpolated_detects_suffix():
    from pathlib import Path

    assert _is_interpolated(Path("slice_z01_interpolated.ome.zarr"))
    assert not _is_interpolated(Path("slice_z01.ome.zarr"))


# ---------------------------------------------------------------------------
# _discover_slices / _discover_transforms
# ---------------------------------------------------------------------------


def test_discover_slices_matches_pattern(tmp_path):
    (tmp_path / "slice_z00.ome.zarr").mkdir()
    (tmp_path / "slice_z01_interpolated.ome.zarr").mkdir()
    (tmp_path / "not_a_slice.txt").touch()

    slices = _discover_slices(tmp_path)

    assert set(slices.keys()) == {0, 1}
    assert slices[0].name == "slice_z00.ome.zarr"
    assert slices[1].name == "slice_z01_interpolated.ome.zarr"


def test_discover_transforms_matches_directories(tmp_path):
    (tmp_path / "slice_z01").mkdir()
    (tmp_path / "slice_z02").mkdir()
    (tmp_path / "not_a_dir_marker.txt").touch()

    transforms = _discover_transforms(tmp_path)

    assert set(transforms.keys()) == {1, 2}


# ---------------------------------------------------------------------------
# _read_overlap_z_offsets
# ---------------------------------------------------------------------------


def test_read_overlap_z_offsets_missing_file_returns_zeros(tmp_path):
    assert _read_overlap_z_offsets(tmp_path / "missing.txt") == (0, 0)


def test_read_overlap_z_offsets_reads_values(tmp_path):
    offsets_file = tmp_path / "offsets.txt"
    offsets_file.write_text("3 7\n")
    assert _read_overlap_z_offsets(offsets_file) == (3, 7)


def test_read_overlap_z_offsets_invalid_content_returns_zeros(tmp_path):
    offsets_file = tmp_path / "offsets.txt"
    offsets_file.write_text("not,a,number\n")
    assert _read_overlap_z_offsets(offsets_file) == (0, 0)
