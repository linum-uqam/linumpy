"""Tests for detect_interface_z and crop_below_interface in linumpy/geometry/."""

import numpy as np
import pytest

from linumpy.geometry.crop import crop_below_interface
from linumpy.geometry.interface import detect_interface_z


def _make_vol_with_interface(n_z=60, n_x=16, n_y=16, interface_z=20):
    """
    Create a synthetic (X, Y, Z) volume with a bright 'tissue' layer
    starting at interface_z.  Used by detect_interface_z.
    """
    vol = np.zeros((n_x, n_y, n_z), dtype=np.float32)
    # Plain signal below interface
    vol[:, :, interface_z:] = 100.0
    # Slight noise everywhere
    rng = np.random.default_rng(0)
    vol += rng.random((n_x, n_y, n_z)).astype(np.float32) * 5.0
    return vol


# ---------------------------------------------------------------------------
# detect_interface_z
# ---------------------------------------------------------------------------


def test_detect_interface_z_returns_int():
    vol = _make_vol_with_interface()
    result = detect_interface_z(vol)
    assert isinstance(result, int)


def test_detect_interface_z_non_negative():
    vol = _make_vol_with_interface()
    result = detect_interface_z(vol)
    assert result >= 0


def test_detect_interface_z_within_volume():
    n_z = 50
    vol = _make_vol_with_interface(n_z=n_z)
    result = detect_interface_z(vol)
    assert result < n_z


def test_detect_interface_z_approximate_position():
    """Interface should be detected near the expected depth."""
    expected = 25
    vol = _make_vol_with_interface(n_z=80, interface_z=expected)
    result = detect_interface_z(vol, sigma_xy=1.0, sigma_z=1.0)
    # Allow ±10 voxel tolerance
    assert abs(result - expected) <= 10


def test_detect_interface_z_empty_volume():
    """All-zero volume: returns 0."""
    vol = np.zeros((8, 8, 30), dtype=np.float32)
    result = detect_interface_z(vol)
    assert result == 0


# ---------------------------------------------------------------------------
# crop_below_interface
# ---------------------------------------------------------------------------


def _make_zxy_vol(n_z=60, n_x=16, n_y=16, interface_z=20):
    """Return (Z, Y, X) volume as produced by read_omezarr."""
    vol_xyz = _make_vol_with_interface(n_z=n_z, n_x=n_x, n_y=n_y, interface_z=interface_z)
    return np.transpose(vol_xyz, (2, 0, 1))  # (Z, Y, X)


def test_crop_below_interface_returns_tuple():
    vol_zxy = _make_zxy_vol()
    result = crop_below_interface(vol_zxy, depth_um=100.0, resolution_um=5.0)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_crop_below_interface_output_shape_depth():
    """With crop_before_interface=True, output Z == depth_px exactly."""
    resolution_um = 5.0
    depth_um = 50.0
    expected_depth_px = round(depth_um / resolution_um)  # 10
    vol_zxy = _make_zxy_vol(n_z=80, interface_z=10)
    vol_crop, _ = crop_below_interface(vol_zxy, depth_um=depth_um, resolution_um=resolution_um, crop_before_interface=True)
    assert vol_crop.shape[0] == pytest.approx(expected_depth_px, abs=1)


def test_crop_below_interface_xy_dims_unchanged():
    """XY dimensions must not change after cropping."""
    vol_zxy = _make_zxy_vol(n_z=60, n_x=20, n_y=24)
    vol_crop, _ = crop_below_interface(vol_zxy, depth_um=100.0, resolution_um=5.0)
    assert vol_crop.shape[1] == 20
    assert vol_crop.shape[2] == 24


def test_crop_below_interface_returns_interface_index():
    """Second return value (interface index) must be int >= 0."""
    vol_zxy = _make_zxy_vol()
    _, avg_iface = crop_below_interface(vol_zxy, depth_um=50.0, resolution_um=5.0)
    assert isinstance(avg_iface, int)
    assert avg_iface >= 0


def test_crop_below_interface_crop_before():
    """With crop_before_interface=True the start is shifted to the interface."""
    vol_zxy = _make_zxy_vol(n_z=80, n_x=16, n_y=16, interface_z=20)
    vol_crop_after, _iface = crop_below_interface(vol_zxy, depth_um=50.0, resolution_um=5.0, crop_before_interface=False)
    vol_crop_before, _ = crop_below_interface(vol_zxy, depth_um=50.0, resolution_um=5.0, crop_before_interface=True)
    # crop_before removes voxels above the interface → fewer Z voxels
    assert vol_crop_before.shape[0] <= vol_crop_after.shape[0]


def test_crop_below_interface_percentile_clip_runs():
    """percentile_clip parameter should not raise."""
    vol_zxy = _make_zxy_vol()
    vol_crop, _ = crop_below_interface(vol_zxy, depth_um=50.0, resolution_um=5.0, percentile_clip=99.0)
    assert vol_crop.shape[1] > 0


# ---------------------------------------------------------------------------
# Regression tests for interface detection edge cases
# ---------------------------------------------------------------------------


def test_detect_interface_z_small_tissue_coverage():
    """Interface must be detected when tissue covers only ~15% of XY."""
    n_z, n_x, n_y = 80, 40, 40
    interface_z = 25
    vol = np.zeros((n_x, n_y, n_z), dtype=np.float32)
    # Place tissue in a small corner patch (6x6 = 36 out of 1600 pixels ≈ 2%)
    vol[:6, :6, interface_z:] = 100.0
    rng = np.random.default_rng(42)
    vol += rng.random((n_x, n_y, n_z)).astype(np.float32) * 2.0
    result = detect_interface_z(vol, sigma_xy=1.0, sigma_z=1.0)
    assert abs(result - interface_z) <= 10, f"Expected interface near {interface_z}, got {result}"


def test_detect_interface_z_no_wrap_artifact():
    """Bright values at the end of Z must not create a false interface at z=0."""
    n_z, n_x, n_y = 80, 16, 16
    interface_z = 30
    vol = np.zeros((n_x, n_y, n_z), dtype=np.float32)
    vol[:, :, interface_z:] = 100.0
    # Make the last few Z slices extra bright -- would create z=0 artifact with wrap padding
    vol[:, :, -5:] = 500.0
    rng = np.random.default_rng(7)
    vol += rng.random((n_x, n_y, n_z)).astype(np.float32) * 2.0
    result = detect_interface_z(vol, sigma_xy=1.0, sigma_z=1.0)
    assert result > 5, f"Interface falsely detected near z=0 ({result}), expected near {interface_z}"


def test_detect_interface_z_deep_artifact_clamped():
    """Interface must NOT be detected past halfway even when intensity rises near the end.

    This is the failure mode observed in sub-22 slices z39-z46, where imaging
    artifacts caused the gradient to peak near z=52/55.  With max_depth_fraction=0.5
    the returned interface must be within the first half of the volume.
    """
    n_z, n_x, n_y = 55, 20, 20
    vol = np.zeros((n_x, n_y, n_z), dtype=np.float32)
    # Simulate artifact: intensity rises steeply near the very end of the volume
    vol[:, :, 50:] = 800.0
    rng = np.random.default_rng(99)
    vol += rng.random((n_x, n_y, n_z)).astype(np.float32) * 5.0
    result = detect_interface_z(vol, sigma_xy=1.0, sigma_z=1.0, max_depth_fraction=0.5)
    assert result < n_z // 2, f"Interface detected past halfway ({result} >= {n_z // 2})"
