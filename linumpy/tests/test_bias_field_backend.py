"""Integration tests for the n4_correct backend dispatcher."""

from __future__ import annotations

import numpy as np
import pytest

from linumpy.gpu import GPU_AVAILABLE
from linumpy.intensity.bias_field import n4_correct, n4_correct_per_section


def _synthetic_volume(shape=(20, 32, 32), seed=0):
    rng = np.random.default_rng(seed)
    z, y, x = shape
    zg, yg, xg = np.mgrid[0:z, 0:y, 0:x].astype(np.float32)
    cz, cy, cx = z / 2, y / 2, x / 2
    r = np.sqrt(((zg - cz) / (z / 3)) ** 2 + ((yg - cy) / (y / 3)) ** 2 + ((xg - cx) / (x / 3)) ** 2)
    truth = np.where(r < 1.0, 1.0, 0.3).astype(np.float32) + rng.normal(0, 0.02, shape).astype(np.float32)
    bias = (1.0 + 0.4 * (zg / z + 0.5 * yg / y - 0.5 * xg / x)).astype(np.float32)
    return (truth * bias).astype(np.float32), r < 1.2


def test_n4_correct_cpu_backend_runs():
    """Default CPU backend (SimpleITK) still runs and returns valid output."""
    vol, mask = _synthetic_volume()
    corrected, bias = n4_correct(vol, mask, shrink_factor=2, n_iterations=[5, 5], backend="cpu")
    assert corrected.shape == vol.shape
    assert bias.shape == vol.shape
    assert np.isfinite(corrected).all()
    assert np.isfinite(bias).all()


def test_n4_correct_gpu_backend_runs_on_cpu_fallback():
    """GPU backend runs on the NumPy path even when CUDA is unavailable."""
    vol, mask = _synthetic_volume()
    corrected, bias = n4_correct(vol, mask, shrink_factor=2, n_iterations=[10, 10], spline_distance_mm=20.0, backend="gpu")
    assert corrected.shape == vol.shape
    assert bias.shape == vol.shape
    assert np.isfinite(corrected).all()
    assert np.isfinite(bias).all()


def test_n4_correct_auto_backend_picks_available():
    """auto backend should run successfully regardless of GPU presence."""
    vol, mask = _synthetic_volume()
    corrected, bias = n4_correct(vol, mask, shrink_factor=2, n_iterations=[5, 5], spline_distance_mm=20.0, backend="auto")
    assert corrected.shape == vol.shape
    assert np.isfinite(corrected).all()
    assert np.isfinite(bias).all()


def test_n4_correct_invalid_backend_raises():
    vol, mask = _synthetic_volume()
    with pytest.raises(ValueError, match="backend"):
        n4_correct(vol, mask, backend="tpu")


def test_n4_correct_per_section_gpu_forces_serial():
    """When backend='gpu', per_section must run serially regardless of n_processes."""
    vol, mask = _synthetic_volume(shape=(20, 24, 24))
    corrected, bias = n4_correct_per_section(
        vol,
        n_serial_slices=2,
        mask=mask,
        n_processes=4,  # should be coerced to 1 internally
        shrink_factor=2,
        n_iterations=[5],
        spline_distance_mm=20.0,
        backend="gpu",
    )
    assert corrected.shape == vol.shape
    assert bias.shape == vol.shape
    assert np.isfinite(corrected).all()


def test_n4_correct_per_section_cpu_unchanged():
    """CPU per_section path still works as before."""
    vol, mask = _synthetic_volume(shape=(20, 24, 24))
    corrected, bias = n4_correct_per_section(
        vol,
        n_serial_slices=2,
        mask=mask,
        n_processes=1,
        shrink_factor=2,
        n_iterations=[5],
        backend="cpu",
    )
    assert corrected.shape == vol.shape
    assert bias.shape == vol.shape
    assert np.isfinite(corrected).all()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_n4_correct_gpu_backend_uses_cuda_when_available():
    """When CUDA is available the gpu backend should still match shape/finite."""
    vol, mask = _synthetic_volume()
    corrected, bias = n4_correct(vol, mask, shrink_factor=2, n_iterations=[5, 5], spline_distance_mm=20.0, backend="gpu")
    assert corrected.shape == vol.shape
    assert bias.shape == vol.shape
    assert np.isfinite(corrected).all()
