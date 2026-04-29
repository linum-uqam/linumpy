"""Tests for linumpy.gpu.bspline."""

from __future__ import annotations

import numpy as np
import pytest

from linumpy.gpu import GPU_AVAILABLE
from linumpy.gpu.bspline import (
    _cubic_bspline_basis,
    bspline_evaluate,
    bspline_fit,
)

# ---------------------------------------------------------------------------
# Basis function sanity
# ---------------------------------------------------------------------------


def test_basis_partition_of_unity():
    """The four cubic B-spline weights must sum to 1 for any t in [0, 1)."""
    t = np.linspace(0.0, 0.999, 50, dtype=np.float32)
    weights = _cubic_bspline_basis(t, np)
    assert weights.shape == (50, 4)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)


def test_basis_nonnegative():
    t = np.linspace(0.0, 0.999, 50, dtype=np.float32)
    weights = _cubic_bspline_basis(t, np)
    assert (weights >= 0).all()


# ---------------------------------------------------------------------------
# bspline_fit + bspline_evaluate (CPU path)
# ---------------------------------------------------------------------------


def test_bspline_constant_field():
    """Fit a constant volume; recovered field must equal the constant everywhere."""
    shape = (12, 16, 16)
    vals = np.full(shape, 0.7, dtype=np.float32)
    coeffs = bspline_fit(vals, weights=None, mask=None, n_control_points=(6, 8, 8), use_gpu=False)
    field = bspline_evaluate(coeffs, shape, use_gpu=False)
    assert np.max(np.abs(field - 0.7)) < 1e-3


def test_bspline_linear_gradient():
    """A linear gradient should be reproduced (approximately) in the interior."""
    shape = (24, 24, 24)
    z = np.arange(shape[0], dtype=np.float32)[:, None, None]
    vals = np.broadcast_to(0.5 + 0.1 * z, shape).astype(np.float32)
    coeffs = bspline_fit(vals, weights=None, mask=None, n_control_points=(8, 8, 8), use_gpu=False)
    field = bspline_evaluate(coeffs, shape, use_gpu=False)

    # Check interior (away from boundary smoothing).  Cubic B-spline kernel
    # regression introduces small bias near boundaries; require the slope
    # in the central region to match within 5%.
    interior = field[6:-6]
    means = interior.mean(axis=(1, 2))
    slope = float(means[-1] - means[0]) / (interior.shape[0] - 1)
    expected_slope = 0.1
    assert abs(slope - expected_slope) / expected_slope < 0.05


def test_bspline_smooth_recovery():
    """A smooth field (sum of Gaussians) should be approximated within 5% rel error."""
    shape = (20, 32, 32)
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing="ij",
    )
    centre = (10.0, 16.0, 16.0)
    sigma = 8.0
    vals = (
        1.0
        + 0.3 * np.exp(-((zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2) / (2 * sigma**2))
    ).astype(np.float32)

    coeffs = bspline_fit(vals, weights=None, mask=None, n_control_points=(8, 12, 12), use_gpu=False)
    field = bspline_evaluate(coeffs, shape, use_gpu=False)

    rel_err = np.max(np.abs(field - vals) / vals)
    assert rel_err < 0.05, f"Max relative error {rel_err:.4f} exceeds 5%"


def test_bspline_mask_respected():
    """Masked-out voxels must not influence the fit."""
    shape = (12, 16, 16)
    vals = np.zeros(shape, dtype=np.float32)
    vals[:, :8, :] = 0.4  # left half: tissue
    vals[:, 8:, :] = 1e6  # right half: should be ignored

    mask = np.zeros(shape, dtype=bool)
    mask[:, :8, :] = True

    coeffs = bspline_fit(vals, weights=None, mask=mask, n_control_points=(6, 8, 8), use_gpu=False)
    field = bspline_evaluate(coeffs, shape, use_gpu=False)
    # In the masked region, fitted value must be near 0.4 (not contaminated by 1e6).
    assert np.max(np.abs(field[:, :8, :] - 0.4)) < 0.1


def test_bspline_evaluate_resampling_shape():
    """Evaluate at a different resolution than the fit; output shape must match."""
    coeffs = np.ones((6, 8, 8), dtype=np.float32) * 0.5
    field = bspline_evaluate(coeffs, target_shape=(20, 32, 32), use_gpu=False)
    assert field.shape == (20, 32, 32)
    np.testing.assert_allclose(field, 0.5, atol=1e-5)


def test_bspline_invalid_control_points():
    """Fewer than 4 control points on any axis should raise."""
    vals = np.ones((10, 10, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        bspline_fit(vals, None, None, n_control_points=(3, 5, 5), use_gpu=False)


# ---------------------------------------------------------------------------
# CPU/GPU agreement
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_bspline_cpu_gpu_agree_fit():
    rng = np.random.default_rng(0)
    shape = (16, 24, 24)
    vals = rng.random(shape, dtype=np.float32)
    cpu = bspline_fit(vals, None, None, n_control_points=(6, 8, 8), use_gpu=False)
    gpu = bspline_fit(vals, None, None, n_control_points=(6, 8, 8), use_gpu=True)
    assert np.max(np.abs(cpu - gpu)) < 1e-4


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_bspline_cpu_gpu_agree_evaluate():
    rng = np.random.default_rng(1)
    coeffs = rng.random((6, 8, 8), dtype=np.float32)
    cpu = bspline_evaluate(coeffs, target_shape=(16, 24, 24), use_gpu=False)
    gpu = bspline_evaluate(coeffs, target_shape=(16, 24, 24), use_gpu=True)
    assert np.max(np.abs(cpu - gpu)) < 1e-4
