"""Tests for linumpy.gpu.n4."""

import numpy as np
import pytest

from linumpy.gpu import GPU_AVAILABLE
from linumpy.gpu.n4 import _build_log_psf, sharpen_residual

# ---------------------------------------------------------------------------
# Histogram sharpening
# ---------------------------------------------------------------------------


def test_psf_is_unit_mass_and_centred():
    psf = _build_log_psf(n_bins=200, bin_width=0.01, fwhm=0.15, xp=np)
    assert psf.shape == (200,)
    np.testing.assert_allclose(psf.sum(), 1.0, atol=1e-6)
    # Maximum should be at the centre bin
    assert int(np.argmax(psf)) == 100


def test_sharpen_preserves_mass_unimodal():
    """Sharpening a unimodal Gaussian distribution should approximately
    preserve the integral of the histogram (mass conservation)."""
    rng = np.random.default_rng(0)
    # 2000 samples from N(0, 0.2)
    log_v = rng.normal(0.0, 0.2, size=2000).astype(np.float32)
    mask = np.ones_like(log_v, dtype=bool)
    sharp = sharpen_residual(log_v, mask, n_bins=200, fwhm_log=0.1, wiener_noise=0.01, use_gpu=False)
    # Sharpened LUT remaps every value; the mean of the mapped values
    # should still be close to the original mean (approximate mass
    # preservation under the LUT).
    assert abs(float(sharp.mean()) - float(log_v.mean())) < 0.05


def test_sharpen_lut_monotone_unimodal():
    """For a unimodal Gaussian, the LUT must be approximately monotone."""
    rng = np.random.default_rng(1)
    log_v = rng.normal(0.0, 0.2, size=4000).astype(np.float32)
    sharp = sharpen_residual(log_v, mask=None, n_bins=200, fwhm_log=0.1, wiener_noise=0.01, use_gpu=False)
    # Sort by input; sharp output must be (approximately) sorted too.
    order = np.argsort(log_v)
    sharp_sorted = sharp[order]
    # Allow small non-monotone wiggle from histogram noise; check Spearman-like
    # monotonicity by counting strict inversions in a smoothed signal.
    smoothed = np.convolve(sharp_sorted, np.ones(50) / 50.0, mode="valid")
    diffs = np.diff(smoothed)
    fraction_increasing = float((diffs >= 0).mean())
    assert fraction_increasing > 0.95, f"Only {fraction_increasing:.3f} of LUT diffs are non-decreasing"


def test_sharpen_narrows_modes_bimodal():
    """A blurred bimodal distribution should be sharpened: the gap between
    its two peaks (after sharpening) should be at least as deep as before."""
    rng = np.random.default_rng(2)
    n = 4000
    samples = np.concatenate(
        [
            rng.normal(-0.3, 0.15, size=n // 2),  # blurred left peak
            rng.normal(0.3, 0.15, size=n // 2),  # blurred right peak
        ]
    ).astype(np.float32)

    sharp = sharpen_residual(samples, mask=None, n_bins=200, fwhm_log=0.2, wiener_noise=0.005, use_gpu=False)

    # Compare bimodality (peak-to-trough ratio) before vs after.
    def _bimodality_ratio(values: np.ndarray) -> float:
        hist, _ = np.histogram(values, bins=80, range=(-0.8, 0.8))
        peak_l = float(hist[:40].max())
        peak_r = float(hist[40:].max())
        trough = float(hist[35:45].min())
        return min(peak_l, peak_r) / max(trough, 1.0)

    ratio_before = _bimodality_ratio(samples)
    ratio_after = _bimodality_ratio(sharp)
    assert ratio_after >= ratio_before * 0.9, (
        f"Sharpening should not flatten modes: before={ratio_before:.3f}, after={ratio_after:.3f}"
    )


def test_sharpen_handles_empty_mask():
    """Empty mask should return input unchanged."""
    log_v = np.linspace(-1.0, 1.0, 100, dtype=np.float32)
    mask = np.zeros_like(log_v, dtype=bool)
    sharp = sharpen_residual(log_v, mask, use_gpu=False)
    np.testing.assert_array_equal(sharp, log_v)


def test_sharpen_handles_constant_volume():
    """A constant volume must produce finite output (no NaN/Inf)."""
    log_v = np.full(500, 0.5, dtype=np.float32)
    sharp = sharpen_residual(log_v, mask=None, use_gpu=False)
    assert np.isfinite(sharp).all()


def test_sharpen_outside_mask_unchanged():
    """Voxels outside the mask must be returned unchanged."""
    rng = np.random.default_rng(3)
    log_v = rng.normal(0.0, 0.2, size=1000).astype(np.float32)
    mask = rng.random(1000) > 0.5
    sharp = sharpen_residual(log_v, mask, use_gpu=False)
    np.testing.assert_array_equal(sharp[~mask], log_v[~mask])


# ---------------------------------------------------------------------------
# CPU/GPU agreement
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_sharpen_cpu_gpu_agree():
    rng = np.random.default_rng(0)
    log_v = rng.normal(0.0, 0.2, size=2000).astype(np.float32)
    cpu = sharpen_residual(log_v, None, use_gpu=False)
    gpu = sharpen_residual(log_v, None, use_gpu=True)
    assert np.max(np.abs(cpu - gpu)) < 1e-3


# ---------------------------------------------------------------------------
# Full N4 driver
# ---------------------------------------------------------------------------


def _make_synthetic_volume(
    shape: tuple[int, int, int] = (32, 64, 64),
    bias_amp: float = 0.6,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (vol_with_bias, true_bias, mask) for testing."""
    rng = np.random.default_rng(seed)
    z, y, x = shape
    zg, yg, xg = np.mgrid[0:z, 0:y, 0:x].astype(np.float32)
    cz, cy, cx = z / 2, y / 2, x / 2
    r = np.sqrt(((zg - cz) / (z / 3)) ** 2 + ((yg - cy) / (y / 3)) ** 2 + ((xg - cx) / (x / 3)) ** 2)
    truth = np.where(r < 1.0, 1.0, 0.3).astype(np.float32)
    truth = truth + rng.normal(0.0, 0.02, size=shape).astype(np.float32)
    mask = r < 1.2

    z_norm = (zg - cz) / z
    y_norm = (yg - cy) / y
    x_norm = (xg - cx) / x
    bias = 1.0 + bias_amp * (z_norm + 0.5 * y_norm - 0.5 * x_norm)
    bias = np.clip(bias, 0.5, 2.0).astype(np.float32)

    biased = truth * bias
    return biased, bias, mask


def test_n4_correct_gpu_recovers_known_bias_cpu():
    from linumpy.gpu.n4 import n4_correct_gpu

    vol, true_bias, mask = _make_synthetic_volume(shape=(24, 48, 48), bias_amp=0.4)
    corrected, est_bias = n4_correct_gpu(
        vol,
        mask,
        shrink_factor=2,
        n_iterations=[20, 20],
        spline_distance_mm=20.0,
        voxel_size_mm=(1.0, 1.0, 1.0),
        use_gpu=False,
    )
    assert est_bias.shape == vol.shape
    assert corrected.shape == vol.shape
    assert np.isfinite(est_bias).all()
    assert np.isfinite(corrected).all()

    ratio = (est_bias / true_bias)[mask]
    cv = float(np.std(ratio) / np.mean(ratio))
    assert cv < 0.10, f"Bias recovery CV too high: {cv:.3f}"


def test_n4_correct_gpu_reduces_residual_spread():
    from linumpy.gpu.n4 import n4_correct_gpu

    vol, _, mask = _make_synthetic_volume(shape=(24, 48, 48), bias_amp=0.5)

    # Restrict to one tissue class (interior) -- true intensity is constant
    # there, so any spread comes from the bias field.
    z, y, x = vol.shape
    zg, yg, xg = np.mgrid[0:z, 0:y, 0:x].astype(np.float32)
    cz, cy, cx = z / 2, y / 2, x / 2
    r = np.sqrt(((zg - cz) / (z / 3)) ** 2 + ((yg - cy) / (y / 3)) ** 2 + ((xg - cx) / (x / 3)) ** 2)
    interior = (r < 0.7) & mask

    corrected, _ = n4_correct_gpu(vol, mask, shrink_factor=2, n_iterations=[20, 20], spline_distance_mm=20.0, use_gpu=False)
    spread_before = float(np.std(vol[interior]) / np.mean(vol[interior]))
    spread_after = float(np.std(corrected[interior]) / np.mean(corrected[interior]))
    assert spread_after <= spread_before * 0.7, f"Spread not reduced: before={spread_before:.3f}, after={spread_after:.3f}"


def test_n4_correct_gpu_no_nan_unmasked_voxels():
    from linumpy.gpu.n4 import n4_correct_gpu

    vol, _, mask = _make_synthetic_volume(shape=(20, 32, 32))
    corrected, bias = n4_correct_gpu(vol, mask, shrink_factor=2, n_iterations=[10], spline_distance_mm=20.0, use_gpu=False)
    assert np.isfinite(corrected).all()
    assert np.isfinite(bias).all()


def test_n4_correct_gpu_deterministic():
    from linumpy.gpu.n4 import n4_correct_gpu

    vol, _, mask = _make_synthetic_volume(shape=(20, 32, 32))
    a, _ = n4_correct_gpu(vol, mask, shrink_factor=2, n_iterations=[10], use_gpu=False)
    b, _ = n4_correct_gpu(vol, mask, shrink_factor=2, n_iterations=[10], use_gpu=False)
    np.testing.assert_array_equal(a, b)


def test_n4_correct_gpu_no_mask():
    from linumpy.gpu.n4 import n4_correct_gpu

    vol, _, _ = _make_synthetic_volume(shape=(20, 32, 32))
    corrected, bias = n4_correct_gpu(vol, mask=None, shrink_factor=2, n_iterations=[10], use_gpu=False)
    assert corrected.shape == vol.shape
    assert np.isfinite(corrected).all()
    assert np.isfinite(bias).all()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_n4_correct_cpu_gpu_agree():
    from linumpy.gpu.n4 import n4_correct_gpu

    vol, _, mask = _make_synthetic_volume(shape=(20, 32, 32))
    cpu, _ = n4_correct_gpu(vol, mask, shrink_factor=2, n_iterations=[10], use_gpu=False)
    gpu, _ = n4_correct_gpu(vol, mask, shrink_factor=2, n_iterations=[10], use_gpu=True)
    rel_err = np.max(np.abs(cpu - gpu)) / max(float(np.max(np.abs(cpu))), 1e-6)
    assert rel_err < 1e-2, f"CPU/GPU divergence: rel_err={rel_err:.3e}"


def test_n4_correct_gpu_out_buffers_alias_input():
    """``out=vol`` overwrites the input host buffer in place, ``bias_out``
    receives the bias field, and the result matches the non-aliased call."""
    from linumpy.gpu.n4 import n4_correct_gpu

    vol, _, mask = _make_synthetic_volume(shape=(20, 32, 32))
    expected_corr, expected_bias = n4_correct_gpu(vol, mask, shrink_factor=2, n_iterations=[10], use_gpu=False)

    vol_inout = vol.copy()
    bias_buf = np.empty_like(vol_inout, dtype=np.float32)
    corr_ret, bias_ret = n4_correct_gpu(
        vol_inout,
        mask,
        shrink_factor=2,
        n_iterations=[10],
        use_gpu=False,
        out=vol_inout,
        bias_out=bias_buf,
    )
    # Returns must be the supplied buffers, not fresh allocations.
    assert corr_ret is vol_inout
    assert bias_ret is bias_buf
    np.testing.assert_allclose(vol_inout, expected_corr, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(bias_buf, expected_bias, rtol=1e-5, atol=1e-5)


def test_n4_correct_gpu_out_buffer_shape_mismatch_raises():
    from linumpy.gpu.n4 import n4_correct_gpu

    vol, _, mask = _make_synthetic_volume(shape=(20, 32, 32))
    bad_out = np.empty((10, 10, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="out must be"):
        n4_correct_gpu(vol, mask, shrink_factor=2, n_iterations=[5], use_gpu=False, out=bad_out)
    bad_bias = np.empty(vol.shape, dtype=np.float64)
    with pytest.raises(ValueError, match="bias_out must be"):
        n4_correct_gpu(vol, mask, shrink_factor=2, n_iterations=[5], use_gpu=False, bias_out=bad_bias)
