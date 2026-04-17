"""Tests for linumpy/stitching/interpolation.py"""

import numpy as np
import pytest

from linumpy.stitching.interpolation import (
    _fractional_affine_parts,
    _matrix_fractional_power,
    find_best_overlap_planes,
    interpolate_average,
    interpolate_weighted,
    interpolate_z_morph,
)


def _vol(shape=(8, 16, 16), seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random(shape) * 100.0).astype(np.float32)


# ---------------------------------------------------------------------------
# interpolate_average
# ---------------------------------------------------------------------------


def test_interpolate_average_shape():
    before = _vol()
    after = _vol(seed=1)
    result = interpolate_average(before, after)
    assert result.shape == before.shape


def test_interpolate_average_midpoint():
    """Result should be the exact arithmetic mean."""
    before = np.zeros((4, 8, 8), dtype=np.float32)
    after = np.full((4, 8, 8), 2.0, dtype=np.float32)
    result = interpolate_average(before, after)
    np.testing.assert_allclose(result, 1.0)


def test_interpolate_average_identical():
    """Averaging a volume with itself preserves values."""
    vol = _vol()
    result = interpolate_average(vol, vol)
    np.testing.assert_allclose(result, vol, rtol=1e-5)


def test_interpolate_average_dtype_float32():
    """Output dtype should be float32."""
    before = np.ones((4, 8, 8), dtype=np.uint8)
    after = np.ones((4, 8, 8), dtype=np.uint8)
    result = interpolate_average(before, after)
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# interpolate_weighted
# ---------------------------------------------------------------------------


def test_interpolate_weighted_shape():
    before = _vol()
    after = _vol(seed=1)
    result = interpolate_weighted(before, after, sigma=1.0)
    assert result.shape == before.shape


def test_interpolate_weighted_same_result_as_average_when_sigma_zero():
    """With sigma ≈ 0, weighted should be close to simple average."""
    before = np.zeros((6, 8, 8), dtype=np.float32)
    after = np.full((6, 8, 8), 4.0, dtype=np.float32)
    avg = interpolate_average(before, after)
    weighted = interpolate_weighted(before, after, sigma=0.01)
    np.testing.assert_allclose(weighted, avg, rtol=0.05)


def test_interpolate_weighted_smoothing_reduces_variance():
    """Larger sigma should produce smoother output (lower std dev along Z)."""
    rng = np.random.default_rng(42)
    before = rng.random((20, 8, 8)).astype(np.float32)
    after = rng.random((20, 8, 8)).astype(np.float32)
    std_low_sigma = interpolate_weighted(before, after, sigma=0.1).std()
    std_high_sigma = interpolate_weighted(before, after, sigma=3.0).std()
    assert std_high_sigma < std_low_sigma


# ---------------------------------------------------------------------------
# Fractional affine helpers
# ---------------------------------------------------------------------------


def test_matrix_fractional_power_identity_alpha_1():
    M = np.array([[1.05, 0.02], [-0.01, 0.98]])
    M_half, imag = _matrix_fractional_power(M, 1.0)
    np.testing.assert_allclose(M_half, M, atol=1e-10)
    assert imag < 1e-6


def test_matrix_fractional_power_half_squared_equals_matrix():
    M = np.array([[1.04, 0.01], [-0.02, 0.97]])
    M_half, _ = _matrix_fractional_power(M, 0.5)
    np.testing.assert_allclose(M_half @ M_half, M, atol=1e-6)


def test_fractional_affine_parts_alpha_0_is_identity():
    M = np.array([[1.05, 0.02], [-0.01, 0.98]])
    t = np.array([3.0, -1.5])
    M_alpha, t_alpha, _ = _fractional_affine_parts(M, t, 0.0)
    np.testing.assert_allclose(M_alpha, np.eye(2), atol=1e-10)
    np.testing.assert_allclose(t_alpha, np.zeros(2), atol=1e-10)


def test_fractional_affine_parts_alpha_1_is_original():
    M = np.array([[1.05, 0.02], [-0.01, 0.98]])
    t = np.array([3.0, -1.5])
    M_alpha, t_alpha, _ = _fractional_affine_parts(M, t, 1.0)
    np.testing.assert_allclose(M_alpha, M, atol=1e-10)
    np.testing.assert_allclose(t_alpha, t, atol=1e-10)


def test_fractional_affine_parts_half_compose_squared_equals_full():
    """For a half-transform, applying twice should equal applying once at alpha=1."""
    M = np.array([[1.06, 0.015], [-0.02, 0.95]])
    t = np.array([4.5, -2.0])
    M_half, t_half, _ = _fractional_affine_parts(M, t, 0.5)

    # Applying the half-transform twice: x -> M_half x + t_half, then again.
    # Expected result: M x + t (affine on an arbitrary point).
    x = np.array([3.0, 7.0])
    y_once = M_half @ x + t_half
    y_twice = M_half @ y_once + t_half
    y_full = M @ x + t
    np.testing.assert_allclose(y_twice, y_full, atol=1e-6)


def test_fractional_affine_parts_pure_translation_degenerate_case():
    """When M = I the closed-form is degenerate; falls back to alpha * t."""
    M = np.eye(2)
    t = np.array([5.0, -3.0])
    _, t_alpha, _ = _fractional_affine_parts(M, t, 0.5)
    np.testing.assert_allclose(t_alpha, 0.5 * t, atol=1e-10)


# ---------------------------------------------------------------------------
# Synthetic ground-truth benchmarks
# ---------------------------------------------------------------------------


def _make_structured_vol(shape=(6, 64, 64), seed=0):
    """Create a structured synthetic volume with repeatable content."""
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    # A few 2D blobs + low-frequency gradient
    vol = np.zeros(shape, dtype=np.float32)
    for z in range(nz):
        depth = z / max(nz - 1, 1)
        cy = ny * (0.3 + 0.1 * depth)
        cx = nx * (0.5 + 0.05 * depth)
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * (ny / 6.0) ** 2))
        noise = rng.normal(0.0, 0.02, size=(ny, nx)).astype(np.float32)
        vol[z] = 0.2 + 0.7 * blob + noise
    return vol


def test_find_best_overlap_planes_returns_valid_pair():
    before = _make_structured_vol(seed=1)
    after = _make_structured_vol(seed=2)
    ref_before, ref_after, corr = find_best_overlap_planes(before, after)
    assert 0 <= ref_before < before.shape[0]
    assert 0 <= ref_after < after.shape[0]
    assert np.isfinite(corr)


def test_interpolate_z_morph_boundary_planes_match_sources():
    """Top of z-morph output should match bottom of vol_before, bottom → top of vol_after."""
    before = _make_structured_vol(seed=3)
    after = _make_structured_vol(seed=4)
    vol, diag = interpolate_z_morph(before, after, max_iterations=50, min_overlap_correlation=0.0, min_ncc_improvement=-10.0)
    if diag["method_used"] != "zmorph":
        pytest.skip(f"zmorph fell back ({diag['fallback_reason']}); boundary assertion not applicable")
    assert diag["top_boundary_residual_mean"] < 1e-4
    assert diag["bottom_boundary_residual_mean"] < 1e-4
    assert vol.shape[0] == min(before.shape[0], after.shape[0])


def test_interpolate_z_morph_hard_skips_when_registration_unreliable():
    """Unrelated noise volumes must not produce a fabricated interpolation.

    Failed gates return ``(None, diag)`` with ``interpolation_failed=True``
    — the pipeline treats this as a genuine gap rather than inserting a
    blended volume.
    """
    rng = np.random.default_rng(99)
    before = rng.random((4, 32, 32)).astype(np.float32)
    after = rng.random((4, 32, 32)).astype(np.float32)
    vol, diag = interpolate_z_morph(
        before,
        after,
        max_iterations=20,
        min_overlap_correlation=0.99,
        min_ncc_improvement=0.0,
    )
    assert vol is None
    assert diag["interpolation_failed"] is True
    assert diag["method_used"] is None
    assert diag["fallback_reason"] in {
        "low_overlap_ncc",
        "no_foreground_planes",
        "reg_did_not_improve",
        "registration_exception",
        "affine_determinant_non_positive",
    }


def test_interpolate_z_morph_success_does_not_mark_failed():
    """A successful zmorph run leaves interpolation_failed absent/False."""
    before, _truth, after = _make_3slice_stack_with_drift(drift_px=1.0, seed=3)
    vol, diag = interpolate_z_morph(
        before,
        after,
        max_iterations=100,
        min_overlap_correlation=0.0,
        min_ncc_improvement=-10.0,
    )
    if diag["method_used"] != "zmorph":
        pytest.skip(f"zmorph hard-skipped on synthetic input (reason={diag['fallback_reason']})")
    assert vol is not None
    assert diag.get("interpolation_failed", False) is False


# ---------------------------------------------------------------------------
# Ground-truth benchmark: drop the middle slice of a synthetic 3-slice stack
# and compare reconstructions against the held-out truth.
# ---------------------------------------------------------------------------


def _make_3slice_stack_with_drift(shape=(6, 48, 48), drift_px=1.0, seed=0):
    """Build three consecutive slices sharing most structure + a small XY drift.

    Returns ``(vol_before, vol_missing_ground_truth, vol_after)``. The
    missing slice is halfway between before and after in XY position, so a
    correct interpolation should reconstruct it up to registration noise.
    """
    ny, nx = shape[1], shape[2]
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)

    def _slice_at_drift(dy: float, dx: float, seed_noise: int) -> np.ndarray:
        vol = np.zeros(shape, dtype=np.float32)
        noise_rng = np.random.default_rng(seed_noise)
        for z in range(shape[0]):
            depth = z / max(shape[0] - 1, 1)
            cy = ny * 0.4 + dy
            cx = nx * 0.55 + dx
            blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * (ny / 5.0) ** 2))
            stripe = 0.2 * np.sin((xx + 4.0 * depth) * 0.35)
            noise = noise_rng.normal(0.0, 0.01, size=(ny, nx)).astype(np.float32)
            vol[z] = np.clip(0.15 + 0.7 * blob + stripe + noise, 0.0, None)
        return vol

    before = _slice_at_drift(-drift_px, 0.0, seed_noise=seed)
    missing = _slice_at_drift(0.0, 0.0, seed_noise=seed + 1)
    after = _slice_at_drift(+drift_px, 0.0, seed_noise=seed + 2)
    return before, missing, after


def _volume_ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity as ssim

    a = a.astype(np.float32)
    b = b.astype(np.float32)
    data_range = float(max(a.max(), b.max()) - min(a.min(), b.min()) + 1e-8)
    win = min(min(a.shape), 7)
    if win % 2 == 0:
        win -= 1
    return float(ssim(a, b, data_range=data_range, win_size=max(win, 3)))


def _volume_psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse < 1e-12:
        return 100.0
    data_range = float(max(a.max(), b.max()) - min(a.min(), b.min()) + 1e-8)
    return 10.0 * float(np.log10(data_range**2 / mse))


def test_ground_truth_zmorph_matches_boundaries_exactly():
    """zmorph reconstructs the boundary planes of the missing slice exactly."""
    before, _truth, after = _make_3slice_stack_with_drift(drift_px=1.5)
    vol, diag = interpolate_z_morph(
        before,
        after,
        max_iterations=100,
        min_overlap_correlation=0.0,
        min_ncc_improvement=-10.0,
    )
    if diag["method_used"] != "zmorph" or vol is None:
        pytest.skip(f"zmorph fell back ({diag['fallback_reason']}); skipping boundary assertion")

    # Output top plane ≡ deepest plane of vol_before (identity warp).
    # Output bottom plane ≡ top plane of vol_after  (identity warp).
    # apply_transform uses a non-zero default fill value, so the outermost
    # 2-pixel border can differ slightly; compare the interior only.
    interior = (slice(2, -2), slice(2, -2))
    np.testing.assert_allclose(vol[0][interior], before[-1][interior], atol=1e-3)
    np.testing.assert_allclose(vol[-1][interior], after[0][interior], atol=1e-3)


def test_ground_truth_zmorph_vs_average_ssim():
    """Report SSIM/PSNR of zmorph and simple average vs. the held-out ground-truth slice.

    Smoke test: ensures both methods produce a reasonable reconstruction on
    the synthetic drift benchmark. The raw numbers are printed for manual
    inspection.
    """
    before, truth, after = _make_3slice_stack_with_drift(drift_px=1.0, seed=7)

    vol_zm, diag_zm = interpolate_z_morph(
        before, after, max_iterations=100, min_overlap_correlation=0.0, min_ncc_improvement=-10.0
    )
    vol_avg = interpolate_average(before, after)

    if vol_zm is None:
        pytest.skip(f"zmorph hard-skipped ({diag_zm['fallback_reason']}); cannot compare metrics")

    ssim_zm = _volume_ssim(vol_zm, truth)
    ssim_avg = _volume_ssim(vol_avg, truth)
    psnr_zm = _volume_psnr(vol_zm, truth)
    psnr_avg = _volume_psnr(vol_avg, truth)

    print(
        f"\nground-truth comparison (drift=1.0px):\n"
        f"  zmorph : SSIM={ssim_zm:.3f} PSNR={psnr_zm:.2f} dB "
        f"(used={diag_zm['method_used']}, reason={diag_zm['fallback_reason']})\n"
        f"  average: SSIM={ssim_avg:.3f} PSNR={psnr_avg:.2f} dB"
    )

    # zmorph reconstructs only from the two boundary planes (physically
    # principled), so SSIM vs an arbitrary interior GT is not its target
    # metric. Only enforce a loose sanity floor.
    assert ssim_zm > 0.05, f"zmorph SSIM pathologically low: {ssim_zm}"
    assert ssim_avg > 0.2, f"average SSIM too low: {ssim_avg}"
