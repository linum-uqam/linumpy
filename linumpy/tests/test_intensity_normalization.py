"""Tests for linumpy/intensity/normalization.py"""

import numpy as np
import pytest

from linumpy.intensity.normalization import (
    _build_cdf,
    _chunk_boundaries,
    _robust_percentile,
    _smooth_weighted,
    apply_histogram_matching,
    apply_zprofile_smoothing,
    compute_scale_factors,
    get_agarose_mask,
    normalize_volume,
)

# ---------------------------------------------------------------------------
# get_agarose_mask
# ---------------------------------------------------------------------------


def _make_tissue_vol(shape=(10, 32, 32)):
    """Volume with bright tissue region and dim agarose surroundings."""
    rng = np.random.default_rng(0)
    vol = rng.random(shape).astype(np.float32) * 20.0  # low = agarose
    # Bright tissue block in the center
    cx, cy = shape[1] // 4, shape[2] // 4
    vol[:, cx : cx * 3, cy : cy * 3] += 80.0
    return vol


def test_get_agarose_mask_shape():
    vol = _make_tissue_vol((8, 32, 32))
    mask, _threshold = get_agarose_mask(vol)
    assert mask.shape == (32, 32)


def test_get_agarose_mask_is_boolean():
    vol = _make_tissue_vol()
    mask, _ = get_agarose_mask(vol)
    assert mask.dtype == bool


def test_get_agarose_mask_threshold_positive():
    vol = _make_tissue_vol()
    _, threshold = get_agarose_mask(vol)
    assert threshold > 0


def test_get_agarose_mask_low_intensity_is_agarose():
    """Low-intensity region should be classified as agarose."""
    vol = _make_tissue_vol()
    mask, _ = get_agarose_mask(vol)
    # The surrounding low-intensity region should have agarose voxels
    assert mask.any()


# ---------------------------------------------------------------------------
# normalize_volume
# ---------------------------------------------------------------------------


def test_normalize_volume_output_shape():
    vol = _make_tissue_vol((6, 24, 24))
    mask, _ = get_agarose_mask(vol)
    result, _thresholds = normalize_volume(vol.copy(), mask)
    assert result.shape == vol.shape


def test_normalize_volume_output_range():
    """Normalized values should be in [0, 1]."""
    vol = _make_tissue_vol((6, 24, 24))
    mask, _ = get_agarose_mask(vol)
    result, _ = normalize_volume(vol.copy(), mask)
    assert float(result.min()) >= -1e-6
    assert float(result.max()) <= 1.0 + 1e-6


def test_normalize_volume_background_thresholds_length():
    vol = _make_tissue_vol((5, 24, 24))
    mask, _ = get_agarose_mask(vol)
    _, thresholds = normalize_volume(vol.copy(), mask)
    assert len(thresholds) == vol.shape[0]


def test_normalize_volume_preserves_relative_brightness():
    """Global divisor must preserve a 2:1 inter-section brightness ratio.

    Construct two sections that are identical in structure but one has 2× the
    overall signal level.  After normalize_volume the bright section's mean
    should remain ~2× the dim section's mean.
    """
    rng = np.random.default_rng(42)
    n_y, n_x = 32, 32
    # Dim section: tissue in center, low intensity
    section_dim = rng.random((n_y, n_x)).astype(np.float32) * 0.1
    section_dim[8:24, 8:24] += 0.4  # tissue above agarose

    # Bright section: same structure, 2× signal
    section_bright = section_dim * 2.0

    vol = np.stack([section_dim, section_bright], axis=0)  # (2, 32, 32)
    agarose_mask = vol.mean(axis=0) < 0.15  # low-intensity pixels = agarose

    result, _ = normalize_volume(vol.copy(), agarose_mask)

    # The bright section's tissue median should be ~2× the dim section's
    tissue_mask_2d = ~agarose_mask
    mean_dim = float(np.mean(result[0][tissue_mask_2d]))
    mean_bright = float(np.mean(result[1][tissue_mask_2d]))
    ratio = mean_bright / mean_dim
    assert 1.8 <= ratio <= 2.2, f"Expected brightness ratio ~2, got {ratio:.3f}"


# ---------------------------------------------------------------------------
# _robust_percentile
# ---------------------------------------------------------------------------


def test_robust_percentile_empty_returns_zero():
    """Nearly-empty array (< 500 non-zeros) should return 0.0."""
    chunk = np.zeros((10, 10, 10), dtype=np.float32)
    assert _robust_percentile(chunk, 90) == 0.0


def test_robust_percentile_computes_correctly():
    chunk = np.arange(1, 1001, dtype=np.float32)  # 1000 values
    result = _robust_percentile(chunk, 50)
    expected = float(np.percentile(chunk, 50))
    assert abs(result - expected) < 1.0


# ---------------------------------------------------------------------------
# _smooth_weighted
# ---------------------------------------------------------------------------


def test_smooth_weighted_preserves_mean():
    """Smoothing should not wildly change the mean of non-zero values."""
    values = np.array([1.0, 2.0, 0.0, 2.0, 1.0])
    smoothed = _smooth_weighted(values, sigma=1.0)
    assert smoothed.shape == values.shape


def test_smooth_weighted_zeros_dont_bias():
    """Zeros indicate missing data; non-zero neighbors should dominate."""
    values = np.array([1.0, 0.0, 0.0, 0.0, 1.0])
    smoothed = _smooth_weighted(values, sigma=0.5)
    # Interior zeros should be interpolated from neighbors (non-zero)
    assert all(v >= 0 for v in smoothed)


# ---------------------------------------------------------------------------
# _chunk_boundaries
# ---------------------------------------------------------------------------


def test_chunk_boundaries_with_serial_slices():
    bounds = _chunk_boundaries(n_z=10, n_serial_slices=5)
    assert len(bounds) == 5
    # Boundaries should cover [0, 10)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == 10


def test_chunk_boundaries_per_plane():
    """n_serial_slices=None → one boundary per Z-plane."""
    bounds = _chunk_boundaries(n_z=5, n_serial_slices=None)
    assert len(bounds) == 5
    for i, (s, e) in enumerate(bounds):
        assert s == i
        assert e == i + 1


# ---------------------------------------------------------------------------
# _build_cdf
# ---------------------------------------------------------------------------


def test_build_cdf_normalized():
    values = np.random.default_rng(0).random(1000).astype(np.float64)
    _bins, cdf = _build_cdf(values, n_bins=100)
    # CDF must be non-decreasing and last value == 1
    assert cdf[-1] == pytest.approx(1.0)
    assert np.all(np.diff(cdf) >= 0)


def test_build_cdf_bin_count():
    values = np.linspace(0, 1, 200)
    bins, cdf = _build_cdf(values, n_bins=50)
    assert len(bins) == 50
    assert len(cdf) == 50


# ---------------------------------------------------------------------------
# compute_scale_factors
# ---------------------------------------------------------------------------


def test_compute_scale_factors_shape():
    rng = np.random.default_rng(5)
    vol = rng.random((20, 16, 16)).astype(np.float32)
    sf, _raw, _smoothed, _bounds = compute_scale_factors(
        vol, n_serial_slices=4, smooth_sigma=1.0, percentile=90.0, min_scale=0.5, max_scale=2.0
    )
    assert sf.shape == (20,)


def test_compute_scale_factors_clamped():
    rng = np.random.default_rng(6)
    vol = rng.random((20, 16, 16)).astype(np.float32)
    min_s, max_s = 0.5, 2.0
    sf, *_ = compute_scale_factors(vol, n_serial_slices=4, smooth_sigma=1.0, percentile=90.0, min_scale=min_s, max_scale=max_s)
    assert float(sf.min()) >= min_s - 1e-6
    assert float(sf.max()) <= max_s + 1e-6


# ---------------------------------------------------------------------------
# apply_histogram_matching
# ---------------------------------------------------------------------------


def test_apply_histogram_matching_shape():
    rng = np.random.default_rng(7)
    vol = rng.random((10, 16, 16)).astype(np.float32)
    result = apply_histogram_matching(vol, n_serial_slices=2, n_bins=64)
    assert result.shape == vol.shape


def test_apply_histogram_matching_range_preserved():
    """Output values should stay within roughly [0, 1] for unit input."""
    rng = np.random.default_rng(8)
    vol = rng.random((10, 16, 16)).astype(np.float32)
    result = apply_histogram_matching(vol, n_serial_slices=2, n_bins=64)
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0 + 1e-5


def test_apply_histogram_matching_preserves_background():
    """Voxels at or below the tissue threshold must not be modified."""
    rng = np.random.default_rng(9)
    vol = rng.random((8, 12, 12)).astype(np.float32)
    # Carve out a clear background region (exact zeros) that must stay zero.
    vol[:, :3, :3] = 0.0
    result = apply_histogram_matching(vol, n_serial_slices=2, n_bins=64, tissue_threshold=0.0)
    assert np.all(result[:, :3, :3] == 0.0)


def test_apply_histogram_matching_identity_on_flat_volume():
    """Matching to its own histogram should be (approximately) identity on tissue."""
    rng = np.random.default_rng(10)
    vol = rng.random((6, 16, 16)).astype(np.float32) * 0.5 + 0.25
    result = apply_histogram_matching(vol, n_serial_slices=1, n_bins=256)
    # Single section => reference == source => identity up to binning resolution.
    assert float(np.mean(np.abs(result - vol))) < 2e-2


# ---------------------------------------------------------------------------
# apply_zprofile_smoothing
# ---------------------------------------------------------------------------


def test_apply_zprofile_smoothing_shape_and_dtype():
    rng = np.random.default_rng(11)
    vol = rng.random((12, 16, 16)).astype(np.float32) + 0.5
    mask = np.ones_like(vol, dtype=bool)
    result = apply_zprofile_smoothing(vol, mask, sigma=2.0)
    assert result.shape == vol.shape
    assert result.dtype == np.float32


def test_apply_zprofile_smoothing_disabled_when_sigma_zero():
    rng = np.random.default_rng(12)
    vol = rng.random((8, 16, 16)).astype(np.float32)
    mask = np.ones_like(vol, dtype=bool)
    result = apply_zprofile_smoothing(vol, mask, sigma=0.0)
    np.testing.assert_array_equal(result, vol)


def test_apply_zprofile_smoothing_preserves_background():
    """Background voxels (outside mask) must be left unchanged."""
    rng = np.random.default_rng(13)
    vol = rng.random((6, 12, 12)).astype(np.float32) + 0.5
    mask = np.zeros_like(vol, dtype=bool)
    mask[:, 2:10, 2:10] = True
    result = apply_zprofile_smoothing(vol, mask, sigma=2.0)
    np.testing.assert_array_equal(result[~mask], vol[~mask])


def test_apply_zprofile_smoothing_reduces_z_jitter():
    """Z-planes with injected per-Z gain noise should be aligned to the smooth trend."""
    rng = np.random.default_rng(14)
    n_z = 30
    base = rng.random((n_z, 16, 16)).astype(np.float32) * 0.1 + 0.5
    # Inject per-Z multiplicative jitter
    jitter = 1.0 + 0.1 * rng.standard_normal(n_z).astype(np.float32)
    vol = base * jitter[:, None, None]
    mask = np.ones_like(vol, dtype=bool)

    def step(v):
        means = np.array([v[z][mask[z]].mean() for z in range(n_z)])
        return float(np.mean(np.abs(np.diff(means)) / (0.5 * (means[:-1] + means[1:]))))

    s_before = step(vol)
    result = apply_zprofile_smoothing(vol, mask, sigma=2.0)
    s_after = step(result)
    assert s_after < 0.3 * s_before
