"""Tests for the Vermeer-2013 depth-resolved attenuation model.

The reference equation (Vermeer et al. 2013, Biomed. Opt. Express, Eq. 17) is

    mu[i] = log(1 + I[i] / sum_{j=i+1..N-1} I[j]) / (2 * dz)

For a pure single-scattering signal ``I[i] = exp(-2 * mu * dz * i)``, the
estimator should recover ``mu`` exactly at every depth at which the deep tail
of the signal is fully contained in the volume.
"""

import numpy as np

from linumpy.intensity.attenuation import get_attenuation_vermeer2013


def test_recovers_uniform_attenuation_from_clean_exponential():
    """A pure exponential decay should recover the true mu within ~1 %."""
    n_z = 500
    dz_m = 10.0e-6  # 10 microns/pixel
    mu_true_per_cm = 50.0  # signal drops to ~0.6 by the end of the volume
    mu_true_per_m = mu_true_per_cm * 100.0

    z = np.arange(n_z)
    aline = np.exp(-2.0 * mu_true_per_m * dz_m * z)
    vol = np.broadcast_to(aline, (4, 4, n_z)).astype(float).copy()

    mu_cm = get_attenuation_vermeer2013(vol, dz=dz_m)

    # Use the central 60 % to avoid the unavoidable tail blow-up at the
    # bottom (Vermeer overshoots when the residual signal beyond the volume
    # is non-negligible compared to the local intensity).
    lo, hi = int(0.1 * n_z), int(0.7 * n_z)
    estimated = mu_cm[2, 2, lo:hi]

    assert np.allclose(estimated, mu_true_per_cm, rtol=0.01), (
        f"mu mismatch: estimated mean={estimated.mean():.3f}, std={estimated.std():.3g}, true={mu_true_per_cm}"
    )


def test_does_not_underestimate_by_factor_two():
    """Regression guard: the previous implementation halved mu."""
    n_z = 500
    dz_m = 10.0e-6
    mu_true_per_cm = 50.0

    z = np.arange(n_z)
    aline = np.exp(-2.0 * mu_true_per_cm * 100.0 * dz_m * z)
    vol = np.broadcast_to(aline, (4, 4, n_z)).astype(float).copy()

    mu_cm = get_attenuation_vermeer2013(vol, dz=dz_m)
    centre = mu_cm[2, 2, 50:300].mean()

    # The historical bug (denominator including I[i]) produced mu ~= 25 here.
    assert centre > 0.75 * mu_true_per_cm, f"mu under-estimated: {centre:.2f} vs true {mu_true_per_cm}"


# --- New methods (Smith 2015, Liu 2019, Li 2020, Faber 2004) -----------


def _exp_volume(mu_per_cm: float = 50.0, dz_m: float = 10.0e-6, n_z: int = 500, shape_xy: tuple = (8, 8)) -> np.ndarray:
    """Synthetic uniform-attenuation OCT volume in (X, Y, Z) order."""
    z = np.arange(n_z)
    aline = np.exp(-2.0 * mu_per_cm * 100.0 * dz_m * z).astype(np.float32)
    return np.broadcast_to(aline, (*shape_xy, n_z)).astype(np.float32).copy()


def test_extended_alias_emits_deprecation_warning():
    """The legacy name still works but warns."""
    import warnings

    from linumpy.intensity.attenuation import (
        get_attenuation_smith2015,
        get_extended_attenuation_vermeer2013,
    )

    vol = _exp_volume()
    mask = np.ones_like(vol, dtype=bool)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = get_extended_attenuation_vermeer2013(vol, mask=mask, k=0, res=10.0)

    assert any(issubclass(x.category, DeprecationWarning) for x in w)
    direct = get_attenuation_smith2015(vol, mask=mask, k=0, res=10.0)
    assert np.array_equal(out, direct)


def test_liu2019_recovers_uniform_attenuation():
    """Liu 2019 with exact-form regularization should match the truth in the bulk."""
    from linumpy.intensity.attenuation import get_attenuation_liu2019

    mu_true = 50.0
    vol = _exp_volume(mu_per_cm=mu_true, n_z=500)
    mask = np.ones_like(vol, dtype=bool)

    mu_cm = get_attenuation_liu2019(vol, mask=mask, k=0, res=10.0, tail_fit_voxels=30)

    # Central 60 % of the depth axis.
    centre = mu_cm[2, 2, 50:350]
    assert np.allclose(centre, mu_true, rtol=0.02), (
        f"liu mismatch: mean={centre.mean():.3f}, std={centre.std():.3g}, true={mu_true}"
    )


def test_liu2019_tail_more_accurate_than_vermeer():
    """The exact-form C should reduce the tail blow-up vs C=0."""
    from linumpy.intensity.attenuation import get_attenuation_liu2019

    mu_true = 50.0
    vol = _exp_volume(mu_per_cm=mu_true, n_z=300)
    mask = np.ones_like(vol, dtype=bool)

    mu_vermeer = get_attenuation_vermeer2013(vol, dz=10.0e-6)
    mu_liu = get_attenuation_liu2019(vol, mask=mask, k=0, res=10.0, tail_fit_voxels=30)

    # Look at the last 20 voxels (excluding the very last one which is 0 by
    # construction in both methods).
    tail_err_vermeer = np.abs(mu_vermeer[2, 2, -25:-5] - mu_true).mean()
    tail_err_liu = np.abs(mu_liu[2, 2, -25:-5] - mu_true).mean()
    assert tail_err_liu < tail_err_vermeer, (
        f"Liu tail error ({tail_err_liu:.2f}) should be < Vermeer tail error ({tail_err_vermeer:.2f})"
    )


def test_li2020_recovers_attenuation_with_noise_floor():
    """Li 2020 should recover mu when a noise floor is present.

    Builds a 600-voxel A-line with mu = 20 /cm so the signal is well
    above a 1e-3 noise floor for the first ~250 voxels; Li's truncation
    should keep that depth range intact and recover mu within ~15 %.
    """
    from linumpy.intensity.attenuation import get_attenuation_li2020

    mu_true = 20.0
    n_z = 600
    dz_m = 10.0e-6
    z = np.arange(n_z)
    aline = np.exp(-2.0 * mu_true * 100.0 * dz_m * z).astype(np.float32)
    vol = np.broadcast_to(aline, (8, 8, n_z)).astype(np.float32).copy()
    rng = np.random.default_rng(0)
    noise_floor = 1e-3
    vol_noisy = vol + noise_floor + 1e-4 * rng.standard_normal(vol.shape).astype(np.float32)
    mask = np.ones_like(vol_noisy, dtype=bool)

    mu_cm = get_attenuation_li2020(vol_noisy, mask=mask, k=0, res=10.0, snr_threshold_db=6.0, tail_fit_voxels=40)
    # Inspect the well-above-noise region (signal at z=80 is ~3e-4,
    # ~30x the noise std). The trailing voxels approach the truncation
    # depth and inherit Vermeer's tail blow-up.
    centre = mu_cm[2, 2, 30:80]
    valid = centre[centre > 0]
    assert valid.size > 40, f"Li truncation removed too many voxels ({valid.size})"
    assert abs(valid.mean() - mu_true) / mu_true < 0.10, (
        f"li mismatch: mean={valid.mean():.3f}, std={valid.std():.3g}, true={mu_true}"
    )


def test_faber2004_runs_with_noncontiguous_mask():
    """Faber 2004 should run on a non-contiguous mask without crashing.

    Regression guard for the historical shape bug where
    ``z[zp::] - z[zp]`` was passed instead of ``z[mask_aline] - z[zp]``,
    which raised ``ValueError: shape mismatch`` whenever the mask had
    interior gaps. Note: the underlying Lorentzian-times-exponential
    LSQ is sensitive to the hard-coded initial guess (z0=0, zR=100,
    mu_t=1e-3) and is not exercised here for numerical accuracy --
    only that the shapes of the data and depth arrays match.
    """
    from linumpy.intensity.attenuation import get_attenuation_faber2004

    mu_true = 50.0  # 1/cm
    mu_true_per_m = mu_true * 100.0
    dz_m = 10.0e-6
    n_z = 60
    z = np.arange(n_z) * dz_m  # metres
    z0 = 30 * dz_m
    zR = 200e-6
    psf = 1.0 / (((z - z0) / zR) ** 2 + 1.0)
    aline = (psf * np.exp(-2.0 * mu_true_per_m * z)).astype(np.float32)
    vol = np.broadcast_to(aline, (3, 3, n_z)).astype(np.float32).copy()

    # Non-contiguous mask: first valid voxel at index 5, gap at 20-22.
    mask = np.ones((3, 3, n_z), dtype=bool)
    mask[:, :, :5] = False
    mask[:, :, 20:23] = False

    attn, r_length, z_focus = get_attenuation_faber2004(vol, mask=mask, dz=dz_m, N=0)
    assert attn.shape == (3, 3)
    assert r_length.shape == (3, 3)
    assert z_focus.shape == (3, 3)
    assert np.all(np.isfinite(attn))
