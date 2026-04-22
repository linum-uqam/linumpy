"""Slice interpolation utilities for missing serial sections.

Single strategy: :func:`interpolate_z_morph` — z-aware morphing via fractional
affine warps (``T**alpha``, :func:`scipy.linalg.fractional_matrix_power`).
Reconstructs a synthetic slice that transitions along Z from ``vol_before[-1]``
to ``vol_after[0]``, matching the physical geometry of serial sectioning.

When any quality gate fails, :func:`interpolate_z_morph` **does not fabricate
a volume**. It returns ``(None, diagnostics)`` with
``diagnostics["interpolation_failed"] = True`` and a specific
``fallback_reason``. The caller must honour this by *not* emitting a
reconstructed slice — a blend of the two neighbours would also be
fabricated data, with the added failure mode of ghost/double-contour
artefacts whenever the two neighbours differ.

Downstream of a failed interpolation, the pipeline treats the slice as a
genuine multi-slice gap: no zarr is produced, the manifest fragment records
``interpolation_failed=true``, and ``slice_config_final.csv`` surfaces the
failure so the final report can flag it.

Returns ``(volume, diagnostics)`` where *volume* is ``None`` on failure and
*diagnostics* is always a JSON-serialisable dict.

See ``docs/SLICE_INTERPOLATION_FEATURE.md`` for the scientific rationale,
failure modes, and the connection to ``slice_config.csv`` (``interpolated``,
``interpolation_failed``, ``interpolation_fallback_reason`` columns).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy.linalg import fractional_matrix_power
from scipy.ndimage import distance_transform_edt, gaussian_filter

from linumpy.registration.transforms import apply_transform, register_2d_images_sitk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalization / NCC helpers
# ---------------------------------------------------------------------------


def _normalize_plane_for_ncc(plane: np.ndarray) -> np.ndarray:
    """Return a zero-mean / unit-std plane suitable for normalised CC."""
    crop = plane.astype(np.float32)
    valid = crop > 0
    if valid.any():
        pmin = float(np.percentile(crop[valid], 5))
        pmax = float(np.percentile(crop[valid], 95))
        crop = np.clip((crop - pmin) / max(pmax - pmin, 1e-8), 0, 1)
    return (crop - crop.mean()) / (crop.std() + 1e-8)


def _ncc(a: np.ndarray, b: np.ndarray, margin_frac: float = 0.25) -> float:
    """Normalised cross-correlation of two 2D images on their central ROI."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch in NCC: {a.shape} vs {b.shape}")
    h, w = a.shape
    margin = int(min(h, w) * margin_frac)
    roi = (slice(margin, h - margin), slice(margin, w - margin))
    an = _normalize_plane_for_ncc(a[roi])
    bn = _normalize_plane_for_ncc(b[roi])
    return float(np.mean(an * bn))


def _foreground_fraction(plane: np.ndarray, threshold: float | None = None) -> float:
    """Fraction of pixels above a background threshold.

    When *threshold* is None, uses the 1st percentile of positive values as a
    soft background estimate. This makes the function robust to common OCT
    volumes that have a non-trivial dark offset.
    """
    if plane.size == 0:
        return 0.0
    if threshold is None:
        positive = plane[plane > 0]
        if positive.size == 0:
            return 0.0
        threshold = float(np.percentile(positive, 1))
    return float((plane > threshold).mean())


# ---------------------------------------------------------------------------
# Affine / fractional-affine helpers
# ---------------------------------------------------------------------------


def _matrix_fractional_power(matrix: np.ndarray, alpha: float, imag_tol: float = 1e-4) -> tuple[np.ndarray, float]:
    """Return ``matrix ** alpha`` as a real matrix, plus the max imaginary magnitude.

    Uses :func:`scipy.linalg.fractional_matrix_power` which internally
    performs a Schur decomposition and is numerically more stable than a
    bare eigen-decomposition with ``.real`` truncation.

    Returns
    -------
    real_part : np.ndarray
        Real part of the fractional power.
    imag_magnitude : float
        Maximum ``|imag|`` relative to ``max(|real|, 1)``. A large value
        indicates the matrix had negative-real-eigenvalue components (e.g.
        reflection) and the real projection is *not* a valid power.
    """
    with np.errstate(all="ignore"):
        m_alpha = fractional_matrix_power(matrix, alpha)
    if np.iscomplexobj(m_alpha):
        scale = max(float(np.max(np.abs(m_alpha.real))), 1.0)
        imag_mag = float(np.max(np.abs(m_alpha.imag)) / scale)
        m_alpha = m_alpha.real
    else:
        imag_mag = 0.0
    return m_alpha, imag_mag


def _fractional_affine_parts(
    matrix: np.ndarray, translation: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute ``T**alpha`` for an affine transform given by (matrix, translation, centre).

    The affine acts as ``T(x) = M(x - c) + c + t``. For fractional alpha, we
    keep the centre fixed and compute:

        M_alpha = M ** alpha
        t_alpha = (I - M_alpha) @ (I - M)^{-1} @ t

    Derivation: write ``T(x) = M x + b`` with ``b = (I - M) c + t``. Iterating
    this k times gives ``M^k x + (sum_{i=0}^{k-1} M^i) b``. The closed form
    ``(I - M^alpha)(I - M)^{-1}`` continuously extends the geometric sum to
    real alpha. Substituting ``b`` back and using a fixed centre gives the
    formula above.

    Sanity checks:
      alpha=1 → M_alpha=M, t_alpha=t.
      alpha=0 → M_alpha=I, t_alpha=0.
      alpha=0.5 → (M_alpha + I) @ t_alpha = t, matching the half-transform
      relation used throughout this module.

    When ``I - M`` is near-singular (pure translation / near-identity
    matrix), we fall back to the linear approximation ``t_alpha ≈ alpha * t``
    which is exact for M = I.
    """
    dim = matrix.shape[0]
    identity = np.eye(dim)

    # Exact shortcuts for alpha∈{0, 1} — avoids numerical drift from
    # fractional_matrix_power that would move identity pixels around and
    # break the "boundary planes match sources exactly" guarantee.
    if alpha == 0.0:
        return identity, np.zeros(dim, dtype=np.float64), 0.0
    if alpha == 1.0:
        return np.asarray(matrix, dtype=np.float64), np.asarray(translation, dtype=np.float64), 0.0

    m_alpha, imag_mag = _matrix_fractional_power(matrix, alpha)

    diff = identity - matrix
    if abs(np.linalg.det(diff)) < 1e-10:
        t_alpha = alpha * np.asarray(translation, dtype=np.float64)
    else:
        acc = np.linalg.solve(diff, np.asarray(translation, dtype=np.float64))
        t_alpha = (identity - m_alpha) @ acc
    return m_alpha, t_alpha, imag_mag


# ---------------------------------------------------------------------------
# Simple interpolators (unchanged API; used as fallbacks)
# ---------------------------------------------------------------------------


def interpolate_average(vol_before: np.ndarray, vol_after: np.ndarray) -> np.ndarray:
    """Simple 50/50 average of two adjacent volumes."""
    return 0.5 * vol_before.astype(np.float32) + 0.5 * vol_after.astype(np.float32)


def interpolate_weighted(vol_before: np.ndarray, vol_after: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Weighted average with Gaussian smoothing along Z."""
    avg = 0.5 * vol_before.astype(np.float32) + 0.5 * vol_after.astype(np.float32)
    return gaussian_filter(avg, sigma=(sigma, 0, 0))


# ---------------------------------------------------------------------------
# Boundary plane / slab selection
# ---------------------------------------------------------------------------


def _build_reference_slab(vol: np.ndarray, z_center: int, slab_size: int) -> np.ndarray:
    """Mean-intensity projection over *slab_size* planes centred at *z_center*.

    Clamps to the volume bounds. A 1-plane slab returns the plane itself.
    """
    nz = vol.shape[0]
    half = max(1, slab_size) // 2
    lo = max(0, z_center - half)
    hi = min(nz, z_center + half + 1)
    return vol[lo:hi].mean(axis=0).astype(np.float32)


def find_best_overlap_planes(
    vol_before: np.ndarray,
    vol_after: np.ndarray,
    search_window: int = 5,
    min_foreground_fraction: float = 0.1,
) -> tuple[int, int, float]:
    """Find the best-correlated plane pair at the volume boundary.

    In serial sectioning the physically adjacent tissue is near the **bottom**
    of *vol_before* and the **top** of *vol_after*. This function searches
    the last ``search_window`` planes of *vol_before* against the first
    ``search_window`` planes of *vol_after* using normalised cross-correlation
    on the central ROI, skipping planes whose foreground fraction is below
    ``min_foreground_fraction``.

    Returns ``(ref_before, ref_after, best_corr)``. When no candidate pair
    passes the foreground filter, the corner planes are returned with a
    correlation of ``-inf``.
    """
    nz_before = vol_before.shape[0]
    nz_after = vol_after.shape[0]

    before_zs = [
        z
        for z in range(max(0, nz_before - search_window), nz_before)
        if _foreground_fraction(vol_before[z]) >= min_foreground_fraction
    ]
    after_zs = [
        z for z in range(min(search_window, nz_after)) if _foreground_fraction(vol_after[z]) >= min_foreground_fraction
    ]

    if not before_zs or not after_zs:
        logger.warning(
            "find_best_overlap_planes: no candidate plane passed the foreground filter (before_zs=%s, after_zs=%s)",
            before_zs,
            after_zs,
        )
        return nz_before - 1, 0, float("-inf")

    h, w = vol_before.shape[1], vol_before.shape[2]
    margin = min(h, w) // 4
    roi = (slice(margin, h - margin), slice(margin, w - margin))
    # Normalise on the ROI (not the full plane) so the resulting arrays are
    # zero-mean / unit-std over the region that actually goes into the NCC.
    # Normalising on the full plane — where OCT backgrounds are mostly zero —
    # leaves the central tissue ROI with a strongly positive mean, which
    # inflates `mean(a*b)` well beyond the [-1, 1] range expected for NCC.
    before_norms = {z: _normalize_plane_for_ncc(vol_before[z][roi]) for z in before_zs}
    after_norms = {z: _normalize_plane_for_ncc(vol_after[z][roi]) for z in after_zs}

    best_corr = -np.inf
    ref_before = before_zs[-1]
    ref_after = after_zs[0]
    for zb in before_zs:
        for za in after_zs:
            corr = float(np.mean(before_norms[zb] * after_norms[za]))
            if corr > best_corr:
                best_corr = corr
                ref_before = zb
                ref_after = za

    return ref_before, ref_after, best_corr


# ---------------------------------------------------------------------------
# 2D affine registration wrapper shared by both interpolators
# ---------------------------------------------------------------------------


def _prepare_2d(plane: np.ndarray) -> np.ndarray:
    """Normalise a 2D plane to [0, 1] for registration."""
    plane = plane.astype(np.float32)
    mn, mx = float(plane.min()), float(plane.max())
    if mx > mn:
        return (plane - mn) / (mx - mn)
    return plane


def _register_boundary(
    fixed_2d: np.ndarray,
    moving_2d: np.ndarray,
    metric: str,
    max_iterations: int,
) -> sitk.Transform:
    transform, _, _ = register_2d_images_sitk(
        fixed_2d,
        moving_2d,
        method="affine",
        metric=metric,
        max_iterations=max_iterations,
        return_3d_transform=False,
        verbose=False,
    )
    return transform


# ---------------------------------------------------------------------------
# Blend helpers
# ---------------------------------------------------------------------------


def _gaussian_feather_blend(
    warped_before: np.ndarray,
    warped_after: np.ndarray,
    w_before: np.ndarray | None = None,
    w_after: np.ndarray | None = None,
) -> np.ndarray:
    """Per-plane distance-transform feather blend.

    Combines an XY edge feather (via the distance transform of each input's
    foreground mask) with optional per-plane z-weights ``w_before`` /
    ``w_after`` (shape ``(nz,)``). The z-weights are authoritative: when
    ``w_before[z] = 1`` and ``w_after[z] = 0`` the output at plane ``z`` is
    exactly ``warped_before[z]`` within its foreground region, even in pixels
    where only ``warped_after`` has data (those pixels remain 0, mirroring
    the input). Out-of-mask regions of one source are filled from the other
    only when the corresponding z-weight is non-zero.
    """
    nz, nx, ny = warped_before.shape
    mask_before = warped_before > 0
    mask_after = warped_after > 0

    dist_before = np.zeros((nz, nx, ny), dtype=np.float32)
    dist_after = np.zeros((nz, nx, ny), dtype=np.float32)
    for z in range(nz):
        if mask_before[z].any():
            dist_before[z] = distance_transform_edt(mask_before[z])
        if mask_after[z].any():
            dist_after[z] = distance_transform_edt(mask_after[z])

    dist_before = gaussian_filter(dist_before, sigma=(0, 2, 2))
    dist_after = gaussian_filter(dist_after, sigma=(0, 2, 2))

    zw_before = np.ones((nz,), dtype=np.float32) if w_before is None else np.asarray(w_before, dtype=np.float32).reshape(-1)
    zw_after = np.ones((nz,), dtype=np.float32) if w_after is None else np.asarray(w_after, dtype=np.float32).reshape(-1)

    weighted_before = dist_before * zw_before.reshape(-1, 1, 1)
    weighted_after = dist_after * zw_after.reshape(-1, 1, 1)

    total = weighted_before + weighted_after + 1e-10
    wb = weighted_before / total
    wa = weighted_after / total

    # Only-X-side regions: fall back to that side at its z-weight (not 1).
    # This keeps boundary planes (one side fully zeroed out) matching the
    # other side exactly without polluting only-X regions with "ghost" data
    # that the active side never intended to contribute.
    only_before = mask_before & ~mask_after
    only_after = mask_after & ~mask_before
    zb_bcast = np.broadcast_to(zw_before.reshape(-1, 1, 1), wb.shape)
    za_bcast = np.broadcast_to(zw_after.reshape(-1, 1, 1), wa.shape)
    wb = np.where(only_before, zb_bcast, wb)
    wa = np.where(only_before, 0.0, wa)
    wb = np.where(only_after, 0.0, wb)
    wa = np.where(only_after, za_bcast, wa)

    return wb * warped_before + wa * warped_after


# ---------------------------------------------------------------------------
# z-morph interpolation: physical-geometry aware
# ---------------------------------------------------------------------------


def _fractional_affine_transform(
    matrix: np.ndarray,
    translation: np.ndarray,
    center: np.ndarray,
    alpha: float,
    dim: int = 2,
) -> sitk.AffineTransform:
    """Construct a SimpleITK affine for ``T**alpha`` with a fixed centre."""
    m_alpha, t_alpha, _imag = _fractional_affine_parts(matrix, translation, alpha)
    tform = sitk.AffineTransform(dim)
    tform.SetMatrix(m_alpha.flatten().tolist())
    tform.SetTranslation(t_alpha.tolist())
    tform.SetCenter(center.tolist())
    return tform


def interpolate_z_morph(
    vol_before: np.ndarray,
    vol_after: np.ndarray,
    output_z: int | None = None,
    metric: str = "MSE",
    max_iterations: int = 1000,
    overlap_search_window: int = 5,
    min_overlap_correlation: float = 0.3,
    reference_slab_size: int = 3,
    min_foreground_fraction: float = 0.1,
    min_ncc_improvement: float = 0.05,
    blend_method: str = "gaussian",
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Z-aware morphing interpolation.

    Registers ``vol_after[0]`` to ``vol_before[-1]`` to get an affine ``T``,
    then for each output plane at fractional depth ``alpha ∈ [0, 1]`` warps
    the before-boundary by ``T**alpha`` and the after-boundary by
    ``T**(alpha - 1)``, cross-fading with weight ``alpha``. Output top/bottom
    planes match the boundary planes exactly.

    **Hard skip on gate failure.** When any quality gate fails
    (``no_foreground_planes``, ``low_overlap_ncc``, ``registration_exception``,
    ``reg_did_not_improve``, ``affine_determinant_non_positive``) the function
    returns ``(None, diagnostics)`` with
    ``diagnostics["interpolation_failed"] = True`` and a specific
    ``fallback_reason``. **No fabricated volume is produced** — blending the
    two neighbours would also be made-up data, with the added failure mode of
    ghost/double-contour artefacts whenever the two neighbours differ.

    See ``docs/SLICE_INTERPOLATION_FEATURE.md`` for the physical model, the
    rationale for the hard-skip behaviour, and parameter-tuning guidance.

    Returns
    -------
    volume : np.ndarray | None
        Interpolated 3D volume, shape ``(output_z or min(nz_before, nz_after), H, W)``,
        or ``None`` when a quality gate fails.
    diagnostics : dict
        JSON-serialisable trace of the attempt.
    """
    nz_before, nx, ny = vol_before.shape
    nz_after = vol_after.shape[0]
    nz_out = output_z if output_z is not None else min(nz_before, nz_after)

    diag: dict[str, Any] = {
        "method": "zmorph",
        "method_used": "zmorph",
        "fallback_reason": None,
        "nz_out": int(nz_out),
        "overlap_search_window": overlap_search_window,
        "reference_slab_size": reference_slab_size,
        "min_foreground_fraction": min_foreground_fraction,
        "min_overlap_correlation": min_overlap_correlation,
        "min_ncc_improvement": min_ncc_improvement,
        "blend_method": blend_method,
        "registration_metric": metric,
        "max_iterations": max_iterations,
    }

    # -- Boundary plane/slab selection --------------------------------------
    ref_before, ref_after, best_corr = find_best_overlap_planes(
        vol_before,
        vol_after,
        search_window=overlap_search_window,
        min_foreground_fraction=min_foreground_fraction,
    )
    diag["ref_before"] = int(ref_before)
    diag["ref_after"] = int(ref_after)
    diag["pre_reg_ncc"] = float(best_corr)

    def _hard_skip(reason: str) -> tuple[None, dict[str, Any]]:
        """Abort interpolation without fabricating a volume.

        A weighted blend of the two neighbours would also be made-up data
        (doubled contours, ghosted structures). Honest behaviour is to emit
        no output and let the pipeline treat the slot as a genuine gap.
        """
        diag["method_used"] = None
        diag["fallback_reason"] = reason
        diag["interpolation_failed"] = True
        logger.warning(
            "[interpolation] zmorph could not produce a reliable output (reason=%s); "
            "emitting no volume (slot will stay empty in the final reconstruction)",
            reason,
        )
        return None, diag

    if not np.isfinite(best_corr) or best_corr < min_overlap_correlation:
        return _hard_skip("no_foreground_planes" if not np.isfinite(best_corr) else "low_overlap_ncc")

    slab_before = _build_reference_slab(vol_before, ref_before, reference_slab_size)
    slab_after = _build_reference_slab(vol_after, ref_after, reference_slab_size)

    fixed_2d = _prepare_2d(slab_after)
    moving_2d = _prepare_2d(slab_before)

    try:
        transform_2d = _register_boundary(fixed_2d, moving_2d, metric=metric, max_iterations=max_iterations)
    except Exception as exc:
        diag["registration_error_message"] = str(exc)
        return _hard_skip("registration_exception")

    affine_2d = sitk.AffineTransform(transform_2d)
    matrix = np.array(affine_2d.GetMatrix()).reshape(2, 2)
    translation = np.array(affine_2d.GetTranslation())
    center = np.array(affine_2d.GetCenter())

    warped_slab_before = apply_transform(slab_before.astype(np.float32), transform_2d)
    post_reg_ncc = _ncc(slab_after, warped_slab_before)
    diag["post_reg_ncc"] = float(post_reg_ncc)
    diag["ncc_improvement"] = float(post_reg_ncc - best_corr)
    diag["affine_matrix"] = matrix.tolist()
    diag["affine_translation"] = translation.tolist()
    diag["affine_determinant"] = float(np.linalg.det(matrix))

    if post_reg_ncc - best_corr < min_ncc_improvement:
        return _hard_skip("reg_did_not_improve")

    det = float(np.linalg.det(matrix))
    if det <= 0.0:
        diag["affine_determinant_invalid"] = True
        return _hard_skip("affine_determinant_non_positive")

    # -- Build the morphed output ------------------------------------------
    top_of_after = vol_after[0].astype(np.float32)
    bottom_of_before = vol_before[-1].astype(np.float32)

    warped_before = np.zeros((nz_out, nx, ny), dtype=np.float32)
    warped_after = np.zeros((nz_out, nx, ny), dtype=np.float32)
    w_before_list: list[float] = []
    max_imag = 0.0

    for z in range(nz_out):
        alpha = z / (nz_out - 1) if nz_out > 1 else 0.5

        # before contribution: warp by T**alpha (alpha ∈ [0, 1])
        m_a, t_a, imag_a = _fractional_affine_parts(matrix, translation, alpha)
        max_imag = max(max_imag, imag_a)
        before_tform = sitk.AffineTransform(2)
        before_tform.SetMatrix(m_a.flatten().tolist())
        before_tform.SetTranslation(t_a.tolist())
        before_tform.SetCenter(center.tolist())
        warped_before[z] = apply_transform(bottom_of_before, before_tform)

        # after contribution: warp by T**(alpha - 1) (alpha - 1 ∈ [-1, 0])
        m_a2, t_a2, imag_b = _fractional_affine_parts(matrix, translation, alpha - 1.0)
        max_imag = max(max_imag, imag_b)
        after_tform = sitk.AffineTransform(2)
        after_tform.SetMatrix(m_a2.flatten().tolist())
        after_tform.SetTranslation(t_a2.tolist())
        after_tform.SetCenter(center.tolist())
        warped_after[z] = apply_transform(top_of_after, after_tform)

        w_before_list.append(1.0 - alpha)

    diag["fractional_power_max_imag"] = float(max_imag)
    if max_imag > 1e-3:
        diag["warning_large_imag_part"] = True

    w_before_arr = np.asarray(w_before_list, dtype=np.float32)
    w_after_arr = 1.0 - w_before_arr

    if blend_method == "linear":
        result = w_before_arr.reshape(-1, 1, 1) * warped_before + w_after_arr.reshape(-1, 1, 1) * warped_after
    elif blend_method == "gaussian":
        result = _gaussian_feather_blend(warped_before, warped_after, w_before=w_before_arr, w_after=w_after_arr)
    else:
        raise ValueError(f"Unknown blend_method: {blend_method}")

    # Quality stats at the two boundaries (should be near-perfect)
    diag["top_boundary_residual_mean"] = float(np.abs(result[0] - bottom_of_before).mean())
    diag["bottom_boundary_residual_mean"] = float(np.abs(result[-1] - top_of_after).mean())

    return result, diag
