# -*- coding: utf-8 -*-
"""
Slice interpolation utilities for missing or degraded serial sections.

Consolidated from linum_interpolate_missing_slice.py.
"""
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, gaussian_filter

from linumpy.stitching.registration import register_2d_images_sitk, apply_transform
from linumpy.utils.image_quality import (
    compute_ssim_3d, compute_edge_score, compute_variance_score
)


def compute_half_affine_transform(transform):
    """Compute a transform that is 'halfway' to the given transform.

    For affine transforms: decomposes the transform matrix via eigendecomposition
    and applies the matrix square root (half rotation + half translation).

    Parameters
    ----------
    transform : sitk.Transform
        Full transform from image A to image B.

    Returns
    -------
    sitk.AffineTransform
        Transform representing half the transformation.
    """
    if isinstance(transform, sitk.CompositeTransform):
        transform = sitk.AffineTransform(transform.GetNthTransform(0))

    dim = transform.GetDimension()

    if dim == 2:
        half_transform = sitk.AffineTransform(2)
        matrix = np.array(transform.GetMatrix()).reshape(2, 2)
        translation = np.array(transform.GetTranslation())
        center = np.array(transform.GetCenter())

        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        sqrt_eigenvalues = np.sqrt(eigenvalues.astype(complex))
        half_matrix = (eigenvectors @ np.diag(sqrt_eigenvalues) @
                       np.linalg.inv(eigenvectors)).real

        # Correct half-translation: h(h(x)) = T(x) requires
        # (H_m + I) * h_t = t  =>  h_t = (H_m + I)^{-1} * t
        half_translation = np.linalg.solve(
            half_matrix + np.eye(2), translation
        )

        half_transform.SetMatrix(half_matrix.flatten().tolist())
        half_transform.SetTranslation(half_translation.tolist())
        half_transform.SetCenter(center.tolist())

    elif dim == 3:
        half_transform = sitk.AffineTransform(3)
        matrix = np.array(transform.GetMatrix()).reshape(3, 3)
        translation = np.array(transform.GetTranslation())
        center = np.array(transform.GetCenter())

        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        sqrt_eigenvalues = np.sqrt(eigenvalues.astype(complex))
        half_matrix = (eigenvectors @ np.diag(sqrt_eigenvalues) @
                       np.linalg.inv(eigenvectors)).real

        half_translation = np.linalg.solve(
            half_matrix + np.eye(3), translation
        )

        half_transform.SetMatrix(half_matrix.flatten().tolist())
        half_transform.SetTranslation(half_translation.tolist())
        half_transform.SetCenter(center.tolist())
    else:
        raise ValueError(f"Unsupported transform dimension: {dim}")

    return half_transform


def interpolate_average(vol_before: np.ndarray, vol_after: np.ndarray) -> np.ndarray:
    """Simple 50/50 average of two adjacent volumes.

    Parameters
    ----------
    vol_before : np.ndarray
        Volume before missing slice (Z, X, Y).
    vol_after : np.ndarray
        Volume after missing slice (Z, X, Y).

    Returns
    -------
    np.ndarray
        Average volume.
    """
    return 0.5 * vol_before.astype(np.float32) + 0.5 * vol_after.astype(np.float32)


def interpolate_weighted(vol_before: np.ndarray,
                         vol_after: np.ndarray,
                         sigma: float = 2.0) -> np.ndarray:
    """Weighted average with Gaussian smoothing along Z.

    Parameters
    ----------
    vol_before : np.ndarray
        Volume before missing slice.
    vol_after : np.ndarray
        Volume after missing slice.
    sigma : float
        Gaussian smoothing sigma along Z-axis.

    Returns
    -------
    np.ndarray
        Weighted average.
    """
    avg = 0.5 * vol_before.astype(np.float32) + 0.5 * vol_after.astype(np.float32)
    return gaussian_filter(avg, sigma=(sigma, 0, 0))


def find_best_overlap_planes(vol_before: np.ndarray,
                             vol_after: np.ndarray,
                             search_window: int = 5):
    """Find the best-correlated plane pair at the boundary between two volumes.

    In serial sectioning, each OCT volume images the tissue surface that
    remains after removing a physical slice. The physically adjacent tissue
    is therefore near the **bottom** of *vol_before* and the **top** of
    *vol_after*. Because the exact cut depth can vary slightly, this function
    searches the last *search_window* z-planes of *vol_before* against the
    first *search_window* z-planes of *vol_after* using normalized
    cross-correlation on the central ROI, and returns the pair with the
    highest correlation together with that correlation score.

    The returned correlation also serves as a quality gate: a low score (e.g.
    below ~0.1) indicates that no reliable structural match was found and the
    caller should fall back to a simpler interpolation strategy.

    Parameters
    ----------
    vol_before : np.ndarray
        3D volume (Z, X, Y) before the missing slice.
    vol_after : np.ndarray
        3D volume (Z, X, Y) after the missing slice.
    search_window : int
        Number of z-planes to search at each boundary. Default 5.

    Returns
    -------
    ref_before : int
        Best z-index in *vol_before*.
    ref_after : int
        Best z-index in *vol_after*.
    best_corr : float
        Normalized cross-correlation at the best pair (range approximately
        [-1, 1]; higher is better).
    """
    nz_before = vol_before.shape[0]
    nz_after = vol_after.shape[0]
    h, w = vol_before.shape[1], vol_before.shape[2]
    margin = min(h, w) // 4
    roi = (slice(margin, h - margin), slice(margin, w - margin))

    def _norm_roi(plane):
        crop = plane[roi].astype(np.float32)
        valid = crop > 0
        if valid.any():
            pmin = float(np.percentile(crop[valid], 5))
            pmax = float(np.percentile(crop[valid], 95))
            crop = np.clip((crop - pmin) / max(pmax - pmin, 1e-8), 0, 1)
        return (crop - crop.mean()) / (crop.std() + 1e-8)

    before_zs = range(max(0, nz_before - search_window), nz_before)
    after_zs = range(0, min(search_window, nz_after))

    before_norms = {z: _norm_roi(vol_before[z]) for z in before_zs}
    after_norms = {z: _norm_roi(vol_after[z]) for z in after_zs}

    best_corr = -np.inf
    ref_before = nz_before - 1
    ref_after = 0

    for zb in before_zs:
        for za in after_zs:
            corr = float(np.mean(before_norms[zb] * after_norms[za]))
            if corr > best_corr:
                best_corr = corr
                ref_before = zb
                ref_after = za

    return ref_before, ref_after, best_corr


def interpolate_registration_based(vol_before: np.ndarray,
                                   vol_after: np.ndarray,
                                   metric: str = 'MSE',
                                   max_iterations: int = 1000,
                                   reference_slice: int | None = None,
                                   blend_method: str = 'gaussian',
                                   overlap_search_window: int = 5,
                                   min_overlap_correlation: float = 0.1) -> np.ndarray:
    """Interpolate a missing slice using registration-based morphing.

    1. Finds the best-correlated plane pair at the volume boundary using
       ``find_best_overlap_planes`` (quality gate + best reference selection)
    2. Registers that pair of 2D planes to obtain the XY alignment transform
    3. Computes the half-transform representing the midpoint transformation
    4. Warps both volumes toward the midpoint
    5. Blends the results using linear or feathered (Gaussian) blending

    If the best overlap correlation is below *min_overlap_correlation*, the
    volumes cannot be reliably aligned and a simple average is returned instead.

    Parameters
    ----------
    vol_before : np.ndarray
        3D volume (Z, X, Y) before the missing slice.
    vol_after : np.ndarray
        3D volume (Z, X, Y) after the missing slice.
    metric : str
        Registration metric: 'MSE', 'CC', or 'MI'.
    max_iterations : int
        Maximum registration iterations.
    reference_slice : int or None
        When provided, overrides automatic plane selection and uses this
        z-index (clamped to each volume's bounds) as the registration
        reference in both volumes. When *None* (default),
        ``find_best_overlap_planes`` selects the best plane pair within
        *overlap_search_window* planes from each boundary.
    blend_method : str
        'linear' (50/50) or 'gaussian' (feathered distance-transform blend).
    overlap_search_window : int
        Number of z-planes to search at each boundary when selecting the
        reference plane pair automatically. Ignored when *reference_slice*
        is set. Default 5.
    min_overlap_correlation : float
        Minimum normalized cross-correlation required to proceed with
        registration. Below this threshold the volumes are considered
        mismatched and a plain average is returned. Default 0.1.

    Returns
    -------
    np.ndarray
        Interpolated 3D volume.
    """
    nz_before, nx, ny = vol_before.shape
    nz_after = vol_after.shape[0]
    nz_out = min(nz_before, nz_after)

    if reference_slice is None:
        ref_before, ref_after, best_corr = find_best_overlap_planes(
            vol_before, vol_after, search_window=overlap_search_window
        )
        if best_corr < min_overlap_correlation:
            print(f"  [interpolation] Overlap correlation {best_corr:.3f} is below threshold "
                  f"{min_overlap_correlation:.3f} — falling back to simple average.")
            return interpolate_average(vol_before[:nz_out], vol_after[:nz_out])
        print(f"  [interpolation] Best overlap: before[{ref_before}] ↔ after[{ref_after}] "
              f"(corr={best_corr:.3f})")
    else:
        ref_before = min(reference_slice, nz_before - 1)
        ref_after = min(reference_slice, nz_after - 1)

    fixed_2d = vol_after[ref_after].astype(np.float32)
    moving_2d = vol_before[ref_before].astype(np.float32)

    mn, mx = fixed_2d.min(), fixed_2d.max()
    if mx > mn:
        fixed_2d = (fixed_2d - mn) / (mx - mn)
    mn, mx = moving_2d.min(), moving_2d.max()
    if mx > mn:
        moving_2d = (moving_2d - mn) / (mx - mn)

    transform_2d, _, _ = register_2d_images_sitk(
        fixed_2d, moving_2d,
        method='affine',
        metric=metric,
        max_iterations=max_iterations,
        return_3d_transform=False,
        verbose=False
    )

    half_transform = compute_half_affine_transform(transform_2d)
    inv_half_transform = half_transform.GetInverse()

    warped_before = np.zeros((nz_out, nx, ny), dtype=np.float32)
    warped_after = np.zeros((nz_out, nx, ny), dtype=np.float32)

    for z in range(nz_out):
        warped_before[z] = apply_transform(vol_before[z].astype(np.float32), half_transform)
        warped_after[z] = apply_transform(vol_after[z].astype(np.float32), inv_half_transform)

    if blend_method == 'linear':
        return 0.5 * warped_before + 0.5 * warped_after

    elif blend_method == 'gaussian':
        mask_before = warped_before > 0
        mask_after = warped_after > 0

        dist_before = np.zeros((nz_out, nx, ny), dtype=np.float32)
        dist_after = np.zeros((nz_out, nx, ny), dtype=np.float32)

        for z in range(nz_out):
            if np.any(mask_before[z]):
                dist_before[z] = distance_transform_edt(mask_before[z])
            if np.any(mask_after[z]):
                dist_after[z] = distance_transform_edt(mask_after[z])

        dist_before = gaussian_filter(dist_before, sigma=(0, 2, 2))
        dist_after = gaussian_filter(dist_after, sigma=(0, 2, 2))

        total_dist = dist_before + dist_after + 1e-10
        w_before = dist_before / total_dist
        w_after = dist_after / total_dist

        only_before = mask_before & ~mask_after
        only_after = mask_after & ~mask_before
        w_before[only_before] = 1.0
        w_after[only_before] = 0.0
        w_before[only_after] = 0.0
        w_after[only_after] = 1.0

        return w_before * warped_before + w_after * warped_after

    raise ValueError(f"Unknown blend_method: {blend_method}")


def assess_degraded_slice_quality(vol_degraded: np.ndarray,
                                  vol_before: np.ndarray,
                                  vol_after: np.ndarray):
    """Automatically assess the quality of a degraded slice.

    Uses SSIM (weight 0.5), edge preservation (0.3), and variance (0.2).

    Parameters
    ----------
    vol_degraded : np.ndarray
        The degraded slice volume.
    vol_before : np.ndarray
        Volume before the degraded slice.
    vol_after : np.ndarray
        Volume after the degraded slice.

    Returns
    -------
    quality_score : float
        Score from 0 (unusable) to 1 (perfect).
    metrics : dict
        Individual metric scores.
    """
    reference = 0.5 * vol_before.astype(np.float32) + 0.5 * vol_after.astype(np.float32)

    ssim_before = compute_ssim_3d(vol_degraded, vol_before)
    ssim_after = compute_ssim_3d(vol_degraded, vol_after)
    ssim_score = (ssim_before + ssim_after) / 2

    edge_score = compute_edge_score(vol_degraded, reference)
    variance_score = compute_variance_score(vol_degraded, reference)

    quality_score = 0.5 * ssim_score + 0.3 * edge_score + 0.2 * variance_score

    metrics = {
        'ssim_before': ssim_before,
        'ssim_after': ssim_after,
        'ssim_mean': ssim_score,
        'edge_preservation': edge_score,
        'variance_ratio': variance_score,
        'overall': quality_score
    }

    return quality_score, metrics


def blend_with_degraded(interpolated: np.ndarray,
                        degraded: np.ndarray,
                        quality_weight: float) -> np.ndarray:
    """Blend an interpolated result with a degraded slice weighted by quality.

    Parameters
    ----------
    interpolated : np.ndarray
        Pure interpolated volume.
    degraded : np.ndarray
        Degraded slice volume.
    quality_weight : float
        Weight for degraded slice (0 = use interpolated, 1 = use degraded).

    Returns
    -------
    np.ndarray
        Blended result.
    """
    w = quality_weight
    return w * degraded.astype(np.float32) + (1 - w) * interpolated.astype(np.float32)
