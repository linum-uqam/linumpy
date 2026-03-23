# -*- coding: utf-8 -*-
"""
Slice interpolation utilities for missing or degraded serial sections.

Consolidated from linum_interpolate_missing_slice.py.
"""
import numpy as np


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
    import SimpleITK as sitk

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

        half_transform.SetMatrix(half_matrix.flatten().tolist())
        half_transform.SetTranslation((translation / 2.0).tolist())
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

        half_transform.SetMatrix(half_matrix.flatten().tolist())
        half_transform.SetTranslation((translation / 2.0).tolist())
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
    from scipy.ndimage import gaussian_filter

    avg = 0.5 * vol_before.astype(np.float32) + 0.5 * vol_after.astype(np.float32)
    return gaussian_filter(avg, sigma=(sigma, 0, 0))


def interpolate_registration_based(vol_before: np.ndarray,
                                   vol_after: np.ndarray,
                                   metric: str = 'MSE',
                                   max_iterations: int = 1000,
                                   reference_slice: int = None,
                                   blend_method: str = 'gaussian') -> np.ndarray:
    """Interpolate a missing slice using registration-based morphing.

    1. Registers 2D slices from vol_before to vol_after
    2. Computes half-transforms for each z-level
    3. Warps both volumes toward the midpoint
    4. Blends the results using linear or feathered (Gaussian) blending

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
        Z-index for registration reference. Default: middle Z.
    blend_method : str
        'linear' (50/50) or 'gaussian' (feathered distance-transform blend).

    Returns
    -------
    np.ndarray
        Interpolated 3D volume.
    """
    import SimpleITK as sitk
    from linumpy.stitching.registration import register_2d_images_sitk, apply_transform

    nz, nx, ny = vol_before.shape

    if reference_slice is None:
        reference_slice = nz // 2

    fixed_2d = vol_after[reference_slice].astype(np.float32)
    moving_2d = vol_before[reference_slice].astype(np.float32)

    mn, mx = fixed_2d.min(), fixed_2d.max()
    if mx > mn:
        fixed_2d = (fixed_2d - mn) / (mx - mn)
    mn, mx = moving_2d.min(), moving_2d.max()
    if mx > mn:
        moving_2d = (moving_2d - mn) / (mx - mn)

    transform_2d, _, error = register_2d_images_sitk(
        fixed_2d, moving_2d,
        method='affine',
        metric=metric,
        max_iterations=max_iterations,
        return_3d_transform=False,
        verbose=False
    )

    half_transform = compute_half_affine_transform(transform_2d)
    inv_half_transform = half_transform.GetInverse()

    warped_before = np.zeros_like(vol_before, dtype=np.float32)
    warped_after = np.zeros_like(vol_after, dtype=np.float32)

    for z in range(nz):
        warped_before[z] = apply_transform(vol_before[z].astype(np.float32), half_transform)
        warped_after[z] = apply_transform(vol_after[z].astype(np.float32), inv_half_transform)

    if blend_method == 'linear':
        return 0.5 * warped_before + 0.5 * warped_after

    elif blend_method == 'gaussian':
        from scipy.ndimage import distance_transform_edt, gaussian_filter

        mask_before = warped_before > 0
        mask_after = warped_after > 0

        dist_before = np.zeros_like(warped_before, dtype=np.float32)
        dist_after = np.zeros_like(warped_after, dtype=np.float32)

        for z in range(warped_before.shape[0]):
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
    from linumpy.utils.image_quality import (
        compute_ssim_3d, compute_edge_score, compute_variance_score
    )

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
