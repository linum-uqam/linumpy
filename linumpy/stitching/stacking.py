# -*- coding: utf-8 -*-
"""
3D slice stacking utilities.

Consolidated from linum_stack_slices_motor.py and linum_stack_motor_only.py.
"""
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def find_z_overlap(fixed_vol: np.ndarray,
                   moving_vol: np.ndarray,
                   slicing_interval_mm: float,
                   search_range_mm: float,
                   resolution_um: float) -> Tuple[int, float]:
    """Find optimal Z-overlap between consecutive slices using cross-correlation.

    Searches around the expected overlap for the best normalized
    cross-correlation score, using the center XY region for speed.

    Parameters
    ----------
    fixed_vol : np.ndarray
        Bottom (fixed) slice volume (Z, Y, X).
    moving_vol : np.ndarray
        Top (moving) slice volume (Z, Y, X).
    slicing_interval_mm : float
        Expected physical slice thickness in mm.
    search_range_mm : float
        Search range around expected position in mm.
    resolution_um : float
        Z resolution in microns per voxel.

    Returns
    -------
    best_overlap : int
        Optimal overlap in Z voxels.
    best_corr : float
        Correlation score at optimal overlap.
    """
    interval_vox = int((slicing_interval_mm * 1000) / resolution_um)
    expected_overlap_vox = min(fixed_vol.shape[0], moving_vol.shape[0]) - interval_vox
    search_range_vox = int((search_range_mm * 1000) / resolution_um)

    min_overlap = max(1, expected_overlap_vox - search_range_vox)
    max_overlap = min(fixed_vol.shape[0], moving_vol.shape[0],
                      expected_overlap_vox + search_range_vox)

    if min_overlap >= max_overlap:
        return expected_overlap_vox, 0.0

    h, w = fixed_vol.shape[1], fixed_vol.shape[2]
    margin = min(h, w) // 4
    y_slice = slice(margin, h - margin)
    x_slice = slice(margin, w - margin)

    best_overlap = expected_overlap_vox
    best_corr = -np.inf

    for overlap in range(min_overlap, max_overlap + 1):
        fixed_region = fixed_vol[-overlap:, y_slice, x_slice]
        moving_region = moving_vol[:overlap, y_slice, x_slice]

        fixed_norm = (fixed_region - fixed_region.mean()) / (fixed_region.std() + 1e-8)
        moving_norm = (moving_region - moving_region.mean()) / (moving_region.std() + 1e-8)

        corr = np.mean(fixed_norm * moving_norm)
        if corr > best_corr:
            best_corr = corr
            best_overlap = overlap

    return best_overlap, best_corr


def apply_2d_transform(image_2d: np.ndarray,
                       transform,
                       rotation_only: bool = False,
                       max_rotation_deg: float = 1.0,
                       override_rotation=None) -> np.ndarray:
    """Apply a SimpleITK 2D/3D transform to a single 2D image (Z-slice).

    Parameters
    ----------
    image_2d : np.ndarray
        2D image to transform.
    transform : sitk.Transform
        SimpleITK transform (extracts 2D rotation/translation from 3D Euler).
    rotation_only : bool
        If True, apply only rotation, ignore translation.
    max_rotation_deg : float
        Maximum rotation in degrees; larger values are clamped. 0 = no clamping.
    override_rotation : float or None
        Use this rotation angle (radians) instead of extracting from transform.

    Returns
    -------
    np.ndarray
        Transformed 2D image.
    """
    import SimpleITK as sitk

    sitk_img = sitk.GetImageFromArray(image_2d.astype(np.float32))

    if transform.GetDimension() == 3:
        if isinstance(transform, sitk.Euler3DTransform) or transform.GetName() == 'Euler3DTransform':
            params = transform.GetParameters()
            angle = params[2] if len(params) > 2 else 0
            tx = params[3] if len(params) > 3 else 0
            ty = params[4] if len(params) > 4 else 0

            if override_rotation is not None:
                angle = override_rotation
            elif max_rotation_deg > 0:
                max_angle_rad = np.radians(max_rotation_deg)
                if abs(angle) > max_angle_rad:
                    angle = np.clip(angle, -max_angle_rad, max_angle_rad)

            center = transform.GetCenter()
            center_2d = [center[0], center[1]]
            tfm_2d = sitk.Euler2DTransform()
            tfm_2d.SetCenter(center_2d)
            tfm_2d.SetAngle(angle)
            if rotation_only:
                tfm_2d.SetTranslation([0, 0])
            else:
                tfm_2d.SetTranslation([tx, ty])
        else:
            tfm_2d = sitk.Euler2DTransform()
            angle = 0
            tx, ty = 0, 0
    else:
        tfm_2d = transform
        if rotation_only and hasattr(tfm_2d, 'SetTranslation'):
            tfm_2d.SetTranslation([0, 0])
        angle = 0
        tx, ty = 0, 0

    tx_final = 0 if rotation_only else tx
    ty_final = 0 if rotation_only else ty
    if abs(angle) < 0.00175 and abs(tx_final) < 1.0 and abs(ty_final) < 1.0:
        return image_2d.copy()

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_img)
    resampler.SetTransform(tfm_2d)
    resampler.SetInterpolator(sitk.sitkLinear)

    nonzero_vals = image_2d[image_2d > 0]
    default_val = float(np.percentile(nonzero_vals, 1)) if len(nonzero_vals) > 0 else 0.0
    resampler.SetDefaultPixelValue(default_val)

    result = resampler.Execute(sitk_img)
    return sitk.GetArrayFromImage(result)


def apply_transform_to_volume(vol: np.ndarray,
                               transform,
                               rotation_only: bool = False,
                               max_rotation_deg: float = 1.0,
                               override_rotation=None) -> np.ndarray:
    """Apply a 2D transform to each Z-slice of a volume.

    Parameters
    ----------
    vol : np.ndarray
        3D volume (Z, Y, X).
    transform : sitk.Transform
        Transform to apply to each slice.
    rotation_only : bool
        If True, apply only rotation.
    max_rotation_deg : float
        Maximum rotation in degrees.
    override_rotation : float or None
        If provided, use this rotation for all slices.

    Returns
    -------
    np.ndarray
        Transformed volume.
    """
    result = np.zeros_like(vol)
    for z in range(vol.shape[0]):
        result[z] = apply_2d_transform(vol[z], transform, rotation_only,
                                       max_rotation_deg, override_rotation)
    return result


def apply_xy_shift(vol: np.ndarray,
                   dx_px: float,
                   dy_px: float,
                   output_shape: Tuple[int, int]):
    """Compute destination region for placing a shifted volume.

    Returns the (possibly cropped) volume data and destination coordinates
    without allocating a full-size output array.

    Parameters
    ----------
    vol : np.ndarray
        3D volume (Z, Y, X).
    dx_px, dy_px : float
        Shift in pixels (X and Y directions).
    output_shape : tuple
        (out_ny, out_nx) output canvas size.

    Returns
    -------
    cropped_vol : np.ndarray or None
        Cropped volume data to write.
    dst_coords : tuple or None
        (y_start, y_end, x_start, x_end) in output coordinates.
    """
    out_ny, out_nx = output_shape
    dx_int, dy_int = int(round(dx_px)), int(round(dy_px))

    dst_y_start = dy_int
    dst_x_start = dx_int
    dst_y_end = dst_y_start + vol.shape[1]
    dst_x_end = dst_x_start + vol.shape[2]

    src_y_start = max(0, -dst_y_start)
    src_y_end = vol.shape[1] - max(0, dst_y_end - out_ny)
    src_x_start = max(0, -dst_x_start)
    src_x_end = vol.shape[2] - max(0, dst_x_end - out_nx)

    dst_y_start = max(0, dst_y_start)
    dst_y_end = min(out_ny, dst_y_end)
    dst_x_start = max(0, dst_x_start)
    dst_x_end = min(out_nx, dst_x_end)

    if src_y_end > src_y_start and src_x_end > src_x_start:
        cropped = vol[:, src_y_start:src_y_end, src_x_start:src_x_end]
        return cropped, (dst_y_start, dst_y_end, dst_x_start, dst_x_end)
    return None, None


def blend_overlap_z(fixed_region: np.ndarray,
                    moving_region: np.ndarray) -> np.ndarray:
    """Blend overlapping Z-region using a cosine (Hann) ramp along Z-axis.

    The weight ramp has zero slope at both endpoints, so there is no abrupt
    intensity change at either boundary of the overlap zone.  At tissue
    boundaries where only one slice has data the full intensity of that slice
    is used unchanged.

    Parameters
    ----------
    fixed_region : np.ndarray
        3D array (Z, Y, X) from the existing stack (bottom portion).
    moving_region : np.ndarray
        3D array (Z, Y, X) from the new slice (top portion).

    Returns
    -------
    np.ndarray
        Blended region with smooth Z-transition.
    """
    nz = fixed_region.shape[0]

    if nz <= 1:
        return moving_region if np.sum(moving_region > 0) >= np.sum(fixed_region > 0) else fixed_region

    # Cosine (Hann) ramp: 0 → 1 with zero slope at both ends
    t = np.linspace(0, np.pi, nz)
    z_weights = 0.5 * (1 - np.cos(t))
    alphas = np.broadcast_to(z_weights[:, np.newaxis, np.newaxis], fixed_region.shape).copy()

    fixed_valid = fixed_region > 0
    moving_valid = moving_region > 0
    both_valid = fixed_valid & moving_valid
    fixed_only = fixed_valid & ~moving_valid
    moving_only = moving_valid & ~fixed_valid

    blended = np.zeros_like(moving_region, dtype=np.float32)
    if np.any(both_valid):
        blended[both_valid] = ((1 - alphas) * fixed_region + alphas * moving_region)[both_valid]
    if np.any(fixed_only):
        blended[fixed_only] = fixed_region[fixed_only]
    if np.any(moving_only):
        blended[moving_only] = moving_region[moving_only]

    return blended


def blend_overlap_xy(existing: np.ndarray,
                     new_data: np.ndarray,
                     method: str = 'none') -> np.ndarray:
    """Blend overlapping XY regions for motor-only stacking.

    Parameters
    ----------
    existing : np.ndarray
        Existing data in the output region.
    new_data : np.ndarray
        Incoming data to blend.
    method : str
        'none' (overwrite), 'average', 'max', or 'feather'.

    Returns
    -------
    np.ndarray
        Blended result.
    """
    if method == 'none':
        mask = new_data != 0
        existing[mask] = new_data[mask]
        return existing
    elif method == 'average':
        both_valid = (existing != 0) & (new_data != 0)
        only_new = (existing == 0) & (new_data != 0)
        existing[both_valid] = (existing[both_valid] + new_data[both_valid]) / 2
        existing[only_new] = new_data[only_new]
        return existing
    elif method == 'max':
        return np.maximum(existing, new_data)
    elif method == 'feather':
        return blend_overlap_xy(existing, new_data, 'average')
    return existing


def refine_z_blend_overlap(existing: np.ndarray,
                            moving_overlap: np.ndarray,
                            max_refinement_px: float) -> Tuple[np.ndarray, float]:
    """Find and apply a small XY shift to align moving_overlap with existing before blending.

    Uses 2D phase correlation on Z-projected overlap regions to detect residual
    XY misalignment at slice boundaries.

    Parameters
    ----------
    existing : np.ndarray
        3D array (Z, Y, X) from current stack at the overlap zone.
    moving_overlap : np.ndarray
        3D array (Z, Y, X) from incoming slice at the overlap zone.
    max_refinement_px : float
        Maximum allowed shift magnitude in pixels.

    Returns
    -------
    refined : np.ndarray
        Shifted moving_overlap with residual XY misalignment corrected.
    magnitude : float
        Shift magnitude applied (pixels), or 0.0 if not applied.
    """
    from scipy.ndimage import shift as ndi_shift
    from linumpy.stitching.registration import pairWisePhaseCorrelation

    fixed_2d = np.mean(existing, axis=0).astype(np.float32)
    moving_2d = np.mean(moving_overlap, axis=0).astype(np.float32)

    valid = (fixed_2d > 0) & (moving_2d > 0)
    if np.sum(valid) < 1000:
        return moving_overlap, 0.0

    try:
        shift = pairWisePhaseCorrelation(fixed_2d, moving_2d)
        dy, dx = float(shift[0]), float(shift[1])
    except Exception as e:
        logger.debug(f"Z-blend phase correlation failed: {e}")
        return moving_overlap, 0.0

    magnitude = np.sqrt(dy**2 + dx**2)

    if magnitude < 0.1:
        return moving_overlap, 0.0

    if magnitude > max_refinement_px:
        logger.debug(f"Z-blend refinement rejected: {magnitude:.2f} px > max {max_refinement_px} px")
        return moving_overlap, 0.0

    refined = ndi_shift(moving_overlap.astype(np.float32), [0, dy, dx],
                        order=1, mode='constant', cval=0)
    return refined, magnitude
