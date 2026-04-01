"""Refinement registration: best-Z search and small rotation/translation correction."""

import numpy as np
import SimpleITK as sitk


def find_best_z(fixed_vol: np.ndarray, moving_slice: np.ndarray, expected_z: int, search_range: int) -> tuple[int, float]:
    """Find the Z-index in fixed_vol that best matches moving_slice.

    Uses normalized cross-correlation in the center region.

    Parameters
    ----------
    fixed_vol : array-like
        Fixed volume (Z, Y, X) or dask/zarr array.
    moving_slice : np.ndarray
        2D slice to match.
    expected_z : int
        Expected Z-index in fixed_vol for the match.
    search_range : int
        Search +/-search_range around expected_z.

    Returns
    -------
    best_z : int
        Z-index giving the best correlation.
    best_corr : float
        Correlation score at best_z.
    """
    nz = fixed_vol.shape[0]
    expected_z = max(0, min(nz - 1, expected_z))
    z_min = max(0, expected_z - search_range)
    z_max = min(nz - 1, expected_z + search_range)

    if z_min >= z_max:
        return max(0, min(nz - 1, expected_z)), 0.0

    h, w = moving_slice.shape
    margin = min(h, w) // 4
    roi = (slice(margin, h - margin), slice(margin, w - margin))

    moving_roi = moving_slice[roi].astype(np.float32)
    valid_mov = moving_roi > 0
    if valid_mov.any():
        pmin = float(np.percentile(moving_roi[valid_mov], 5))
        pmax = float(np.percentile(moving_roi[valid_mov], 95))
        moving_roi = np.clip((moving_roi - pmin) / max(pmax - pmin, 1e-8), 0, 1)

    moving_norm = (moving_roi - moving_roi.mean()) / (moving_roi.std() + 1e-8)

    best_z = expected_z
    best_corr = -np.inf

    for z in range(z_min, z_max + 1):
        fixed_slice = np.array(fixed_vol[z])
        fixed_roi = fixed_slice[roi].astype(np.float32)
        valid_fix = fixed_roi > 0
        if valid_fix.any():
            pmin = float(np.percentile(fixed_roi[valid_fix], 5))
            pmax = float(np.percentile(fixed_roi[valid_fix], 95))
            fixed_roi = np.clip((fixed_roi - pmin) / max(pmax - pmin, 1e-8), 0, 1)

        fixed_norm = (fixed_roi - fixed_roi.mean()) / (fixed_roi.std() + 1e-8)
        corr = float(np.mean(fixed_norm * moving_norm))

        if corr > best_corr:
            best_corr = corr
            best_z = z

    return max(0, min(nz - 1, best_z)), best_corr


def register_refinement(
    fixed: np.ndarray,
    moving: np.ndarray,
    enable_rotation: bool = True,
    max_rotation_deg: float = 5.0,
    max_translation_px: float = 20.0,
    fixed_mask: np.ndarray | None = None,
    moving_mask: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    """Compute small rotation and translation refinement using SimpleITK.

    Parameters
    ----------
    fixed, moving : np.ndarray
        2D images for registration (should be normalized to [0, 1]).
    enable_rotation : bool
        Enable Euler2D rotation (default True). False = translation only.
    max_rotation_deg : float
        Maximum allowed rotation in degrees.
    max_translation_px : float
        Maximum allowed translation in pixels.
    fixed_mask, moving_mask : np.ndarray or None
        Optional tissue masks multiplied into images before registration.

    Returns
    -------
    tx, ty : float
        Translation refinement in pixels.
    angle_deg : float
        Rotation angle in degrees.
    metric : float
        Registration metric value.
    """
    fixed_std = np.std(fixed[fixed > 0]) if np.any(fixed > 0) else 0
    moving_std = np.std(moving[moving > 0]) if np.any(moving > 0) else 0
    if fixed_std < 0.01 or moving_std < 0.01:
        return 0.0, 0.0, 0.0, 0.0

    fixed_masked = fixed * fixed_mask.astype(np.float32) if fixed_mask is not None else fixed
    moving_masked = moving * moving_mask.astype(np.float32) if moving_mask is not None else moving

    fixed_sitk = sitk.GetImageFromArray(fixed_masked.astype(np.float32))
    moving_sitk = sitk.GetImageFromArray(moving_masked.astype(np.float32))

    if enable_rotation:
        transform = sitk.Euler2DTransform()
        center = [fixed.shape[1] / 2.0, fixed.shape[0] / 2.0]
        transform.SetCenter(center)
    else:
        transform = sitk.TranslationTransform(2)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsCorrelation()
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=200, convergenceMinimumValue=1e-6, convergenceWindowSize=10
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(transform, inPlace=False)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])

    try:
        final = reg.Execute(fixed_sitk, moving_sitk)
        metric = reg.GetMetricValue()

        inner = final.GetNthTransform(0) if final.GetName() == "CompositeTransform" else final

        if enable_rotation:
            euler = sitk.Euler2DTransform(inner)
            angle_deg = np.degrees(euler.GetAngle())
            tx, ty = euler.GetTranslation()
        else:
            tx, ty = inner.GetOffset()
            angle_deg = 0.0

        mag = np.sqrt(tx**2 + ty**2)
        if mag > max_translation_px:
            scale = max_translation_px / mag
            tx, ty = tx * scale, ty * scale

        if abs(angle_deg) > max_rotation_deg:
            angle_deg = float(np.clip(angle_deg, -max_rotation_deg, max_rotation_deg))

        return tx, ty, angle_deg, metric

    except Exception:
        return 0.0, 0.0, 0.0, float("inf")
