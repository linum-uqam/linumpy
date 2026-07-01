"""SimpleITK-based image registration and transform application."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import SimpleITK as sitk


def itk_registration(
    vol1: np.ndarray,
    vol2: np.ndarray,
    _offset: tuple[float, float, float] = (0, 0, 0),
    metric: str = "MSQ",
    verbose: bool = False,
    match_histograms: bool = False,
    mask_fixed: np.ndarray | None = None,
    mask_moving: np.ndarray | None = None,
) -> tuple[list[float], float]:
    """Use ITK::ImageRegistrationMethod.MutualInformation.

    Parameters
    ----------
    vol1 : ndarray
        Fixed image / volume
    vol2 : ndarray
        Moving image / volume
    offset : (3,) tuple
        Offset position relating the two volumes
    metric : str
        Similarity metric to use for registration. Available metrics are : MSQ, JHMI, MMI, ANTsCorr, corr (default)
    verbose : bool
        Verbose flag
    match_histograms : bool
        If True, match histograms of moving and fixed volumes before registration.
    mask_fixed : ndarray, optional
        Mask to apply to the fixed volume.
    mask_moving : ndarray, optional
        Mask to apply to the moving volume.

    Returns
    -------
    (3,) tuple
        Estimated deltas between these two volumes
    float
        Similarity metric

    """
    # Make the itk images
    fixed_vol = sitk.GetImageFromArray(vol1)
    moving_vol = sitk.GetImageFromArray(vol2)

    # Normalization
    sitk.Normalize(fixed_vol)
    sitk.Normalize(moving_vol)

    # Match Histograms
    if match_histograms:
        moving_vol = sitk.HistogramMatching(moving_vol, fixed_vol)

    # Image Registration Filter
    reg = sitk.ImageRegistrationMethod()
    reg.SetInitialTransform(sitk.TranslationTransform(fixed_vol.GetDimension()))
    reg.SetInterpolator(sitk.sitkLinear)

    if metric == "MSQ":
        reg.SetMetricAsMeanSquares()
    elif metric == "JHMI":
        reg.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=50)
    elif metric == "MMI":
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    elif metric == "ANTsCorr":
        reg.SetMetricAsANTSNeighborhoodCorrelation(20)
    elif metric == "corr":
        reg.SetMetricAsCorrelation()
    else:  # Correlation
        reg.SetMetricAsCorrelation()

    learning_rate = 1.0
    min_step = 1
    n_iterations = 500
    reg.SetOptimizerAsRegularStepGradientDescent(learning_rate, min_step, n_iterations)
    reg.SetShrinkFactorsPerLevel(shrinkFactors=[1])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Adding fixed and moving mask if defined
    if mask_fixed is not None:
        print("Add fixed mask")
        reg.SetMetricFixedMask(sitk.GetImageFromArray(mask_fixed.astype(int)))
    if mask_moving is not None:
        print("Add moving mask")
        reg.SetMetricMovingMask(sitk.GetImageFromArray(mask_moving.astype(int)))

    final_transform = reg.Execute(fixed_vol, moving_vol)

    if verbose:
        print(reg, final_transform)

    deltas: list = []
    if vol1.ndim == 3:
        dx = final_transform.GetParameters()[0]
        dy = final_transform.GetParameters()[1]
        dz = final_transform.GetParameters()[2]
        deltas = [dx, dy, dz]

    elif vol1.ndim == 2:
        dx = -final_transform.GetParameters()[1]
        dy = -final_transform.GetParameters()[0]
        deltas = [dx, dy]

    MI = reg.GetMetricValue()
    return deltas, MI


def align_images_sitk(im1: np.ndarray, im2: np.ndarray) -> tuple[list[float], float]:
    """Align two images using SimpleITK translation registration."""
    # Parameters
    learning_rate = 4.0
    min_step = 0.01
    max_iteration = 200

    fixed = sitk.GetImageFromArray(im1)
    moving = sitk.GetImageFromArray(im2)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsCorrelation()
    reg.SetOptimizerAsRegularStepGradientDescent(
        learning_rate, min_step, max_iteration, relaxationFactor=0.5, gradientMagnitudeTolerance=1e-9
    )
    reg.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    reg.SetInterpolator(sitk.sitkLinear)

    out_tx = reg.Execute(fixed, moving)

    dx = -out_tx.GetParameters()[1]
    dy = -out_tx.GetParameters()[0]
    deltas = [dx, dy]
    m = reg.GetMetricValue()
    return deltas, m


def register_2d_images_sitk(
    ref_image: np.ndarray,
    moving_image: np.ndarray,
    method: str = "euler",
    metric: str = "MSE",
    max_iterations: int = 2500,
    min_step: float = 1e-12,
    grad_mag_tol: float = 1e-12,
    fixed_mask: np.ndarray | None = None,
    moving_mask: np.ndarray | None = None,
    return_3d_transform: bool = False,
    verbose: bool = False,
    initial_translation: tuple[float, float] | None = None,
    initial_step: float | None = None,
) -> tuple[sitk.Transform, str, float]:
    """Register 2D `moving_image` to `ref_image`.

    Parameters
    ----------
    ref_image: numpy.ndarray
        The reference image.
    moving_image: numpy.ndarray
        The image to register.
    method: str
        The type of transform for registration. Choices are: "euler",
        "affine" or "translation".
    metric: str
        The metric to optimize. Choices are:
            - "MSE": Mean-squared error
            - "MI": Mattes mutual information
            - "AntsCC": Ants neighbourhood cross-correlation
            - "CC": Cross-correlation
    max_iterations: int
        Maximum number of iterations at each level of the multiscale
        pyramid (3 levels).
    min_step: float
        Minimum step size for the gradient descent.
    grad_mag_tol: float
        Gradient magnitude tolerance for gradient descent.
    fixed_mask: numpy.ndarray
        Optional mask to apply to the reference image during registration.
    moving_mask: numpy.ndarray
        Optional mask to apply to the moving image during registration.
    return_3d_transform: bool
        If True, will return a 3D transform instead of 2D. Useful when the
        transform is applied to a 3D image.
    verbose: bool
        If True, will log registrations metric at each iteration.
    initial_translation: tuple or None
        Optional initial translation (tx, ty) to use as starting point.
        If None, uses CenteredTransformInitializer.
    initial_step: float or None
        Initial step size for the optimizer. If None, uses 4.0 pixels.
        When initial_translation is provided, a smaller step (e.g., 1.0) is recommended.

    Returns
    -------
    out_transform: sitk.sitkTransform
        Transform for bringing `moving_image` onto `ref_image`.
    stop_condition: str
        String describing optimizer stopping condition.
    error: float
        Registration metric value at the end of registration process.
    """
    # Type cast everything to float32
    ref_image = ref_image.astype(np.float32)
    moving_image = moving_image.astype(np.float32)

    fixed_sitk_image = sitk.GetImageFromArray(ref_image)
    moving_sitk_image = sitk.GetImageFromArray(moving_image)

    reg = sitk.ImageRegistrationMethod()

    if fixed_mask is not None:
        fixed_sitk_mask = sitk.GetImageFromArray(fixed_mask.astype(np.uint8))
        reg.SetMetricFixedMask(fixed_sitk_mask)
    if moving_mask is not None:
        moving_sitk_mask = sitk.GetImageFromArray(moving_mask.astype(np.uint8))
        reg.SetMetricMovingMask(moving_sitk_mask)

    if metric.lower() == "mse":
        reg.SetMetricAsMeanSquares()
    elif metric.lower() == "cc":
        reg.SetMetricAsCorrelation()
    elif metric.lower() == "antscc":
        reg.SetMetricAsANTSNeighborhoodCorrelation(radius=20)
    elif metric.lower() == "mi":
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Use smaller step size when we have an initial translation estimate (to avoid drifting away)
    step_size = (1.0 if initial_translation is not None else 4.0) if initial_step is None else initial_step

    reg.SetOptimizerAsRegularStepGradientDescent(step_size, min_step, max_iterations, 0.5, grad_mag_tol)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([3, 2, 1])

    # this step is essential for the registration to work properly
    # determines the scale of each parameter in the optimizer
    reg.SetOptimizerScalesFromIndexShift()

    if method == "euler":
        sitk_transform = sitk.Euler2DTransform()
    elif method == "affine":
        sitk_transform = sitk.AffineTransform(2)
    elif method == "translation":
        sitk_transform = sitk.TranslationTransform(2)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Initialize transform - use provided translation or centered initializer
    if initial_translation is not None:
        # Set center to image center
        center = [fixed_sitk_image.GetWidth() / 2.0, fixed_sitk_image.GetHeight() / 2.0]
        if method == "euler":
            assert isinstance(sitk_transform, sitk.Euler2DTransform)
            sitk_transform.SetCenter(center)
            sitk_transform.SetTranslation(initial_translation)
        elif method == "affine":
            assert isinstance(sitk_transform, sitk.AffineTransform)
            sitk_transform.SetCenter(center)
            sitk_transform.SetTranslation(initial_translation)
        elif method == "translation":
            assert isinstance(sitk_transform, sitk.TranslationTransform)
            sitk_transform.SetOffset(initial_translation)
    else:
        sitk_transform = sitk.CenteredTransformInitializer(fixed_sitk_image, moving_sitk_image, sitk_transform)

    reg.SetInitialTransform(sitk_transform)

    reg.SetInterpolator(sitk.sitkLinear)
    if verbose:
        reg.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(reg))

    out_transform = reg.Execute(fixed_sitk_image, moving_sitk_image)
    stop_condition = reg.GetOptimizerStopConditionDescription()
    error = reg.GetMetricValue()

    if return_3d_transform:
        if method == "euler":
            angle_rad = out_transform.GetAngle()
            center_of_rotation = out_transform.GetCenter()
            translation = out_transform.GetTranslation()
            transform_3d = sitk.Euler3DTransform()
            transform_3d.SetCenter([center_of_rotation[0], center_of_rotation[1], 0.0])
            transform_3d.SetRotation(0.0, 0.0, angle_rad)
            transform_3d.SetTranslation([translation[0], translation[1], 0.0])
        elif method == "translation":
            translation = out_transform.GetOffset()
            transform_3d = sitk.TranslationTransform(3)
            transform_3d.SetOffset([translation[0], translation[1], 0.0])
        elif method == "affine":
            transform_3d = sitk.AffineTransform(3)
            translation = out_transform.GetTranslation()
            transform_3d.SetCenter((*out_transform.GetCenter(), 0.0))
            transform_3d.SetTranslation([translation[0], translation[1], 0.0])
            matrix_2d = out_transform.GetMatrix()
            matrix_3d = np.zeros((3, 3))
            matrix_3d[:2, :2] = np.array(matrix_2d).reshape((2, 2))
            matrix_3d[2, 2] = 1.0
            transform_3d.SetMatrix(matrix_3d.flatten().tolist())
        else:
            raise ValueError(f"Unknown method: {method}")
        out_transform = transform_3d

    return out_transform, stop_condition, error


def command_iteration(method: sitk.ImageRegistrationMethod) -> None:
    """Invoke when the optimization has an iteration."""
    if method.GetOptimizerIteration() == 0:
        print("Estimated Scales: ", method.GetOptimizerScales())
    print(f"{method.GetOptimizerIteration():3} " + f"= {method.GetMetricValue():7.5f} " + f": {method.GetOptimizerPosition()}")


def apply_transform(
    moving_image: np.ndarray,
    transform: sitk.Transform,
    reference_image: np.ndarray | None = None,
    reference_spacing: Sequence[float] | None = None,
    moving_spacing: Sequence[float] | None = None,
) -> np.ndarray:
    """Apply `transform` to `moving_image`.

    By default the image is transformed inside its own domain (preserving
    backward compatibility).  When ``reference_image`` is provided, the moving
    image is resampled onto the reference grid instead — required when the
    transform maps between two volumes that do not share a domain (e.g.
    inter-subject registration with different shapes / spacing).

    Parameters
    ----------
    moving_image
        Moving image to transform (numpy ZYX).
    transform
        SimpleITK transform mapping the reference (or moving) domain to
        moving voxel coordinates.
    reference_image
        Optional reference volume defining the output grid.  Only its shape
        and spacing are used; intensities are ignored.
    reference_spacing
        Voxel spacing ``(sz, sy, sx)`` in mm for ``reference_image``.
        Required when ``reference_image`` is given.
    moving_spacing
        Voxel spacing ``(sz, sy, sx)`` in mm for ``moving_image``.  Required
        when ``reference_image`` is given so the moving image carries the
        correct physical extent.
    """
    if reference_image is not None:
        if reference_spacing is None or moving_spacing is None:
            raise ValueError("reference_spacing and moving_spacing are required when reference_image is given.")
        moving_image_sitk = _numpy_to_sitk_image(moving_image, moving_spacing)
        reference_sitk = _numpy_to_sitk_image(reference_image, reference_spacing)
    else:
        moving_image_sitk = sitk.GetImageFromArray(moving_image)
        reference_sitk = moving_image_sitk

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)

    # Use edge value instead of zero to avoid black dots at boundaries
    nonzero_vals = moving_image[moving_image > 0]
    default_val = float(np.percentile(nonzero_vals, 1)) if len(nonzero_vals) > 0 else 0.0
    resampler.SetDefaultPixelValue(default_val)

    resampler.SetTransform(transform)

    out = resampler.Execute(moving_image_sitk)
    out = sitk.GetArrayFromImage(out)

    return out


def _numpy_to_sitk_image(volume: np.ndarray, spacing: Sequence[float]) -> sitk.Image:
    """Convert a numpy ZYX volume to a SimpleITK image with identity frame.

    Spacing is in numpy order ``(sz, sy, sx)``; SITK uses ``(sx, sy, sz)``.
    Origin is ``(0, 0, 0)`` and direction is identity — matching the
    project-wide RAS-aligned OME-Zarr convention.
    """
    vol_sitk = sitk.GetImageFromArray(volume)
    vol_sitk.SetSpacing([float(spacing[2]), float(spacing[1]), float(spacing[0])])
    vol_sitk.SetOrigin([0.0, 0.0, 0.0])
    vol_sitk.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    return vol_sitk


def register_3d_rigid_to_volume(
    fixed_volume: np.ndarray,
    fixed_spacing: Sequence[float],
    moving_volume: np.ndarray,
    moving_spacing: Sequence[float],
    metric: str = "MI",
    max_iterations: int = 1000,
    initial_rotation_deg: Sequence[float] = (0.0, 0.0, 0.0),
    crop_to_bbox: bool = True,
    verbose: bool = False,
    progress_callback: Callable[[Any], None] | None = None,
) -> tuple[sitk.Euler3DTransform, str, float]:
    """Rigid 3-D registration between two arbitrary volumes (no atlas).

    Both volumes are interpreted as ZYX numpy arrays sitting in an identity
    physical frame (origin = 0, direction = identity, spacing in mm).  The
    returned ``Euler3DTransform`` maps ``fixed`` coordinates to ``moving``
    coordinates and can be passed to :func:`apply_transform` together with
    ``reference_image=fixed_volume`` to resample the moving subject onto the
    fixed grid.

    The recipe mirrors :func:`linumpy.reference.allen.register_3d_rigid_to_allen`
    minus the Allen-specific resampling and centroid-fallback logic: the two
    volumes are expected to already share an approximate RAS frame at a
    compatible resolution.

    Parameters
    ----------
    fixed_volume, moving_volume
        Numpy ZYX volumes.
    fixed_spacing, moving_spacing
        ``(sz, sy, sx)`` in mm.
    metric
        ``'MI'`` (default, Mattes), ``'MSE'``, ``'CC'``, or ``'AntsCC'``.
    max_iterations
        Maximum optimizer iterations per pyramid level.
    initial_rotation_deg
        ``(rx, ry, rz)`` degrees applied before optimisation.
    crop_to_bbox
        Crop both volumes to their non-zero tissue bounding box before
        registration.  Translation offsets are restored on output so the
        transform is valid for the full original volumes.
    verbose
        Print progress.
    progress_callback
        Called once per iteration with the SITK registration method.

    Returns
    -------
    transform, stop_condition, metric_value
    """
    fixed_vol = np.asarray(fixed_volume)
    moving_vol = np.asarray(moving_volume)

    # Optional crop to non-zero bounding box.  Translation offsets are added
    # back on the final transform's translation so it remains valid for the
    # uncropped volumes.
    margin_voxels = 10
    fixed_crop_origin_mm = (0.0, 0.0, 0.0)
    moving_crop_origin_mm = (0.0, 0.0, 0.0)
    if crop_to_bbox:
        fixed_vol, fixed_crop_origin_mm = _crop_to_bbox(fixed_vol, fixed_spacing, margin_voxels)
        moving_vol, moving_crop_origin_mm = _crop_to_bbox(moving_vol, moving_spacing, margin_voxels)
        if verbose:
            print(f"Cropped fixed → {fixed_vol.shape}, moving → {moving_vol.shape}")

    fixed_sitk = _numpy_to_sitk_image(fixed_vol, fixed_spacing)
    moving_sitk = _numpy_to_sitk_image(moving_vol, moving_spacing)

    fixed_image = sitk.Normalize(sitk.Cast(fixed_sitk, sitk.sitkFloat32))
    moving_image = sitk.Normalize(sitk.Cast(moving_sitk, sitk.sitkFloat32))

    registration = sitk.ImageRegistrationMethod()

    metric_u = metric.upper()
    if metric_u == "MI":
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    elif metric_u == "MSE":
        registration.SetMetricAsMeanSquares()
    elif metric_u == "CC":
        registration.SetMetricAsCorrelation()
    elif metric_u == "ANTSCC":
        registration.SetMetricAsANTSNeighborhoodCorrelation(radius=20)
    else:
        raise ValueError(f"Unknown metric: {metric}. Choose from: MI, MSE, CC, AntsCC")

    registration.SetMetricSamplingStrategy(registration.REGULAR)
    registration.SetMetricSamplingPercentage(0.25)

    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=0.5,
        minStep=1e-4,
        numberOfIterations=max_iterations,
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-8,
    )
    registration.SetOptimizerScalesFromPhysicalShift()

    registration.SetShrinkFactorsPerLevel([8, 4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([4, 2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInterpolator(sitk.sitkLinear)

    # Centroid-based initial alignment: align the centre-of-tissue of each
    # volume in physical space.  Falls back to geometric centre when a volume
    # is entirely zero (shouldn't happen for real data but keeps the function
    # safe).
    fixed_center = _tissue_centroid_mm(fixed_vol, fixed_spacing)
    moving_center = _tissue_centroid_mm(moving_vol, moving_spacing)

    initial_transform = sitk.Euler3DTransform()
    # SITK XYZ order; numpy axes are ZYX.
    initial_transform.SetCenter([fixed_center[2], fixed_center[1], fixed_center[0]])
    initial_transform.SetTranslation(
        [
            moving_center[2] - fixed_center[2],
            moving_center[1] - fixed_center[1],
            moving_center[0] - fixed_center[0],
        ]
    )
    initial_transform.SetRotation(
        float(np.deg2rad(initial_rotation_deg[0])),
        float(np.deg2rad(initial_rotation_deg[1])),
        float(np.deg2rad(initial_rotation_deg[2])),
    )
    if verbose:
        print(f"Centroid init: fixed={fixed_center} mm, moving={moving_center} mm")
        print(f"Initial translation: {initial_transform.GetTranslation()} mm (SITK XYZ)")

    registration.SetInitialTransform(initial_transform)

    if verbose or progress_callback is not None:

        def _on_iter(method: Any) -> None:
            if verbose:
                if method.GetOptimizerIteration() == 0:
                    print(f"Estimated scales: {method.GetOptimizerScales()}")
                print(
                    f"Iter {method.GetOptimizerIteration():3d} "
                    f"= {method.GetMetricValue():7.5f} : {method.GetOptimizerPosition()}"
                )
            if progress_callback is not None:
                progress_callback(method)

        registration.AddCommand(sitk.sitkIterationEvent, lambda: _on_iter(registration))

    final_transform = registration.Execute(fixed_image, moving_image)
    stop = registration.GetOptimizerStopConditionDescription()
    error = registration.GetMetricValue()

    # Restore crop offsets so the transform is valid for the uncropped
    # volumes (both still anchored at SITK origin = 0, identity direction).
    # Derivation.  Let R, c, t_crop be the rigid params learned on the
    # cropped images.  Cropped fixed coord p_fc maps to uncropped fixed coord
    # p_ff = p_fc + fixed_off, and similarly for moving.  We want
    #     T_full(p_ff) = T_crop(p_ff - fixed_off) + moving_off
    #                  = R(p_ff - fixed_off - c) + c + t_crop + moving_off.
    # Substituting c' = c + fixed_off gives
    #     T_full(p_ff) = R(p_ff - c') + c' + (t_crop + moving_off - fixed_off).
    # So c_new = c + fixed_off and t_new = t_crop + moving_off - fixed_off.
    if crop_to_bbox and (any(fixed_crop_origin_mm) or any(moving_crop_origin_mm)):
        # SITK XYZ; our offsets are stored in numpy ZYX, so reverse them.
        fixed_off_xyz = np.array([fixed_crop_origin_mm[2], fixed_crop_origin_mm[1], fixed_crop_origin_mm[0]])
        moving_off_xyz = np.array([moving_crop_origin_mm[2], moving_crop_origin_mm[1], moving_crop_origin_mm[0]])
        c_old = np.array(final_transform.GetCenter())
        params = list(final_transform.GetParameters())
        t_old = np.array(params[3:6])
        c_new = c_old + fixed_off_xyz
        t_new = t_old + moving_off_xyz - fixed_off_xyz
        final_transform.SetCenter(c_new.tolist())
        params[3], params[4], params[5] = float(t_new[0]), float(t_new[1]), float(t_new[2])
        final_transform.SetParameters(params)
        if verbose:
            print(f"Crop-restored translation (SITK XYZ): {tuple(params[3:6])} mm")

    if verbose:
        print(f"Done: {stop}; metric={error:.6f}")

    return final_transform, stop, error


def _crop_to_bbox(
    volume: np.ndarray, spacing: Sequence[float], margin_voxels: int
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Crop ``volume`` to its non-zero bounding box.

    Returns the cropped array and the physical offset ``(z, y, x)`` of the
    crop origin in mm.  If the volume is entirely zero the original array
    and a zero offset are returned.
    """
    nonzero = np.nonzero(volume)
    if len(nonzero[0]) == 0:
        return volume, (0.0, 0.0, 0.0)
    slices = tuple(
        slice(
            max(0, int(idx.min()) - margin_voxels),
            min(volume.shape[ax], int(idx.max()) + margin_voxels + 1),
        )
        for ax, idx in enumerate(nonzero)
    )
    cropped = volume[slices]
    offset_mm = (
        slices[0].start * float(spacing[0]),
        slices[1].start * float(spacing[1]),
        slices[2].start * float(spacing[2]),
    )
    return cropped, offset_mm


def _tissue_centroid_mm(volume: np.ndarray, spacing: Sequence[float]) -> tuple[float, float, float]:
    """Return the centroid of non-zero voxels in mm (numpy ZYX order).

    Falls back to the geometric centre when the volume is entirely zero.
    """
    nonzero = np.argwhere(volume > 0)
    if len(nonzero) == 0:
        z, y, x = volume.shape
        return (z / 2.0 * spacing[0], y / 2.0 * spacing[1], x / 2.0 * spacing[2])
    cz, cy, cx = nonzero.mean(axis=0)
    return (float(cz) * spacing[0], float(cy) * spacing[1], float(cx) * spacing[2])
