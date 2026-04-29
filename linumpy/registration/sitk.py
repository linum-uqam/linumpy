"""SimpleITK-based image registration and transform application."""

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


def apply_transform(moving_image: np.ndarray, transform: sitk.Transform) -> np.ndarray:
    """Apply `transform` to `moving_image`.

    The image is transformed inside its own domain.

    Parameters
    ----------
    moving_image: numpy.ndarray
        Moving image to transform.
    transform: sitk.sitkTransform
        Transform to apply to `moving_image`.
    """
    moving_image_sitk = sitk.GetImageFromArray(moving_image)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(moving_image_sitk)  # transforms the image inside its domain
    resampler.SetInterpolator(sitk.sitkLinear)

    # Use edge value instead of zero to avoid black dots at boundaries
    nonzero_vals = moving_image[moving_image > 0]
    default_val = float(np.percentile(nonzero_vals, 1)) if len(nonzero_vals) > 0 else 0.0
    resampler.SetDefaultPixelValue(default_val)

    resampler.SetTransform(transform)

    out = resampler.Execute(moving_image_sitk)
    out = sitk.GetArrayFromImage(out)

    return out
