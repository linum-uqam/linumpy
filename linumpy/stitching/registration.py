#! /usr/bin/env python
# -*- coding: utf-8 -*-

import SimpleITK as sitk
import numpy as np
from skimage.feature import peak_local_max

from linumpy.stitching.stitch_utils import getOverlap


def pairWisePhaseCorrelation(
        vol1, vol2, nPeaks=8, returnCC=False
):  # TODO: Test for 3D images
    """Find the translation between image pairs using phase correlation and cross-correlation.
    Parameters
    ----------
    vol1 : ndimage
        Fixed image / volume
    vol2 : ndimage
        Moving image / volume
    nPeaks : int
        Number of phase correlation peaks to sample
    returnCC : bool
        Return cross-correlation score

    Returns
    -------
    list
        Translation of vol2 -/- vol1 in each direction

    Notes
    -----
    - Works in 2D for now. Needs to be tested in 3D.

    References
    ----------
    Preibisch S. et al. (2008) Fast Stitching of Large 3D Biological Datasets (ImageJ Proceesings)
    """

    # Extend images by 1/4 of their size in each direction
    vol_shape = vol1.shape
    new_shape = np.array(vol_shape) * 1.25
    pad_size = np.ceil(0.5 * (new_shape - vol_shape)).astype(int)
    pad_width = list()
    for pad in pad_size:
        pad_width.append((pad, pad))
    vol1_p = np.pad(vol1, pad_width, mode="reflect")
    vol2_p = np.pad(vol2, pad_width, mode="reflect")

    # Apply a window on the image extension
    vol1_p = applyHanningWindow(vol1_p, pad_size)
    vol2_p = applyHanningWindow(vol2_p, pad_size)

    # TODO: Add zero-padding up to the next power of two or up to a given size ...

    # Phase correlation matrix Q computation
    Q_num = np.fft.fft2(vol2_p) * np.conjugate(np.fft.fft2(vol1_p))
    Q_denum = np.abs(Q_num)
    with np.errstate(divide="ignore"):
        Q_freq = np.divide(Q_num, Q_denum)
        Q_freq[Q_denum == 0] = 0
    Q = np.fft.ifft2(Q_freq)

    # Find the main peak
    pmax = np.amax(Q)
    indices = np.where(Q == pmax)

    # Find the first N peaks
    coordinates = peak_local_max(
        np.abs(Q), min_distance=1, num_peaks=nPeaks, exclude_border=False
    )  # max value in the whole image

    deltasList = list()
    for indices in coordinates:
        deltas = list()
        for idx, s in zip(indices, vol1_p.shape):
            deltas.append(int(-idx + s / 2))

        # Check if it is outside the original image
        for ii in range(len(deltas)):
            if deltas[ii] > vol_shape[ii]:
                print(("deltas larger than imshape", deltas[ii], vol_shape[ii]))
                deltas[ii] -= vol_shape[ii]
        deltasList.append(deltas)

    # Try all translation permutations and find which one has the highest correlation.
    translations = list()
    for deltas in deltasList:
        if vol1.ndim == 2:
            dx, dy = deltas[:]
            translations.extend(
                [
                    [dx, dy],
                    [dx - np.sign(dx) * int(vol1_p.shape[0] / 2), dy],
                    [dx, dy - np.sign(dy) * int(vol1_p.shape[1] / 2)],
                    [
                        dx - np.sign(dx) * int(vol1_p.shape[0] / 2),
                        dy - np.sign(dy) * int(vol1_p.shape[1] / 2),
                    ],
                ]
            )
        else:
            dx, dy, dz = deltas[:]
            nxp = np.sign(dx) * int(vol1_p.shape[0] / 2)
            nyp = np.sign(dy) * int(vol1_p.shape[1] / 2)
            nzp = np.sign(dz) * int(vol1_p.shape[2] / 2)
            translations.extend(
                [
                    [dx, dy, dz],
                    [dx - nxp, dy, dz],
                    [dx, dy - nyp, dz],
                    [dx - nxp, dy - nyp, dz],
                    [dx, dy, dz - nzp],
                    [dx, dy - nyp, dz - nzp],
                    [dx - nxp, dy, dz - nzp],
                    [dx - nxp, dy - nyp, dz - nzp],
                ]
            )
    corrScore = list()
    for this_delta in translations:
        pos1 = [0] * vol1.ndim
        ov1, ov2, _, _ = getOverlap(vol1, vol2, pos1, this_delta)
        try:
            corr = crossCorrelation(ov1, ov2)
        except:
            corr = 0
        corrScore.append(corr)

    corrScore = np.array(corrScore)
    corrScore[np.isnan(corrScore)] = 0

    idx = np.where(corrScore == corrScore.max())[0][0]

    if returnCC:
        return translations[idx], corrScore[idx]
    else:
        return translations[idx]


def crossCorrelation(vol1, vol2, mask=None):
    """Computes the normalized cross-correlation between two ndarrays

    Parameters
    ----------
    vol1 : ndarray
        Fixed volume
    vol2 : ndarray
        Moving volume
    mask : ndarray
        Mask where the cross-correlation is computed. Assumed to be everywhere.

    Returns
    -------
    float
        Cross correlation between the volumes

    Notes
    -----
    - If a mask is given, the weighted NCC is computed instead of the NCC.
    - vol1, vol2 and mask should have the same shape.
    - mask is normalized before using it in the NCC computation.
    """
    if mask is None:
        mask = np.ones_like(vol1, dtype=float)

    # Normalizing the mask
    if mask.sum() > 0:
        mask = mask / float(mask.sum())
    else:
        return 0.0  # The mask is empty

    try:  # Using the WNCC, i.e. using a weighted sum instead of an average.
        covAB = np.sum(
            (vol1 - np.sum(vol1 * mask)) * (vol2 - np.sum(vol2 * mask)) * mask
        )
        sA = np.sqrt(np.sum((vol1 - np.sum(vol1 * mask)) ** 2.0 * mask))
        sB = np.sqrt(np.sum((vol2 - np.sum(vol2 * mask)) ** 2.0 * mask))

        return covAB / float(sA * sB)
    except:
        return 0.0


def applyHanningWindow(im, padshape):
    """Apply an hanning window to image

    Parameters
    ----------
    im : ndarray
         ndarray to modify

    Returns
    -------
    ndarray
        Modified ndarray.

    """
    ndim = im.ndim
    if ndim == 2:
        im = np.reshape(im, (im.shape[0], im.shape[1], 1))

    nx, ny, nz = im.shape
    for ii in range(ndim):
        pad = padshape[ii]
        s = im.shape[ii]

        h = np.hanning(pad * 2)
        h_full = np.ones((s,))
        h_full[0:pad] = h[0:pad]
        h_full[-pad::] = h[pad::]

        # Reshape and tile
        reshape_size = [1, 1, 1]
        reshape_size[ii] = s
        tile_size = [nx, ny, nz]
        tile_size[ii] = 1

        h_full = np.tile(np.reshape(h_full, reshape_size), tile_size)
        im = im * h_full
    if ndim == 2:
        im = np.squeeze(im)

    return im


def ITKRegistration(
        vol1,
        vol2,
        offset=(0, 0, 0),
        metric="MSQ",
        verbose=False,
        matchHistograms=False,
        maskFixed=None,
        maskMoving=None,
):
    """Uses ITK::ImageRegistrationMethod.MutualInformation

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

    Returns
    -------
    (3,) tuple
        Estimated deltas between these two volumes
    float
        Similarity metric

    """
    # Equalize histogram

    # Make the itk images
    fixed_vol = sitk.GetImageFromArray(vol1)
    moving_vol = sitk.GetImageFromArray(vol2)

    # Normalization
    sitk.Normalize(fixed_vol)
    sitk.Normalize(moving_vol)

    # Match Histograms
    if matchHistograms:
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

    learningRate = 1.0
    minStep = 1
    nIterations = 500
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate, minStep, nIterations)
    reg.SetShrinkFactorsPerLevel(shrinkFactors=[1])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Adding fixed and moving mask if defined
    if maskFixed is not None:
        print("Add fixed mask")
        reg.SetMetricFixedMask(sitk.GetImageFromArray(maskFixed.astype(int)))
    if maskMoving is not None:
        print("Add moving mask")
        reg.SetMetricMovingMask(sitk.GetImageFromArray(maskMoving.astype(int)))

    finalTransform = reg.Execute(fixed_vol, moving_vol)

    if verbose:
        print(reg, finalTransform)

    if vol1.ndim == 3:
        dx = finalTransform.GetParameters()[0]
        dy = finalTransform.GetParameters()[1]
        dz = finalTransform.GetParameters()[2]
        deltas = [dx, dy, dz]

    elif vol1.ndim == 2:
        dx = -finalTransform.GetParameters()[1]
        dy = -finalTransform.GetParameters()[0]
        deltas = [dx, dy]

    MI = reg.GetMetricValue()
    return deltas, MI


def align_images_sitk(im1, im2):

    # Parameters
    learning_rate = 4.0
    min_step = 0.01
    max_iteration = 200

    fixed = sitk.GetImageFromArray(im1)
    moving = sitk.GetImageFromArray(im2)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.SetOptimizerAsRegularStepGradientDescent(learning_rate, min_step, max_iteration, relaxationFactor=0.5,
                                               gradientMagnitudeTolerance=1e-9)
    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)

    outTx = R.Execute(fixed, moving)

    dx = -outTx.GetParameters()[1]
    dy = -outTx.GetParameters()[0]
    deltas = [dx, dy]
    m = R.GetMetricValue()
    return deltas, m


def register_2d_images_sitk(ref_image, moving_image, method='euler',
                            metric='MSE', max_iterations=2500,
                            min_step=1e-12, grad_mag_tol=1e-12,
                            fixed_mask=None, moving_mask=None,
                            return_3d_transform=False, verbose=False,
                            initial_translation=None, initial_step=None):
    """
    Register 2D `moving_image` to `ref_image`.

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
        Optional mask to apply to the reference image during
        registration.
    moving_mask: numpy.ndarray
        Optional mask to apply to the moving image during
        registration.
    return_3d_transform: bool
        If True, will return a 3D transform instead of 2D. Useful
        when the transform is applied to a 3D image.
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

    R = sitk.ImageRegistrationMethod()

    if fixed_mask is not None:
        fixed_sitk_mask = sitk.GetImageFromArray(fixed_mask.astype(np.uint8))
        R.SetMetricFixedMask(fixed_sitk_mask)
    if moving_mask is not None:
        moving_sitk_mask = sitk.GetImageFromArray(moving_mask.astype(np.uint8))
        R.SetMetricMovingMask(moving_sitk_mask)

    if metric.lower() == 'mse':
        R.SetMetricAsMeanSquares()
    elif metric.lower() == 'cc':
        R.SetMetricAsCorrelation()
    elif metric.lower() == 'antscc':
        R.SetMetricAsANTSNeighborhoodCorrelation(radius=20)
    elif metric.lower() == 'mi':
        R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    else:
        raise ValueError("Unknown metric: {}".format(metric))

    # Use smaller step size when we have an initial translation estimate (to avoid drifting away)
    if initial_step is None:
        step_size = 1.0 if initial_translation is not None else 4.0
    else:
        step_size = initial_step

    R.SetOptimizerAsRegularStepGradientDescent(step_size, min_step, max_iterations,
                                               0.5, grad_mag_tol)
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([3, 2, 1])

    # this step is essential for the registration to work properly
    # determines the scale of each parameter in the optimizer
    R.SetOptimizerScalesFromIndexShift()

    if method == 'euler':
        sitk_transform = sitk.Euler2DTransform()
    elif method == 'affine':
        sitk_transform = sitk.AffineTransform(2)
    elif method == 'translation':
        sitk_transform = sitk.TranslationTransform(2)
    else:
        raise ValueError("Unknown method: {}".format(method))

    # Initialize transform - use provided translation or centered initializer
    if initial_translation is not None:
        # Set center to image center
        center = [fixed_sitk_image.GetWidth() / 2.0, fixed_sitk_image.GetHeight() / 2.0]
        if method == 'euler':
            sitk_transform.SetCenter(center)
            sitk_transform.SetTranslation(initial_translation)
        elif method == 'affine':
            sitk_transform.SetCenter(center)
            sitk_transform.SetTranslation(initial_translation)
        elif method == 'translation':
            sitk_transform.SetOffset(initial_translation)
    else:
        sitk_transform = sitk.CenteredTransformInitializer(
            fixed_sitk_image,
            moving_sitk_image,
            sitk_transform
        )

    R.SetInitialTransform(sitk_transform)

    R.SetInterpolator(sitk.sitkLinear)
    if verbose:
        R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    out_transform = R.Execute(fixed_sitk_image, moving_sitk_image)
    stop_condition = R.GetOptimizerStopConditionDescription()
    error = R.GetMetricValue()

    if return_3d_transform:
        if method == 'euler':
            angle_rad = out_transform.GetAngle()
            center_of_rotation = out_transform.GetCenter()
            translation = out_transform.GetTranslation()
            transform_3d = sitk.Euler3DTransform()
            transform_3d.SetCenter([center_of_rotation[0], center_of_rotation[1], 0.0])
            transform_3d.SetRotation(0.0, 0.0, angle_rad)
            transform_3d.SetTranslation([translation[0], translation[1], 0.0])
        elif method == 'translation':
            translation = out_transform.GetOffset()
            transform_3d = sitk.TranslationTransform(3)
            transform_3d.SetOffset([translation[0], translation[1], 0.0])
        elif method == 'affine':
            transform_3d = sitk.AffineTransform(3)
            translation = out_transform.GetTranslation()
            transform_3d.SetCenter(out_transform.GetCenter() + (0.0,))
            transform_3d.SetTranslation([translation[0], translation[1], 0.0])
            matrix_2d = out_transform.GetMatrix()
            matrix_3d = np.zeros((3, 3))
            matrix_3d[:2, :2] = np.array(matrix_2d).reshape((2, 2))
            matrix_3d[2, 2] = 1.0
            transform_3d.SetMatrix(matrix_3d.flatten().tolist())
        else:
            raise ValueError("Unknown method: {}".format(method))
        out_transform = transform_3d

    return out_transform, stop_condition, error


def command_iteration(method):
    """ Callback invoked when the optimization has an iteration """
    if method.GetOptimizerIteration() == 0:
        print("Estimated Scales: ", method.GetOptimizerScales())
    print(f"{method.GetOptimizerIteration():3} "
          + f"= {method.GetMetricValue():7.5f} "
          + f": {method.GetOptimizerPosition()}")


def apply_transform(moving_image, transform):
    """
    Apply `transform` to `moving_image`. The image is
    transformed inside its own domain.

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
    if len(nonzero_vals) > 0:
        default_val = float(np.percentile(nonzero_vals, 1))
    else:
        default_val = 0.0
    resampler.SetDefaultPixelValue(default_val)

    resampler.SetTransform(transform)

    out = resampler.Execute(moving_image_sitk)
    out = sitk.GetArrayFromImage(out)

    return out


def find_best_z(fixed_vol, moving_slice: np.ndarray,
                expected_z: int, search_range: int,
                mask=None):
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
    mask : np.ndarray or None
        Optional 2D tissue mask applied to correlation.

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

    if mask is not None:
        moving_roi = moving_roi * mask[roi].astype(np.float32)

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

        if mask is not None:
            fixed_roi = fixed_roi * mask[roi].astype(np.float32)

        fixed_norm = (fixed_roi - fixed_roi.mean()) / (fixed_roi.std() + 1e-8)
        corr = float(np.mean(fixed_norm * moving_norm))

        if corr > best_corr:
            best_corr = corr
            best_z = z

    return max(0, min(nz - 1, best_z)), best_corr


def register_refinement(fixed: np.ndarray, moving: np.ndarray,
                        enable_rotation: bool = True,
                        max_rotation_deg: float = 5.0,
                        max_translation_px: float = 20.0,
                        fixed_mask=None, moving_mask=None):
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
        learningRate=1.0, numberOfIterations=200,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(transform, inPlace=False)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])

    try:
        final = reg.Execute(fixed_sitk, moving_sitk)
        metric = reg.GetMetricValue()

        inner = final.GetNthTransform(0) if final.GetName() == 'CompositeTransform' else final

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

    except Exception as e:
        return 0.0, 0.0, 0.0, float('inf')


def create_transform(tx: float, ty: float, angle_deg: float, center):
    """Create a 3D SimpleITK Euler transform from 2D parameters.

    Parameters
    ----------
    tx, ty : float
        Translation in pixels.
    angle_deg : float
        Rotation angle in degrees (around Z axis).
    center : sequence
        (cx, cy) rotation center.

    Returns
    -------
    sitk.Euler3DTransform
    """

    transform = sitk.Euler3DTransform()
    transform.SetCenter([center[0], center[1], 0.0])
    transform.SetRotation(0.0, 0.0, np.radians(angle_deg))
    transform.SetTranslation([tx, ty, 0.0])
    return transform
