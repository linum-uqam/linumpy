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
    #vol1 = icorr.eqhist(vol1)
    #vol2 = icorr.eqhist(vol2)

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
    # reg.SetOptimizerAsConjugateGradientLineSearch(learningRate, nIterations)
    # reg.SetOptimizerAsGradientDescentLineSearch(learningRate, minStep)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate, minStep, nIterations)
    #reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel(shrinkFactors=[1])
    # reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[3,2,1])
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
    #plt.subplot(121)
    #plt.imshow(im1)
    #plt.subplot(122)
    #plt.imshow(im2)
    #plt.show()

    # Parameters
    learning_rate = 4.0
    min_step = 0.01
    max_iteration = 200


    fixed = sitk.GetImageFromArray(im1)
    moving = sitk.GetImageFromArray(im2)

    R = sitk.ImageRegistrationMethod()
    #R.SetMetricAsMeanSquares()
    R.SetMetricAsCorrelation()
    #R.SetOptimizerAsGradientDescent(learning_rate, max_iteration)
    R.SetOptimizerAsRegularStepGradientDescent(learning_rate, min_step, max_iteration, relaxationFactor=0.5, gradientMagnitudeTolerance=1e-9)
    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    #R.SetOptimizerScales([8,4,2,1])
    R.SetInterpolator(sitk.sitkLinear)

    outTx = R.Execute(fixed, moving)

    #print("-------")
    #print(outTx)
    #print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    #print(f" Iteration: {R.GetOptimizerIteration()}")
    #print(f" Metric value: {R.GetMetricValue()}")

    # Resampling
    #resampler = sitk.ResampleImageFilter()
    #resampler.SetReferenceImage(fixed)
    #resampler.SetInterpolator(sitk.sitkLinear)
    #resampler.SetDefaultPixelValue(1.0)
    #resampler.SetTransform(outTx)
    #out = resampler.Execute(moving)

    # Create composite
    #simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    #simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    #cimg = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0)
    #plt.imshow(sitk.GetArrayFromImage(cimg)); plt.show()


    dx = -outTx.GetParameters()[1]
    dy = -outTx.GetParameters()[0]
    deltas = [dx, dy]
    m = R.GetMetricValue()
    return deltas, m


def register_2d_images_sitk(ref_image, moving_image, method='euler',
                            metric='MSE', max_iterations=10000, 
                            min_step=1e-12, grad_mag_tol=1e-12,
                            return_3d_transform=False, verbose=False):
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
    return_3d_transform: bool
        If True, will return a 3D transform instead of 2D. Useful
        when the transform is applied to a 3D image.
    verbose: bool
        If True, will log registrations metric at each iteration.

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

    R.SetOptimizerAsRegularStepGradientDescent(4.0, min_step, max_iterations,
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

    # sitk_transform.SetParameters([0.0] * sitk_transform.GetNumberOfParameters())
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
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)

    out = resampler.Execute(moving_image_sitk)
    out = sitk.GetArrayFromImage(out)

    return out