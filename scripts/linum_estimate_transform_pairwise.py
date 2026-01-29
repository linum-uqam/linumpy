#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pairwise slice registration for 3D reconstruction.

Estimates the 2D transform (translation, rotation, or affine) to align consecutive
slices in a serial sectioning dataset. Uses phase correlation for robust initial
alignment followed by intensity-based refinement.

Output:
- transform.tfm: SimpleITK transform file
- offsets.txt: Z-index correspondence between fixed and moving volumes
"""
import linumpy._thread_config  # noqa: F401

import argparse
import logging
import os

import numpy as np
import SimpleITK as sitk

from linumpy.io.zarr import read_omezarr
from linumpy.stitching.registration import pairWisePhaseCorrelation
from linumpy.utils.io import assert_output_exists, add_overwrite_arg
from linumpy.utils.metrics import collect_pairwise_registration_metrics

from linumpy._thread_config import configure_all_libraries
configure_all_libraries()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Argument Parsing
# =============================================================================

def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_fixed', help='Fixed volume (.ome.zarr)')
    p.add_argument('in_moving', help='Moving volume (.ome.zarr)')
    p.add_argument('out_directory', help='Output directory')

    # Masks
    p.add_argument('--use_masks', action='store_true', help='Use masks for registration')
    p.add_argument('--moving_mask', type=str, default=None)
    p.add_argument('--fixed_mask', type=str, default=None)
    p.add_argument('--mask_mode', choices=['multiply', 'sitk', 'none'], default='multiply',
                   help='How to use masks: multiply (pre-multiply images), '
                        'sitk (use SimpleITK mask parameters), none (ignore) [%(default)s]')

    # Registration settings
    p.add_argument('--moving_slice_index', type=int, default=0,
                   help='Z-index in moving volume to use [%(default)s]')
    p.add_argument('--transform', choices=['translation', 'euler', 'affine'],
                   default='translation', help='Transform type [%(default)s]')
    p.add_argument('--metric', choices=['MSE', 'CC', 'MI'], default='CC',
                   help='Similarity metric [%(default)s]')
    p.add_argument('--max_translation', type=float, default=50.0,
                   help='Max allowed translation in pixels [%(default)s]')
    p.add_argument('--max_rotation', type=float, default=2.0,
                   help='Max allowed rotation in degrees [%(default)s]')

    # Z-matching
    p.add_argument('--slicing_interval', type=float, default=0.200,
                   help='Physical distance between slices in mm [%(default)s]')
    p.add_argument('--slice_gap_multiplier', type=int, default=1,
                   help='Multiplier when slices are skipped [%(default)s]')
    p.add_argument('--allowed_drifting', type=float, default=0.050,
                   help='Z-drift tolerance in mm [%(default)s]')
    p.add_argument('--z_bias', type=float, default=0.15,
                   help='Penalty weight for deviation from expected z (0-1) [%(default)s]')

    # Output
    p.add_argument('--out_transform', default='transform.tfm')
    p.add_argument('--out_offsets', default='offsets.txt')
    p.add_argument('--screenshot', default=None, help='Debug screenshot path')

    add_overwrite_arg(p)
    return p


# =============================================================================
# Image Processing
# =============================================================================

def normalize_image(image, robust=True):
    """
    Normalize image to [0, 1] using percentile clipping.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    robust : bool
        If True, uses a more robust normalization that better handles
        varying tissue content between slices.
    """
    valid = image > 0
    if not np.any(valid):
        return np.zeros_like(image, dtype=np.float32)

    if robust:
        # Use narrower percentile range for more stable normalization
        # This reduces sensitivity to varying tissue content
        pmin = np.percentile(image[valid], 5)
        pmax = np.percentile(image[valid], 95)
    else:
        pmin = np.percentile(image[valid], 1)
        pmax = np.percentile(image, 99)

    if pmax <= pmin:
        return np.zeros_like(image, dtype=np.float32)

    normalized = (image.astype(np.float32) - pmin) / (pmax - pmin)
    return np.clip(normalized, 0, 1)



def compute_phase_correlation(fixed, moving, downsample=4):
    """
    Compute translation using phase correlation.

    Uses downsampled images for speed, returns translation in original coordinates.
    Applies windowing to reduce edge effects that can bias translation estimates.
    """
    from scipy.ndimage import zoom

    # Downsample for speed
    if downsample > 1:
        fixed_ds = zoom(fixed, 1/downsample, order=1)
        moving_ds = zoom(moving, 1/downsample, order=1)
    else:
        fixed_ds, moving_ds = fixed, moving

    # Apply Hanning window to reduce edge effects
    # Edge effects can cause systematic bias in phase correlation
    window_y = np.hanning(fixed_ds.shape[0])
    window_x = np.hanning(fixed_ds.shape[1])
    window = np.outer(window_y, window_x)

    fixed_windowed = fixed_ds * window
    moving_windowed = moving_ds * window

    try:
        deltas = pairWisePhaseCorrelation(fixed_windowed, moving_windowed)
        # Scale back to original resolution
        tx = float(deltas[1]) * downsample
        ty = float(deltas[0]) * downsample
        return tx, ty
    except Exception as e:
        logger.warning(f"Phase correlation failed: {e}")
        return 0.0, 0.0


# =============================================================================
# Registration
# =============================================================================

def register_translation(fixed, moving, fixed_mask=None, moving_mask=None,
                         initial_translation=None, metric='CC', max_iter=500,
                         mask_mode='multiply'):
    """
    Register using translation-only transform.

    Parameters
    ----------
    mask_mode : str
        How to use masks: 'multiply' (pre-multiply images, recommended),
        'sitk' (use SimpleITK mask parameters), 'none' (ignore masks).

    Returns (tx, ty, metric_value).
    """
    # Apply masks by multiplication if requested (recommended approach)
    # This avoids issues with SimpleITK mask handling at tissue boundaries
    if mask_mode == 'multiply':
        if fixed_mask is not None:
            fixed = fixed * fixed_mask.astype(np.float32)
        if moving_mask is not None:
            moving = moving * moving_mask.astype(np.float32)
        # Don't pass masks to SimpleITK
        fixed_mask = None
        moving_mask = None

    fixed_sitk = sitk.GetImageFromArray(fixed.astype(np.float32))
    moving_sitk = sitk.GetImageFromArray(moving.astype(np.float32))

    # Initial transform
    transform = sitk.TranslationTransform(2)
    if initial_translation is not None:
        transform.SetOffset([float(initial_translation[0]), float(initial_translation[1])])

    # Registration setup
    registration = sitk.ImageRegistrationMethod()

    if metric == 'CC':
        registration.SetMetricAsCorrelation()
    elif metric == 'MI':
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    else:
        registration.SetMetricAsMeanSquares()

    # Use masks in SimpleITK only if mask_mode is 'sitk'
    if mask_mode == 'sitk':
        if fixed_mask is not None:
            fixed_mask_sitk = sitk.GetImageFromArray(fixed_mask.astype(np.uint8))
            registration.SetMetricFixedMask(fixed_mask_sitk)
        if moving_mask is not None:
            moving_mask_sitk = sitk.GetImageFromArray(moving_mask.astype(np.uint8))
            registration.SetMetricMovingMask(moving_mask_sitk)

    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=0.01,
        numberOfIterations=max_iter
    )
    registration.SetOptimizerScalesFromPhysicalShift()

    registration.SetInitialTransform(transform, inPlace=False)
    registration.SetInterpolator(sitk.sitkLinear)

    # Multi-resolution
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])

    try:
        final_transform = registration.Execute(fixed_sitk, moving_sitk)
        offset = final_transform.GetOffset()
        metric_value = registration.GetMetricValue()
        return offset[0], offset[1], metric_value
    except Exception as e:
        logger.warning(f"Registration failed: {e}")
        return 0.0, 0.0, float('inf')


def register_rigid(fixed, moving, fixed_mask=None, moving_mask=None,
                   initial_translation=None, metric='CC', max_iter=500,
                   mask_mode='multiply'):
    """
    Register using rigid (rotation + translation) transform.

    Parameters
    ----------
    mask_mode : str
        How to use masks: 'multiply' (pre-multiply images, recommended),
        'sitk' (use SimpleITK mask parameters), 'none' (ignore masks).

    Returns (tx, ty, angle_deg, metric_value).
    """
    # Apply masks by multiplication if requested (recommended approach)
    if mask_mode == 'multiply':
        if fixed_mask is not None:
            fixed = fixed * fixed_mask.astype(np.float32)
        if moving_mask is not None:
            moving = moving * moving_mask.astype(np.float32)
        # Don't pass masks to SimpleITK
        fixed_mask = None
        moving_mask = None

    fixed_sitk = sitk.GetImageFromArray(fixed.astype(np.float32))
    moving_sitk = sitk.GetImageFromArray(moving.astype(np.float32))

    # Initial transform centered on image
    transform = sitk.Euler2DTransform()
    center = [fixed.shape[1] / 2.0, fixed.shape[0] / 2.0]
    transform.SetCenter(center)

    if initial_translation is not None:
        transform.SetTranslation([float(initial_translation[0]), float(initial_translation[1])])

    # Registration setup
    registration = sitk.ImageRegistrationMethod()

    if metric == 'CC':
        registration.SetMetricAsCorrelation()
    elif metric == 'MI':
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    else:
        registration.SetMetricAsMeanSquares()

    # Use masks in SimpleITK only if mask_mode is 'sitk'
    if mask_mode == 'sitk':
        if fixed_mask is not None:
            fixed_mask_sitk = sitk.GetImageFromArray(fixed_mask.astype(np.uint8))
            registration.SetMetricFixedMask(fixed_mask_sitk)
        if moving_mask is not None:
            moving_mask_sitk = sitk.GetImageFromArray(moving_mask.astype(np.uint8))
            registration.SetMetricMovingMask(moving_mask_sitk)

    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=0.001,
        numberOfIterations=max_iter
    )
    registration.SetOptimizerScalesFromPhysicalShift()

    registration.SetInitialTransform(transform, inPlace=False)
    registration.SetInterpolator(sitk.sitkLinear)

    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])

    try:
        final_transform = registration.Execute(fixed_sitk, moving_sitk)
        params = final_transform.GetParameters()
        angle_rad = params[0]
        tx, ty = params[1], params[2]
        metric_value = registration.GetMetricValue()
        return tx, ty, np.degrees(angle_rad), metric_value
    except Exception as e:
        logger.warning(f"Rigid registration failed: {e}")
        return 0.0, 0.0, 0.0, float('inf')


def create_3d_transform(tx, ty, angle_deg=0.0, transform_type='translation'):
    """Create a 3D SimpleITK transform from 2D parameters."""
    if transform_type == 'translation' or angle_deg == 0.0:
        transform = sitk.TranslationTransform(3)
        transform.SetOffset([tx, ty, 0.0])
    else:
        transform = sitk.Euler3DTransform()
        transform.SetRotation(0.0, 0.0, np.radians(angle_deg))
        transform.SetTranslation([tx, ty, 0.0])
    return transform


# =============================================================================
# Main Registration Logic
# =============================================================================

def _downsample_mask(mask, scale):
    if mask is None:
        return None
    from scipy.ndimage import zoom
    return zoom(mask.astype(np.float32), scale, order=0) > 0


def find_best_z_match(fixed_vol, moving_image, expected_z, search_range, metric='CC',
                      fixed_mask_vol=None, moving_mask=None, z_bias=0.0):
    """
    Find the best matching z-index in fixed volume for the moving image.

    Uses normalized cross-correlation on downsampled images for speed.
    Optionally masks and biases toward the expected z to avoid interface snapping.
    """
    from scipy.ndimage import zoom

    best_z = expected_z
    best_score = -float('inf')

    # Downsample for speed
    moving_ds = zoom(moving_image, 0.25, order=1)
    moving_norm = normalize_image(moving_ds)
    moving_mask_ds = _downsample_mask(moving_mask, 0.25)

    for z in range(max(0, expected_z - search_range),
                   min(fixed_vol.shape[0], expected_z + search_range + 1)):
        fixed_slice = np.array(fixed_vol[z])
        fixed_ds = zoom(fixed_slice, 0.25, order=1)
        fixed_norm = normalize_image(fixed_ds)

        fixed_mask_ds = None
        if fixed_mask_vol is not None:
            fixed_mask_ds = _downsample_mask(np.array(fixed_mask_vol[z]) > 0, 0.25)

        # Compute correlation with optional mask intersection
        if np.std(fixed_norm) > 0 and np.std(moving_norm) > 0:
            if fixed_mask_ds is not None and moving_mask_ds is not None:
                mask = np.logical_and(fixed_mask_ds, moving_mask_ds)
                if np.count_nonzero(mask) > 100:
                    fixed_vals = fixed_norm[mask]
                    moving_vals = moving_norm[mask]
                else:
                    fixed_vals = fixed_norm.ravel()
                    moving_vals = moving_norm.ravel()
            else:
                fixed_vals = fixed_norm.ravel()
                moving_vals = moving_norm.ravel()

            score = np.corrcoef(fixed_vals, moving_vals)[0, 1]
            if z_bias > 0 and search_range > 0:
                score -= z_bias * (abs(z - expected_z) / float(search_range))
            if score > best_score:
                best_score = score
                best_z = z

    logger.info(f"Best z-match: {best_z} (correlation: {best_score:.3f})")
    return best_z


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load volumes
    logger.info(f"Loading fixed: {args.in_fixed}")
    fixed_vol, res = read_omezarr(args.in_fixed)
    logger.info(f"Loading moving: {args.in_moving}")
    moving_vol, _ = read_omezarr(args.in_moving)

    # Create output directory
    assert_output_exists(args.out_directory, parser, args)
    os.makedirs(args.out_directory)

    # Get moving image
    moving_raw = np.array(moving_vol[args.moving_slice_index])
    moving_image = normalize_image(moving_raw)

    # Load masks if provided
    fixed_mask = None
    moving_mask = None
    fixed_mask_vol = None
    if args.use_masks:
        if args.moving_mask:
            moving_mask_vol, _ = read_omezarr(args.moving_mask)
            moving_mask = np.array(moving_mask_vol[args.moving_slice_index]) > 0
        if args.fixed_mask:
            fixed_mask_vol, _ = read_omezarr(args.fixed_mask)

    # Calculate expected z-index correspondence
    interval_vox = int(np.ceil(args.slicing_interval / res[0]))
    total_interval = interval_vox * args.slice_gap_multiplier
    expected_z = total_interval + args.moving_slice_index
    search_range = int(np.ceil(args.allowed_drifting / res[0])) * args.slice_gap_multiplier

    logger.info(f"Expected z: {expected_z}, search range: ±{search_range}")

    # Find best z-match
    best_z = find_best_z_match(
        fixed_vol,
        moving_raw,
        expected_z,
        search_range,
        fixed_mask_vol=fixed_mask_vol,
        moving_mask=moving_mask,
        z_bias=args.z_bias
    )

    # Get fixed image at best z
    fixed_raw = np.array(fixed_vol[best_z])
    fixed_image = normalize_image(fixed_raw)

    # Get fixed mask if provided
    if args.use_masks and fixed_mask_vol is not None:
        fixed_mask = np.array(fixed_mask_vol[best_z]) > 0


    # Step 1: Phase correlation for initial translation estimate
    logger.info("Computing phase correlation...")
    init_tx, init_ty = compute_phase_correlation(fixed_image, moving_image)
    init_mag = np.sqrt(init_tx**2 + init_ty**2)
    logger.info(f"Phase correlation: tx={init_tx:.1f}, ty={init_ty:.1f} (mag={init_mag:.1f})")

    # Step 2: Refine with intensity-based registration
    if args.use_masks:
        logger.info(f"Using masks with mode: {args.mask_mode}")
    if args.transform == 'translation':
        logger.info("Refining with translation registration...")
        tx, ty, metric_val = register_translation(
            fixed_image, moving_image,
            fixed_mask, moving_mask,
            initial_translation=(init_tx, init_ty),
            metric=args.metric,
            mask_mode=args.mask_mode
        )
        angle_deg = 0.0
    else:
        logger.info(f"Refining with {args.transform} registration...")
        tx, ty, angle_deg, metric_val = register_rigid(
            fixed_image, moving_image,
            fixed_mask, moving_mask,
            initial_translation=(init_tx, init_ty),
            metric=args.metric,
            mask_mode=args.mask_mode
        )

    # Validate result
    translation_mag = np.sqrt(tx**2 + ty**2)
    logger.info(f"Final: tx={tx:.1f}, ty={ty:.1f}, rot={angle_deg:.2f}°, mag={translation_mag:.1f}")

    # Check bounds
    if translation_mag > args.max_translation:
        logger.warning(f"Translation {translation_mag:.1f} exceeds max {args.max_translation}")
        logger.warning("Using phase correlation result instead")
        tx, ty = init_tx, init_ty
        angle_deg = 0.0
        translation_mag = init_mag

    if abs(angle_deg) > args.max_rotation:
        logger.warning(f"Rotation {angle_deg:.2f}° exceeds max {args.max_rotation}°")
        angle_deg = 0.0

    # Still exceeds? Fall back to identity
    if translation_mag > args.max_translation:
        logger.warning("Phase correlation also exceeds threshold - using identity")
        tx, ty, angle_deg = 0.0, 0.0, 0.0

    # Create and save transform
    transform = create_3d_transform(tx, ty, angle_deg, args.transform)
    sitk.WriteTransform(transform, os.path.join(args.out_directory, args.out_transform))

    # Save offsets
    np.savetxt(os.path.join(args.out_directory, args.out_offsets),
               np.array([best_z, args.moving_slice_index]), fmt='%d')

    # Collect metrics
    collect_pairwise_registration_metrics(
        registration_error=float(metric_val) if metric_val != float('inf') else 0.0,
        tx=float(tx),
        ty=float(ty),
        rotation_deg=float(angle_deg),
        best_z_index=int(best_z),
        expected_z_index=int(expected_z),
        output_path=args.out_directory,
        fixed_path=args.in_fixed,
        moving_path=args.in_moving,
        params={'transform_type': args.transform, 'metric': args.metric}
    )

    logger.info(f"Transform saved to {args.out_directory}")

    # Screenshot if requested
    if args.screenshot:
        import matplotlib.pyplot as plt

        # Apply transform for visualization
        moving_sitk = sitk.GetImageFromArray(moving_image.astype(np.float32))
        transform_2d = sitk.TranslationTransform(2, [tx, ty])
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(moving_sitk)
        resampler.SetTransform(transform_2d)
        resampler.SetInterpolator(sitk.sitkLinear)
        registered = sitk.GetArrayFromImage(resampler.Execute(moving_sitk))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(fixed_image, cmap='gray')
        axes[0].set_title(f'Fixed (z={best_z})')
        axes[1].imshow(moving_image, cmap='gray')
        axes[1].set_title('Moving')
        axes[2].imshow(registered, cmap='gray')
        axes[2].set_title(f'Registered (tx={tx:.1f}, ty={ty:.1f})')

        for ax in axes:
            ax.axis('off')

        fig.suptitle(f'Registration: metric={metric_val:.4f}')
        fig.tight_layout()
        fig.savefig(args.screenshot, dpi=150)
        plt.close(fig)
        logger.info(f"Screenshot saved to {args.screenshot}")


if __name__ == '__main__':
    main()
