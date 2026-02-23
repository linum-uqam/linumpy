#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stack 3D slices using motor positions for XY alignment and simplified Z-matching.

This script implements motor-position-based 3D reconstruction:
1. XY ALIGNMENT: Uses shifts_xy.csv (motor positions) - precise and consistent
2. Z-MATCHING: Finds optimal overlap depth using correlation - simplified

This replaces the complex pairwise registration approach when motor positions
are reliable. The XY shifts from the microscope stage are more precise than
image-based registration for positioning.

The Z-matching finds where consecutive slices should overlap by correlating
the bottom of one slice with the top of the next.
"""

import linumpy._thread_config  # noqa: F401

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from linumpy.io.zarr import read_omezarr, AnalysisOmeZarrWriter
from linumpy.utils.io import add_overwrite_arg, assert_output_exists
from linumpy.utils.metrics import collect_stack_metrics

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_slices_dir',
                   help='Directory containing slice volumes (.ome.zarr)')
    p.add_argument('in_shifts',
                   help='CSV file with XY shifts (shifts_xy.csv)')
    p.add_argument('out_stack',
                   help='Output stacked volume (.ome.zarr)')

    # Registration refinements (optional)
    p.add_argument('--transforms_dir', type=str, default=None,
                   help='Directory containing pairwise registration outputs.\n'
                        'If provided, applies rotation/translation refinements.')
    p.add_argument('--rotation_only', action='store_true',
                   help='Apply only rotation from registration transforms, ignore translation.\n'
                        'Use this to prevent XY drift when motor positions are trusted.')
    p.add_argument('--max_rotation_deg', type=float, default=1.0,
                   help='Maximum rotation to apply per slice (degrees). Larger rotations\n'
                        'are clamped to prevent registration errors from causing drift. [%(default)s]')
    p.add_argument('--accumulate_translations', action='store_true',
                   help='Accumulate pairwise translations cumulatively across slices.\n'
                        'Each slice gets the sum of all preceding pairwise translations.\n'
                        'This propagates corrections through the stack, fixing cumulative\n'
                        'drift and motor position errors. Rotation stays per-slice.')
    p.add_argument('--max_pairwise_translation', type=float, default=0,
                   help='Maximum reliable pairwise translation magnitude (pixels).\n'
                        'Translations at or above this value are assumed to be registration\n'
                        'failures (hitting the optimizer boundary) and excluded from\n'
                        'accumulation. Set to registration_max_translation. 0 = disabled.\n'
                        '[%(default)s]')
    p.add_argument('--smooth_window', type=int, default=0,
                   help='Smooth cumulative translations with a moving average of this window\n'
                        'size (in slices). Reduces XY jitter between consecutive slices,\n'
                        'improving blend quality. 0 disables smoothing. [%(default)s]')
    p.add_argument('--skip_error_transforms', action='store_true',
                   help='Skip registration transforms flagged as overall_status="error"\n'
                        'in pairwise_registration_metrics.json.  Error-status registrations\n'
                        'are typically spurious (e.g. registered against an interpolated\n'
                        'slice) and applying them introduces large rotation/translation\n'
                        'artifacts at those slice boundaries.')
    p.add_argument('--skip_warning_transforms', action='store_true',
                   help='Also skip transforms with overall_status="warning".\n'
                        'Warning-status registrations hit the optimizer boundary (e.g. large\n'
                        'translation clamped at max_translation_px), making their fixed_z/\n'
                        'moving_z Z-offsets unreliable. Discarding them falls back to the\n'
                        'default moving_z_first_index, preventing Z gaps caused by bad\n'
                        'Z-overlap estimates from failed registrations.')
    p.add_argument('--no_xy_shift', action='store_true',
                   help='Skip XY shifting from motor positions.\n'
                        'Use when slices are already in common space (e.g., from bring_to_common_space).')

    # Z-matching parameters
    p.add_argument('--slicing_interval_mm', type=float, default=0.200,
                   help='Physical slice thickness in mm [%(default)s]')
    p.add_argument('--search_range_mm', type=float, default=0.100,
                   help='Search range for Z-matching in mm [%(default)s]')
    p.add_argument('--use_expected_overlap', action='store_true',
                   help='Use expected overlap from slicing_interval instead of correlation')
    p.add_argument('--moving_z_first_index', type=int, default=8,
                   help='Starting Z-index in moving volume to skip noisy data [%(default)s]')

    # Blending
    p.add_argument('--blend', action='store_true',
                   help='Blend overlapping regions using diffusion method')
    p.add_argument('--blend_depth', type=int, default=None,
                   help='Number of z-slices to blend (default: auto from overlap)')

    # Output options
    p.add_argument('--pyramid_resolutions', type=float, nargs='+',
                   default=[10, 25, 50, 100],
                   help='Target resolutions for pyramid levels in microns')
    p.add_argument('--make_isotropic', action='store_true', default=True,
                   help='Resample to isotropic voxels')
    p.add_argument('--no_isotropic', dest='make_isotropic', action='store_false')

    # Debug
    p.add_argument('--max_slices', type=int, default=None,
                   help='Maximum slices to process (for testing)')
    p.add_argument('--output_z_matches', type=str, default=None,
                   help='Output CSV with Z-matching results')

    add_overwrite_arg(p)
    return p


def load_shifts(shifts_path):
    """Load shifts CSV and compute cumulative XY shifts."""
    df = pd.read_csv(shifts_path)

    # Get all slice IDs
    all_ids = sorted(set(df['fixed_id'].tolist() + df['moving_id'].tolist()))

    # Build shift lookup
    shift_lookup = {}
    for _, row in df.iterrows():
        fixed_id = int(row['fixed_id'])
        moving_id = int(row['moving_id'])
        shift_lookup[(fixed_id, moving_id)] = (row['x_shift_mm'], row['y_shift_mm'])

    # Compute cumulative shifts
    cumsum = {all_ids[0]: (0.0, 0.0)}
    for i in range(len(all_ids) - 1):
        fixed_id = all_ids[i]
        moving_id = all_ids[i + 1]

        if (fixed_id, moving_id) in shift_lookup:
            dx_mm, dy_mm = shift_lookup[(fixed_id, moving_id)]
        else:
            logger.warning(f"No shift for {fixed_id} -> {moving_id}, using 0")
            dx_mm, dy_mm = 0.0, 0.0

        prev_dx, prev_dy = cumsum[fixed_id]
        cumsum[moving_id] = (prev_dx + dx_mm, prev_dy + dy_mm)

    return cumsum, all_ids


def load_registration_transforms(transforms_dir, slice_ids,
                                 skip_error_status=False,
                                 skip_warning_status=False):
    """
    Load pairwise registration transforms from directory.

    Parameters
    ----------
    transforms_dir : Path
        Directory containing registration outputs (subdirs per slice)
    slice_ids : list
        List of slice IDs to load transforms for
    skip_error_status : bool
        If True, discard transforms whose pairwise_registration_metrics.json
        reports overall_status == 'error'.  These are typically registrations
        that failed (e.g. registered against an interpolated/synthetic slice)
        and would introduce spurious rotations into the stack.
    skip_warning_status : bool
        If True, also discard transforms with overall_status == 'warning'.
        Warning-status registrations hit the optimizer boundary (e.g. large
        translation or rotation) and their Z-offsets (fixed_z/moving_z) are
        unreliable, causing incorrect Z-overlap computation during stacking.
        Discarding them falls back to the default moving_z_first_index.

    Returns
    -------
    dict
        Mapping from slice_id to (transform, z_offset) tuple
    """
    import json

    transforms_dir = Path(transforms_dir)
    transforms = {}

    for slice_id in slice_ids[1:]:  # First slice has no transform
        # Find transform directory for this slice
        # Pattern: slice_z{id}_* or similar
        matching_dirs = list(transforms_dir.glob(f"*z{slice_id:02d}*")) + \
                       list(transforms_dir.glob(f"*z{slice_id}*"))

        if not matching_dirs:
            logger.warning(f"No transform found for slice {slice_id}")
            transforms[slice_id] = None
            continue

        transform_dir = matching_dirs[0]

        # Load transform file
        tfm_files = list(transform_dir.glob("*.tfm"))
        offset_files = list(transform_dir.glob("*.txt"))

        if not tfm_files:
            logger.warning(f"No .tfm file in {transform_dir}")
            transforms[slice_id] = None
            continue

        try:
            # Check registration quality status before loading the transform
            if skip_error_status or skip_warning_status:
                metrics_files = list(transform_dir.glob("pairwise_registration_metrics.json"))
                if metrics_files:
                    with open(metrics_files[0]) as f:
                        metrics = json.load(f)
                    status = metrics.get("overall_status", "ok")
                    should_skip = (status == "error" and skip_error_status) or \
                                  (status == "warning" and skip_warning_status)
                    if should_skip:
                        logger.warning(
                            f"Slice {slice_id}: skipping transform with "
                            f"overall_status='{status}' (unreliable registration)"
                        )
                        transforms[slice_id] = None
                        continue

            tfm = sitk.ReadTransform(str(tfm_files[0]))

            # Load z-offsets if available
            # offsets.txt contains [fixed_z, moving_z]
            # - fixed_z: Z-index in fixed volume where overlap region starts
            # - moving_z: Z-index in moving volume where overlap region starts
            # These indicate WHERE the volumes overlap, not how much.
            fixed_z = None
            moving_z = None
            if offset_files:
                offsets = np.loadtxt(str(offset_files[0]))
                if len(offsets) >= 2:
                    fixed_z = int(offsets[0])
                    moving_z = int(offsets[1])
                    logger.debug(f"Slice {slice_id}: fixed_z={fixed_z}, moving_z={moving_z}")

            transforms[slice_id] = (tfm, fixed_z, moving_z)
            logger.debug(f"Loaded transform for slice {slice_id}")

        except Exception as e:
            logger.warning(f"Could not load transform for slice {slice_id}: {e}")
            transforms[slice_id] = None

    return transforms


def apply_2d_transform(image_2d, transform, rotation_only=False, max_rotation_deg=1.0,
                       override_rotation=None):
    """
    Apply a SimpleITK 2D/3D transform to a 2D image.

    Parameters
    ----------
    image_2d : np.ndarray
        2D image to transform
    transform : sitk.Transform
        SimpleITK transform (extracts 2D rotation/translation)
    rotation_only : bool
        If True, apply only rotation, ignore translation
    max_rotation_deg : float
        Maximum allowed rotation in degrees. Larger rotations are clamped.
        Set to 0 to disable clamping.
    override_rotation : float or None
        If provided, use this rotation angle (radians) instead of the one
        extracted from the transform. Already clamped by the caller.
    Returns
    -------
    np.ndarray
        Transformed 2D image
    """
    # Convert to SimpleITK
    sitk_img = sitk.GetImageFromArray(image_2d.astype(np.float32))

    # Create 2D transform from 3D if needed
    if transform.GetDimension() == 3:
        # Extract 2D parameters from 3D Euler transform
        if isinstance(transform, sitk.Euler3DTransform) or transform.GetName() == 'Euler3DTransform':
            params = transform.GetParameters()
            # Euler3D: [rotX, rotY, rotZ, transX, transY, transZ]
            angle = params[2] if len(params) > 2 else 0  # rotZ
            tx = params[3] if len(params) > 3 else 0
            ty = params[4] if len(params) > 4 else 0

            if override_rotation is not None:
                # Use pre-smoothed rotation; skip per-slice clamping (already done)
                angle = override_rotation
            elif max_rotation_deg > 0:
                # Clamp rotation if it exceeds max_rotation_deg
                max_angle_rad = np.radians(max_rotation_deg)
                if abs(angle) > max_angle_rad:
                    logger.warning(f"Clamping rotation {np.degrees(angle):.2f}° to ±{max_rotation_deg}°")
                    angle = np.clip(angle, -max_angle_rad, max_angle_rad)

            center = transform.GetCenter()
            center_2d = [center[0], center[1]]

            tfm_2d = sitk.Euler2DTransform()
            tfm_2d.SetCenter(center_2d)
            tfm_2d.SetAngle(angle)
            if rotation_only:
                tfm_2d.SetTranslation([0, 0])  # Ignore translation
            else:
                tfm_2d.SetTranslation([tx, ty])
        else:
            # Try to use as-is or create identity
            tfm_2d = sitk.Euler2DTransform()
            angle = 0
    else:
        tfm_2d = transform
        if rotation_only and hasattr(tfm_2d, 'SetTranslation'):
            tfm_2d.SetTranslation([0, 0])
        angle = 0

    # Check if transform is essentially identity (skip if very small)
    # This avoids introducing boundary artifacts from trivial rotations
    tx_final = 0 if rotation_only else tx
    ty_final = 0 if rotation_only else ty

    # If rotation is very small (< 0.1 degrees) and translation small (< 1 pixel), skip
    if abs(angle) < 0.00175 and abs(tx_final) < 1.0 and abs(ty_final) < 1.0:  # 0.1 degrees in radians
        return image_2d.copy()

    # Apply transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_img)
    resampler.SetTransform(tfm_2d)
    resampler.SetInterpolator(sitk.sitkLinear)

    # Use nearest neighbor extrapolation at boundaries to avoid black dots
    # Get a representative non-zero value for the default
    nonzero_vals = image_2d[image_2d > 0]
    if len(nonzero_vals) > 0:
        # Use a small positive value instead of zero to mark extrapolated regions
        default_val = float(np.percentile(nonzero_vals, 1))
    else:
        default_val = 0.0
    resampler.SetDefaultPixelValue(default_val)

    result = resampler.Execute(sitk_img)
    result_arr = sitk.GetArrayFromImage(result)

    return result_arr


def apply_transform_to_volume(vol, transform, rotation_only=False, max_rotation_deg=1.0,
                              override_rotation=None):
    """
    Apply 2D transform to each Z-slice of a volume.

    Parameters
    ----------
    vol : np.ndarray
        3D volume (Z, Y, X)
    transform : sitk.Transform
        Transform to apply to each slice
    rotation_only : bool
        If True, apply only rotation, ignore translation
    max_rotation_deg : float
        Maximum allowed rotation in degrees. Larger rotations are clamped.
    override_rotation : float or None
        If provided, use this rotation angle (radians) for all slices
        instead of extracting it from the transform.

    Returns
    -------
    np.ndarray
        Transformed volume
    """
    result = np.zeros_like(vol)
    for z in range(vol.shape[0]):
        result[z] = apply_2d_transform(vol[z], transform, rotation_only, max_rotation_deg,
                                       override_rotation)
    return result


def find_z_overlap(fixed_vol, moving_vol, slicing_interval_mm, search_range_mm, resolution_um):
    """
    Find optimal Z-overlap between consecutive slices using correlation.

    Parameters
    ----------
    fixed_vol : np.ndarray
        Bottom (fixed) slice volume
    moving_vol : np.ndarray
        Top (moving) slice volume
    slicing_interval_mm : float
        Expected physical slice thickness
    search_range_mm : float
        Search range around expected position
    resolution_um : float
        Z resolution in microns

    Returns
    -------
    int
        Optimal overlap in z-voxels
    float
        Correlation score
    """
    # Convert to voxels
    # Expected overlap = volume_depth - slicing_interval (not the interval itself)
    interval_vox = int((slicing_interval_mm * 1000) / resolution_um)
    expected_overlap_vox = min(fixed_vol.shape[0], moving_vol.shape[0]) - interval_vox
    search_range_vox = int((search_range_mm * 1000) / resolution_um)

    # Search range
    min_overlap = max(1, expected_overlap_vox - search_range_vox)
    max_overlap = min(fixed_vol.shape[0], moving_vol.shape[0],
                      expected_overlap_vox + search_range_vox)

    if min_overlap >= max_overlap:
        return expected_overlap_vox, 0.0

    # Use middle XY region for correlation (faster and more robust)
    h, w = fixed_vol.shape[1], fixed_vol.shape[2]
    margin = min(h, w) // 4
    y_slice = slice(margin, h - margin)
    x_slice = slice(margin, w - margin)

    best_overlap = expected_overlap_vox
    best_corr = -np.inf

    for overlap in range(min_overlap, max_overlap + 1):
        # Get overlapping regions
        fixed_region = fixed_vol[-overlap:, y_slice, x_slice]
        moving_region = moving_vol[:overlap, y_slice, x_slice]

        # Normalize
        fixed_norm = (fixed_region - fixed_region.mean()) / (fixed_region.std() + 1e-8)
        moving_norm = (moving_region - moving_region.mean()) / (moving_region.std() + 1e-8)

        # Correlation
        corr = np.mean(fixed_norm * moving_norm)

        if corr > best_corr:
            best_corr = corr
            best_overlap = overlap

    return best_overlap, best_corr


def apply_xy_shift(vol, dx_px, dy_px, output_shape):
    """
    Compute the destination region for placing a shifted volume.

    Returns the (possibly cropped) volume data and destination coordinates,
    without allocating a full-size output array.

    Parameters
    ----------
    vol : np.ndarray
        3D volume (Z, Y, X)
    dx_px, dy_px : float
        Shift in pixels
    output_shape : tuple
        (out_ny, out_nx) - output canvas size

    Returns
    -------
    cropped_vol : np.ndarray or None
        The cropped volume data to write
    dst_coords : tuple or None
        (y_start, y_end, x_start, x_end) in output coordinates
    """
    out_ny, out_nx = output_shape

    # Integer part of shift
    dx_int, dy_int = int(round(dx_px)), int(round(dy_px))

    # Compute destination coordinates
    dst_y_start = dy_int
    dst_x_start = dx_int
    dst_y_end = dst_y_start + vol.shape[1]
    dst_x_end = dst_x_start + vol.shape[2]

    # Compute source crop if destination is out of bounds
    src_y_start = max(0, -dst_y_start)
    src_y_end = vol.shape[1] - max(0, dst_y_end - out_ny)
    src_x_start = max(0, -dst_x_start)
    src_x_end = vol.shape[2] - max(0, dst_x_end - out_nx)

    # Clip destination to output bounds
    dst_y_start = max(0, dst_y_start)
    dst_y_end = min(out_ny, dst_y_end)
    dst_x_start = max(0, dst_x_start)
    dst_x_end = min(out_nx, dst_x_end)

    # Check if there's any valid region
    if src_y_end > src_y_start and src_x_end > src_x_start:
        cropped = vol[:, src_y_start:src_y_end, src_x_start:src_x_end]
        return cropped, (dst_y_start, dst_y_end, dst_x_start, dst_x_end)
    else:
        return None, None


def compute_output_shape(slice_files, cumsum_px, first_vol_shape):
    """Compute output volume shape to fit all slices."""
    xmin, xmax, ymin, ymax = [0], [first_vol_shape[2]], [0], [first_vol_shape[1]]

    for slice_id, (dx, dy) in cumsum_px.items():
        # Assuming all slices have similar XY dimensions
        xmin.append(dx)
        xmax.append(dx + first_vol_shape[2])
        ymin.append(dy)
        ymax.append(dy + first_vol_shape[1])

    x0 = min(xmin)
    y0 = min(ymin)
    nx = int(np.ceil(max(xmax) - x0))
    ny = int(np.ceil(max(ymax) - y0))

    return ny, nx, x0, y0


def blend_overlap(fixed_region, moving_region):
    """Blend overlapping Z-region using linear interpolation along Z-axis.

    For Z-stack blending, we want a smooth transition from fixed (bottom of fixed volume)
    to moving (top of moving volume) along the Z direction.

    At tissue boundaries where only one slice has data, we use full intensity
    (no fading) since there is only one contribution.

    Parameters
    ----------
    fixed_region : np.ndarray
        3D array (Z, Y, X) from the existing stack (bottom slice's bottom portion)
    moving_region : np.ndarray
        3D array (Z, Y, X) from the new slice (top portion to blend in)

    Returns
    -------
    np.ndarray
        Blended region with smooth Z transition
    """
    nz = fixed_region.shape[0]

    if nz <= 1:
        # No blending possible with single slice
        # Return whichever has more valid data
        if np.sum(moving_region > 0) >= np.sum(fixed_region > 0):
            return moving_region
        else:
            return fixed_region

    # Create linear weights along Z-axis
    # At z=0 (top of overlap), we want mostly fixed (alpha=0)
    # At z=nz-1 (bottom of overlap), we want mostly moving (alpha=1)
    z_weights = np.linspace(0, 1, nz)

    # Expand weights to match 3D shape (Z, Y, X)
    alphas = z_weights[:, np.newaxis, np.newaxis]
    alphas = np.broadcast_to(alphas, fixed_region.shape).copy()

    # Create tissue masks
    fixed_valid = fixed_region > 0
    moving_valid = moving_region > 0
    both_valid = fixed_valid & moving_valid
    fixed_only = fixed_valid & ~moving_valid
    moving_only = moving_valid & ~fixed_valid

    # Initialize output (zeros where neither has data)
    blended = np.zeros_like(moving_region, dtype=np.float32)

    # Where both have data, apply standard weighted blend
    if np.any(both_valid):
        blended[both_valid] = ((1 - alphas) * fixed_region + alphas * moving_region)[both_valid]

    # Where only one slice has data, use full intensity.
    # No fading needed — there's only one contribution, so we preserve it as-is.
    if np.any(fixed_only):
        blended[fixed_only] = fixed_region[fixed_only]

    if np.any(moving_only):
        blended[moving_only] = moving_region[moving_only]

    return blended


def main():
    p = _build_arg_parser()
    args = p.parse_args()

    slices_dir = Path(args.in_slices_dir)
    output_path = Path(args.out_stack)

    assert_output_exists(output_path, p, args)

    # Find slice files
    slice_files_list = sorted(slices_dir.glob('*.ome.zarr'))
    if not slice_files_list:
        p.error(f"No .ome.zarr files found in {slices_dir}")

    # Extract slice IDs
    pattern = re.compile(r'slice_z(\d+)')
    slice_files = {}
    for f in slice_files_list:
        match = pattern.search(f.name)
        if match:
            slice_id = int(match.group(1))
            slice_files[slice_id] = f

    if not slice_files:
        p.error(f"No files matched slice pattern in {slices_dir}")

    available_ids = sorted(slice_files.keys())
    if args.max_slices:
        available_ids = available_ids[:args.max_slices]
        slice_files = {k: slice_files[k] for k in available_ids}

    logger.info(f"Found {len(slice_files)} slices: {available_ids[0]} to {available_ids[-1]}")

    # Load shifts
    logger.info(f"Loading shifts from {args.in_shifts}")
    cumsum_mm, all_shift_ids = load_shifts(args.in_shifts)

    # Get resolution from first slice
    # NOTE: read_omezarr returns resolution in MILLIMETERS (OME-NGFF standard)
    first_id = available_ids[0]
    first_vol, first_res = read_omezarr(str(slice_files[first_id]), level=0)
    first_vol = np.array(first_vol[:])

    # Resolution in mm (from OME-NGFF metadata)
    res_z_mm = first_res[0] if len(first_res) >= 1 else 0.010  # default 10 µm
    res_y_mm = first_res[1] if len(first_res) >= 2 else first_res[0]
    res_x_mm = first_res[2] if len(first_res) >= 3 else first_res[0]

    logger.info(f"Resolution: Z={res_z_mm*1000:.2f} µm, Y={res_y_mm*1000:.2f} µm, X={res_x_mm*1000:.2f} µm")

    # Handle XY shifts
    if args.no_xy_shift:
        # Slices are already in common space, no XY shifting needed
        logger.info("Skipping XY shifts (--no_xy_shift specified, slices already in common space)")
        cumsum_px = {slice_id: (0.0, 0.0) for slice_id in available_ids}
        out_ny, out_nx = first_vol.shape[1], first_vol.shape[2]
        x0, y0 = 0, 0
    else:
        # Convert shifts (in mm) to pixels: shift_mm / res_mm = pixels
        cumsum_px = {}
        for slice_id in available_ids:
            if slice_id in cumsum_mm:
                dx_mm, dy_mm = cumsum_mm[slice_id]
            else:
                logger.warning(f"No shift for slice {slice_id}, using (0, 0)")
                dx_mm, dy_mm = 0.0, 0.0
            # mm / mm = pixels
            cumsum_px[slice_id] = (dx_mm / res_x_mm, dy_mm / res_y_mm)

        # Center shifts
        middle_id = available_ids[len(available_ids) // 2]
        center_dx, center_dy = cumsum_px[middle_id]
        cumsum_px = {k: (dx - center_dx, dy - center_dy) for k, (dx, dy) in cumsum_px.items()}

        # Compute output XY shape
        out_ny, out_nx, x0, y0 = compute_output_shape(slice_files, cumsum_px, first_vol.shape)

        # Adjust shifts by origin
        cumsum_px = {k: (dx - x0, dy - y0) for k, (dx, dy) in cumsum_px.items()}

    logger.info(f"Output XY shape: {out_ny} x {out_nx}")

    # Load registration transforms if provided
    registration_transforms = {}
    if args.transforms_dir:
        transforms_dir = Path(args.transforms_dir)
        if transforms_dir.exists():
            logger.info(f"Loading registration transforms from {transforms_dir}")
            registration_transforms = load_registration_transforms(
                transforms_dir, available_ids,
                skip_error_status=args.skip_error_transforms,
                skip_warning_status=args.skip_warning_transforms)
            n_loaded = sum(1 for v in registration_transforms.values() if v is not None)
            logger.info(f"Loaded {n_loaded} transforms for refinement")
        else:
            logger.warning(f"Transforms directory not found: {transforms_dir}")

    # Accumulate translations cumulatively if requested
    # Translations are moved from the transforms into cumsum_px so that:
    # 1. The output canvas is sized to accommodate the cumulative shifts
    # 2. Transforms only apply rotation (no content lost at slice edges)
    if args.accumulate_translations and registration_transforms:
        # First pass: extract all pairwise translations
        pairwise_translations = {}
        for slice_id in available_ids[1:]:
            if slice_id in registration_transforms and registration_transforms[slice_id] is not None:
                transform, fixed_z, moving_z = registration_transforms[slice_id]
                params = list(transform.GetParameters())
                tx = params[3] if len(params) > 3 else 0
                ty = params[4] if len(params) > 4 else 0
                pairwise_translations[slice_id] = (tx, ty)

        # Filter unreliable translations before accumulation
        # Translations at the registration boundary are optimizer failures, not real corrections
        if pairwise_translations and args.max_pairwise_translation > 0:
            boundary = args.max_pairwise_translation * 0.95  # 95% of boundary = likely clamped
            n_excluded = 0
            for slice_id in list(pairwise_translations.keys()):
                tx, ty = pairwise_translations[slice_id]
                mag = np.sqrt(tx**2 + ty**2)
                if mag >= boundary:
                    logger.warning(f"Slice {slice_id}: excluding boundary translation "
                                   f"tx={tx:.1f}, ty={ty:.1f} (mag={mag:.1f} >= {boundary:.1f})")
                    pairwise_translations[slice_id] = (0.0, 0.0)
                    n_excluded += 1
            n_total = len(pairwise_translations)
            logger.info(f"Translation filter: excluded {n_excluded}/{n_total} pairs "
                        f"at boundary (>= {boundary:.1f} px)")

        # Second pass: accumulate filtered translations
        cumulative_tx, cumulative_ty = 0.0, 0.0
        n_accumulated = 0
        for slice_id in available_ids[1:]:
            if slice_id in pairwise_translations:
                tx, ty = pairwise_translations[slice_id]
                cumulative_tx += tx
                cumulative_ty += ty
                if tx != 0 or ty != 0:
                    n_accumulated += 1
                logger.debug(f"Slice {slice_id}: pairwise tx={tx:.2f}, ty={ty:.2f} -> "
                             f"cumulative tx={cumulative_tx:.2f}, ty={cumulative_ty:.2f}")
            # Every slice from this point gets the current cumulative correction
            # Sign is negated: SimpleITK tx=+N shifts content LEFT (fetches from x+N),
            # but cumsum_px dx=+N places content RIGHT. To achieve the same effect
            # as the transform, we subtract.
            prev_dx, prev_dy = cumsum_px[slice_id]
            cumsum_px[slice_id] = (prev_dx - cumulative_tx, prev_dy - cumulative_ty)
        logger.info(f"Accumulated translations for {n_accumulated} slices "
                     f"(final cumulative: tx={cumulative_tx:.2f}, ty={cumulative_ty:.2f})")

        # Smooth cumulative translations to reduce per-slice XY jitter
        if args.smooth_window > 0:
            ids_list = sorted(cumsum_px.keys())
            x_vals = np.array([cumsum_px[sid][0] for sid in ids_list])
            y_vals = np.array([cumsum_px[sid][1] for sid in ids_list])

            w = args.smooth_window
            kernel = np.ones(w) / w
            x_smooth = np.convolve(x_vals, kernel, mode='same')
            y_smooth = np.convolve(y_vals, kernel, mode='same')

            # Keep original values at edges where the kernel doesn't fully overlap
            half_w = w // 2
            x_smooth[:half_w] = x_vals[:half_w]
            x_smooth[-half_w:] = x_vals[-half_w:]
            y_smooth[:half_w] = y_vals[:half_w]
            y_smooth[-half_w:] = y_vals[-half_w:]

            max_correction = 0.0
            for j, sid in enumerate(ids_list):
                correction = np.sqrt((x_smooth[j] - x_vals[j])**2 + (y_smooth[j] - y_vals[j])**2)
                max_correction = max(max_correction, correction)
                cumsum_px[sid] = (float(x_smooth[j]), float(y_smooth[j]))

            logger.info(f"Smoothed translations with window={w} "
                        f"(max correction: {max_correction:.1f} px)")

        # Recompute output XY shape to fit the shifted slices
        out_ny, out_nx, x0, y0 = compute_output_shape(slice_files, cumsum_px, first_vol.shape)
        cumsum_px = {k: (dx - x0, dy - y0) for k, (dx, dy) in cumsum_px.items()}
        logger.info(f"Adjusted output XY shape for accumulated translations: {out_ny} x {out_nx}")

    # Smooth per-slice rotations to reduce jitter from isolated correction outliers.
    # Rotations are applied independently per slice, so alternating ±1-2° corrections
    # (or a single large outlier like z27 at -2.1° surrounded by ~0° slices) create
    # visible notching at tissue boundaries throughout the whole volume.
    # This runs regardless of accumulate_translations.
    smoothed_rotations = {}
    if args.smooth_window > 0 and registration_transforms:
        ids_with_tfm = [sid for sid in available_ids
                        if sid in registration_transforms
                        and registration_transforms[sid] is not None]
        if ids_with_tfm:
            angle_ids = sorted(ids_with_tfm)
            raw_angles = []
            for sid in angle_ids:
                tfm_tuple = registration_transforms[sid]
                tfm, _, _ = tfm_tuple
                params = list(tfm.GetParameters())
                a = params[2] if len(params) > 2 else 0.0
                # Clamp before smoothing (same cap as apply_2d_transform)
                if args.max_rotation_deg > 0:
                    max_rad = np.radians(args.max_rotation_deg)
                    a = float(np.clip(a, -max_rad, max_rad))
                raw_angles.append(a)
            raw_angles = np.array(raw_angles)
            w = args.smooth_window
            kernel = np.ones(w) / w
            smooth_angles = np.convolve(raw_angles, kernel, mode='same')
            half_w = w // 2
            smooth_angles[:half_w] = raw_angles[:half_w]
            smooth_angles[-half_w:] = raw_angles[-half_w:]
            max_rot_corr = float(np.max(np.abs(smooth_angles - raw_angles)))
            logger.info(f"Smoothed rotations with window={w} "
                        f"(max correction: {np.degrees(max_rot_corr):.3f}°)")
            for j, sid in enumerate(angle_ids):
                smoothed_rotations[sid] = float(smooth_angles[j])

    # First pass: find Z overlaps (use registration z-offsets if available)
    logger.info("Finding Z-overlaps between consecutive slices...")
    z_matches = []
    total_z = first_vol.shape[0]

    # Cache volume shapes to avoid re-reading during smoothing
    volume_shapes = {first_id: first_vol.shape}

    prev_vol = first_vol
    prev_id = first_id

    for i, slice_id in enumerate(tqdm(available_ids[1:], desc="Z-matching")):
        vol, _ = read_omezarr(str(slice_files[slice_id]), level=0)
        vol = np.array(vol[:])
        volume_shapes[slice_id] = vol.shape  # Cache shape

        # Check if we have registration-derived Z-indices
        fixed_z = None
        moving_z = None
        if slice_id in registration_transforms and registration_transforms[slice_id] is not None:
            _, fixed_z, moving_z = registration_transforms[slice_id]

        if args.use_expected_overlap:
            # Expected overlap from known slicing interval and volume depth
            # overlap = trimmed_volume_depth - slicing_interval_in_voxels
            # This ensures each slice contributes exactly slicing_interval new voxels
            moving_z = moving_z if moving_z is not None else args.moving_z_first_index
            interval_voxels = int(args.slicing_interval_mm / res_z_mm)
            overlap = vol.shape[0] - (moving_z or 0) - interval_voxels
            overlap = max(0, overlap)
            corr = 0.0
            logger.debug(f"Slice {slice_id}: expected overlap={overlap} voxels "
                         f"(vol_depth={vol.shape[0]}, moving_z={moving_z}, interval={interval_voxels})")
        elif fixed_z is not None:
            # We have registration-derived indices
            # fixed_z: Z-index in prev_vol where overlap starts
            # moving_z: Z-index in vol where overlap starts (skipping noisy initial slices)
            # The overlap depth is: prev_vol.shape[0] - fixed_z
            prev_nz = prev_vol.shape[0]
            overlap = max(0, prev_nz - fixed_z)
            corr = 1.0  # Assume good correlation since registration found it
            logger.debug(f"Slice {slice_id}: fixed_z={fixed_z}, moving_z={moving_z}, overlap={overlap} voxels")
        else:
            # find_z_overlap expects resolution in µm for its internal calculation
            res_z_um = res_z_mm * 1000
            overlap, corr = find_z_overlap(
                prev_vol, vol,
                args.slicing_interval_mm, args.search_range_mm, res_z_um
            )
            moving_z = args.moving_z_first_index  # Use default

        z_matches.append({
            'fixed_id': prev_id,
            'moving_id': slice_id,
            'overlap_voxels': overlap,
            'moving_z_start': moving_z,  # Z-index in moving volume where to start
            'correlation': corr
        })

        # Account for moving_z_start when computing total depth
        # We add (vol_depth - moving_z - overlap) new voxels
        moving_z_val = moving_z if moving_z is not None else 0
        contribution = vol.shape[0] - moving_z_val - overlap
        total_z += max(0, contribution)
        prev_vol = vol
        prev_id = slice_id

    # Save Z-matches if requested
    if args.output_z_matches:
        pd.DataFrame(z_matches).to_csv(args.output_z_matches, index=False)
        logger.info(f"Z-matches saved to {args.output_z_matches}")

    # Smooth Z-overlaps: detect and correct outliers
    overlaps = np.array([m['overlap_voxels'] for m in z_matches])
    if len(overlaps) > 3:
        median_overlap = np.median(overlaps)
        # Detect outliers (deviates more than 30% from median)
        # Using 30% to catch more outliers that cause visible jumps
        outlier_threshold = 0.3 * median_overlap
        smoothed = False
        logger.info(f"Z-overlap smoothing: median={median_overlap:.1f}, threshold=±{outlier_threshold:.1f}")
        for i, match in enumerate(z_matches):
            deviation = abs(match['overlap_voxels'] - median_overlap)
            if deviation > outlier_threshold:
                old_overlap = match['overlap_voxels']
                # Replace with local median or global median
                if i > 0 and i < len(z_matches) - 1:
                    local_median = np.median([z_matches[i-1]['overlap_voxels'],
                                              z_matches[i+1]['overlap_voxels'] if i+1 < len(z_matches) else median_overlap])
                    match['overlap_voxels'] = int(local_median)
                else:
                    match['overlap_voxels'] = int(median_overlap)
                logger.warning(f"Slice {match['moving_id']}: corrected outlier overlap {old_overlap} -> {match['overlap_voxels']} voxels (deviation={deviation:.1f})")
                smoothed = True

        # Recompute total_z after smoothing if any corrections were made
        if smoothed:
            # Use cached shapes instead of re-reading volumes
            total_z = volume_shapes[first_id][0]
            for match in z_matches:
                slice_id = match['moving_id']
                moving_z_val = match.get('moving_z_start', 0) or 0
                overlap = match['overlap_voxels']
                vol_nz = volume_shapes[slice_id][0]
                contribution = vol_nz - moving_z_val - overlap
                total_z += max(0, contribution)
            logger.info(f"Recomputed total Z after smoothing: {total_z}")

    # Log Z-match summary
    overlaps = [m['overlap_voxels'] for m in z_matches]
    logger.info(f"Z-overlap: mean={np.mean(overlaps):.1f}, std={np.std(overlaps):.1f} voxels")

    # Second pass: assemble volume
    logger.info(f"Assembling volume: {total_z} x {out_ny} x {out_nx}")
    output_shape = (total_z, out_ny, out_nx)

    output = AnalysisOmeZarrWriter(
        str(output_path), output_shape,
        chunk_shape=(100, 100, 100),
        dtype=np.float32
    )

    # Place first slice
    first_dx, first_dy = cumsum_px[first_id]
    first_vol_f32 = first_vol.astype(np.float32)
    shifted_first, first_coords = apply_xy_shift(first_vol_f32, first_dx, first_dy, (out_ny, out_nx))

    if shifted_first is not None:
        y0, y1, x0, x1 = first_coords
        output[:first_vol.shape[0], y0:y1, x0:x1] = shifted_first
        logger.info(f"  First slice: shift=({first_dx:.1f}, {first_dy:.1f}) px, xy=[{y0}:{y1}, {x0}:{x1}]")

    z_cursor = first_vol.shape[0]

    # Stack remaining slices
    for i, match in enumerate(tqdm(z_matches, desc="Stacking")):
        slice_id = match['moving_id']
        overlap = match['overlap_voxels']
        moving_z_start = match.get('moving_z_start', 0) or 0

        vol, _ = read_omezarr(str(slice_files[slice_id]), level=0)
        vol = np.array(vol[:]).astype(np.float32)

        # Skip initial noisy z-slices in moving volume
        if moving_z_start > 0:
            vol = vol[moving_z_start:]
            logger.debug(f"Slice {slice_id}: skipped first {moving_z_start} z-slices")

        # Apply registration transform (rotation/small translation refinement) if available
        if slice_id in registration_transforms and registration_transforms[slice_id] is not None:
            transform, _, _ = registration_transforms[slice_id]
            # When accumulating translations, they're handled via cumsum_px;
            # only rotation is applied from the transform
            use_rotation_only = args.rotation_only or args.accumulate_translations
            override_rot = smoothed_rotations.get(slice_id)  # None if no smoothing
            vol = apply_transform_to_volume(vol, transform,
                                           rotation_only=use_rotation_only,
                                           max_rotation_deg=args.max_rotation_deg,
                                           override_rotation=override_rot)
            if use_rotation_only:
                logger.debug(f"Applied rotation-only transform to slice {slice_id} (max_rot={args.max_rotation_deg}°)")
            else:
                logger.debug(f"Applied registration transform to slice {slice_id}")

        # Apply XY shift (from motor positions)
        dx, dy = cumsum_px[slice_id]
        shifted, dst_coords = apply_xy_shift(vol, dx, dy, (out_ny, out_nx))

        if shifted is None:
            logger.warning(f"Slice {slice_id} is outside output bounds, skipping")
            continue

        dst_y0, dst_y1, dst_x0, dst_x1 = dst_coords

        # Determine Z range for this slice
        z_start = z_cursor - overlap
        z_end = z_start + shifted.shape[0]

        # Ensure we don't exceed output bounds
        if z_end > output_shape[0]:
            z_end = output_shape[0]
            shifted = shifted[:z_end - z_start]

        if args.blend and overlap > 0 and z_start < z_cursor:
            # Blend overlap region
            overlap_z_start = z_start
            overlap_z_end = min(z_cursor, z_end)
            overlap_depth = overlap_z_end - overlap_z_start

            if overlap_depth > 0:
                # Get overlap regions from output and shifted
                existing = np.array(output[overlap_z_start:overlap_z_end, dst_y0:dst_y1, dst_x0:dst_x1])
                moving_overlap = shifted[:overlap_depth]

                # Intensity matching: adjust moving slice to match existing in overlap
                # This reduces visible bands at slice transitions
                existing_valid = existing > 0
                moving_valid = moving_overlap > 0
                both_valid = existing_valid & moving_valid

                if np.sum(both_valid) > 1000:  # Need enough pixels for reliable statistics
                    existing_median = np.median(existing[both_valid])
                    moving_median = np.median(moving_overlap[both_valid])

                    if moving_median > 1e-6 and existing_median > 1e-6:
                        scale = existing_median / moving_median
                        # Clamp scale to prevent extreme corrections
                        scale = np.clip(scale, 0.5, 2.0)
                        if abs(scale - 1.0) > 0.01:
                            # Apply scaling to the entire shifted volume, not just overlap
                            shifted = shifted * scale
                            moving_overlap = shifted[:overlap_depth]
                            logger.debug(f"Slice {slice_id}: intensity scale={scale:.3f}")

                # Blend
                blended = blend_overlap(existing, moving_overlap)
                output[overlap_z_start:overlap_z_end, dst_y0:dst_y1, dst_x0:dst_x1] = blended

                # Add non-overlapping part
                if z_end > z_cursor:
                    output[z_cursor:z_end, dst_y0:dst_y1, dst_x0:dst_x1] = shifted[overlap_depth:]
        else:
            # No blending - just write to specific region
            output[z_start:z_end, dst_y0:dst_y1, dst_x0:dst_x1] = shifted

        z_cursor = z_end

        logger.debug(f"  Slice {slice_id}: z=[{z_start}:{z_end}], xy=[{dst_y0}:{dst_y1}, {dst_x0}:{dst_x1}]")

    # Finalize with pyramid
    logger.info("Generating pyramid levels...")
    output.finalize(
        first_res,
        target_resolutions_um=args.pyramid_resolutions,
        make_isotropic=args.make_isotropic
    )

    # Collect metrics
    z_offsets = np.array([m['overlap_voxels'] for m in z_matches])
    collect_stack_metrics(
        output_shape=output_shape,
        z_offsets=z_offsets,
        num_slices=len(available_ids),
        resolution=list(first_res),
        output_path=str(output_path),
        blend_enabled=args.blend,
        normalize_enabled=False
    )

    logger.info(f"Done! Output saved to {output_path}")


if __name__ == '__main__':
    main()
