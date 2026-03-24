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
from linumpy.stitching.stacking import (
    find_z_overlap, enforce_z_consistency, apply_2d_transform,
    apply_transform_to_volume, apply_xy_shift, blend_overlap_z,
    refine_z_blend_overlap
)
from linumpy.utils.io import add_overwrite_arg, assert_output_exists
from linumpy.utils.metrics import collect_stack_metrics
from linumpy.shifts.utils import load_shifts_csv

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
    p.add_argument('--z_overlap_min_corr', type=float, default=0.5,
                   help='When using correlation-based Z-overlap (not --use_expected_overlap),\n'
                        'fall back to expected overlap if the best correlation is below this\n'
                        'threshold. Prevents failed tissue contact from causing wrong\n'
                        'Z-positioning. 0 = always trust correlation result. [%(default)s]')
    p.add_argument('--moving_z_first_index', type=int, default=8,
                   help='Starting Z-index in moving volume to skip noisy data [%(default)s]')

    # Blending
    p.add_argument('--blend', action='store_true',
                   help='Blend overlapping regions using a cosine (Hann) ramp')
    p.add_argument('--blend_depth', type=int, default=None,
                   help='Number of z-slices to blend (default: auto from overlap)')
    p.add_argument('--blend_refinement_px', type=float, default=0,
                   help='Enable Z-blend refinement: phase-correlation-based XY shift\n'
                        'correction applied in the overlap zone before blending, analogous\n'
                        'to stitch_3d_with_refinement for tiles. Set to the maximum\n'
                        'allowed shift in pixels (e.g. 10). 0 disables. [%(default)s]')
    p.add_argument('--blend_z_refine_vox', type=int, default=0,
                   help='Z-blend position search: scan N voxels below the expected overlap\n'
                        'boundary (when --use_expected_overlap) for the best-correlated tissue\n'
                        'plane and set the blend there. Z-spacing stays fixed at slicing_interval;\n'
                        'only the blend zone moves. Useful when tissue overlap is smaller than\n'
                        'the imaging depth implies (e.g. deeper cuts). 0 = disabled. [%(default)s]')

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

    p.add_argument('--confidence_high', type=float, default=0.6,
                   help='Registration confidence above which the full transform is applied.\n'
                        'Between confidence_low and confidence_high, rotation-only is forced\n'
                        'regardless of --rotation_only. Based on registration_confidence in\n'
                        'pairwise_registration_metrics.json. [%(default)s]')
    p.add_argument('--confidence_low', type=float, default=0.3,
                   help='Registration confidence below which the transform is skipped entirely.\n'
                        'Prevents bad registrations from introducing XY drift. [%(default)s]')

    add_overwrite_arg(p)
    return p


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
            # Read registration quality metrics (always, to extract confidence score)
            confidence = 1.0
            metrics_files = list(transform_dir.glob("pairwise_registration_metrics.json"))
            if metrics_files:
                with open(metrics_files[0]) as f:
                    metrics_data = json.load(f)
                status = metrics_data.get("overall_status", "ok")
                try:
                    confidence = float(
                        metrics_data["metrics"]["registration_confidence"]["value"]
                    )
                except (KeyError, TypeError, ValueError):
                    confidence = 1.0  # fallback for older JSONs without confidence score

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

            transforms[slice_id] = (tfm, fixed_z, moving_z, confidence)
            logger.debug(f"Loaded transform for slice {slice_id} (confidence={confidence:.2f})")

        except Exception as e:
            logger.warning(f"Could not load transform for slice {slice_id}: {e}")
            transforms[slice_id] = None

    return transforms


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
    cumsum_mm, all_shift_ids = load_shifts_csv(args.in_shifts)

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
                transform, fixed_z, moving_z, _ = registration_transforms[slice_id]
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
                tfm, _, _, _ = tfm_tuple
                params = list(tfm.GetParameters())
                a = params[2] if len(params) > 2 else 0.0
                # Clamp before smoothing (same cap as apply_2d_transform)
                if args.max_rotation_deg > 0:
                    max_rad = np.radians(args.max_rotation_deg)
                    a = float(np.clip(a, -max_rad, max_rad))
                raw_angles.append(a)
            raw_angles = np.array(raw_angles)
            # Clamp window to data length: np.convolve mode='same' returns
            # max(M, N) elements, so a kernel larger than the data produces
            # smooth_angles longer than raw_angles and the subtraction fails.
            w = min(args.smooth_window, len(raw_angles))
            if w < 2:
                smooth_angles = raw_angles.copy()
            else:
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
            _, fixed_z, moving_z, _ = registration_transforms[slice_id]

        if args.use_expected_overlap:
            # Expected overlap from known slicing interval and volume depth.
            # ALWAYS use the physical default moving_z (moving_z_first_index),
            # NOT the registration-derived value.  Registration-derived moving_z
            # can vary between slices and cause inconsistent Z-spacing even when
            # the user has explicitly requested physics-based expected overlap.
            moving_z = args.moving_z_first_index
            interval_voxels = int(args.slicing_interval_mm / res_z_mm)
            overlap = vol.shape[0] - (moving_z or 0) - interval_voxels
            overlap = max(0, overlap)
            corr = 0.0
            logger.debug(f"Slice {slice_id}: expected overlap={overlap} voxels "
                         f"(vol_depth={vol.shape[0]}, moving_z={moving_z} [fixed], interval={interval_voxels})")
            # Optionally search below expected_overlap for the best-correlated tissue
            # boundary to blend at, while keeping z-spacing fixed at slicing_interval.
            # This handles cases where the actual tissue overlap is smaller than the
            # imaging depth implies (i.e. the cut removed more tissue than expected).
            # Skip refinement for low-confidence slices — spurious correlation matches
            # at degraded tissue boundaries cause Z-jumps.
            blend_overlap = overlap
            slice_confidence = None
            if slice_id in registration_transforms and registration_transforms[slice_id] is not None:
                slice_confidence = registration_transforms[slice_id][3]
            refine_ok = (slice_confidence is None or slice_confidence >= args.confidence_low)
            if args.blend_z_refine_vox > 0 and overlap > 0 and refine_ok:
                search_vox = args.blend_z_refine_vox
                min_ov = max(1, overlap - search_vox)
                max_ov = overlap  # cap at expected to preserve slicing_interval z-spacing
                crop_z = moving_z or 0
                h, w = prev_vol.shape[1], prev_vol.shape[2]
                margin = min(h, w) // 4
                y_sl = slice(margin, h - margin)
                x_sl = slice(margin, w - margin)
                best_ref_corr = -np.inf
                for ov in range(min_ov, max_ov + 1):
                    f_reg = prev_vol[-ov:, y_sl, x_sl]
                    m_reg = vol[crop_z: crop_z + ov, y_sl, x_sl]
                    if m_reg.shape[0] < ov:
                        break
                    f_n = (f_reg - f_reg.mean()) / (f_reg.std() + 1e-8)
                    m_n = (m_reg - m_reg.mean()) / (m_reg.std() + 1e-8)
                    c = float(np.mean(f_n * m_n))
                    if c > best_ref_corr:
                        best_ref_corr = c
                        blend_overlap = ov
                logger.debug(f"Slice {slice_id}: blend_z_refine: expected_overlap={overlap}, "
                             f"blend_overlap={blend_overlap} (corr={best_ref_corr:.3f})")
            elif not refine_ok:
                logger.info(f"Slice {slice_id}: skipping blend_z_refine (confidence "
                            f"{slice_confidence:.3f} < {args.confidence_low})")
        elif fixed_z is not None:
            # We have registration-derived indices
            # fixed_z: Z-index in prev_vol where overlap starts
            # moving_z: Z-index in vol where overlap starts (skipping noisy initial slices)
            # The overlap depth is: prev_vol.shape[0] - fixed_z
            prev_nz = prev_vol.shape[0]
            overlap = max(0, prev_nz - fixed_z)
            blend_overlap = overlap
            corr = 1.0  # Assume good correlation since registration found it
            logger.debug(f"Slice {slice_id}: fixed_z={fixed_z}, moving_z={moving_z}, overlap={overlap} voxels")
        else:
            # find_z_overlap expects resolution in µm for its internal calculation
            res_z_um = res_z_mm * 1000
            overlap, corr = find_z_overlap(
                prev_vol, vol,
                args.slicing_interval_mm, args.search_range_mm, res_z_um
            )
            # Fall back to expected overlap when correlation is too low to trust
            if args.z_overlap_min_corr > 0 and corr < args.z_overlap_min_corr:
                interval_voxels = int(args.slicing_interval_mm / res_z_mm)
                crop_z = args.moving_z_first_index or 0
                fallback_overlap = max(0, vol.shape[0] - crop_z - interval_voxels)
                logger.warning(
                    f"Slice {slice_id}: Z-overlap correlation {corr:.3f} < "
                    f"z_overlap_min_corr={args.z_overlap_min_corr:.2f}, "
                    f"falling back to expected overlap {fallback_overlap} (was: {overlap})"
                )
                overlap = fallback_overlap
                corr = 0.0
            blend_overlap = overlap
            moving_z = args.moving_z_first_index  # Use default

        z_matches.append({
            'fixed_id': prev_id,
            'moving_id': slice_id,
            'overlap_voxels': overlap,
            'blend_overlap_voxels': blend_overlap,
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

    # Enforce Z-consistency: replace outlier overlaps using neighbor interpolation.
    # High-confidence registrations (confidence >= confidence_high) are protected.
    confidence_per_slice = {
        sid: tfm_tuple[3]
        for sid, tfm_tuple in registration_transforms.items()
        if tfm_tuple is not None
    }
    overlaps_before = [m['overlap_voxels'] for m in z_matches]
    logger.info(
        f"Z-overlap consistency check: median={np.median(overlaps_before):.1f}, "
        f"std={np.std(overlaps_before):.1f} voxels"
    )
    z_matches, z_corrections = enforce_z_consistency(
        z_matches,
        confidence_per_slice=confidence_per_slice,
        outlier_threshold_frac=0.30,
        confidence_protect_threshold=args.confidence_high,
    )
    if z_corrections:
        for c in z_corrections:
            logger.warning(
                f"Slice {c['moving_id']}: corrected outlier {c['field']} "
                f"{c['old_value']} -> {c['new_value']}"
            )
        # Recompute total_z after corrections
        total_z = volume_shapes[first_id][0]
        for match in z_matches:
            sid = match['moving_id']
            mz = match.get('moving_z_start', 0) or 0
            ov = match['overlap_voxels']
            vol_nz = volume_shapes[sid][0]
            total_z += max(0, vol_nz - mz - ov)
        logger.info(f"Recomputed total Z after consistency enforcement: {total_z}")

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
        # blend_overlap may be < overlap when z-blend refinement found a tighter tissue match
        blend_overlap = min(match.get('blend_overlap_voxels', overlap), overlap)
        moving_z_start = match.get('moving_z_start', 0) or 0

        vol, _ = read_omezarr(str(slice_files[slice_id]), level=0)
        vol = np.array(vol[:]).astype(np.float32)

        # Skip initial noisy z-slices in moving volume
        if moving_z_start > 0:
            vol = vol[moving_z_start:]
            logger.debug(f"Slice {slice_id}: skipped first {moving_z_start} z-slices")

        # Apply registration transform (rotation/small translation refinement) if available
        if slice_id in registration_transforms and registration_transforms[slice_id] is not None:
            transform, _, _, confidence = registration_transforms[slice_id]
            # Adaptive degradation: skip, force rotation-only, or apply full transform
            # based on the per-registration confidence score.
            if args.confidence_low is not None and confidence < args.confidence_low:
                logger.warning(f"Slice {slice_id}: skipping transform "
                               f"(confidence={confidence:.2f} < confidence_low={args.confidence_low:.2f})")
            else:
                if args.confidence_high is not None and confidence < args.confidence_high:
                    use_rotation_only = True
                    logger.debug(f"Slice {slice_id}: forcing rotation-only "
                                 f"(confidence={confidence:.2f} < confidence_high={args.confidence_high:.2f})")
                else:
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

        if args.blend and blend_overlap > 0 and z_start < z_cursor:
            # Blend the region [z_cursor - blend_overlap, z_cursor].
            # When blend_overlap == overlap this is the standard behaviour.
            # When blend_overlap < overlap (z-blend refinement found a tighter tissue
            # match), the leading part of the overlap [z_start, z_cursor - blend_overlap]
            # retains the existing fixed-volume data rather than blending non-matching tissue.
            s_blend_start = overlap - blend_overlap  # index into shifted where blend starts
            overlap_z_start = z_cursor - blend_overlap
            overlap_z_end = z_cursor
            overlap_depth = blend_overlap

            if overlap_depth > 0:
                # Get overlap regions from output and shifted
                existing = np.array(output[overlap_z_start:overlap_z_end, dst_y0:dst_y1, dst_x0:dst_x1])
                moving_overlap = shifted[s_blend_start:s_blend_start + overlap_depth]

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
                            moving_overlap = shifted[s_blend_start:s_blend_start + overlap_depth]
                            logger.debug(f"Slice {slice_id}: intensity scale={scale:.3f}")

                # Z-blend refinement: correct residual XY misalignment in the overlap zone
                if args.blend_refinement_px > 0:
                    moving_overlap, ref_mag = refine_z_blend_overlap(
                        existing, moving_overlap, args.blend_refinement_px
                    )
                    if ref_mag > 0:
                        logger.debug(f"Slice {slice_id}: z-blend XY refinement {ref_mag:.2f} px")

                # Blend
                blended = blend_overlap_z(existing, moving_overlap)
                output[overlap_z_start:overlap_z_end, dst_y0:dst_y1, dst_x0:dst_x1] = blended

                # New contribution (always shifted[overlap:] to preserve z-spacing)
                if z_end > z_cursor:
                    output[z_cursor:z_end, dst_y0:dst_y1, dst_x0:dst_x1] = shifted[overlap:]
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
