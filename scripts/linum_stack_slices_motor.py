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
from scipy.ndimage import correlate
from tqdm import tqdm

from linumpy.io.zarr import read_omezarr, AnalysisOmeZarrWriter
from linumpy.utils.mosaic_grid import getDiffusionBlendingWeights
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

    # Z-matching parameters
    p.add_argument('--slicing_interval_mm', type=float, default=0.200,
                   help='Physical slice thickness in mm [%(default)s]')
    p.add_argument('--search_range_mm', type=float, default=0.100,
                   help='Search range for Z-matching in mm [%(default)s]')
    p.add_argument('--use_expected_overlap', action='store_true',
                   help='Use expected overlap from slicing_interval instead of correlation')

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


def load_registration_transforms(transforms_dir, slice_ids):
    """
    Load pairwise registration transforms from directory.

    Parameters
    ----------
    transforms_dir : Path
        Directory containing registration outputs (subdirs per slice)
    slice_ids : list
        List of slice IDs to load transforms for

    Returns
    -------
    dict
        Mapping from slice_id to (transform, z_offset) tuple
    """
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
            tfm = sitk.ReadTransform(str(tfm_files[0]))

            # Load z-offsets if available
            # offsets.txt contains [fixed_z, moving_z]
            # The overlap is: fixed_z - moving_z (how many voxels from fixed match moving)
            z_overlap = None
            if offset_files:
                offsets = np.loadtxt(str(offset_files[0]))
                if len(offsets) >= 2:
                    fixed_z = int(offsets[0])
                    moving_z = int(offsets[1])
                    # Overlap is the number of voxels that overlap between slices
                    z_overlap = fixed_z - moving_z
                    logger.debug(f"Slice {slice_id}: fixed_z={fixed_z}, moving_z={moving_z}, overlap={z_overlap}")

            transforms[slice_id] = (tfm, z_overlap)
            logger.debug(f"Loaded transform for slice {slice_id}")

        except Exception as e:
            logger.warning(f"Could not load transform for slice {slice_id}: {e}")
            transforms[slice_id] = None

    return transforms


def apply_2d_transform(image_2d, transform):
    """
    Apply a SimpleITK 2D/3D transform to a 2D image.

    Parameters
    ----------
    image_2d : np.ndarray
        2D image to transform
    transform : sitk.Transform
        SimpleITK transform (extracts 2D rotation/translation)

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

            center = transform.GetCenter()
            center_2d = [center[0], center[1]]

            tfm_2d = sitk.Euler2DTransform()
            tfm_2d.SetCenter(center_2d)
            tfm_2d.SetAngle(angle)
            tfm_2d.SetTranslation([tx, ty])
        else:
            # Try to use as-is or create identity
            tfm_2d = sitk.Euler2DTransform()
    else:
        tfm_2d = transform

    # Apply transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_img)
    resampler.SetTransform(tfm_2d)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)

    result = resampler.Execute(sitk_img)
    return sitk.GetArrayFromImage(result)


def apply_transform_to_volume(vol, transform):
    """
    Apply 2D transform to each Z-slice of a volume.

    Parameters
    ----------
    vol : np.ndarray
        3D volume (Z, Y, X)
    transform : sitk.Transform
        Transform to apply to each slice

    Returns
    -------
    np.ndarray
        Transformed volume
    """
    result = np.zeros_like(vol)
    for z in range(vol.shape[0]):
        result[z] = apply_2d_transform(vol[z], transform)
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
    expected_overlap_vox = int((slicing_interval_mm * 1000) / resolution_um)
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
    """Blend overlapping Z-region using diffusion weights."""
    # Create tissue mask for blending
    fixed_mask = fixed_region > np.percentile(fixed_region, 10)
    moving_mask = moving_region > np.percentile(moving_region, 10)

    # Get blending weights
    try:
        alphas = getDiffusionBlendingWeights(fixed_mask, moving_mask, factor=2)
    except Exception:
        # Fallback to linear blending
        nz = fixed_region.shape[0]
        alphas = np.linspace(0, 1, nz)[:, None, None]

    blended = (1 - alphas) * fixed_region + alphas * moving_region
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
            registration_transforms = load_registration_transforms(transforms_dir, available_ids)
            n_loaded = sum(1 for v in registration_transforms.values() if v is not None)
            logger.info(f"Loaded {n_loaded} transforms for refinement")
        else:
            logger.warning(f"Transforms directory not found: {transforms_dir}")

    # First pass: find Z overlaps (use registration z-offsets if available)
    logger.info("Finding Z-overlaps between consecutive slices...")
    z_matches = []
    total_z = first_vol.shape[0]

    prev_vol = first_vol
    prev_id = first_id

    for i, slice_id in enumerate(tqdm(available_ids[1:], desc="Z-matching")):
        vol, _ = read_omezarr(str(slice_files[slice_id]), level=0)
        vol = np.array(vol[:])

        # Check if we have a registration-derived Z-overlap
        reg_overlap = None
        if slice_id in registration_transforms and registration_transforms[slice_id] is not None:
            _, reg_overlap = registration_transforms[slice_id]

        if reg_overlap is not None and reg_overlap > 0:
            # Use registration-derived overlap (fixed_z - moving_z)
            overlap = reg_overlap
            corr = 1.0  # Assume good correlation since registration found it
            logger.debug(f"Slice {slice_id}: using registration overlap={overlap} voxels")
        elif args.use_expected_overlap:
            # slicing_interval_mm / res_z_mm = overlap in voxels
            overlap = int(args.slicing_interval_mm / res_z_mm)
            corr = 0.0
        else:
            # find_z_overlap expects resolution in µm for its internal calculation
            res_z_um = res_z_mm * 1000
            overlap, corr = find_z_overlap(
                prev_vol, vol,
                args.slicing_interval_mm, args.search_range_mm, res_z_um
            )

        z_matches.append({
            'fixed_id': prev_id,
            'moving_id': slice_id,
            'overlap_voxels': overlap,
            'correlation': corr
        })

        total_z += vol.shape[0] - overlap
        prev_vol = vol
        prev_id = slice_id

    # Save Z-matches if requested
    if args.output_z_matches:
        pd.DataFrame(z_matches).to_csv(args.output_z_matches, index=False)
        logger.info(f"Z-matches saved to {args.output_z_matches}")

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

        vol, _ = read_omezarr(str(slice_files[slice_id]), level=0)
        vol = np.array(vol[:]).astype(np.float32)

        # Apply registration transform (rotation/small translation refinement) if available
        if slice_id in registration_transforms and registration_transforms[slice_id] is not None:
            transform, _ = registration_transforms[slice_id]
            vol = apply_transform_to_volume(vol, transform)
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
