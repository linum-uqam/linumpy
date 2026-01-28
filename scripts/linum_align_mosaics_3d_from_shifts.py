#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Using xy shifts file, bring all mosaics in `in_mosaics_dir` to a common space. Each
volume is resampled to a common shape and its content is translated following the
transforms in xy shifts. All transformed mosaics are saved to `out_directory`.

Optionally accepts a slice configuration file to filter which slices to process.
When slices are skipped, their shifts are accumulated to maintain proper alignment.
"""
# Configure thread limits before numpy/scipy imports
import linumpy._thread_config  # noqa: F401

import argparse
import csv
import re
from os.path import join as pjoin
from os.path import split as psplit
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd

from linumpy.io.zarr import read_omezarr, save_omezarr
from linumpy.utils.io import add_overwrite_arg, assert_output_exists
from linumpy.utils_images import apply_xy_shift


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_mosaics_dir',
                   help='Directory containing mosaics to bring to common space.')
    p.add_argument('in_shifts',
                   help='Spreadsheet containing xy shifts (.csv).')
    p.add_argument('out_directory',
                   help='Output directory containing the aligned mosaics.')
    p.add_argument('--slice_config', default=None,
                   help='Optional slice configuration file (.csv) to filter slices.\n'
                        'Expected columns: slice_id, use (true/false), notes (optional)')

    # Outlier filtering options
    p.add_argument('--filter_outliers', action='store_true',
                   help='Detect and filter outlier shifts that cause excessive drift.')
    p.add_argument('--max_shift_mm', type=float, default=0.5,
                   help='Maximum allowed pairwise shift in mm. Larger shifts are clamped. [%(default)s]')
    p.add_argument('--outlier_method', choices=['clamp', 'median', 'zero', 'local', 'iqr'], default='iqr',
                   help='How to handle outlier shifts:\n'
                        '  clamp: Limit to max_shift_mm\n'
                        '  median: Replace with global median of non-outliers\n'
                        '  zero: Replace with zero\n'
                        '  local: Replace with local median of neighbors\n'
                        '  iqr: Auto-detect outliers using IQR and replace with local median [%(default)s]')
    p.add_argument('--iqr_multiplier', type=float, default=1.5,
                   help='IQR multiplier for outlier detection (only with --outlier_method iqr). [%(default)s]')

    # Drift centering
    p.add_argument('--no_center_drift', action='store_true',
                   help='Do not center drift around middle slice.\n'
                        'By default, drift is centered to prevent slices from moving out of volume.')

    add_overwrite_arg(p)
    return p


def load_slice_config(config_path):
    """Load slice configuration and return set of slice IDs to use."""
    slices_to_use = set()
    with open(config_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            slice_id = int(row['slice_id'])
            use = row['use'].lower().strip() in ('true', '1', 'yes')
            if use:
                slices_to_use.add(slice_id)
    return slices_to_use


def filter_outlier_shifts(shifts_df, max_shift_mm=0.5, method='median', iqr_multiplier=1.5):
    """
    Detect and filter outlier shifts that cause excessive drift.

    Parameters
    ----------
    shifts_df : pd.DataFrame
        DataFrame with columns: fixed_id, moving_id, x_shift_mm, y_shift_mm
    max_shift_mm : float
        Maximum allowed pairwise shift in mm (used when method != 'iqr')
    method : str
        How to handle outliers: 'clamp', 'median', 'zero', 'local', or 'iqr'
        - clamp: Limit to max_shift_mm while preserving direction
        - median: Replace with global median of non-outliers
        - zero: Replace with zero shift
        - local: Replace with local median (average of neighbors)
        - iqr: Use IQR-based outlier detection and replace with local median
    iqr_multiplier : float
        Multiplier for IQR-based outlier detection (default: 1.5)

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with outlier shifts corrected
    """
    df = shifts_df.copy()

    # Calculate magnitude of each shift
    shift_mag = np.sqrt(df['x_shift_mm']**2 + df['y_shift_mm']**2)

    # Find outliers based on method
    if method == 'iqr':
        # IQR-based outlier detection
        q1 = shift_mag.quantile(0.25)
        q3 = shift_mag.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + iqr_multiplier * iqr
        outlier_mask = shift_mag > upper_bound
        print(f"IQR-based detection: Q1={q1:.3f}, Q3={q3:.3f}, IQR={iqr:.3f}, threshold={upper_bound:.3f} mm")
    else:
        outlier_mask = shift_mag > max_shift_mm

    n_outliers = outlier_mask.sum()

    if n_outliers == 0:
        threshold_str = f"IQR threshold" if method == 'iqr' else f"{max_shift_mm} mm"
        print(f"No outlier shifts detected (threshold: {threshold_str})")
        return df

    print(f"Detected {n_outliers} outlier shifts:")
    for idx in df[outlier_mask].index:
        row = df.loc[idx]
        mag = shift_mag[idx]
        print(f"  {int(row['fixed_id'])}->{int(row['moving_id'])}: "
              f"({row['x_shift_mm']:.3f}, {row['y_shift_mm']:.3f}) mm, magnitude={mag:.3f} mm")

    if method == 'clamp':
        # Clamp to maximum allowed magnitude while preserving direction
        for idx in df[outlier_mask].index:
            scale = max_shift_mm / shift_mag[idx]
            df.loc[idx, 'x_shift_mm'] *= scale
            df.loc[idx, 'y_shift_mm'] *= scale
            if 'x_shift' in df.columns:
                df.loc[idx, 'x_shift'] *= scale
                df.loc[idx, 'y_shift'] *= scale
        print(f"  -> Clamped to {max_shift_mm} mm")

    elif method == 'median':
        # Replace with global median of all non-outlier shifts
        non_outlier = df[~outlier_mask]
        median_x = non_outlier['x_shift_mm'].median()
        median_y = non_outlier['y_shift_mm'].median()

        for idx in df[outlier_mask].index:
            df.loc[idx, 'x_shift_mm'] = median_x
            df.loc[idx, 'y_shift_mm'] = median_y
            if 'x_shift' in df.columns:
                df.loc[idx, 'x_shift'] = non_outlier['x_shift'].median()
                df.loc[idx, 'y_shift'] = non_outlier['y_shift'].median()
        print(f"  -> Replaced with global median: ({median_x:.3f}, {median_y:.3f}) mm")

    elif method == 'zero':
        # Replace with zero
        for idx in df[outlier_mask].index:
            df.loc[idx, 'x_shift_mm'] = 0.0
            df.loc[idx, 'y_shift_mm'] = 0.0
            if 'x_shift' in df.columns:
                df.loc[idx, 'x_shift'] = 0.0
                df.loc[idx, 'y_shift'] = 0.0
        print(f"  -> Replaced with zero")

    elif method in ['local', 'iqr']:
        # Replace with local median (average of valid neighbors)
        # This preserves local trends while removing spikes
        for idx in df[outlier_mask].index:
            # Find neighbors (up to 2 on each side)
            pos = df.index.get_loc(idx)
            neighbor_vals_x = []
            neighbor_vals_y = []
            neighbor_vals_px_x = []
            neighbor_vals_px_y = []

            for offset in [-2, -1, 1, 2]:
                neighbor_pos = pos + offset
                if 0 <= neighbor_pos < len(df):
                    neighbor_idx = df.index[neighbor_pos]
                    if not outlier_mask[neighbor_idx]:  # Only use non-outlier neighbors
                        neighbor_vals_x.append(df.loc[neighbor_idx, 'x_shift_mm'])
                        neighbor_vals_y.append(df.loc[neighbor_idx, 'y_shift_mm'])
                        if 'x_shift' in df.columns:
                            neighbor_vals_px_x.append(df.loc[neighbor_idx, 'x_shift'])
                            neighbor_vals_px_y.append(df.loc[neighbor_idx, 'y_shift'])

            if neighbor_vals_x:
                local_x = np.median(neighbor_vals_x)
                local_y = np.median(neighbor_vals_y)
                old_x = df.loc[idx, 'x_shift_mm']
                old_y = df.loc[idx, 'y_shift_mm']
                df.loc[idx, 'x_shift_mm'] = local_x
                df.loc[idx, 'y_shift_mm'] = local_y
                if 'x_shift' in df.columns and neighbor_vals_px_x:
                    df.loc[idx, 'x_shift'] = np.median(neighbor_vals_px_x)
                    df.loc[idx, 'y_shift'] = np.median(neighbor_vals_px_y)
                row = shifts_df.loc[idx]
                print(f"  {int(row['fixed_id'])}->{int(row['moving_id'])}: "
                      f"({old_x:.3f}, {old_y:.3f}) -> ({local_x:.3f}, {local_y:.3f}) mm (local median)")
            else:
                # No valid neighbors, use global median
                non_outlier = df[~outlier_mask]
                df.loc[idx, 'x_shift_mm'] = non_outlier['x_shift_mm'].median()
                df.loc[idx, 'y_shift_mm'] = non_outlier['y_shift_mm'].median()
                if 'x_shift' in df.columns:
                    df.loc[idx, 'x_shift'] = non_outlier['x_shift'].median()
                    df.loc[idx, 'y_shift'] = non_outlier['y_shift'].median()
                print(f"  -> (no valid neighbors, used global median)")

    # Report new cumulative drift
    new_cumsum_x = df['x_shift_mm'].cumsum()
    new_cumsum_y = df['y_shift_mm'].cumsum()
    print(f"New total drift: ({new_cumsum_x.iloc[-1]:.3f}, {new_cumsum_y.iloc[-1]:.3f}) mm")

    return df


def build_cumulative_shifts(shifts_df, selected_slice_ids, resolution, center_drift=True):
    """
    Build cumulative shifts for selected slices, properly handling skipped slices.
    
    The shifts file contains pairwise shifts between consecutive slices in the 
    original dataset. When slices are skipped, we need to accumulate the 
    intermediate shifts.
    
    Parameters
    ----------
    shifts_df : pd.DataFrame
        DataFrame with columns: fixed_id, moving_id, x_shift, y_shift, x_shift_mm, y_shift_mm
        The x_shift/y_shift are in pixels, x_shift_mm/y_shift_mm are in mm
    selected_slice_ids : list
        List of slice IDs that will be processed (sorted)
    resolution : tuple
        Resolution (res_z, res_y, res_x) in microns/pixel from the OME-Zarr metadata.
        Used to convert mm shifts to pixels at the current resolution.
    center_drift : bool
        If True, center the cumulative drift around the middle slice to prevent
        slices from drifting out of the volume.

    Returns
    -------
    dict
        Mapping from slice_id to (cumulative_dx, cumulative_dy) in pixels
    """
    # Build a mapping from (fixed_id, moving_id) to (dx_mm, dy_mm)
    # We use mm values for consistency, then convert to pixels using the resolution
    shift_lookup = {}
    for _, row in shifts_df.iterrows():
        fixed_id = int(row['fixed_id'])
        moving_id = int(row['moving_id'])
        shift_lookup[(fixed_id, moving_id)] = (row['x_shift_mm'], row['y_shift_mm'])

    # Get all slice IDs mentioned in the shifts file
    all_slice_ids = set()
    for _, row in shifts_df.iterrows():
        all_slice_ids.add(int(row['fixed_id']))
        all_slice_ids.add(int(row['moving_id']))
    all_slice_ids = sorted(all_slice_ids)

    # Build cumulative shifts for all slices first (in mm)
    cumsum_all = {all_slice_ids[0]: (0.0, 0.0)}
    for i in range(len(all_slice_ids) - 1):
        fixed_id = all_slice_ids[i]
        moving_id = all_slice_ids[i + 1]

        if (fixed_id, moving_id) in shift_lookup:
            dx_mm, dy_mm = shift_lookup[(fixed_id, moving_id)]
        else:
            # If no shift available, assume zero shift
            print(f"Warning: No shift found for pair ({fixed_id}, {moving_id}), using 0")
            dx_mm, dy_mm = 0.0, 0.0

        prev_dx, prev_dy = cumsum_all[fixed_id]
        cumsum_all[moving_id] = (prev_dx + dx_mm, prev_dy + dy_mm)

    # Extract cumulative shifts only for selected slices and convert to pixels
    # Resolution from OME-Zarr can be in microns OR millimeters depending on the writer.
    # We detect which by checking the magnitude:
    # - If res < 1, it's likely in mm (e.g., 0.01 mm = 10 µm)
    # - If res >= 1, it's likely in µm (e.g., 10 µm)
    res_x_raw, res_y_raw = resolution[-2], resolution[-1]

    # Detect unit and convert to microns if needed
    if res_x_raw < 1.0:
        # Resolution is in mm, convert to microns
        res_x_um = res_x_raw * 1000.0
        res_y_um = res_y_raw * 1000.0
        print(f"Resolution detected as mm: {res_x_raw:.4f} mm = {res_x_um:.2f} µm/px")
    else:
        # Resolution is already in microns
        res_x_um = res_x_raw
        res_y_um = res_y_raw
        print(f"Resolution detected as µm: {res_x_um:.2f} µm/px")

    # Now convert mm shifts to pixels: pixels = mm * 1000 / µm_per_pixel
    mm_to_px_x = 1000.0 / res_x_um  # pixels per mm
    mm_to_px_y = 1000.0 / res_y_um  # pixels per mm

    cumsum_selected = {}
    for slice_id in selected_slice_ids:
        if slice_id in cumsum_all:
            dx_mm, dy_mm = cumsum_all[slice_id]
            cumsum_selected[slice_id] = (dx_mm * mm_to_px_x, dy_mm * mm_to_px_y)
        else:
            print(f"Warning: Slice {slice_id} not found in shifts file, using 0 shift")
            cumsum_selected[slice_id] = (0.0, 0.0)

    print(f"Conversion: {mm_to_px_x:.2f} px/mm")

    # Center the drift around the middle slice if requested
    if center_drift and len(cumsum_selected) > 0:
        # Find the middle slice
        middle_idx = len(selected_slice_ids) // 2
        middle_slice_id = selected_slice_ids[middle_idx]
        center_dx, center_dy = cumsum_selected[middle_slice_id]

        print(f"Centering drift around slice {middle_slice_id} (offset: {center_dx:.1f}, {center_dy:.1f} px)")

        # Subtract the middle slice's offset from all slices
        for slice_id in cumsum_selected:
            dx, dy = cumsum_selected[slice_id]
            cumsum_selected[slice_id] = (dx - center_dx, dy - center_dy)

    return cumsum_selected


def compute_common_shape(mosaic_files, slice_ids, cumsum_shifts):
    """
    Compute the common shape needed to fit all aligned mosaics.
    
    Parameters
    ----------
    mosaic_files : dict
        Mapping from slice_id to mosaic file path
    slice_ids : list
        List of slice IDs to process
    cumsum_shifts : dict
        Mapping from slice_id to (cumulative_dx, cumulative_dy)
        
    Returns
    -------
    tuple
        (nx, ny, x0, y0) - shape and origin offsets
    """
    xmin, xmax, ymin, ymax = [], [], [], []

    for slice_id in slice_ids:
        img, _ = read_omezarr(mosaic_files[slice_id])
        dx, dy = cumsum_shifts[slice_id]

        # dx, dy represent WHERE the slice should be positioned
        # For numpy array (Z, H, W): shape[-2] is height (Y), shape[-1] is width (X)
        # So the slice's bounding box:
        #   X: from dx to dx + width
        #   Y: from dy to dy + height
        xmin.append(dx)
        xmax.append(dx + img.shape[-1])  # width = shape[-1] = X dimension
        ymin.append(dy)
        ymax.append(dy + img.shape[-2])  # height = shape[-2] = Y dimension

    x0 = min(xmin)
    y0 = min(ymin)
    nx = int(max(xmax) - x0)
    ny = int(max(ymax) - y0)

    return nx, ny, x0, y0


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_output_exists(args.out_directory, parser, args)

    # Create output directory
    Path(args.out_directory).mkdir(parents=True)

    # Get all .ome.zarr files in in_mosaics_dir and build mapping
    in_mosaics_dir = Path(args.in_mosaics_dir)
    mosaics_list = sorted([p for p in in_mosaics_dir.glob('*.ome.zarr')])

    # Extract slice IDs from filenames and build slice_id -> file mapping
    pattern = r".*z(\d+).*"
    mosaic_files = {}
    for f in mosaics_list:
        match = re.match(pattern, f.name)
        if match:
            slice_id = int(match.group(1))
            mosaic_files[slice_id] = f

    if not mosaic_files:
        parser.error(f"No mosaic files found in {in_mosaics_dir}")

    # Determine which slices to process
    available_slice_ids = sorted(mosaic_files.keys())

    if args.slice_config:
        # Load slice config and filter
        config_path = Path(args.slice_config)
        if not config_path.exists():
            parser.error(f"Slice config file not found: {config_path}")

        slices_to_use = load_slice_config(config_path)
        selected_slice_ids = [s for s in available_slice_ids if s in slices_to_use]

        if not selected_slice_ids:
            parser.error("No slices selected after applying slice config filter")

        excluded = set(available_slice_ids) - set(selected_slice_ids)
        if excluded:
            print(f"Excluding slices from config: {sorted(excluded)}")
    else:
        selected_slice_ids = available_slice_ids

    print(f"Processing {len(selected_slice_ids)} slices: {selected_slice_ids}")

    # Load shifts file
    shifts_df = pd.read_csv(args.in_shifts)

    # Report original cumulative drift
    orig_cumsum_x = shifts_df['x_shift_mm'].cumsum()
    orig_cumsum_y = shifts_df['y_shift_mm'].cumsum()
    print(f"Original total drift (all slices): ({orig_cumsum_x.iloc[-1]:.3f}, {orig_cumsum_y.iloc[-1]:.3f}) mm")

    # Filter outliers if requested
    # NOTE: Outlier filtering operates on ALL shifts in the file, not just selected slices.
    # This is intentional: a bad registration between slices A→B where B is excluded
    # would still affect the cumulative drift for slice C (where A→B→C).
    # The drift centering (below) DOES use only selected slices.
    if args.filter_outliers:
        print(f"\nFiltering outlier shifts (method: {args.outlier_method})")
        shifts_df = filter_outlier_shifts(
            shifts_df,
            max_shift_mm=args.max_shift_mm,
            method=args.outlier_method,
            iqr_multiplier=args.iqr_multiplier
        )

    # Validate that shifts file contains required slices
    shifts_slice_ids = set()
    for _, row in shifts_df.iterrows():
        shifts_slice_ids.add(int(row['fixed_id']))
        shifts_slice_ids.add(int(row['moving_id']))

    missing_in_shifts = set(selected_slice_ids) - shifts_slice_ids
    if missing_in_shifts:
        print(f"Warning: Slices {sorted(missing_in_shifts)} not found in shifts file, will use 0 shift")

    # Get resolution from first mosaic
    first_slice_id = selected_slice_ids[0]
    _, res = read_omezarr(mosaic_files[first_slice_id])

    # Build cumulative shifts for selected slices
    center_drift = not args.no_center_drift
    cumsum_shifts = build_cumulative_shifts(shifts_df, selected_slice_ids, res, center_drift=center_drift)

    # Compute common shape
    nx, ny, x0, y0 = compute_common_shape(mosaic_files, selected_slice_ids, cumsum_shifts)

    print(f"Common space: shape=(width={nx}, height={ny}), origin offset=({x0:.1f}, {y0:.1f}) px")

    # Process each selected slice
    for slice_id in selected_slice_ids:
        mosaic_file = mosaic_files[slice_id]
        img, res = read_omezarr(mosaic_file)
        # Reference array shape is (Z, height, width) = (Z, ny, nx)
        reference = np.zeros((img.shape[0], ny, nx), dtype=img.dtype)

        dx, dy = cumsum_shifts[slice_id]
        # Shift data into the common space:
        # x0, y0 are the minimum coordinates (possibly negative after centering)
        # We need to shift by -x0, -y0 to move everything to positive space,
        # then apply the slice's shift to position it correctly
        # Final position in output = dx - x0
        dx_shifted = dx - x0
        dy_shifted = dy - y0

        # Note: apply_xy_shift uses SimpleITK's TranslationTransform.
        # For a 3D numpy array (Z, Y, X), SimpleITK reorders to (X, Y, Z).
        # translation[0] affects X (columns), translation[1] affects Y (rows).
        # The function signature is apply_xy_shift(img, ref, dx, dy).
        # To move content to position (dx_shifted, dy_shifted), we pass negative values
        # because the transform moves the sampling grid, not the content.
        aligned = apply_xy_shift(img[:], reference, -dx_shifted, -dy_shifted)

        _, filename = psplit(mosaic_file)
        outfile = pjoin(args.out_directory, filename)
        save_omezarr(da.from_array(aligned), outfile, res, chunks=img.chunks)

        print(f"  Processed slice {slice_id:02d}: cumulative_shift=({dx:.1f}, {dy:.1f}) px, "
              f"applied_shift=({dx_shifted:.1f}, {dy_shifted:.1f}) px")


if __name__ == '__main__':
    main()
