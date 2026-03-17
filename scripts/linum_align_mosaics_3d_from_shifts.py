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
from linumpy.utils.shifts import filter_outlier_shifts, filter_step_outliers, build_cumulative_shifts
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

    p.add_argument('--excluded_slice_mode',
                   choices=['keep', 'local_median', 'median', 'zero'],
                   default='keep',
                   help='How to handle shifts that involve excluded slices:\n'
                        '  keep: use original shifts (default)\n'
                        '  local_median: replace with local median of neighbors\n'
                        '  median: replace with global median of non-excluded shifts\n'
                        '  zero: replace with zero')
    p.add_argument('--excluded_slice_window', type=int, default=2,
                   help='Neighbor window for excluded-slice replacement [%(default)s]')

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
    p.add_argument('--max_step_mm', type=float, default=0.0,
                   help='Maximum allowed per-step shift in mm. 0 disables. [%(default)s]')
    p.add_argument('--step_window', type=int, default=2,
                   help='Neighbor window for step outlier replacement [%(default)s]')
    p.add_argument('--step_method', choices=['clamp', 'local_median', 'local_mad'],
                   default='local_median',
                   help='How to handle per-step spikes:\n'
                        '  clamp: Limit magnitude to max_step_mm (requires max_step_mm > 0)\n'
                        '  local_median: Replace steps above max_step_mm with local median '
                        '(requires max_step_mm > 0)\n'
                        '  local_mad: Detect AND replace local outliers using MAD-based scoring; '
                        'no fixed threshold needed, controlled by --step_mad_threshold '
                        '[%(default)s]')
    p.add_argument('--step_mad_threshold', type=float, default=3.0,
                   help='Number of local MADs above the local median that triggers outlier '
                        'detection (only with --step_method local_mad). A step is flagged '
                        'when its magnitude exceeds local_median + threshold * local_MAD. '
                        '[%(default)s]')

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


def _replace_with_local_median(df, idx, window, skip_mask=None):
    pos = df.index.get_loc(idx)
    neighbor_vals_x = []
    neighbor_vals_y = []
    neighbor_vals_px_x = []
    neighbor_vals_px_y = []

    for offset in range(-window, window + 1):
        if offset == 0:
            continue
        neighbor_pos = pos + offset
        if 0 <= neighbor_pos < len(df):
            neighbor_idx = df.index[neighbor_pos]
            if skip_mask is not None and skip_mask.get(neighbor_idx, False):
                continue
            neighbor_vals_x.append(df.loc[neighbor_idx, 'x_shift_mm'])
            neighbor_vals_y.append(df.loc[neighbor_idx, 'y_shift_mm'])
            if 'x_shift' in df.columns:
                neighbor_vals_px_x.append(df.loc[neighbor_idx, 'x_shift'])
                neighbor_vals_px_y.append(df.loc[neighbor_idx, 'y_shift'])

    if not neighbor_vals_x:
        return None

    result = {
        'x_shift_mm': float(np.median(neighbor_vals_x)),
        'y_shift_mm': float(np.median(neighbor_vals_y))
    }
    if neighbor_vals_px_x:
        result['x_shift'] = float(np.median(neighbor_vals_px_x))
        result['y_shift'] = float(np.median(neighbor_vals_px_y))
    return result


def handle_excluded_slice_shifts(shifts_df, excluded_slice_ids, mode='keep', window=2):
    if not excluded_slice_ids or mode == 'keep':
        return shifts_df

    df = shifts_df.copy()
    excluded_set = set(int(s) for s in excluded_slice_ids)
    mask = (
        df['fixed_id'].astype(int).isin(excluded_set) |
        df['moving_id'].astype(int).isin(excluded_set)
    )
    n_pairs = int(mask.sum())
    if n_pairs == 0:
        print("No shifts involve excluded slices")
        return df

    print(f"Handling {n_pairs} shifts involving excluded slices (mode: {mode})")

    if mode == 'zero':
        df.loc[mask, ['x_shift_mm', 'y_shift_mm']] = 0.0
        if 'x_shift' in df.columns:
            df.loc[mask, ['x_shift', 'y_shift']] = 0.0
        return df

    non_masked = df[~mask]
    if non_masked.empty:
        print("Warning: all shifts involve excluded slices; falling back to zeros")
        df.loc[mask, ['x_shift_mm', 'y_shift_mm']] = 0.0
        if 'x_shift' in df.columns:
            df.loc[mask, ['x_shift', 'y_shift']] = 0.0
        return df

    if mode == 'median':
        med_x = float(non_masked['x_shift_mm'].median())
        med_y = float(non_masked['y_shift_mm'].median())
        df.loc[mask, 'x_shift_mm'] = med_x
        df.loc[mask, 'y_shift_mm'] = med_y
        if 'x_shift' in df.columns:
            df.loc[mask, 'x_shift'] = float(non_masked['x_shift'].median())
            df.loc[mask, 'y_shift'] = float(non_masked['y_shift'].median())
        return df

    # local_median
    skip_mask = {idx: True for idx in df[mask].index}
    for idx in df[mask].index:
        replacement = _replace_with_local_median(df, idx, window, skip_mask=skip_mask)
        if replacement is None:
            df.loc[idx, 'x_shift_mm'] = float(non_masked['x_shift_mm'].median())
            df.loc[idx, 'y_shift_mm'] = float(non_masked['y_shift_mm'].median())
            if 'x_shift' in df.columns:
                df.loc[idx, 'x_shift'] = float(non_masked['x_shift'].median())
                df.loc[idx, 'y_shift'] = float(non_masked['y_shift'].median())
            continue
        df.loc[idx, 'x_shift_mm'] = replacement['x_shift_mm']
        df.loc[idx, 'y_shift_mm'] = replacement['y_shift_mm']
        if 'x_shift' in replacement:
            df.loc[idx, 'x_shift'] = replacement['x_shift']
            df.loc[idx, 'y_shift'] = replacement['y_shift']

    return df


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
        width = img.shape[-1]
        height = img.shape[-2]

        # So the slice's bounding box:
        #   X: from dx to dx + width
        #   Y: from dy to dy + height
        xmin.append(dx)
        xmax.append(dx + width)
        ymin.append(dy)
        ymax.append(dy + height)

    x0 = min(xmin)
    y0 = min(ymin)
    nx = int(np.ceil(max(xmax) - x0))
    ny = int(np.ceil(max(ymax) - y0))

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
        excluded = set()

    print(f"Processing {len(selected_slice_ids)} slices: {selected_slice_ids}")

    # Load shifts file
    shifts_df = pd.read_csv(args.in_shifts)

    if excluded:
        shifts_df = handle_excluded_slice_shifts(
            shifts_df,
            excluded_slice_ids=excluded,
            mode=args.excluded_slice_mode,
            window=args.excluded_slice_window
        )

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

    if (args.max_step_mm and args.max_step_mm > 0) or args.step_method == 'local_mad':
        print(f"\nFiltering step outliers (method: {args.step_method})")
        shifts_df = filter_step_outliers(
            shifts_df,
            max_step_mm=args.max_step_mm,
            window=args.step_window,
            method=args.step_method,
            mad_threshold=args.step_mad_threshold
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

        # Load image data
        img_data = img[:]

        # Reference array shape is (Z, height, width) = (Z, ny, nx)
        reference = np.zeros((img_data.shape[0], ny, nx), dtype=img_data.dtype)

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
        aligned = apply_xy_shift(img_data, reference, -dx_shifted, -dy_shifted)

        _, filename = psplit(mosaic_file)
        outfile = pjoin(args.out_directory, filename)
        save_omezarr(da.from_array(aligned), outfile, res, chunks=img.chunks)

        print(f"  Processed slice {slice_id:02d}: cumulative_shift=({dx:.1f}, {dy:.1f}) px, "
              f"applied_shift=({dx_shifted:.1f}, {dy_shifted:.1f}) px")


if __name__ == '__main__':
    main()
