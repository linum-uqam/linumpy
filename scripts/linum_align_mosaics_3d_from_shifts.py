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
from linumpy.shifts.utils import build_cumulative_shifts
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

    # Drift centering
    p.add_argument('--no_center_drift', action='store_true',
                   help='Do not center drift around middle slice.\n'
                        'By default, drift is centered to prevent slices from moving out of volume.')

    p.add_argument('--refine_unreliable', action='store_true',
                   help='For transitions flagged as unreliable (reliable=0 in the shifts CSV),\n'
                        'replace the metadata-derived shift with a 2-D phase cross-correlation\n'
                        'estimate computed from the stitched mosaics.  Requires scikit-image.')
    p.add_argument('--refine_max_discrepancy_px', type=float, default=0,
                   help='When --refine_unreliable is active, reject the image-based estimate and\n'
                        'keep the original motor estimate if the two differ by more than this\n'
                        'many pixels (L2 norm). 0 = accept all image-based estimates (default).\n'
                        'Recommended: 50. Guards against phase-correlation failures on large-\n'
                        'offset or low-overlap transitions where the image estimate is wrong.')

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


def _estimate_shift_by_registration(fixed_path, moving_path):
    """Estimate the XY shift between two 3D mosaics via 2-D phase cross-correlation.

    Computes a max-projection over the central 20 % of Z-slices for each
    mosaic, zero-pads both projections to the same shape, then calls
    ``skimage.registration.phase_cross_correlation``.

    Parameters
    ----------
    fixed_path, moving_path : path-like
        Paths to the two ``.ome.zarr`` mosaics.

    Returns
    -------
    dx_mm, dy_mm : float
        Estimated XY shift in mm (same sign convention as the shifts CSV:
        positive X means the moving mosaic is to the left of the fixed one).
    dx_px, dy_px : float
        Same shift in pixels.
    """
    from skimage.registration import phase_cross_correlation

    fixed_vol, res = read_omezarr(fixed_path)
    moving_vol, _ = read_omezarr(moving_path)

    fixed_data = np.array(fixed_vol)
    moving_data = np.array(moving_vol)

    def _proj(arr):
        nz = arr.shape[0]
        z0 = max(0, nz // 2 - max(1, nz // 10))
        z1 = min(nz, nz // 2 + max(1, nz // 10))
        return arr[z0:z1].max(axis=0).astype(np.float32)

    fixed_proj = _proj(fixed_data)
    moving_proj = _proj(moving_data)

    # Pad both to the same (max) shape so that phase_cross_correlation requirements are met
    h = max(fixed_proj.shape[0], moving_proj.shape[0])
    w = max(fixed_proj.shape[1], moving_proj.shape[1])

    def _pad(arr, th, tw):
        ph = th - arr.shape[0]
        pw = tw - arr.shape[1]
        return np.pad(arr, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)))

    fixed_padded = _pad(fixed_proj, h, w)
    moving_padded = _pad(moving_proj, h, w)

    shift, _, _ = phase_cross_correlation(fixed_padded, moving_padded, upsample_factor=10)

    # phase_cross_correlation returns (row_shift, col_shift) = (dy, dx) in pixels.
    # A positive dy means the moving image is shifted downward (larger row index = larger Y).
    # A positive dx means the moving image is shifted rightward (larger col = larger X).
    # The CSV convention is shift = fixed_pos - moving_pos, so:
    #   if moving is to the right by dx_px → x_shift_mm is negative.
    # The phase_cross_correlation shift[1] already has the right sign for our convention:
    # it reports how much to translate moving to align to fixed.
    dy_px, dx_px = float(shift[0]), float(shift[1])

    res_x_mm = res[-1] if res[-1] < 1.0 else res[-1] / 1000.0
    res_y_mm = res[-2] if res[-2] < 1.0 else res[-2] / 1000.0

    dx_mm = dx_px * res_x_mm
    dy_mm = dy_px * res_y_mm

    return dx_mm, dy_mm, dx_px, dy_px


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

    # Refine unreliable transitions with image-based registration if requested
    if args.refine_unreliable and 'reliable' in shifts_df.columns:
        unreliable_mask = shifts_df['reliable'].astype(int) == 0
        n_unreliable = int(unreliable_mask.sum())
        if n_unreliable > 0:
            print(f"Refining {n_unreliable} unreliable transitions via image registration...")
            for idx in shifts_df[unreliable_mask].index:
                fixed_id = int(shifts_df.loc[idx, 'fixed_id'])
                moving_id = int(shifts_df.loc[idx, 'moving_id'])
                if fixed_id not in mosaic_files or moving_id not in mosaic_files:
                    print(f"  Skipping z{fixed_id:02d}→z{moving_id:02d}: mosaic file(s) not found")
                    continue
                try:
                    dx_mm, dy_mm, dx_px, dy_px = _estimate_shift_by_registration(
                        mosaic_files[fixed_id], mosaic_files[moving_id]
                    )
                    # Check discrepancy between image estimate and original motor estimate
                    orig_dx_mm = shifts_df.loc[idx, 'x_shift_mm']
                    orig_dy_mm = shifts_df.loc[idx, 'y_shift_mm']
                    if args.refine_max_discrepancy_px > 0 and 'x_shift' in shifts_df.columns:
                        orig_dx_px = float(shifts_df.loc[idx, 'x_shift'])
                        orig_dy_px = float(shifts_df.loc[idx, 'y_shift'])
                        discrepancy_px = np.sqrt((dx_px - orig_dx_px) ** 2 + (dy_px - orig_dy_px) ** 2)
                        if discrepancy_px > args.refine_max_discrepancy_px:
                            print(f"  z{fixed_id:02d}→z{moving_id:02d}: image estimate discarded "
                                  f"(discrepancy={discrepancy_px:.1f} px > "
                                  f"{args.refine_max_discrepancy_px:.0f} px threshold); "
                                  f"keeping motor estimate ({orig_dx_mm:.3f}, {orig_dy_mm:.3f}) mm")
                            continue
                    print(f"  z{fixed_id:02d}→z{moving_id:02d}: metadata=({orig_dx_mm:.3f}, "
                          f"{orig_dy_mm:.3f}) mm → "
                          f"registered=({dx_mm:.3f}, {dy_mm:.3f}) mm")
                    shifts_df.loc[idx, 'x_shift_mm'] = dx_mm
                    shifts_df.loc[idx, 'y_shift_mm'] = dy_mm
                    if 'x_shift' in shifts_df.columns:
                        shifts_df.loc[idx, 'x_shift'] = dx_px
                        shifts_df.loc[idx, 'y_shift'] = dy_px
                except Exception as exc:
                    print(f"  Warning: registration failed for z{fixed_id:02d}→z{moving_id:02d} ({exc}); "
                          f"keeping metadata shift")
        else:
            print("No unreliable transitions found in shifts file; --refine_unreliable has no effect")
    elif args.refine_unreliable:
        print("Warning: --refine_unreliable requested but shifts CSV has no 'reliable' column; "
              "re-run linum_estimate_xy_shift_from_metadata.py to generate it")

    # Report original cumulative drift
    orig_cumsum_x = shifts_df['x_shift_mm'].cumsum()
    orig_cumsum_y = shifts_df['y_shift_mm'].cumsum()
    print(f"Original total drift (all slices): ({orig_cumsum_x.iloc[-1]:.3f}, {orig_cumsum_y.iloc[-1]:.3f}) mm")

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
