#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Using xy shifts file, bring all mosaics in `in_mosaics_dir` to a common space. Each
volume is resampled to a common shape and its content is translated following the
transforms in xy shifts. All transformed mosaics are saved to `out_directory`.

Optionally accepts a slice configuration file to filter which slices to process.
When slices are skipped, their shifts are accumulated to maintain proper alignment.
"""
import argparse
import csv
from pathlib import Path
from os.path import split as psplit
from os.path import join as pjoin
import re
import pandas as pd
import numpy as np
from linumpy.io.zarr import read_omezarr, save_omezarr
from linumpy.utils_images import apply_xy_shift
from linumpy.utils.io import add_overwrite_arg, assert_output_exists
import dask.array as da


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


def build_cumulative_shifts(shifts_df, selected_slice_ids, resolution):
    """
    Build cumulative shifts for selected slices, properly handling skipped slices.
    
    The shifts file contains pairwise shifts between consecutive slices in the 
    original dataset. When slices are skipped, we need to accumulate the 
    intermediate shifts.
    
    Parameters
    ----------
    shifts_df : pd.DataFrame
        DataFrame with columns: fixed_id, moving_id, x_shift_mm, y_shift_mm
    selected_slice_ids : list
        List of slice IDs that will be processed (sorted)
    resolution : tuple
        Resolution (res_x, res_y) in mm/pixel for converting shifts to pixels
        
    Returns
    -------
    dict
        Mapping from slice_id to (cumulative_dx, cumulative_dy) in pixels
    """
    # Build a mapping from (fixed_id, moving_id) to (dx_mm, dy_mm)
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
    
    # Build cumulative shifts for all slices first
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
    res_x, res_y = resolution[-2], resolution[-1]
    cumsum_selected = {}
    for slice_id in selected_slice_ids:
        if slice_id in cumsum_all:
            dx_mm, dy_mm = cumsum_all[slice_id]
            cumsum_selected[slice_id] = (dx_mm / res_x, dy_mm / res_y)
        else:
            print(f"Warning: Slice {slice_id} not found in shifts file, using 0 shift")
            cumsum_selected[slice_id] = (0.0, 0.0)
    
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
        
        xmin.append(-dx)
        xmax.append(-dx + img.shape[-2])
        ymin.append(-dy)
        ymax.append(-dy + img.shape[-1])
    
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
    cumsum_shifts = build_cumulative_shifts(shifts_df, selected_slice_ids, res)
    
    # Compute common shape
    nx, ny, x0, y0 = compute_common_shape(mosaic_files, selected_slice_ids, cumsum_shifts)
    
    print(f"Common space shape: ({nx}, {ny})")
    
    # Process each selected slice
    for slice_id in selected_slice_ids:
        mosaic_file = mosaic_files[slice_id]
        img, res = read_omezarr(mosaic_file)
        reference = np.zeros((img.shape[0], nx, ny), dtype=img.dtype)
        
        dx, dy = cumsum_shifts[slice_id]
        dx_shifted = dx + x0
        dy_shifted = dy + y0
        
        aligned = apply_xy_shift(img[:], reference, dy_shifted, dx_shifted)
        
        _, filename = psplit(mosaic_file)
        outfile = pjoin(args.out_directory, filename)
        save_omezarr(da.from_array(aligned), outfile, res, chunks=img.chunks)
        
        print(f"  Processed slice {slice_id:02d}: shift=({dx:.1f}, {dy:.1f}) px")


if __name__ == '__main__':
    main()
