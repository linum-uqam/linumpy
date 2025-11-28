#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a slice configuration file for controlling which slices are used
in the 3D reconstruction pipeline.

This script can detect slices from:
1. A directory containing mosaic grids (*.ome.zarr files with z## in the name)
2. A directory containing raw tiles (tile_x*_y*_z* folders)
3. An existing shifts_xy.csv file

The output is a CSV file with columns:
- slice_id: The slice identifier (e.g., 00, 01, 02)
- use: Boolean whether to use this slice (true/false)
- notes: Optional notes for documentation

Example usage:
    # From mosaic grids directory
    linum_generate_slice_config.py /path/to/mosaics slice_config.csv

    # From raw tiles directory
    linum_generate_slice_config.py /path/to/raw_tiles slice_config.csv --from_tiles

    # From existing shifts file
    linum_generate_slice_config.py /path/to/shifts_xy.csv slice_config.csv --from_shifts
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np

from linumpy.reconstruction import get_tiles_ids
from linumpy.utils.io import add_overwrite_arg, assert_output_exists


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input",
                   help="Input directory (mosaic grids or raw tiles) or shifts CSV file")
    p.add_argument("output_file",
                   help="Output slice configuration CSV file")
    
    source_group = p.add_mutually_exclusive_group()
    source_group.add_argument("--from_tiles", action="store_true",
                              help="Input is a raw tiles directory")
    source_group.add_argument("--from_shifts", action="store_true",
                              help="Input is an existing shifts_xy.csv file")
    
    p.add_argument("--exclude", nargs="+", type=int, default=[],
                   help="List of slice IDs to exclude (set use=false)")
    
    add_overwrite_arg(p)
    return p


def get_slice_ids_from_mosaics(directory: Path) -> list:
    """Extract slice IDs from mosaic grid filenames."""
    pattern = r".*z(\d+).*\.ome\.zarr$"
    slice_ids = []
    
    for f in directory.iterdir():
        if f.is_dir() and f.suffix == '.zarr':
            match = re.match(pattern, f.name)
            if match:
                slice_id = int(match.group(1))
                slice_ids.append(slice_id)
    
    return sorted(set(slice_ids))


def get_slice_ids_from_tiles(directory: Path) -> list:
    """Extract slice IDs from raw tile directories."""
    _, tile_ids = get_tiles_ids(directory)
    z_values = np.unique([ids[2] for ids in tile_ids])
    return sorted(z_values.tolist())


def get_slice_ids_from_shifts(shifts_file: Path) -> list:
    """Extract slice IDs from an existing shifts_xy.csv file."""
    slice_ids = set()
    
    with open(shifts_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            slice_ids.add(int(row['fixed_id']))
            slice_ids.add(int(row['moving_id']))
    
    return sorted(slice_ids)


def write_slice_config(output_file: Path, slice_ids: list, exclude_ids: list = None):
    """Write the slice configuration file."""
    if exclude_ids is None:
        exclude_ids = []
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['slice_id', 'use', 'notes'])
        
        for slice_id in slice_ids:
            use = 'false' if slice_id in exclude_ids else 'true'
            writer.writerow([f'{slice_id:02d}', use, ''])


def main():
    p = _build_arg_parser()
    args = p.parse_args()
    
    input_path = Path(args.input)
    output_file = Path(args.output_file)
    
    assert_output_exists(output_file, p, args)
    
    # Detect slice IDs based on input type
    if args.from_shifts:
        if not input_path.is_file():
            p.error(f"Shifts file not found: {input_path}")
        slice_ids = get_slice_ids_from_shifts(input_path)
        print(f"Found {len(slice_ids)} slices in shifts file: {input_path}")
    elif args.from_tiles:
        if not input_path.is_dir():
            p.error(f"Tiles directory not found: {input_path}")
        slice_ids = get_slice_ids_from_tiles(input_path)
        print(f"Found {len(slice_ids)} slices in tiles directory: {input_path}")
    else:
        # Default: assume mosaic grids directory
        if not input_path.is_dir():
            p.error(f"Mosaics directory not found: {input_path}")
        slice_ids = get_slice_ids_from_mosaics(input_path)
        print(f"Found {len(slice_ids)} slices in mosaics directory: {input_path}")
    
    if not slice_ids:
        p.error("No slices found in input. Check the input path and type.")
    
    # Write the configuration file
    write_slice_config(output_file, slice_ids, args.exclude)
    
    print(f"Slice configuration written to: {output_file}")
    if args.exclude:
        print(f"Excluded slices: {args.exclude}")
    print(f"Slice IDs: {[f'{s:02d}' for s in slice_ids]}")


if __name__ == "__main__":
    main()

