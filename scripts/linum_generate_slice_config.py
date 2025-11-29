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
- galvo_confidence: (optional) Galvo shift detection confidence (0-1)
- galvo_fix: (optional) Whether galvo fix would be applied (true/false)
- notes: Optional notes for documentation

Example usage:
    # From mosaic grids directory
    linum_generate_slice_config.py /path/to/mosaics slice_config.csv

    # From raw tiles directory
    linum_generate_slice_config.py /path/to/raw_tiles slice_config.csv --from_tiles

    # From existing shifts file
    linum_generate_slice_config.py /path/to/shifts_xy.csv slice_config.csv --from_shifts
    
    # With galvo detection (requires raw tiles)
    linum_generate_slice_config.py /path/to/raw_tiles slice_config.csv --from_tiles --detect_galvo
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
    
    # Galvo detection options
    galvo_group = p.add_argument_group('Galvo Detection',
                                       'Detect galvo shift artifacts in raw tiles')
    galvo_group.add_argument("--detect_galvo", action="store_true",
                             help="Run galvo shift detection (requires --from_tiles or raw tiles dir)")
    galvo_group.add_argument("--tiles_dir", type=str, default=None,
                             help="Raw tiles directory for galvo detection (if input is shifts file)")
    galvo_group.add_argument("--galvo_threshold", type=float, default=0.3,
                             help="Confidence threshold for galvo fix (default: 0.3)")
    
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


def detect_galvo_for_slices(tiles_dir: Path, slice_ids: list, threshold: float = 0.3) -> dict:
    """
    Detect galvo shift artifacts for each slice.
    
    Parameters
    ----------
    tiles_dir : Path
        Directory containing raw tiles
    slice_ids : list
        List of slice IDs to analyze
    threshold : float
        Confidence threshold for applying fix
        
    Returns
    -------
    dict
        Mapping from slice_id to {'confidence': float, 'would_fix': bool}
    """
    from linumpy.microscope.oct import OCT
    from linumpy.preproc.xyzcorr import detect_galvo_shift
    from tqdm.auto import tqdm
    
    results = {}
    
    for z in tqdm(slice_ids, desc="Detecting galvo shift"):
        try:
            # Get tiles for this slice
            tiles, _ = get_tiles_ids(tiles_dir, z=z)
            
            if not tiles:
                results[z] = {'confidence': 0.0, 'would_fix': False, 'error': 'no_tiles'}
                continue
            
            # Sample one tile from the middle of the slice
            sample_tile = tiles[len(tiles) // 2]
            
            oct = OCT(sample_tile)
            n_extra = oct.info.get('n_extra', 0)
            
            if n_extra == 0:
                results[z] = {'confidence': 0.0, 'would_fix': False, 'error': 'no_extra_alines'}
                continue
            
            # Load raw data without fixes
            vol = oct.load_image(crop=False, fix_galvo_shift=False, fix_camera_shift=False)
            aip = vol.mean(axis=0)
            
            # Run detection
            shift, confidence = detect_galvo_shift(aip, n_pixel_return=n_extra, return_confidence=True)
            
            results[z] = {
                'confidence': confidence,
                'would_fix': confidence >= threshold,
                'shift': shift
            }
        except Exception as e:
            results[z] = {'confidence': 0.0, 'would_fix': False, 'error': str(e)}
    
    return results


def write_slice_config(output_file: Path, slice_ids: list, exclude_ids: list = None, 
                       galvo_results: dict = None):
    """Write the slice configuration file."""
    if exclude_ids is None:
        exclude_ids = []
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header depends on whether galvo detection was run
        if galvo_results:
            writer.writerow(['slice_id', 'use', 'galvo_confidence', 'galvo_fix', 'notes'])
        else:
            writer.writerow(['slice_id', 'use', 'notes'])
        
        for slice_id in slice_ids:
            use = 'false' if slice_id in exclude_ids else 'true'
            
            if galvo_results and slice_id in galvo_results:
                galvo = galvo_results[slice_id]
                confidence = f"{galvo['confidence']:.3f}"
                would_fix = 'true' if galvo.get('would_fix', False) else 'false'
                note = galvo.get('error', '')
                writer.writerow([f'{slice_id:02d}', use, confidence, would_fix, note])
            elif galvo_results:
                writer.writerow([f'{slice_id:02d}', use, '0.000', 'false', 'not_analyzed'])
            else:
                writer.writerow([f'{slice_id:02d}', use, ''])


def main():
    p = _build_arg_parser()
    args = p.parse_args()
    
    input_path = Path(args.input)
    output_file = Path(args.output_file)
    
    assert_output_exists(output_file, p, args)
    
    # Determine tiles directory for galvo detection
    tiles_dir = None
    if args.tiles_dir:
        tiles_dir = Path(args.tiles_dir)
    elif args.from_tiles:
        tiles_dir = input_path
    
    # Validate galvo detection requirements
    if args.detect_galvo and tiles_dir is None:
        p.error("--detect_galvo requires --from_tiles or --tiles_dir to specify raw tiles location")
    
    if args.detect_galvo and tiles_dir and not tiles_dir.is_dir():
        p.error(f"Tiles directory not found: {tiles_dir}")
    
    # Detect slice IDs based on input type
    if args.from_shifts:
        if not input_path.exists():
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
    
    # Run galvo detection if requested
    galvo_results = None
    if args.detect_galvo:
        print(f"\nRunning galvo shift detection (threshold={args.galvo_threshold})...")
        galvo_results = detect_galvo_for_slices(tiles_dir, slice_ids, args.galvo_threshold)
        
        # Print summary
        fix_count = sum(1 for r in galvo_results.values() if r.get('would_fix', False))
        skip_count = len(galvo_results) - fix_count
        print(f"\nGalvo Detection Summary:")
        print(f"  Fix would be applied: {fix_count} slices")
        print(f"  Fix would be skipped: {skip_count} slices")
    
    # Write the configuration file
    write_slice_config(output_file, slice_ids, args.exclude, galvo_results)
    
    print(f"\nSlice configuration written to: {output_file}")
    if args.exclude:
        print(f"Excluded slices: {args.exclude}")
    print(f"Slice IDs: {[f'{s:02d}' for s in slice_ids]}")


if __name__ == "__main__":
    main()

