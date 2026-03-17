#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assess slice quality for 3D mosaic grids and optionally update slice configuration.

This script analyzes mosaic grid slices to detect quality issues that might affect
reconstruction. It uses multiple metrics to identify problematic slices:

1. **SSIM (Structural Similarity)**: Compares each slice to its neighbors
2. **Edge Preservation**: Detects if edge structures are preserved compared to neighbors
3. **Variance Consistency**: Checks for unusual signal variance (data loss/corruption)
4. **First Slice Detection**: Automatically identifies calibration slices (thicker/different)

The output can be:
- A new slice_config.csv with quality scores and recommendations
- An update to an existing slice_config.csv with quality assessments
- A quality report for review

Example usage:
    # Assess quality and create/update slice config
    linum_assess_slice_quality.py /path/to/mosaics slice_config.csv

    # Assess and exclude low-quality slices automatically
    linum_assess_slice_quality.py /path/to/mosaics slice_config.csv --min_quality 0.3

    # Exclude first N calibration slices
    linum_assess_slice_quality.py /path/to/mosaics slice_config.csv --exclude_first 1

    # Update existing config with quality info
    linum_assess_slice_quality.py /path/to/mosaics slice_config.csv --update_existing
"""
# Configure thread limits before numpy/scipy imports
import linumpy._thread_config  # noqa: F401

import argparse
import csv
import re
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
from tqdm.auto import tqdm

from linumpy.io.zarr import read_omezarr
from linumpy.utils.io import add_overwrite_arg, assert_output_exists
from linumpy.utils.image_quality import (
    assess_slice_quality,
    detect_calibration_slice,
)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input",
                   help="Input directory containing mosaic grids (*.ome.zarr)")
    p.add_argument("output_file",
                   help="Output slice configuration CSV file")

    # Quality assessment options
    quality_group = p.add_argument_group('Quality Assessment')
    quality_group.add_argument("--min_quality", type=float, default=0.0,
                               help="Minimum quality score to include slice (0-1). "
                                    "Default: 0.0 (include all, just report)")
    quality_group.add_argument("--sample_depth", type=int, default=5,
                               help="Number of z-planes to sample per slice for "
                                    "faster assessment. Default: 5 (0=all)")
    quality_group.add_argument("--pyramid_level", type=int, default=0,
                               help="Pyramid level to use for assessment (0=full res). "
                                    "Higher levels are faster but less accurate. Default: 0")

    # Calibration slice options
    calib_group = p.add_argument_group('Calibration Slice Handling')
    calib_group.add_argument("--exclude_first", type=int, default=1,
                             help="Exclude first N slices as calibration slices. "
                                  "Default: 1 (first slice is usually calibration)")
    calib_group.add_argument("--detect_calibration", action="store_true",
                             help="Automatically detect calibration slices by their "
                                  "different thickness/structure")
    calib_group.add_argument("--calibration_thickness_ratio", type=float, default=1.5,
                             help="Slices with thickness ratio > this are flagged as "
                                  "calibration. Default: 1.5")

    # Update/merge options
    update_group = p.add_argument_group('Update Existing Config')
    update_group.add_argument("--update_existing", action="store_true",
                              help="Update an existing slice_config.csv with quality info")
    update_group.add_argument("--existing_config", type=str, default=None,
                              help="Path to existing slice config to update")

    # Output options
    output_group = p.add_argument_group('Output Options')
    output_group.add_argument("--report_only", action="store_true",
                              help="Only print report, don't write config file")
    output_group.add_argument("-v", "--verbose", action="store_true",
                              help="Print detailed quality metrics per slice")

    add_overwrite_arg(p)
    return p


def get_mosaic_files(directory: Path) -> Dict[int, Path]:
    """Find all mosaic grid files and extract slice IDs."""
    pattern = r".*z(\d+).*\.ome\.zarr$"
    mosaics = {}

    for f in directory.iterdir():
        if f.is_dir() and f.suffix == '.zarr':
            match = re.match(pattern, f.name)
            if match:
                slice_id = int(match.group(1))
                mosaics[slice_id] = f

    return dict(sorted(mosaics.items()))


def read_existing_config(config_path: Path) -> Dict[int, Dict[str, Any]]:
    """Read an existing slice configuration file."""
    config = {}
    with open(config_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            slice_id = int(row['slice_id'])
            config[slice_id] = dict(row)
    return config


def write_slice_config_with_quality(output_file: Path, slice_ids: List[int],
                                    quality_results: Dict[int, Dict[str, Any]],
                                    exclude_ids: List[int],
                                    existing_config: Optional[Dict[int, Dict[str, Any]]] = None):
    """Write the slice configuration file with quality metrics."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        header = ['slice_id', 'use', 'quality_score', 'ssim_mean', 'edge_score',
                  'variance_score', 'depth', 'exclude_reason']

        # Add galvo columns if present in existing config
        has_galvo = False
        if existing_config:
            sample = next(iter(existing_config.values()), {})
            if 'galvo_confidence' in sample:
                has_galvo = True
                header.insert(3, 'galvo_confidence')
                header.insert(4, 'galvo_fix')

        writer.writerow(header)

        for slice_id in slice_ids:
            quality = quality_results.get(slice_id, {})

            use = 'true'
            reason = ''

            if slice_id in exclude_ids:
                use = 'false'
                if quality.get('is_calibration', False):
                    reason = 'calibration_slice'
                elif quality.get('overall', 1.0) < quality.get('min_threshold', 0):
                    reason = 'low_quality'
                elif quality.get('exclude_first', False):
                    reason = 'first_slice_excluded'
                else:
                    reason = 'manually_excluded'

            # Preserve existing use status if updating
            if existing_config and slice_id in existing_config:
                existing = existing_config[slice_id]
                if existing.get('use', 'true').lower() in ['false', '0', 'no']:
                    use = 'false'
                    if not reason:
                        reason = existing.get('notes', existing.get('exclude_reason', 'previously_excluded'))

            row = [
                f'{slice_id:02d}',
                use,
                f"{quality.get('overall', 0.0):.3f}",
                f"{quality.get('ssim_mean', 0.0):.3f}",
                f"{quality.get('edge_score', 0.0):.3f}",
                f"{quality.get('variance_score', 0.0):.3f}",
                str(quality.get('depth', 0)),
                reason
            ]

            # Add galvo columns if present
            if has_galvo:
                existing = existing_config.get(slice_id, {}) if existing_config else {}
                galvo_conf = existing.get('galvo_confidence', '0.000')
                galvo_fix = existing.get('galvo_fix', 'false')
                row.insert(3, galvo_conf)
                row.insert(4, galvo_fix)

            writer.writerow(row)


def main():
    p = _build_arg_parser()
    args = p.parse_args()

    input_path = Path(args.input)
    output_file = Path(args.output_file)

    if not args.report_only:
        assert_output_exists(output_file, p, args)

    if not input_path.is_dir():
        p.error(f"Input directory not found: {input_path}")

    # Find mosaic files
    print(f"Scanning for mosaic grids in: {input_path}")
    mosaic_files = get_mosaic_files(input_path)

    if not mosaic_files:
        p.error(f"No mosaic grid files found in {input_path}")

    slice_ids = sorted(mosaic_files.keys())
    print(f"Found {len(slice_ids)} slices: {[f'{s:02d}' for s in slice_ids]}")

    # Load existing config if updating
    existing_config = None
    if args.update_existing:
        config_to_load = args.existing_config if args.existing_config else output_file
        if Path(config_to_load).exists():
            existing_config = read_existing_config(Path(config_to_load))
            print(f"Loaded existing config with {len(existing_config)} entries")

    # Identify slices to exclude
    exclude_ids = set()

    # Exclude first N slices
    if args.exclude_first > 0:
        first_slices = slice_ids[:args.exclude_first]
        exclude_ids.update(first_slices)
        print(f"Excluding first {args.exclude_first} slice(s) as calibration: {first_slices}")

    # Load volumes
    print(f"\nLoading slices (pyramid_level={args.pyramid_level})...")
    volumes: Dict[int, np.ndarray] = {}
    for slice_id in tqdm(slice_ids, desc="Loading slices"):
        try:
            vol, _ = read_omezarr(mosaic_files[slice_id], level=args.pyramid_level)
            volumes[slice_id] = vol
        except Exception as e:
            print(f"  Warning: Could not load slice {slice_id:02d}: {e}")
            volumes[slice_id] = None

    # Detect calibration slices if requested
    calibration_slices = []
    if args.detect_calibration:
        print(f"Detecting calibration slices (thickness ratio > {args.calibration_thickness_ratio})...")
        valid_volumes = {sid: vol for sid, vol in volumes.items() if vol is not None}
        calibration_slices = detect_calibration_slice(valid_volumes, args.calibration_thickness_ratio)
        if calibration_slices:
            exclude_ids.update(calibration_slices)
            print(f"Detected calibration slices: {calibration_slices}")

    # Assess quality for each slice
    print(f"\nAssessing slice quality (sample_depth={args.sample_depth})...")
    quality_results: Dict[int, Dict[str, Any]] = {}

    for i, slice_id in enumerate(tqdm(slice_ids, desc="Assessing quality")):
        vol = volumes.get(slice_id)

        if vol is None:
            quality_results[slice_id] = {
                'overall': 0.0, 'ssim_mean': 0.0, 'edge_score': 0.0,
                'variance_score': 0.0, 'depth': 0, 'has_data': False,
                'error': 'load_failed'
            }
            continue

        # Get neighbor volumes
        vol_before = volumes.get(slice_ids[i - 1]) if i > 0 else None
        vol_after = volumes.get(slice_ids[i + 1]) if i < len(slice_ids) - 1 else None

        # Compute quality using consolidated module
        overall, metrics = assess_slice_quality(vol, vol_before, vol_after, args.sample_depth)

        # Add metadata
        metrics['is_calibration'] = slice_id in calibration_slices
        metrics['exclude_first'] = slice_id in slice_ids[:args.exclude_first]
        metrics['min_threshold'] = args.min_quality

        quality_results[slice_id] = metrics

        # Exclude if below quality threshold
        if args.min_quality > 0 and overall < args.min_quality:
            exclude_ids.add(slice_id)

    # Print quality report
    print("\n" + "=" * 70)
    print("SLICE QUALITY REPORT")
    print("=" * 70)
    print(f"{'Slice':<8} {'Quality':<10} {'SSIM':<10} {'Edge':<10} {'Var':<10} {'Depth':<8} {'Status':<15}")
    print("-" * 70)

    for slice_id in slice_ids:
        q = quality_results.get(slice_id, {})
        status = []
        if slice_id in exclude_ids:
            if q.get('is_calibration'):
                status.append('CALIBRATION')
            elif q.get('exclude_first'):
                status.append('FIRST_SLICE')
            elif q.get('overall', 1.0) < args.min_quality:
                status.append('LOW_QUALITY')
            else:
                status.append('EXCLUDED')
        else:
            status.append('OK')

        status_str = ','.join(status)

        print(f"{slice_id:02d}      {q.get('overall', 0):.3f}      "
              f"{q.get('ssim_mean', 0):.3f}      {q.get('edge_score', 0):.3f}      "
              f"{q.get('variance_score', 0):.3f}      {q.get('depth', 0):<8} {status_str}")

    print("-" * 70)
    print(f"Total slices: {len(slice_ids)}")
    print(f"Excluded: {len(exclude_ids)}")
    print(f"Included: {len(slice_ids) - len(exclude_ids)}")

    if args.min_quality > 0:
        low_quality = [s for s in slice_ids
                       if quality_results.get(s, {}).get('overall', 1.0) < args.min_quality]
        if low_quality:
            print(f"Low quality slices (< {args.min_quality}): {low_quality}")

    # Write config file
    if not args.report_only:
        write_slice_config_with_quality(
            output_file, slice_ids, quality_results,
            list(exclude_ids), existing_config
        )
        print(f"\nSlice configuration written to: {output_file}")

    # Return exit code based on whether any slices were excluded
    if exclude_ids:
        print(f"\nExcluded slice IDs: {sorted(exclude_ids)}")


if __name__ == "__main__":
    main()
