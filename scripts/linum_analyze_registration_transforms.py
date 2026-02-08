#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze pairwise registration transforms to detect rotation drift and alignment issues.

This diagnostic tool aggregates rotation and translation data from all pairwise
registration outputs to identify:
- Cumulative rotation drift (edges drifting apart)
- Sudden rotation jumps (slice misalignment)
- Systematic rotation bias (oblique cutting artifacts)
- Translation vs rotation correlation (dilation indicators)

Useful for troubleshooting 3D reconstruction artifacts like "overhangs" and edge
mismatches in obliquely-cut samples (e.g., 45° between sagittal/coronal).
"""
import linumpy._thread_config  # noqa: F401

import argparse
import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from linumpy.utils.io import add_overwrite_arg

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_directory',
                   help='Path to register_pairwise output directory or pipeline output directory')
    p.add_argument('out_directory',
                   help='Output directory for analysis results')

    p.add_argument('--resolution', type=float, default=10.0,
                   help='Resolution in µm/pixel [%(default)s]')
    p.add_argument('--rotation_threshold', type=float, default=2.0,
                   help='Flag rotations above this threshold (degrees) [%(default)s]')
    p.add_argument('--cumulative_threshold', type=float, default=5.0,
                   help='Flag cumulative rotation above this threshold (degrees) [%(default)s]')
    p.add_argument('--include_tfm', action='store_true',
                   help='Also parse transform.tfm files for rotation (if JSON missing)')

    add_overwrite_arg(p)
    return p


def find_registration_dirs(base_path):
    """Find all pairwise registration directories."""
    base = Path(base_path)

    # If base doesn't exist, raise error
    if not base.exists():
        raise FileNotFoundError(f"Path does not exist: {base_path}")

    # Check if this is already the register_pairwise directory
    if base.name == 'register_pairwise':
        reg_dir = base
    elif (base / 'register_pairwise').exists():
        reg_dir = base / 'register_pairwise'
    else:
        # Search for register_pairwise subdirectory
        candidates = list(base.glob('**/register_pairwise'))
        if candidates:
            reg_dir = candidates[0]
        else:
            # Maybe we're directly in a directory containing slice_z* dirs
            # This happens when Nextflow stages files with path("register_pairwise/*")
            slice_dirs = sorted([d for d in base.iterdir() if d.is_dir() and 'slice_z' in d.name])
            if slice_dirs:
                logger.info(f"Found {len(slice_dirs)} slice directories directly in {base_path}")
                return slice_dirs

            # Also check if there are JSON files directly here (flat structure)
            json_files = list(base.glob('**/pairwise_registration_metrics.json'))
            if json_files:
                # Return the parent directories of the JSON files
                slice_dirs = sorted(set(f.parent for f in json_files))
                logger.info(f"Found {len(slice_dirs)} directories with registration metrics")
                return slice_dirs

            raise FileNotFoundError(f"No register_pairwise directory or slice_z* directories found in {base_path}")

    # Find all slice directories
    slice_dirs = sorted([d for d in reg_dir.iterdir() if d.is_dir() and 'slice_z' in d.name])
    if not slice_dirs:
        # Try searching recursively
        slice_dirs = sorted([d.parent for d in reg_dir.glob('**/pairwise_registration_metrics.json')])
        slice_dirs = sorted(set(slice_dirs))  # Remove duplicates

    return slice_dirs


def parse_slice_id(dirname):
    """Extract slice ID from directory name like 'slice_z05_normalize'."""
    match = re.search(r'slice_z(\d+)', dirname)
    return int(match.group(1)) if match else None


def load_metrics_from_json(json_path):
    """Load registration metrics from JSON file."""
    with open(json_path) as f:
        data = json.load(f)

    metrics = data.get('metrics', {})
    return {
        'rotation': metrics.get('rotation', {}).get('value', None),
        'translation_x': metrics.get('translation_x', {}).get('value', None),
        'translation_y': metrics.get('translation_y', {}).get('value', None),
        'translation_magnitude': metrics.get('translation_magnitude', {}).get('value', None),
        'z_drift': metrics.get('z_drift', {}).get('value', None),
        'registration_error': metrics.get('registration_error', {}).get('value', None),
        'fixed_volume': metrics.get('fixed_volume', {}).get('value', None),
        'moving_volume': metrics.get('moving_volume', {}).get('value', None),
    }


def load_rotation_from_tfm(tfm_path):
    """Extract rotation angle from SimpleITK transform file."""
    import SimpleITK as sitk

    try:
        transform = sitk.ReadTransform(str(tfm_path))

        # For Euler2DTransform, parameter[0] is the rotation angle in radians
        if 'Euler2D' in transform.GetName():
            params = transform.GetParameters()
            if len(params) >= 1:
                return np.degrees(params[0])

        # For AffineTransform, extract rotation from matrix
        elif 'Affine' in transform.GetName():
            params = transform.GetParameters()
            # 2D affine: [a00, a01, a10, a11, tx, ty]
            if len(params) >= 4:
                a00, a01 = params[0], params[1]
                # Rotation angle from matrix components
                return np.degrees(np.arctan2(-a01, a00))

        return None
    except Exception as e:
        logger.warning(f"Could not parse transform file {tfm_path}: {e}")
        return None


def collect_registration_data(slice_dirs, include_tfm=False):
    """Collect registration data from all slice directories."""
    records = []

    for slice_dir in slice_dirs:
        slice_id = parse_slice_id(slice_dir.name)
        if slice_id is None:
            continue

        record = {
            'slice_id': slice_id,
            'directory': slice_dir.name,
        }

        # Try JSON first
        json_path = slice_dir / 'pairwise_registration_metrics.json'
        if json_path.exists():
            metrics = load_metrics_from_json(json_path)
            record.update(metrics)

        # Optionally try TFM file if rotation is missing
        if include_tfm and record.get('rotation') is None:
            tfm_path = slice_dir / 'transform.tfm'
            if tfm_path.exists():
                rotation = load_rotation_from_tfm(tfm_path)
                if rotation is not None:
                    record['rotation'] = rotation
                    record['rotation_source'] = 'tfm'

        records.append(record)

    df = pd.DataFrame(records)
    df = df.sort_values('slice_id').reset_index(drop=True)
    return df


def analyze_rotation_drift(df, rotation_threshold=2.0, cumulative_threshold=5.0):
    """Analyze rotation patterns and detect issues."""
    analysis = {
        'n_slices': len(df),
        'rotation_stats': {},
        'issues': [],
        'cumulative_drift': None,
    }

    if 'rotation' not in df.columns or df['rotation'].isna().all():
        analysis['issues'].append("No rotation data available")
        return analysis

    rotations = df['rotation'].dropna()

    # Basic statistics
    analysis['rotation_stats'] = {
        'mean': float(rotations.mean()),
        'std': float(rotations.std()),
        'min': float(rotations.min()),
        'max': float(rotations.max()),
        'median': float(rotations.median()),
    }

    # Cumulative rotation (drift)
    cumulative = rotations.cumsum()
    analysis['cumulative_drift'] = {
        'total': float(cumulative.iloc[-1]) if len(cumulative) > 0 else 0,
        'max_absolute': float(cumulative.abs().max()) if len(cumulative) > 0 else 0,
    }

    # Detect large individual rotations
    large_rotations = df[df['rotation'].abs() > rotation_threshold]
    if len(large_rotations) > 0:
        analysis['issues'].append(
            f"Found {len(large_rotations)} slices with rotation > {rotation_threshold}°"
        )
        analysis['large_rotation_slices'] = large_rotations['slice_id'].tolist()

    # Check cumulative drift
    if abs(analysis['cumulative_drift']['total']) > cumulative_threshold:
        analysis['issues'].append(
            f"High cumulative rotation drift: {analysis['cumulative_drift']['total']:.2f}°"
        )

    # Check for systematic rotation bias
    if abs(analysis['rotation_stats']['mean']) > 0.5:
        analysis['issues'].append(
            f"Systematic rotation bias detected: mean={analysis['rotation_stats']['mean']:.3f}°"
        )

    return analysis


def analyze_translation_rotation_correlation(df):
    """Check if translation and rotation are correlated (dilation indicator)."""
    if 'rotation' not in df.columns or 'translation_magnitude' not in df.columns:
        return {}

    valid = df.dropna(subset=['rotation', 'translation_magnitude'])
    if len(valid) < 5:
        return {}

    correlation = np.corrcoef(valid['rotation'].abs(), valid['translation_magnitude'])[0, 1]

    return {
        'rotation_translation_correlation': float(correlation),
        'interpretation': 'high' if abs(correlation) > 0.5 else 'low',
    }


def generate_report(df, analysis, correlation, output_dir):
    """Generate text report."""
    lines = [
        "=" * 70,
        "REGISTRATION TRANSFORM ANALYSIS",
        "=" * 70,
        "",
        "OVERVIEW",
        "-" * 50,
        f"Total slices analyzed: {analysis['n_slices']}",
        f"Slices with rotation data: {df['rotation'].notna().sum()}",
        "",
    ]

    # Rotation statistics
    if analysis['rotation_stats']:
        lines.extend([
            "ROTATION STATISTICS",
            "-" * 50,
            f"Mean rotation:   {analysis['rotation_stats']['mean']:.4f}°",
            f"Std deviation:   {analysis['rotation_stats']['std']:.4f}°",
            f"Min rotation:    {analysis['rotation_stats']['min']:.4f}°",
            f"Max rotation:    {analysis['rotation_stats']['max']:.4f}°",
            f"Median rotation: {analysis['rotation_stats']['median']:.4f}°",
            "",
        ])

    # Cumulative drift
    if analysis['cumulative_drift']:
        lines.extend([
            "CUMULATIVE ROTATION DRIFT",
            "-" * 50,
            f"Total drift:       {analysis['cumulative_drift']['total']:.4f}°",
            f"Max absolute:      {analysis['cumulative_drift']['max_absolute']:.4f}°",
            "",
        ])

    # Correlation analysis
    if correlation:
        lines.extend([
            "ROTATION-TRANSLATION CORRELATION",
            "-" * 50,
            f"Correlation coefficient: {correlation.get('rotation_translation_correlation', 'N/A'):.4f}",
            f"Interpretation: {correlation.get('interpretation', 'N/A')}",
            "(High correlation may indicate dilation/scaling issues)",
            "",
        ])

    # Issues detected
    lines.extend([
        "ISSUES DETECTED",
        "-" * 50,
    ])
    if analysis['issues']:
        for issue in analysis['issues']:
            lines.append(f"  ⚠ {issue}")
    else:
        lines.append("  ✓ No significant issues detected")
    lines.append("")

    # Large rotation slices
    if 'large_rotation_slices' in analysis:
        lines.extend([
            "SLICES WITH LARGE ROTATIONS",
            "-" * 50,
        ])
        for sid in analysis['large_rotation_slices']:
            row = df[df['slice_id'] == sid].iloc[0]
            lines.append(f"  Slice {sid:02d}: rotation={row['rotation']:.3f}°")
        lines.append("")

    lines.append("=" * 70)

    report_path = output_dir / 'rotation_analysis.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    logger.info(f"Report saved to {report_path}")
    return report_path


def generate_plots(df, output_dir):
    """Generate visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Per-slice rotation
    ax1 = axes[0, 0]
    if 'rotation' in df.columns:
        valid = df.dropna(subset=['rotation'])
        ax1.bar(valid['slice_id'], valid['rotation'], alpha=0.7, color='steelblue')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.axhline(y=2, color='red', linestyle='--', linewidth=0.5, label='±2° threshold')
        ax1.axhline(y=-2, color='red', linestyle='--', linewidth=0.5)
        ax1.set_xlabel('Slice ID')
        ax1.set_ylabel('Rotation (degrees)')
        ax1.set_title('Per-Slice Rotation')
        ax1.legend()

    # 2. Cumulative rotation drift
    ax2 = axes[0, 1]
    if 'rotation' in df.columns:
        valid = df.dropna(subset=['rotation']).sort_values('slice_id')
        cumulative = valid['rotation'].cumsum()
        ax2.plot(valid['slice_id'], cumulative, 'b-', linewidth=2, label='Cumulative rotation')
        ax2.fill_between(valid['slice_id'], 0, cumulative, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Slice ID')
        ax2.set_ylabel('Cumulative Rotation (degrees)')
        ax2.set_title('Cumulative Rotation Drift')
        ax2.legend()

    # 3. Translation magnitude vs rotation
    ax3 = axes[1, 0]
    if 'rotation' in df.columns and 'translation_magnitude' in df.columns:
        valid = df.dropna(subset=['rotation', 'translation_magnitude'])
        ax3.scatter(valid['rotation'].abs(), valid['translation_magnitude'],
                   alpha=0.6, c=valid['slice_id'], cmap='viridis')
        ax3.set_xlabel('|Rotation| (degrees)')
        ax3.set_ylabel('Translation Magnitude (pixels)')
        ax3.set_title('Translation vs Rotation (colored by slice ID)')
        plt.colorbar(ax3.collections[0], ax=ax3, label='Slice ID')

    # 4. Translation components
    ax4 = axes[1, 1]
    if 'translation_x' in df.columns and 'translation_y' in df.columns:
        valid = df.dropna(subset=['translation_x', 'translation_y'])
        ax4.scatter(valid['translation_x'], valid['translation_y'],
                   alpha=0.6, c=valid['slice_id'], cmap='viridis')
        ax4.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax4.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
        ax4.set_xlabel('Translation X (pixels)')
        ax4.set_ylabel('Translation Y (pixels)')
        ax4.set_title('Translation Vector Components')
        ax4.set_aspect('equal')

    plt.tight_layout()
    plot_path = output_dir / 'rotation_analysis.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()

    logger.info(f"Plots saved to {plot_path}")
    return plot_path


def main():
    p = _build_arg_parser()
    args = p.parse_args()

    output_dir = Path(args.out_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find and collect registration data
    logger.info(f"Searching for registration data in {args.in_directory}")
    slice_dirs = find_registration_dirs(args.in_directory)
    logger.info(f"Found {len(slice_dirs)} registration directories")

    df = collect_registration_data(slice_dirs, include_tfm=args.include_tfm)
    logger.info(f"Collected data for {len(df)} slices")

    # Save raw data
    csv_path = output_dir / 'registration_transforms.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Raw data saved to {csv_path}")

    # Analyze rotation
    analysis = analyze_rotation_drift(df, args.rotation_threshold, args.cumulative_threshold)
    correlation = analyze_translation_rotation_correlation(df)

    # Generate outputs
    generate_report(df, analysis, correlation, output_dir)
    generate_plots(df, output_dir)

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    if analysis['cumulative_drift']:
        print(f"Total rotation drift: {analysis['cumulative_drift']['total']:.2f}°")
    if analysis['issues']:
        print(f"Issues found: {len(analysis['issues'])}")
        for issue in analysis['issues']:
            print(f"  - {issue}")
    else:
        print("No significant issues detected")


if __name__ == '__main__':
    main()
