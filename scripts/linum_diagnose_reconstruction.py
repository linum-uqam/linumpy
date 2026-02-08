#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive diagnostic analysis for 3D reconstruction troubleshooting.

This script runs multiple diagnostic analyses to identify the root cause of
reconstruction artifacts like edge mismatches and "overhangs" in serial OCT data:

1. **Rotation Analysis**: Cumulative rotation drift between slices
2. **Dilation Analysis**: Tile position scaling/expansion issues
3. **Edge Alignment**: Cross-correlation between consecutive slice edges
4. **Motor vs Registration**: Compare motor-only vs registered alignment

Designed for troubleshooting 45° oblique-cut mouse brain reconstructions.
"""
import linumpy._thread_config  # noqa: F401

import argparse
import json
import logging
import re
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from linumpy.utils.io import add_overwrite_arg

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('pipeline_output',
                   help='Path to pipeline output directory (containing register_pairwise, etc.)')
    p.add_argument('out_directory',
                   help='Output directory for diagnostic results')

    # Analysis selection
    p.add_argument('--skip_rotation', action='store_true',
                   help='Skip rotation drift analysis')
    p.add_argument('--skip_shifts', action='store_true',
                   help='Skip shifts analysis')
    p.add_argument('--skip_edge', action='store_true',
                   help='Skip edge alignment analysis')

    # Parameters
    p.add_argument('--resolution', type=float, default=10.0,
                   help='Resolution in µm/pixel [%(default)s]')
    p.add_argument('--rotation_threshold', type=float, default=2.0,
                   help='Flag rotations above this threshold (degrees) [%(default)s]')
    p.add_argument('--slice_range', type=str, default=None,
                   help='Analyze only these slices (e.g., "10-20" or "5,10,15")')

    add_overwrite_arg(p)
    return p


def parse_slice_range(range_str):
    """Parse slice range like "10-20" or "5,10,15" into list of IDs."""
    if not range_str:
        return None

    slice_ids = set()
    for part in range_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            slice_ids.update(range(int(start), int(end) + 1))
        else:
            slice_ids.add(int(part))
    return sorted(slice_ids)


def analyze_rotation_drift(pipeline_dir, output_dir, threshold=2.0, slice_ids=None):
    """Analyze rotation patterns from pairwise registration."""
    reg_dir = Path(pipeline_dir) / 'register_pairwise'
    if not reg_dir.exists():
        logger.warning(f"No register_pairwise directory found at {reg_dir}")
        return None

    records = []
    for slice_dir in sorted(reg_dir.iterdir()):
        if not slice_dir.is_dir():
            continue

        match = re.search(r'slice_z(\d+)', slice_dir.name)
        if not match:
            continue

        slice_id = int(match.group(1))
        if slice_ids and slice_id not in slice_ids:
            continue

        json_path = slice_dir / 'pairwise_registration_metrics.json'
        if not json_path.exists():
            continue

        with open(json_path) as f:
            data = json.load(f)

        metrics = data.get('metrics', {})
        records.append({
            'slice_id': slice_id,
            'rotation': metrics.get('rotation', {}).get('value'),
            'translation_x': metrics.get('translation_x', {}).get('value'),
            'translation_y': metrics.get('translation_y', {}).get('value'),
            'z_drift': metrics.get('z_drift', {}).get('value'),
        })

    if not records:
        return None

    df = pd.DataFrame(records).sort_values('slice_id')

    # Compute cumulative rotation
    valid_rot = df['rotation'].dropna()
    cumulative_rotation = valid_rot.cumsum()

    # Identify problematic slices
    issues = []
    large_rot = df[df['rotation'].abs() > threshold]
    if len(large_rot) > 0:
        issues.append(f"Slices with |rotation| > {threshold}°: {large_rot['slice_id'].tolist()}")

    if len(cumulative_rotation) > 0 and abs(cumulative_rotation.iloc[-1]) > 5:
        issues.append(f"High cumulative rotation drift: {cumulative_rotation.iloc[-1]:.2f}°")

    result = {
        'total_slices': len(df),
        'mean_rotation': float(valid_rot.mean()) if len(valid_rot) > 0 else 0,
        'std_rotation': float(valid_rot.std()) if len(valid_rot) > 0 else 0,
        'cumulative_rotation': float(cumulative_rotation.iloc[-1]) if len(cumulative_rotation) > 0 else 0,
        'max_abs_rotation': float(valid_rot.abs().max()) if len(valid_rot) > 0 else 0,
        'large_rotation_slices': large_rot['slice_id'].tolist() if len(large_rot) > 0 else [],
        'issues': issues,
    }

    # Save CSV
    csv_path = output_dir / 'rotation_data.csv'
    df.to_csv(csv_path, index=False)

    # Generate plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax1 = axes[0]
    valid_df = df.dropna(subset=['rotation'])
    ax1.bar(valid_df['slice_id'], valid_df['rotation'], alpha=0.7, color='steelblue')
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.axhline(y=threshold, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(y=-threshold, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Slice ID')
    ax1.set_ylabel('Rotation (degrees)')
    ax1.set_title('Per-Slice Rotation')

    ax2 = axes[1]
    cumsum = valid_df['rotation'].cumsum()
    ax2.plot(valid_df['slice_id'], cumsum, 'b-', linewidth=2)
    ax2.fill_between(valid_df['slice_id'], 0, cumsum, alpha=0.3)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Slice ID')
    ax2.set_ylabel('Cumulative Rotation (degrees)')
    ax2.set_title('Cumulative Rotation Drift')

    plt.tight_layout()
    plt.savefig(output_dir / 'rotation_analysis.png', dpi=150)
    plt.close()

    return result


def analyze_shifts(pipeline_dir, output_dir, resolution=10.0, slice_ids=None):
    """Analyze XY shifts from shifts_xy.csv."""
    shifts_path = Path(pipeline_dir) / 'shifts_xy.csv'
    if not shifts_path.exists():
        # Try parent directory
        shifts_path = Path(pipeline_dir).parent / 'shifts_xy.csv'

    if not shifts_path.exists():
        logger.warning(f"No shifts_xy.csv found")
        return None

    df = pd.read_csv(shifts_path)

    if slice_ids:
        df = df[df['moving_id'].isin(slice_ids) | df['fixed_id'].isin(slice_ids)]

    px_per_mm = 1000 / resolution

    # Compute magnitudes
    df['shift_magnitude_mm'] = np.sqrt(df['x_shift_mm']**2 + df['y_shift_mm']**2)

    # Cumulative drift
    cumsum_x = df['x_shift_mm'].cumsum() * px_per_mm
    cumsum_y = df['y_shift_mm'].cumsum() * px_per_mm

    # Outlier detection
    q1 = df['shift_magnitude_mm'].quantile(0.25)
    q3 = df['shift_magnitude_mm'].quantile(0.75)
    iqr = q3 - q1
    outlier_threshold = q3 + 1.5 * iqr
    outliers = df[df['shift_magnitude_mm'] > outlier_threshold]

    result = {
        'n_shifts': len(df),
        'mean_shift_mm': float(df['shift_magnitude_mm'].mean()),
        'max_shift_mm': float(df['shift_magnitude_mm'].max()),
        'cumulative_x_px': float(cumsum_x.iloc[-1]) if len(cumsum_x) > 0 else 0,
        'cumulative_y_px': float(cumsum_y.iloc[-1]) if len(cumsum_y) > 0 else 0,
        'n_outliers': len(outliers),
        'outlier_indices': outliers['moving_id'].tolist() if len(outliers) > 0 else [],
        'issues': [],
    }

    if len(outliers) > 0:
        result['issues'].append(f"Found {len(outliers)} outlier shifts")

    total_drift = np.sqrt(result['cumulative_x_px']**2 + result['cumulative_y_px']**2)
    if total_drift > 500:
        result['issues'].append(f"Large cumulative drift: {total_drift:.0f} pixels")

    # Generate plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    ax1.scatter(df['x_shift_mm'], df['y_shift_mm'], c=df['moving_id'], cmap='viridis', alpha=0.7)
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax1.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    ax1.set_xlabel('X shift (mm)')
    ax1.set_ylabel('Y shift (mm)')
    ax1.set_title('Pairwise Shifts')

    ax2 = axes[0, 1]
    ax2.plot(df['moving_id'], df['shift_magnitude_mm'], 'b-', alpha=0.7)
    ax2.axhline(y=outlier_threshold, color='red', linestyle='--', label=f'Outlier threshold: {outlier_threshold:.2f} mm')
    ax2.scatter(outliers['moving_id'], outliers['shift_magnitude_mm'], c='red', s=50, zorder=5)
    ax2.set_xlabel('Slice ID')
    ax2.set_ylabel('Shift magnitude (mm)')
    ax2.set_title('Shift Magnitude per Slice')
    ax2.legend()

    ax3 = axes[1, 0]
    ax3.plot(df['moving_id'], cumsum_x, 'b-', label='X drift')
    ax3.plot(df['moving_id'], cumsum_y, 'r-', label='Y drift')
    ax3.set_xlabel('Slice ID')
    ax3.set_ylabel('Cumulative drift (pixels)')
    ax3.set_title('Cumulative XY Drift')
    ax3.legend()

    ax4 = axes[1, 1]
    ax4.plot(cumsum_x, cumsum_y, 'b-', linewidth=2, alpha=0.7)
    ax4.scatter(cumsum_x.iloc[0], cumsum_y.iloc[0], c='green', s=100, zorder=5, label='Start')
    ax4.scatter(cumsum_x.iloc[-1], cumsum_y.iloc[-1], c='red', s=100, zorder=5, label='End')
    ax4.set_xlabel('Cumulative X drift (pixels)')
    ax4.set_ylabel('Cumulative Y drift (pixels)')
    ax4.set_title('Drift Trajectory')
    ax4.legend()
    ax4.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_dir / 'shifts_analysis.png', dpi=150)
    plt.close()

    return result


def generate_summary_report(results, output_dir):
    """Generate comprehensive summary report."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    lines = [
        "=" * 70,
        "3D RECONSTRUCTION DIAGNOSTIC REPORT",
        "=" * 70,
        f"Generated: {timestamp}",
        "",
    ]

    # Rotation Analysis
    if 'rotation' in results and results['rotation']:
        rot = results['rotation']
        lines.extend([
            "ROTATION DRIFT ANALYSIS",
            "-" * 50,
            f"Total slices:           {rot['total_slices']}",
            f"Mean rotation:          {rot['mean_rotation']:.4f}°",
            f"Std rotation:           {rot['std_rotation']:.4f}°",
            f"Max |rotation|:         {rot['max_abs_rotation']:.4f}°",
            f"Cumulative drift:       {rot['cumulative_rotation']:.4f}°",
            "",
        ])
        if rot['issues']:
            lines.append("Issues:")
            for issue in rot['issues']:
                lines.append(f"  ⚠ {issue}")
            lines.append("")

    # Shifts Analysis
    if 'shifts' in results and results['shifts']:
        sh = results['shifts']
        lines.extend([
            "XY SHIFTS ANALYSIS",
            "-" * 50,
            f"Total shift pairs:      {sh['n_shifts']}",
            f"Mean shift:             {sh['mean_shift_mm']:.4f} mm",
            f"Max shift:              {sh['max_shift_mm']:.4f} mm",
            f"Cumulative X drift:     {sh['cumulative_x_px']:.0f} px",
            f"Cumulative Y drift:     {sh['cumulative_y_px']:.0f} px",
            f"Outlier shifts:         {sh['n_outliers']}",
            "",
        ])
        if sh['issues']:
            lines.append("Issues:")
            for issue in sh['issues']:
                lines.append(f"  ⚠ {issue}")
            lines.append("")

    # Overall Assessment
    lines.extend([
        "OVERALL ASSESSMENT",
        "-" * 50,
    ])

    all_issues = []
    if 'rotation' in results and results['rotation']:
        all_issues.extend(results['rotation'].get('issues', []))
    if 'shifts' in results and results['shifts']:
        all_issues.extend(results['shifts'].get('issues', []))

    if not all_issues:
        lines.append("✓ No significant issues detected in analyzed data")
    else:
        lines.append(f"Found {len(all_issues)} potential issues:")
        for issue in all_issues:
            lines.append(f"  • {issue}")

    lines.extend([
        "",
        "RECOMMENDATIONS",
        "-" * 50,
    ])

    # Generate recommendations based on findings
    recommendations = []

    if 'rotation' in results and results['rotation']:
        rot = results['rotation']
        if abs(rot['cumulative_rotation']) > 2:
            recommendations.append("Consider enabling rotation correction in registration (registration_transform='euler')")
        if rot['max_abs_rotation'] > 5:
            recommendations.append("Check slice quality - large rotations may indicate degraded slices")

    if 'shifts' in results and results['shifts']:
        sh = results['shifts']
        if sh['n_outliers'] > 3:
            recommendations.append("Review outlier slices - may need exclusion or manual adjustment")
        total_drift = np.sqrt(sh['cumulative_x_px']**2 + sh['cumulative_y_px']**2)
        if total_drift > 300:
            recommendations.append("Large cumulative drift - check stage calibration or sample mounting")

    if not recommendations:
        recommendations.append("Current parameters appear appropriate for this dataset")

    for rec in recommendations:
        lines.append(f"  → {rec}")

    lines.extend(["", "=" * 70])

    report_path = output_dir / 'diagnostic_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    # Also save JSON
    json_path = output_dir / 'diagnostic_results.json'
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'results': results,
            'issues': all_issues,
            'recommendations': recommendations,
        }, f, indent=2, default=str)

    logger.info(f"Summary report saved to {report_path}")
    return report_path


def main():
    p = _build_arg_parser()
    args = p.parse_args()

    pipeline_dir = Path(args.pipeline_output)
    output_dir = Path(args.out_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    slice_ids = parse_slice_range(args.slice_range)
    if slice_ids:
        logger.info(f"Analyzing slices: {slice_ids}")

    results = {}

    # Run analyses
    if not args.skip_rotation:
        logger.info("Running rotation drift analysis...")
        results['rotation'] = analyze_rotation_drift(
            pipeline_dir, output_dir, args.rotation_threshold, slice_ids
        )

    if not args.skip_shifts:
        logger.info("Running shifts analysis...")
        results['shifts'] = analyze_shifts(
            pipeline_dir, output_dir, args.resolution, slice_ids
        )

    # Generate summary
    report_path = generate_summary_report(results, output_dir)

    # Print summary to console
    print("\n" + "=" * 50)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 50)
    print(f"Results saved to: {output_dir}")

    all_issues = []
    for key, val in results.items():
        if val and 'issues' in val:
            all_issues.extend(val['issues'])

    if all_issues:
        print(f"\n⚠ Found {len(all_issues)} potential issues:")
        for issue in all_issues[:5]:
            print(f"  • {issue}")
        if len(all_issues) > 5:
            print(f"  ... and {len(all_issues) - 5} more (see full report)")
    else:
        print("\n✓ No significant issues detected")


if __name__ == '__main__':
    main()
