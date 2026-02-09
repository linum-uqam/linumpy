#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate dilation analysis results from multiple slices.

This script reads dilation analysis JSON files from multiple slices and
computes summary statistics and recommended correction factors for the
3D reconstruction pipeline.

Outputs:
- Summary statistics across all slices
- Recommended global scale correction factors
- Per-slice correction factors (for advanced use)
- Visualization of scale variation across slices
"""
import linumpy._thread_config  # noqa: F401

import argparse
import json
import logging
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
    p.add_argument('input_directory',
                   help='Directory containing per-slice dilation analysis results.\n'
                        'Expected structure: input_dir/{slice_id}/dilation_analysis/dilation_analysis.json')
    p.add_argument('output_directory',
                   help='Output directory for aggregated results')

    p.add_argument('--pattern', type=str, default='*/dilation_analysis/dilation_analysis.json',
                   help='Glob pattern to find JSON files [%(default)s]')
    p.add_argument('--target_scale', type=float, default=1.0,
                   help='Target scale factor (default 1.0 = motor positions are correct)')

    add_overwrite_arg(p)
    return p


def load_dilation_results(input_dir, pattern):
    """Load all dilation analysis JSON files from directory."""
    input_path = Path(input_dir)
    json_files = sorted(input_path.glob(pattern))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found with pattern: {pattern}")

    results = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            # Extract slice ID from path if not in data
            if data.get('slice_id') is None:
                # Try to extract from path
                parts = json_file.parts
                for part in parts:
                    if part.isdigit():
                        data['slice_id'] = part
                        break
            results.append(data)

    logger.info(f"Loaded {len(results)} dilation analysis results")
    return results


def compute_aggregate_statistics(results):
    """Compute aggregate statistics across all slices."""
    scale_y = [r['scale_factors']['scale_y'] for r in results]
    scale_x = [r['scale_factors']['scale_x'] for r in results]
    mean_scale = [r['scale_factors']['mean_scale'] for r in results]
    anisotropy = [r['scale_factors']['anisotropy'] for r in results]
    r_squared_y = [r['scale_factors']['r_squared_y'] for r in results]
    r_squared_x = [r['scale_factors']['r_squared_x'] for r in results]

    # Residuals
    mean_residual = [r['residuals']['mean_residual'] for r in results]
    max_residual = [r['residuals']['max_residual'] for r in results]

    # Distortions
    has_progressive = [r['distortions']['has_progressive_error'] for r in results]
    gradient_y = [r['distortions']['gradient_y'] for r in results]
    gradient_x = [r['distortions']['gradient_x'] for r in results]

    stats = {
        'n_slices': len(results),
        'scale_y': {
            'mean': float(np.mean(scale_y)),
            'std': float(np.std(scale_y)),
            'min': float(np.min(scale_y)),
            'max': float(np.max(scale_y)),
            'median': float(np.median(scale_y)),
        },
        'scale_x': {
            'mean': float(np.mean(scale_x)),
            'std': float(np.std(scale_x)),
            'min': float(np.min(scale_x)),
            'max': float(np.max(scale_x)),
            'median': float(np.median(scale_x)),
        },
        'mean_scale': {
            'mean': float(np.mean(mean_scale)),
            'std': float(np.std(mean_scale)),
            'min': float(np.min(mean_scale)),
            'max': float(np.max(mean_scale)),
            'median': float(np.median(mean_scale)),
        },
        'anisotropy': {
            'mean': float(np.mean(anisotropy)),
            'std': float(np.std(anisotropy)),
            'max': float(np.max(anisotropy)),
        },
        'fit_quality': {
            'mean_r2_y': float(np.mean(r_squared_y)),
            'mean_r2_x': float(np.mean(r_squared_x)),
            'min_r2': float(min(min(r_squared_y), min(r_squared_x))),
        },
        'residuals': {
            'mean': float(np.mean(mean_residual)),
            'max': float(np.max(max_residual)),
        },
        'progressive_error': {
            'n_slices_affected': sum(has_progressive),
            'mean_gradient_y': float(np.mean(gradient_y)),
            'mean_gradient_x': float(np.mean(gradient_x)),
        },
    }

    return stats


def compute_correction_factors(stats, target_scale=1.0):
    """Compute recommended correction factors."""
    # Use median for robustness against outliers
    correction_y = target_scale / stats['scale_y']['median']
    correction_x = target_scale / stats['scale_x']['median']

    # Alternative: use mean
    correction_y_mean = target_scale / stats['scale_y']['mean']
    correction_x_mean = target_scale / stats['scale_x']['mean']

    return {
        'recommended': {
            'scale_y': float(correction_y),
            'scale_x': float(correction_x),
            'description': 'Based on median scale factors (robust to outliers)',
        },
        'alternative_mean': {
            'scale_y': float(correction_y_mean),
            'scale_x': float(correction_x_mean),
            'description': 'Based on mean scale factors',
        },
        'deviation_from_unity': {
            'y_percent': float((1.0 - stats['scale_y']['median']) * 100),
            'x_percent': float((1.0 - stats['scale_x']['median']) * 100),
            'description': 'How much smaller mosaics are vs expected (%)',
        },
    }


def compute_per_slice_factors(results, target_scale=1.0):
    """Compute per-slice correction factors for advanced use."""
    per_slice = []
    for r in results:
        slice_id = r.get('slice_id', 'unknown')
        scale_y = r['scale_factors']['scale_y']
        scale_x = r['scale_factors']['scale_x']

        per_slice.append({
            'slice_id': slice_id,
            'measured_scale_y': float(scale_y),
            'measured_scale_x': float(scale_x),
            'correction_y': float(target_scale / scale_y),
            'correction_x': float(target_scale / scale_x),
            'r_squared_y': float(r['scale_factors']['r_squared_y']),
            'r_squared_x': float(r['scale_factors']['r_squared_x']),
        })

    return per_slice


def generate_report(stats, corrections, per_slice, output_dir):
    """Generate text report."""
    lines = [
        "=" * 70,
        "AGGREGATED DILATION ANALYSIS REPORT",
        "=" * 70,
        "",
        f"Total slices analyzed: {stats['n_slices']}",
        "",
        "SCALE FACTOR SUMMARY",
        "-" * 50,
        "",
        "Y-direction (rows):",
        f"  Mean:   {stats['scale_y']['mean']:.6f}",
        f"  Median: {stats['scale_y']['median']:.6f}",
        f"  Std:    {stats['scale_y']['std']:.6f}",
        f"  Range:  [{stats['scale_y']['min']:.6f}, {stats['scale_y']['max']:.6f}]",
        "",
        "X-direction (columns):",
        f"  Mean:   {stats['scale_x']['mean']:.6f}",
        f"  Median: {stats['scale_x']['median']:.6f}",
        f"  Std:    {stats['scale_x']['std']:.6f}",
        f"  Range:  [{stats['scale_x']['min']:.6f}, {stats['scale_x']['max']:.6f}]",
        "",
        "Overall:",
        f"  Mean scale:     {stats['mean_scale']['mean']:.6f}",
        f"  Mean anisotropy: {stats['anisotropy']['mean']:.6f}",
        "",
        "INTERPRETATION",
        "-" * 50,
    ]

    # Interpretation
    dev_y = corrections['deviation_from_unity']['y_percent']
    dev_x = corrections['deviation_from_unity']['x_percent']

    if abs(dev_y) > 1.0 or abs(dev_x) > 1.0:
        lines.append(f"⚠ SIGNIFICANT DILATION DETECTED")
        lines.append(f"  Y-direction: {abs(dev_y):.2f}% {'contraction' if dev_y > 0 else 'expansion'}")
        lines.append(f"  X-direction: {abs(dev_x):.2f}% {'contraction' if dev_x > 0 else 'expansion'}")
        lines.append("")
        lines.append("  This will cause edge misalignment in 3D reconstruction.")
        lines.append("  Apply the recommended correction factors below.")
    else:
        lines.append("✓ Scale factors close to 1.0 - minimal dilation detected")

    if stats['anisotropy']['mean'] > 0.01:
        lines.append("")
        lines.append(f"⚠ ANISOTROPIC SCALING: X and Y scales differ by {stats['anisotropy']['mean']*100:.2f}%")
        lines.append("  Use different correction factors for X and Y directions.")

    lines.extend([
        "",
        "RECOMMENDED CORRECTION FACTORS",
        "-" * 50,
        f"Scale Y: {corrections['recommended']['scale_y']:.6f}",
        f"Scale X: {corrections['recommended']['scale_x']:.6f}",
        "",
        "Usage in pipeline:",
        f"  linum_align_mosaics_3d_from_shifts.py ... \\",
        f"    --scale_y {corrections['recommended']['scale_y']:.4f} \\",
        f"    --scale_x {corrections['recommended']['scale_x']:.4f}",
        "",
        "FIT QUALITY",
        "-" * 50,
        f"Mean R² (Y): {stats['fit_quality']['mean_r2_y']:.6f}",
        f"Mean R² (X): {stats['fit_quality']['mean_r2_x']:.6f}",
        f"Min R²:      {stats['fit_quality']['min_r2']:.6f}",
    ])

    if stats['fit_quality']['min_r2'] < 0.99:
        lines.append("⚠ Some slices have lower fit quality - check individual results")
    else:
        lines.append("✓ Good linear fit quality across all slices")

    lines.extend([
        "",
        "PROGRESSIVE ERROR",
        "-" * 50,
        f"Slices with progressive error: {stats['progressive_error']['n_slices_affected']}/{stats['n_slices']}",
        f"Mean gradient Y: {stats['progressive_error']['mean_gradient_y']:.4f} px/tile",
        f"Mean gradient X: {stats['progressive_error']['mean_gradient_x']:.4f} px/tile",
    ])

    if stats['progressive_error']['n_slices_affected'] > stats['n_slices'] // 2:
        lines.append("⚠ Progressive error is systematic - scale correction will help")

    lines.extend([
        "",
        "=" * 70,
    ])

    report_path = output_dir / 'aggregated_dilation_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    logger.info(f"Report saved to {report_path}")
    return report_path


def generate_plots(results, output_dir):
    """Generate visualization plots."""
    slice_ids = [str(r.get('slice_id', i)) for i, r in enumerate(results)]
    scale_y = [r['scale_factors']['scale_y'] for r in results]
    scale_x = [r['scale_factors']['scale_x'] for r in results]
    mean_scale = [r['scale_factors']['mean_scale'] for r in results]
    anisotropy = [r['scale_factors']['anisotropy'] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Scale factors across slices
    ax1 = axes[0, 0]
    x = range(len(slice_ids))
    ax1.plot(x, scale_y, 'o-', label='Scale Y', color='blue')
    ax1.plot(x, scale_x, 's-', label='Scale X', color='red')
    ax1.axhline(y=1.0, color='green', linestyle='--', label='Ideal (1.0)')
    ax1.axhline(y=np.mean(scale_y), color='blue', linestyle=':', alpha=0.5)
    ax1.axhline(y=np.mean(scale_x), color='red', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Slice')
    ax1.set_ylabel('Scale Factor')
    ax1.set_title('Scale Factors Across Slices')
    ax1.set_xticks(x)
    ax1.set_xticklabels(slice_ids, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Mean scale and anisotropy
    ax2 = axes[0, 1]
    ax2.bar(x, mean_scale, color='purple', alpha=0.7, label='Mean Scale')
    ax2.axhline(y=1.0, color='green', linestyle='--', label='Ideal (1.0)')
    ax2.axhline(y=np.mean(mean_scale), color='purple', linestyle=':', label=f'Average ({np.mean(mean_scale):.4f})')
    ax2.set_xlabel('Slice')
    ax2.set_ylabel('Mean Scale Factor')
    ax2.set_title('Mean Scale Factor per Slice')
    ax2.set_xticks(x)
    ax2.set_xticklabels(slice_ids, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Anisotropy
    ax3 = axes[1, 0]
    ax3.bar(x, anisotropy, color='orange', alpha=0.7)
    ax3.axhline(y=np.mean(anisotropy), color='red', linestyle='--',
                label=f'Mean ({np.mean(anisotropy):.4f})')
    ax3.set_xlabel('Slice')
    ax3.set_ylabel('Anisotropy (|Scale_Y - Scale_X|)')
    ax3.set_title('Scale Anisotropy per Slice')
    ax3.set_xticks(x)
    ax3.set_xticklabels(slice_ids, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Distribution of scale factors
    ax4 = axes[1, 1]
    ax4.hist(scale_y, bins=15, alpha=0.5, label='Scale Y', color='blue')
    ax4.hist(scale_x, bins=15, alpha=0.5, label='Scale X', color='red')
    ax4.axvline(x=1.0, color='green', linestyle='--', label='Ideal (1.0)')
    ax4.axvline(x=np.median(scale_y), color='blue', linestyle=':',
                label=f'Median Y ({np.median(scale_y):.4f})')
    ax4.axvline(x=np.median(scale_x), color='red', linestyle=':',
                label=f'Median X ({np.median(scale_x):.4f})')
    ax4.set_xlabel('Scale Factor')
    ax4.set_ylabel('Count')
    ax4.set_title('Distribution of Scale Factors')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig.suptitle('Aggregated Dilation Analysis', fontsize=14)
    plt.tight_layout()

    plot_path = output_dir / 'aggregated_dilation_analysis.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()

    logger.info(f"Plots saved to {plot_path}")
    return plot_path


def main():
    p = _build_arg_parser()
    args = p.parse_args()

    input_dir = Path(args.input_directory)
    output_dir = Path(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_dilation_results(input_dir, args.pattern)

    # Compute statistics
    stats = compute_aggregate_statistics(results)
    corrections = compute_correction_factors(stats, args.target_scale)
    per_slice = compute_per_slice_factors(results, args.target_scale)

    # Save JSON results
    output_data = {
        'statistics': stats,
        'corrections': corrections,
        'per_slice': per_slice,
    }

    json_path = output_dir / 'aggregated_dilation_analysis.json'
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"JSON saved to {json_path}")

    # Save per-slice CSV for easy import
    df = pd.DataFrame(per_slice)
    csv_path = output_dir / 'per_slice_correction_factors.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"CSV saved to {csv_path}")

    # Generate report and plots
    generate_report(stats, corrections, per_slice, output_dir)
    generate_plots(results, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Analyzed {stats['n_slices']} slices")
    print(f"\nMeasured scale factors:")
    print(f"  Y: {stats['scale_y']['median']:.4f} (median)")
    print(f"  X: {stats['scale_x']['median']:.4f} (median)")
    print(f"\nRecommended correction factors:")
    print(f"  Y: {corrections['recommended']['scale_y']:.4f}")
    print(f"  X: {corrections['recommended']['scale_x']:.4f}")
    print(f"\nDeviation from expected:")
    print(f"  Y: {corrections['deviation_from_unity']['y_percent']:.2f}% contraction")
    print(f"  X: {corrections['deviation_from_unity']['x_percent']:.2f}% contraction")


if __name__ == '__main__':
    main()
