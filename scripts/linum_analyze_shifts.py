#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze XY shifts from a shifts file and generate a drift analysis report.

Produces:
- Summary statistics of pairwise shifts
- Outlier detection using IQR method
- Cumulative drift analysis
- Visualization of drift patterns

Useful for debugging alignment issues and understanding sample drift during acquisition.
"""
import linumpy._thread_config  # noqa: F401

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from linumpy.utils.io import add_overwrite_arg, assert_output_exists

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_shifts', help='Input shifts CSV file (shifts_xy.csv)')
    p.add_argument('out_directory', help='Output directory for analysis results')

    p.add_argument('--resolution', type=float, default=10.0,
                   help='Resolution in µm/pixel for converting mm to pixels [%(default)s]')
    p.add_argument('--iqr_multiplier', type=float, default=1.5,
                   help='IQR multiplier for outlier detection [%(default)s]')
    p.add_argument('--slice_config', default=None,
                   help='Optional slice config file to mark excluded slices')

    add_overwrite_arg(p)
    return p


def load_shifts(shifts_path):
    """Load shifts CSV file."""
    df = pd.read_csv(shifts_path)
    required_cols = ['fixed_id', 'moving_id', 'x_shift_mm', 'y_shift_mm']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    return df


def detect_outliers(df, iqr_multiplier=1.5):
    """Detect outliers using IQR method on shift magnitude."""
    shift_mag = np.sqrt(df['x_shift_mm']**2 + df['y_shift_mm']**2)
    q1 = shift_mag.quantile(0.25)
    q3 = shift_mag.quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + iqr_multiplier * iqr
    outlier_mask = shift_mag > upper_bound
    return outlier_mask, upper_bound, q1, q3, iqr


def filter_with_local_median(df, outlier_mask):
    """Replace outliers with local median of neighbors."""
    df_filtered = df.copy()
    for idx in df[outlier_mask].index:
        pos = df.index.get_loc(idx)
        neighbors_x, neighbors_y = [], []
        for offset in [-2, -1, 1, 2]:
            neighbor_pos = pos + offset
            if 0 <= neighbor_pos < len(df):
                neighbor_idx = df.index[neighbor_pos]
                if not outlier_mask[neighbor_idx]:
                    neighbors_x.append(df.loc[neighbor_idx, 'x_shift_mm'])
                    neighbors_y.append(df.loc[neighbor_idx, 'y_shift_mm'])
        if neighbors_x:
            df_filtered.loc[idx, 'x_shift_mm'] = np.median(neighbors_x)
            df_filtered.loc[idx, 'y_shift_mm'] = np.median(neighbors_y)
    return df_filtered


def generate_report(df, df_filtered, outlier_mask, stats, resolution, output_dir):
    """Generate text report."""
    px_per_mm = 1000 / resolution

    report_lines = [
        "=" * 60,
        "SHIFTS ANALYSIS REPORT",
        "=" * 60,
        "",
        "OVERVIEW",
        "-" * 40,
        f"Total shift pairs: {len(df)}",
        f"Resolution: {resolution} µm/pixel",
        "",
        "PAIRWISE SHIFT STATISTICS",
        "-" * 40,
        f"X shift (mm): Mean={df['x_shift_mm'].mean():.4f}, Std={df['x_shift_mm'].std():.4f}",
        f"Y shift (mm): Mean={df['y_shift_mm'].mean():.4f}, Std={df['y_shift_mm'].std():.4f}",
        "",
        "OUTLIER DETECTION (IQR Method)",
        "-" * 40,
        f"Q1={stats['q1']:.4f}, Q3={stats['q3']:.4f}, IQR={stats['iqr']:.4f}",
        f"Upper bound: {stats['upper_bound']:.4f} mm",
        f"Outliers detected: {outlier_mask.sum()}",
    ]

    if outlier_mask.sum() > 0:
        report_lines.append("")
        report_lines.append("Outlier shifts:")
        shift_mag = np.sqrt(df['x_shift_mm']**2 + df['y_shift_mm']**2)
        for idx in df[outlier_mask].index:
            row = df.loc[idx]
            mag = shift_mag[idx]
            report_lines.append(
                f"  {int(row['fixed_id'])}->{int(row['moving_id'])}: "
                f"({row['x_shift_mm']:.3f}, {row['y_shift_mm']:.3f}) mm, mag={mag:.3f} mm"
            )

    # Cumulative drift
    cumsum_x_orig = df['x_shift_mm'].cumsum()
    cumsum_y_orig = df['y_shift_mm'].cumsum()
    cumsum_x_filt = df_filtered['x_shift_mm'].cumsum()
    cumsum_y_filt = df_filtered['y_shift_mm'].cumsum()

    report_lines.extend([
        "",
        "CUMULATIVE DRIFT",
        "-" * 40,
        f"Before filtering: X={cumsum_x_orig.iloc[-1]:.3f} mm, Y={cumsum_y_orig.iloc[-1]:.3f} mm",
        f"After filtering:  X={cumsum_x_filt.iloc[-1]:.3f} mm, Y={cumsum_y_filt.iloc[-1]:.3f} mm",
        "",
        f"In pixels (at {resolution} µm/pixel):",
        f"  Before: X={cumsum_x_orig.iloc[-1] * px_per_mm:.0f} px, Y={cumsum_y_orig.iloc[-1] * px_per_mm:.0f} px",
        f"  After:  X={cumsum_x_filt.iloc[-1] * px_per_mm:.0f} px, Y={cumsum_y_filt.iloc[-1] * px_per_mm:.0f} px",
    ])

    # Centered drift
    mid_idx = len(cumsum_x_filt) // 2
    centered_x = cumsum_x_filt - cumsum_x_filt.iloc[mid_idx]
    centered_y = cumsum_y_filt - cumsum_y_filt.iloc[mid_idx]

    report_lines.extend([
        "",
        f"CENTERED DRIFT (around slice {mid_idx})",
        "-" * 40,
        f"X range: {centered_x.min() * px_per_mm:.0f} to {centered_x.max() * px_per_mm:.0f} px",
        f"Y range: {centered_y.min() * px_per_mm:.0f} to {centered_y.max() * px_per_mm:.0f} px",
        "",
        "=" * 60,
    ])

    report_text = "\n".join(report_lines)

    # Save report
    report_path = os.path.join(output_dir, 'shifts_analysis.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)

    return report_text


def generate_plots(df, df_filtered, outlier_mask, stats, resolution, output_dir):
    """Generate visualization plots."""
    px_per_mm = 1000 / resolution
    upper_bound = stats['upper_bound']

    # Calculate cumulative drift
    cumsum_x_orig = df['x_shift_mm'].cumsum()
    cumsum_y_orig = df['y_shift_mm'].cumsum()
    cumsum_x_filt = df_filtered['x_shift_mm'].cumsum()
    cumsum_y_filt = df_filtered['y_shift_mm'].cumsum()

    mid_idx = len(cumsum_x_filt) // 2
    centered_x = cumsum_x_filt - cumsum_x_filt.iloc[mid_idx]
    centered_y = cumsum_y_filt - cumsum_y_filt.iloc[mid_idx]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Pairwise shifts
    ax = axes[0, 0]
    ax.plot(df['moving_id'], df['x_shift_mm'], 'b.-', label='X shift (original)', alpha=0.7)
    ax.plot(df['moving_id'], df['y_shift_mm'], 'r.-', label='Y shift (original)', alpha=0.7)
    ax.plot(df['moving_id'], df_filtered['x_shift_mm'], 'b-', label='X shift (filtered)', linewidth=2)
    ax.plot(df['moving_id'], df_filtered['y_shift_mm'], 'r-', label='Y shift (filtered)', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axhline(y=upper_bound, color='g', linestyle=':', label=f'IQR threshold ({upper_bound:.2f}mm)')
    ax.axhline(y=-upper_bound, color='g', linestyle=':')
    ax.set_xlabel('Slice ID')
    ax.set_ylabel('Shift (mm)')
    ax.set_title('Pairwise Shifts')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Cumulative drift
    ax = axes[0, 1]
    ax.plot(df['moving_id'], cumsum_x_orig, 'b--', label='X original', alpha=0.5)
    ax.plot(df['moving_id'], cumsum_y_orig, 'r--', label='Y original', alpha=0.5)
    ax.plot(df['moving_id'], cumsum_x_filt, 'b-', label='X filtered', linewidth=2)
    ax.plot(df['moving_id'], cumsum_y_filt, 'r-', label='Y filtered', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Slice ID')
    ax.set_ylabel('Cumulative Drift (mm)')
    ax.set_title('Cumulative Drift')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Centered cumulative drift in pixels
    ax = axes[1, 0]
    ax.plot(df['moving_id'], centered_x * px_per_mm, 'b-', label='X (centered)', linewidth=2)
    ax.plot(df['moving_id'], centered_y * px_per_mm, 'r-', label='Y (centered)', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Slice ID')
    ax.set_ylabel(f'Centered Drift (pixels at {resolution}µm)')
    ax.set_title('Centered Cumulative Drift')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Drift trajectory
    ax = axes[1, 1]
    ax.plot(cumsum_x_filt * px_per_mm, cumsum_y_filt * px_per_mm, 'g-', linewidth=2)
    ax.plot(cumsum_x_filt.iloc[0] * px_per_mm, cumsum_y_filt.iloc[0] * px_per_mm,
            'go', markersize=10, label='Start')
    ax.plot(cumsum_x_filt.iloc[-1] * px_per_mm, cumsum_y_filt.iloc[-1] * px_per_mm,
            'ro', markersize=10, label='End')
    ax.plot(cumsum_x_filt.iloc[mid_idx] * px_per_mm, cumsum_y_filt.iloc[mid_idx] * px_per_mm,
            'ko', markersize=10, label='Middle')
    ax.set_xlabel('X position (pixels)')
    ax.set_ylabel('Y position (pixels)')
    ax.set_title('Drift Trajectory (filtered)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, 'drift_analysis.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved plot: {plot_path}")
    return plot_path


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Create output directory
    assert_output_exists(args.out_directory, parser, args)
    os.makedirs(args.out_directory)

    # Load shifts
    logger.info(f"Loading shifts from {args.in_shifts}")
    df = load_shifts(args.in_shifts)
    logger.info(f"Loaded {len(df)} shift pairs")

    # Detect outliers
    outlier_mask, upper_bound, q1, q3, iqr = detect_outliers(df, args.iqr_multiplier)
    logger.info(f"Detected {outlier_mask.sum()} outliers (IQR bound: {upper_bound:.3f} mm)")

    # Filter outliers
    df_filtered = filter_with_local_median(df, outlier_mask)

    # Statistics
    stats = {
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'upper_bound': upper_bound
    }

    # Generate report
    report = generate_report(df, df_filtered, outlier_mask, stats, args.resolution, args.out_directory)
    print(report)

    # Generate plots
    generate_plots(df, df_filtered, outlier_mask, stats, args.resolution, args.out_directory)

    # Save filtered shifts (useful for debugging)
    filtered_path = os.path.join(args.out_directory, 'shifts_filtered.csv')
    df_filtered.to_csv(filtered_path, index=False)
    logger.info(f"Saved filtered shifts: {filtered_path}")

    logger.info(f"Analysis complete. Results saved to {args.out_directory}")


if __name__ == '__main__':
    main()
