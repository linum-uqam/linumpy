#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze rotation patterns from acquisition XY shifts data.

This script examines the shifts_xy.csv file to detect rotation patterns that
occur during acquisition. By analyzing the direction of shift vectors across
slices, we can identify:

1. **Systematic angular drift**: Shift vectors rotating over time (stage drift)
2. **Oscillating rotation**: Back-and-forth rotation pattern (mechanical backlash)
3. **Sudden rotation jumps**: Sample movement during acquisition

The detected acquisition rotation can be compared with the final pairwise
registration rotation to assess how well the registration is compensating.

For obliquely-mounted samples (e.g., 45° from standard planes), the shift
vector direction should remain relatively constant if there's no rotation.
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
    p.add_argument('in_shifts',
                   help='Input shifts CSV file (shifts_xy.csv)')
    p.add_argument('out_directory',
                   help='Output directory for analysis results')

    p.add_argument('--resolution', type=float, default=10.0,
                   help='Resolution in µm/pixel [%(default)s]')
    p.add_argument('--registration_dir', type=str, default=None,
                   help='Path to register_pairwise directory for comparison')
    p.add_argument('--expected_angle', type=float, default=None,
                   help='Expected shift angle in degrees (e.g., 45 for oblique mount)')
    p.add_argument('--window_size', type=int, default=5,
                   help='Window size for local rotation estimation [%(default)s]')

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


def compute_shift_angles(df):
    """
    Compute the angle of each shift vector.

    Returns angles in degrees, where 0° = positive X direction,
    90° = positive Y direction, etc.
    """
    angles = np.degrees(np.arctan2(df['y_shift_mm'], df['x_shift_mm']))
    return angles


def compute_angular_velocity(angles, window_size=5):
    """
    Compute the rate of change of shift angle (angular velocity).

    Uses a rolling window to smooth the derivative.
    """
    # Handle angle wraparound (-180 to 180)
    angles_unwrapped = np.unwrap(np.radians(angles))
    angles_unwrapped = np.degrees(angles_unwrapped)

    # Compute derivative
    angular_velocity = np.gradient(angles_unwrapped)

    # Smooth with rolling window
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        angular_velocity_smooth = np.convolve(angular_velocity, kernel, mode='same')
    else:
        angular_velocity_smooth = angular_velocity

    return angular_velocity, angular_velocity_smooth


def compute_cumulative_rotation(angles):
    """
    Compute cumulative rotation from shift angle changes.

    This estimates how much rotation has accumulated from the first slice.
    """
    # Convert to numpy array if pandas Series
    angles_arr = np.asarray(angles)

    # Use unwrapped angles to handle wraparound
    angles_unwrapped = np.unwrap(np.radians(angles_arr))
    angles_unwrapped = np.degrees(angles_unwrapped)

    # Cumulative change from first angle
    cumulative = angles_unwrapped - angles_unwrapped[0]

    return pd.Series(cumulative)


def detect_rotation_patterns(angles, angular_velocity):
    """
    Detect different rotation patterns in the data.
    """
    patterns = {
        'systematic_drift': False,
        'oscillation': False,
        'sudden_jumps': [],
    }

    # Check for systematic drift (mean angular velocity significantly different from 0)
    mean_av = np.mean(angular_velocity)
    if abs(mean_av) > 0.5:  # More than 0.5 degrees per slice on average
        patterns['systematic_drift'] = True
        patterns['drift_rate'] = float(mean_av)

    # Check for oscillation (sign changes in angular velocity)
    sign_changes = np.sum(np.diff(np.sign(angular_velocity)) != 0)
    oscillation_ratio = sign_changes / len(angular_velocity)
    if oscillation_ratio > 0.4:  # More than 40% sign changes
        patterns['oscillation'] = True
        patterns['oscillation_frequency'] = float(oscillation_ratio)

    # Detect sudden jumps (angular velocity > 5 degrees in one step)
    jump_threshold = 5.0
    jumps = np.where(np.abs(angular_velocity) > jump_threshold)[0]
    if len(jumps) > 0:
        patterns['sudden_jumps'] = jumps.tolist()

    return patterns


def load_registration_rotations(reg_dir):
    """Load rotation values from pairwise registration metrics."""
    import re

    reg_path = Path(reg_dir)

    if not reg_path.exists():
        logger.warning(f"Registration directory does not exist: {reg_dir}")
        return None

    records = []

    # Try to find slice directories - either directly or in subdirectories
    slice_dirs = []

    # Check if there are slice_z directories directly
    for item in sorted(reg_path.iterdir()):
        if item.is_dir() and 'slice_z' in item.name:
            slice_dirs.append(item)

    # If no direct slice dirs, search recursively for JSON files
    if not slice_dirs:
        json_files = list(reg_path.glob('**/pairwise_registration_metrics.json'))
        slice_dirs = sorted(set(f.parent for f in json_files))

    if not slice_dirs:
        logger.warning(f"No slice directories found in {reg_dir}")
        return None

    for slice_dir in slice_dirs:
        match = re.search(r'slice_z(\d+)', slice_dir.name)
        if not match:
            continue

        slice_id = int(match.group(1))
        json_path = slice_dir / 'pairwise_registration_metrics.json'

        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            metrics = data.get('metrics', {})
            rotation = metrics.get('rotation', {}).get('value')
            records.append({'slice_id': slice_id, 'registration_rotation': rotation})

    if records:
        return pd.DataFrame(records).sort_values('slice_id')
    return None


def analyze_acquisition_rotation(df, expected_angle=None):
    """
    Main analysis of rotation from acquisition shifts.

    Note: The shift vectors represent relative displacement between slices,
    which can vary in direction due to drift. We analyze:
    1. Shift angle consistency (are shifts in a consistent direction?)
    2. Angular velocity (how much does the shift direction change?)
    3. Patterns that might indicate actual sample rotation
    """
    # Compute shift angles
    angles = compute_shift_angles(df)

    # Compute angular velocity
    angular_velocity, angular_velocity_smooth = compute_angular_velocity(angles)

    # Compute cumulative rotation
    cumulative_rotation = compute_cumulative_rotation(angles)

    # Detect patterns
    patterns = detect_rotation_patterns(angles, angular_velocity)

    # Compute shift magnitudes
    magnitudes = np.sqrt(df['x_shift_mm']**2 + df['y_shift_mm']**2)

    # Statistics
    analysis = {
        'n_shifts': len(df),
        'angle_stats': {
            'mean': float(angles.mean()),
            'std': float(angles.std()),
            'min': float(angles.min()),
            'max': float(angles.max()),
            'range': float(angles.max() - angles.min()),
        },
        'magnitude_stats': {
            'mean': float(magnitudes.mean()),
            'std': float(magnitudes.std()),
            'min': float(magnitudes.min()),
            'max': float(magnitudes.max()),
        },
        'cumulative_rotation': {
            'total': float(cumulative_rotation.iloc[-1]),
            'max_absolute': float(cumulative_rotation.abs().max()),
        },
        'patterns': patterns,
        'interpretation': {},
    }

    # Interpretation: High angle std with low magnitude std suggests drift direction changes
    # Low angle std with consistent magnitude suggests systematic drift in one direction
    angle_std = angles.std()
    mag_std = magnitudes.std() / magnitudes.mean() if magnitudes.mean() > 0 else 0

    if angle_std < 30:
        analysis['interpretation']['shift_consistency'] = 'consistent'
        analysis['interpretation']['shift_consistency_note'] = \
            'Shifts are in a consistent direction - sample position is drifting uniformly'
    elif angle_std < 90:
        analysis['interpretation']['shift_consistency'] = 'moderate'
        analysis['interpretation']['shift_consistency_note'] = \
            'Moderate variation in shift direction - some drift + possible rotation'
    else:
        analysis['interpretation']['shift_consistency'] = 'highly_variable'
        analysis['interpretation']['shift_consistency_note'] = \
            'Shift directions vary widely - significant drift pattern changes or sample movement'

    # Compare with expected angle if provided
    if expected_angle is not None:
        angle_deviation = angles.mean() - expected_angle
        analysis['expected_angle'] = expected_angle
        analysis['mean_deviation_from_expected'] = float(angle_deviation)

    return analysis, angles, angular_velocity_smooth, cumulative_rotation


def generate_report(analysis, reg_comparison, output_dir):
    """Generate text report."""
    lines = [
        "=" * 70,
        "ACQUISITION ROTATION ANALYSIS",
        "=" * 70,
        "",
        "This analysis examines the shift vectors between consecutive slices",
        "to detect rotation and drift patterns during acquisition.",
        "",
        "NOTE: 'Shift angle' is the direction of the position change between",
        "slices. Varying angles indicate changing drift direction, which may",
        "result from sample rotation OR complex drift patterns.",
        "",
        "SHIFT VECTOR STATISTICS",
        "-" * 50,
        f"Number of shifts:     {analysis['n_shifts']}",
        f"Mean shift angle:     {analysis['angle_stats']['mean']:.2f}°",
        f"Std deviation:        {analysis['angle_stats']['std']:.2f}°",
        f"Min angle:            {analysis['angle_stats']['min']:.2f}°",
        f"Max angle:            {analysis['angle_stats']['max']:.2f}°",
        f"Angle range:          {analysis['angle_stats']['range']:.2f}°",
        "",
        "SHIFT MAGNITUDE STATISTICS",
        "-" * 50,
        f"Mean magnitude:       {analysis['magnitude_stats']['mean']:.4f} mm",
        f"Std deviation:        {analysis['magnitude_stats']['std']:.4f} mm",
        f"Min magnitude:        {analysis['magnitude_stats']['min']:.4f} mm",
        f"Max magnitude:        {analysis['magnitude_stats']['max']:.4f} mm",
        "",
    ]

    if 'expected_angle' in analysis:
        lines.extend([
            "EXPECTED ANGLE COMPARISON",
            "-" * 50,
            f"Expected angle:       {analysis['expected_angle']:.1f}°",
            f"Deviation from expected: {analysis['mean_deviation_from_expected']:.2f}°",
            "",
        ])

    lines.extend([
        "CUMULATIVE ANGLE CHANGE",
        "-" * 50,
        f"Total angle change:   {analysis['cumulative_rotation']['total']:.2f}°",
        f"Max absolute:         {analysis['cumulative_rotation']['max_absolute']:.2f}°",
        "",
        "INTERPRETATION",
        "-" * 50,
        f"Shift consistency: {analysis['interpretation']['shift_consistency'].upper()}",
        f"  {analysis['interpretation']['shift_consistency_note']}",
        "",
        "PATTERN DETECTION",
        "-" * 50,
    ])

    patterns = analysis['patterns']
    if patterns['systematic_drift']:
        lines.append(f"⚠ Systematic angular drift: {patterns['drift_rate']:.3f}°/slice")
    if patterns['oscillation']:
        lines.append(f"⚠ Oscillation detected (frequency: {patterns['oscillation_frequency']:.2f})")
    if patterns['sudden_jumps']:
        n_jumps = len(patterns['sudden_jumps'])
        lines.append(f"⚠ {n_jumps} sudden direction changes (>5°/step)")
        if n_jumps <= 10:
            lines.append(f"   At slices: {patterns['sudden_jumps']}")
        else:
            lines.append(f"   First 10 at slices: {patterns['sudden_jumps'][:10]}...")
    if not (patterns['systematic_drift'] or patterns['oscillation'] or patterns['sudden_jumps']):
        lines.append("✓ No significant rotation patterns detected")

    # Registration comparison
    if reg_comparison is not None:
        lines.extend([
            "",
            "COMPARISON WITH REGISTRATION",
            "-" * 50,
        ])
        if 'acquisition_vs_registration' in reg_comparison:
            avr = reg_comparison['acquisition_vs_registration']
            lines.extend([
                f"Acquisition cumulative angle change: {avr['acquisition_cumulative']:.2f}°",
                f"Registration cumulative rotation: {avr['registration_cumulative']:.2f}°",
                f"Correlation: {avr['correlation']:.3f}",
            ])
            if abs(avr['correlation']) < 0.3:
                lines.append("→ Weak correlation: acquisition angle changes ≠ actual rotation")
            elif avr['correlation'] > 0.5:
                lines.append("→ Positive correlation: registration tracking drift patterns")
            elif avr['correlation'] < -0.5:
                lines.append("→ Negative correlation: registration may be compensating")

    lines.extend(["", "=" * 70])

    report_path = output_dir / 'acquisition_rotation_analysis.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    logger.info(f"Report saved to {report_path}")
    return report_path


def generate_plots(df, angles, angular_velocity, cumulative_rotation, reg_df, output_dir):
    """Generate visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    slice_ids = df['moving_id'].values

    # 1. Shift vectors (quiver plot)
    ax1 = axes[0, 0]
    # Normalize for visualization
    magnitudes = np.sqrt(df['x_shift_mm']**2 + df['y_shift_mm']**2)
    norm_x = df['x_shift_mm'] / magnitudes.max()
    norm_y = df['y_shift_mm'] / magnitudes.max()

    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
    for i in range(len(df)):
        ax1.arrow(0, 0, norm_x.iloc[i], norm_y.iloc[i],
                  head_width=0.02, head_length=0.01,
                  fc=colors[i], ec=colors[i], alpha=0.7)
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_aspect('equal')
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax1.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    ax1.set_xlabel('Normalized X shift')
    ax1.set_ylabel('Normalized Y shift')
    ax1.set_title('Shift Vector Directions\n(colored by slice order: purple→yellow)')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=slice_ids.min(), vmax=slice_ids.max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, label='Slice ID')

    # 2. Shift angle over slices
    ax2 = axes[0, 1]
    ax2.plot(slice_ids, angles, 'b-', linewidth=1, alpha=0.7, label='Raw angle')
    ax2.axhline(y=angles.mean(), color='red', linestyle='--', label=f'Mean: {angles.mean():.1f}°')
    ax2.fill_between(slice_ids, angles.mean() - angles.std(), angles.mean() + angles.std(),
                     alpha=0.2, color='red', label=f'±1 std: {angles.std():.1f}°')
    ax2.set_xlabel('Slice ID')
    ax2.set_ylabel('Shift Angle (degrees)')
    ax2.set_title('Shift Vector Angle vs Slice')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Angular velocity (rate of angle change)
    ax3 = axes[1, 0]
    ax3.plot(slice_ids, angular_velocity, 'g-', linewidth=1.5, label='Angular velocity (smoothed)')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.axhline(y=2, color='red', linestyle='--', alpha=0.5)
    ax3.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Slice ID')
    ax3.set_ylabel('Angular Velocity (°/slice)')
    ax3.set_title('Rate of Angle Change\n(rotation between consecutive slices)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Cumulative rotation comparison
    ax4 = axes[1, 1]
    ax4.plot(slice_ids, cumulative_rotation, 'b-', linewidth=2, label='Acquisition (from shifts)')

    if reg_df is not None and 'registration_rotation' in reg_df.columns:
        # Align registration data with shift data
        reg_cumulative = reg_df['registration_rotation'].cumsum()
        ax4.plot(reg_df['slice_id'], reg_cumulative, 'r-', linewidth=2,
                label='Registration (cumulative)')

    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Slice ID')
    ax4.set_ylabel('Cumulative Rotation (degrees)')
    ax4.set_title('Cumulative Rotation:\nAcquisition vs Registration')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'acquisition_rotation_analysis.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()

    logger.info(f"Plots saved to {plot_path}")
    return plot_path


def compare_with_registration(cumulative_rotation, reg_df, slice_ids):
    """Compare acquisition rotation with registration rotation."""
    if reg_df is None or len(reg_df) == 0:
        return None

    # Get registration cumulative rotation
    reg_cumulative = reg_df['registration_rotation'].fillna(0).cumsum()

    # Align indices
    common_slices = set(slice_ids) & set(reg_df['slice_id'])
    if len(common_slices) < 5:
        return None

    acq_values = []
    reg_values = []

    for sid in sorted(common_slices):
        acq_idx = np.where(slice_ids == sid)[0]
        reg_idx = reg_df[reg_df['slice_id'] == sid].index

        if len(acq_idx) > 0 and len(reg_idx) > 0:
            acq_values.append(cumulative_rotation.iloc[acq_idx[0]])
            reg_values.append(reg_cumulative.iloc[reg_idx[0]])

    if len(acq_values) < 5:
        return None

    correlation = np.corrcoef(acq_values, reg_values)[0, 1]

    return {
        'acquisition_vs_registration': {
            'acquisition_cumulative': float(acq_values[-1]),
            'registration_cumulative': float(reg_values[-1]),
            'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
        }
    }


def main():
    p = _build_arg_parser()
    args = p.parse_args()

    output_dir = Path(args.out_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load shifts
    logger.info(f"Loading shifts from {args.in_shifts}")
    df = load_shifts(args.in_shifts)
    logger.info(f"Loaded {len(df)} shift pairs")

    # Main analysis
    analysis, angles, angular_velocity, cumulative_rotation = analyze_acquisition_rotation(
        df, expected_angle=args.expected_angle
    )

    # Load registration data if available
    reg_df = None
    if args.registration_dir:
        logger.info(f"Loading registration data from {args.registration_dir}")
        reg_df = load_registration_rotations(args.registration_dir)
        if reg_df is not None:
            logger.info(f"Loaded registration data for {len(reg_df)} slices")

    # Compare with registration
    reg_comparison = compare_with_registration(
        cumulative_rotation, reg_df, df['moving_id'].values
    )

    # Save raw data
    output_df = df.copy()
    output_df['shift_angle'] = angles
    output_df['angular_velocity'] = angular_velocity
    output_df['cumulative_rotation'] = cumulative_rotation
    csv_path = output_dir / 'acquisition_rotation_data.csv'
    output_df.to_csv(csv_path, index=False)
    logger.info(f"Data saved to {csv_path}")

    # Save analysis JSON
    json_path = output_dir / 'acquisition_rotation_analysis.json'
    with open(json_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    # Generate outputs
    generate_report(analysis, reg_comparison, output_dir)
    generate_plots(df, angles, angular_velocity, cumulative_rotation, reg_df, output_dir)

    # Print summary
    print("\n" + "=" * 50)
    print("ACQUISITION ROTATION SUMMARY")
    print("=" * 50)
    print(f"Mean shift angle: {analysis['angle_stats']['mean']:.1f}° (std: {analysis['angle_stats']['std']:.1f}°)")
    print(f"Angle range: {analysis['angle_stats']['range']:.1f}°")
    print(f"Cumulative rotation: {analysis['cumulative_rotation']['total']:.2f}°")

    if analysis['patterns']['systematic_drift']:
        print(f"⚠ Systematic drift: {analysis['patterns']['drift_rate']:.3f}°/slice")
    if analysis['patterns']['oscillation']:
        print(f"⚠ Oscillation detected")
    if analysis['patterns']['sudden_jumps']:
        print(f"⚠ {len(analysis['patterns']['sudden_jumps'])} sudden rotation jumps")


if __name__ == '__main__':
    main()
