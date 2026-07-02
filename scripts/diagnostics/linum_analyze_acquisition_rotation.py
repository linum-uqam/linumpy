#!/usr/bin/env python3
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

import linumpy.config.threads  # noqa: F401

import argparse
import json
import logging
from pathlib import Path

from linumpy.cli.args import add_overwrite_arg
from linumpy.diagnostics.acquisition_rotation import (
    analyze_acquisition_rotation,
    compare_with_registration,
    compute_angular_velocity,
    compute_cumulative_rotation,
    compute_shift_angles,
    detect_rotation_patterns,
    generate_plots,
    generate_report,
    load_registration_rotations,
    load_shifts,
)

__all__ = [
    "analyze_acquisition_rotation",
    "compare_with_registration",
    "compute_angular_velocity",
    "compute_cumulative_rotation",
    "compute_shift_angles",
    "detect_rotation_patterns",
    "generate_plots",
    "generate_report",
    "load_registration_rotations",
    "load_shifts",
    "main",
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("in_shifts", help="Input shifts CSV file (shifts_xy.csv)")
    p.add_argument("out_directory", help="Output directory for analysis results")

    p.add_argument("--resolution", type=float, default=10.0, help="Resolution in µm/pixel [%(default)s]")
    p.add_argument("--registration_dir", type=str, default=None, help="Path to register_pairwise directory for comparison")
    p.add_argument(
        "--expected_angle", type=float, default=None, help="Expected shift angle in degrees (e.g., 45 for oblique mount)"
    )
    p.add_argument("--window_size", type=int, default=5, help="Window size for local rotation estimation [%(default)s]")

    add_overwrite_arg(p)
    return p


def main() -> None:
    """Run function."""
    p = _build_arg_parser()
    args = p.parse_args()

    output_dir = Path(args.out_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load shifts
    logger.info("Loading shifts from %s", args.in_shifts)
    df = load_shifts(args.in_shifts)
    logger.info("Loaded %s shift pairs", len(df))

    # Main analysis
    analysis, angles, angular_velocity, cumulative_rotation = analyze_acquisition_rotation(
        df, expected_angle=args.expected_angle
    )

    # Load registration data if available
    reg_df = None
    if args.registration_dir:
        logger.info("Loading registration data from %s", args.registration_dir)
        reg_df = load_registration_rotations(args.registration_dir)
        if reg_df is not None:
            logger.info("Loaded registration data for %s slices", len(reg_df))

    # Compare with registration
    reg_comparison = compare_with_registration(cumulative_rotation, reg_df, df["moving_id"].values)

    # Save raw data
    output_df = df.copy()
    output_df["shift_angle"] = angles
    output_df["angular_velocity"] = angular_velocity
    output_df["cumulative_rotation"] = cumulative_rotation
    csv_path = output_dir / "acquisition_rotation_data.csv"
    output_df.to_csv(csv_path, index=False)
    logger.info("Data saved to %s", csv_path)

    # Save analysis JSON
    json_path = output_dir / "acquisition_rotation_analysis.json"
    with Path(json_path).open("w") as f:
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

    if analysis["patterns"]["systematic_drift"]:
        print(f"⚠ Systematic drift: {analysis['patterns']['drift_rate']:.3f}°/slice")
    if analysis["patterns"]["oscillation"]:
        print("⚠ Oscillation detected")
    if analysis["patterns"]["sudden_jumps"]:
        print(f"⚠ {len(analysis['patterns']['sudden_jumps'])} sudden rotation jumps")


if __name__ == "__main__":
    main()
