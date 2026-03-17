#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze tile dilation/scaling by comparing expected vs actual tile positions.

This diagnostic tool examines the relationship between motor positions and
registration-derived positions to detect:

1. **Global dilation**: Tiles spread more/less than expected (scale factor ≠ 1)
2. **Anisotropic scaling**: Different scale factors in X vs Y directions
3. **Progressive drift**: Error accumulating across the mosaic
4. **Local distortions**: Non-linear deformations in specific regions

For troubleshooting 3D reconstruction artifacts in serial OCT microscopy,
particularly for obliquely-cut samples where physical vs measured positions
may diverge due to tissue deformation after slicing.
"""
import linumpy._thread_config  # noqa: F401

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from linumpy.io.zarr import read_omezarr
from linumpy.utils.io import add_overwrite_arg

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('input_volume',
                   help='Path to mosaic grid volume (.ome.zarr)')
    p.add_argument('input_transform',
                   help='Path to registration transform (.npy)')
    p.add_argument('out_directory',
                   help='Output directory for analysis results')

    p.add_argument('--overlap_fraction', type=float, default=0.1,
                   help='Expected overlap fraction between tiles [%(default)s]')
    p.add_argument('--resolution', type=float, default=10.0,
                   help='Resolution in µm/pixel [%(default)s]')
    p.add_argument('--slice_id', type=str, default=None,
                   help='Slice identifier for labeling outputs')

    add_overwrite_arg(p)
    return p


def compute_expected_positions(nx, ny, tile_height, tile_width, overlap_fraction):
    """Compute expected tile positions based on motor grid."""
    step_y = tile_height * (1.0 - overlap_fraction)
    step_x = tile_width * (1.0 - overlap_fraction)

    positions = []
    for i in range(nx):
        for j in range(ny):
            positions.append((i * step_y, j * step_x))

    return np.array(positions)


def compute_actual_positions(nx, ny, transform):
    """Compute actual tile positions from registration transform."""
    positions = []
    for i in range(nx):
        for j in range(ny):
            pos = np.dot(transform, [i, j])
            positions.append(pos)

    return np.array(positions)


def estimate_scale_factors(expected, actual):
    """Estimate scale factors by comparing expected vs actual positions."""
    # Use linear regression to estimate scale factor
    # actual = scale * expected + offset

    from scipy import stats

    # Y direction (rows)
    slope_y, intercept_y, r_y, p_y, se_y = stats.linregress(expected[:, 0], actual[:, 0])

    # X direction (cols)
    slope_x, intercept_x, r_x, p_x, se_x = stats.linregress(expected[:, 1], actual[:, 1])

    return {
        'scale_y': slope_y,
        'scale_x': slope_x,
        'offset_y': intercept_y,
        'offset_x': intercept_x,
        'r_squared_y': r_y**2,
        'r_squared_x': r_x**2,
        'anisotropy': abs(slope_y - slope_x),
        'mean_scale': (slope_y + slope_x) / 2,
    }


def analyze_residuals(expected, actual, scale_factors):
    """Analyze residuals after removing estimated scale."""
    # Predicted positions using estimated scale
    predicted_y = scale_factors['scale_y'] * expected[:, 0] + scale_factors['offset_y']
    predicted_x = scale_factors['scale_x'] * expected[:, 1] + scale_factors['offset_x']

    # Residuals
    residual_y = actual[:, 0] - predicted_y
    residual_x = actual[:, 1] - predicted_x
    residual_mag = np.sqrt(residual_y**2 + residual_x**2)

    return {
        'residual_y': residual_y,
        'residual_x': residual_x,
        'residual_magnitude': residual_mag,
        'mean_residual': float(np.mean(residual_mag)),
        'max_residual': float(np.max(residual_mag)),
        'std_residual': float(np.std(residual_mag)),
    }


def detect_local_distortions(expected, actual, nx, ny):
    """Check for local distortions (non-linear deformations)."""
    diff = actual - expected

    # Reshape to grid
    diff_grid_y = diff[:, 0].reshape(nx, ny)
    diff_grid_x = diff[:, 1].reshape(nx, ny)

    # Check for gradient (progressive error)
    gradient_y = np.gradient(diff_grid_y, axis=0).mean()
    gradient_x = np.gradient(diff_grid_x, axis=1).mean()

    # Check for curvature (non-linear distortion)
    curvature_y = np.gradient(np.gradient(diff_grid_y, axis=0), axis=0).std()
    curvature_x = np.gradient(np.gradient(diff_grid_x, axis=1), axis=1).std()

    return {
        'gradient_y': float(gradient_y),
        'gradient_x': float(gradient_x),
        'curvature_y': float(curvature_y),
        'curvature_x': float(curvature_x),
        'has_progressive_error': bool(abs(gradient_y) > 0.5 or abs(gradient_x) > 0.5),
        'has_curvature': bool(curvature_y > 1.0 or curvature_x > 1.0),
    }


def generate_report(analysis, scale_factors, residuals, distortions, output_dir, slice_id=None):
    """Generate text report."""
    slice_label = f" (Slice {slice_id})" if slice_id else ""

    lines = [
        "=" * 70,
        f"TILE DILATION/SCALING ANALYSIS{slice_label}",
        "=" * 70,
        "",
        "SCALE FACTOR ANALYSIS",
        "-" * 50,
        f"Scale factor Y (rows):   {scale_factors['scale_y']:.6f}",
        f"Scale factor X (cols):   {scale_factors['scale_x']:.6f}",
        f"Mean scale factor:       {scale_factors['mean_scale']:.6f}",
        f"Anisotropy (|Sy - Sx|):  {scale_factors['anisotropy']:.6f}",
        f"Offset Y:                {scale_factors['offset_y']:.2f} px",
        f"Offset X:                {scale_factors['offset_x']:.2f} px",
        f"R² fit quality Y:        {scale_factors['r_squared_y']:.6f}",
        f"R² fit quality X:        {scale_factors['r_squared_x']:.6f}",
        "",
        "INTERPRETATION",
        "-" * 50,
    ]

    # Interpret scale factors
    scale_deviation = abs(scale_factors['mean_scale'] - 1.0)
    if scale_deviation < 0.001:
        lines.append("✓ Scale factor ~1.0: No significant dilation detected")
    elif scale_factors['mean_scale'] > 1.0:
        lines.append(f"⚠ Scale > 1.0: Tiles spread MORE than expected ({scale_deviation*100:.2f}% expansion)")
        lines.append("  → Possible cause: Tissue relaxation/expansion after cutting")
    else:
        lines.append(f"⚠ Scale < 1.0: Tiles spread LESS than expected ({scale_deviation*100:.2f}% contraction)")
        lines.append("  → Possible cause: Stage calibration error or tissue shrinkage")

    if scale_factors['anisotropy'] > 0.005:
        lines.append(f"⚠ Anisotropic scaling detected: X/Y scales differ by {scale_factors['anisotropy']:.4f}")
        lines.append("  → May cause edge misalignment in 3D reconstruction")

    lines.extend([
        "",
        "RESIDUAL ANALYSIS (after scale correction)",
        "-" * 50,
        f"Mean residual:  {residuals['mean_residual']:.2f} px",
        f"Max residual:   {residuals['max_residual']:.2f} px",
        f"Std residual:   {residuals['std_residual']:.2f} px",
    ])

    lines.extend([
        "",
        "LOCAL DISTORTION ANALYSIS",
        "-" * 50,
        f"Progressive error (gradient Y): {distortions['gradient_y']:.4f} px/tile",
        f"Progressive error (gradient X): {distortions['gradient_x']:.4f} px/tile",
        f"Non-linearity (curvature Y):    {distortions['curvature_y']:.4f}",
        f"Non-linearity (curvature X):    {distortions['curvature_x']:.4f}",
    ])

    if distortions['has_progressive_error']:
        lines.append("⚠ Progressive error detected: Position error grows across mosaic")
    if distortions['has_curvature']:
        lines.append("⚠ Non-linear distortion detected: Local deformations present")

    lines.extend([
        "",
        "=" * 70,
    ])

    report_path = output_dir / 'dilation_analysis.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    logger.info(f"Report saved to {report_path}")
    return report_path


def generate_plots(expected, actual, residuals, nx, ny, output_dir, slice_id=None):
    """Generate visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    diff = actual - expected

    # 1. Vector field showing displacement
    ax1 = axes[0, 0]
    ax1.quiver(expected[:, 1], expected[:, 0], diff[:, 1], diff[:, 0],
               angles='xy', scale_units='xy', scale=1, alpha=0.7)
    ax1.scatter(expected[:, 1], expected[:, 0], c='blue', s=20, alpha=0.5, label='Expected')
    ax1.scatter(actual[:, 1], actual[:, 0], c='red', s=20, alpha=0.5, label='Actual')
    ax1.set_xlabel('X position (pixels)')
    ax1.set_ylabel('Y position (pixels)')
    ax1.set_title('Tile Displacement Vectors')
    ax1.legend()
    ax1.invert_yaxis()

    # 2. Displacement magnitude heatmap
    ax2 = axes[0, 1]
    diff_mag = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)
    diff_grid = diff_mag.reshape(nx, ny)
    im = ax2.imshow(diff_grid, cmap='hot', interpolation='nearest')
    ax2.set_xlabel('Tile X index')
    ax2.set_ylabel('Tile Y index')
    ax2.set_title('Displacement Magnitude (pixels)')
    plt.colorbar(im, ax=ax2)

    # 3. Expected vs Actual positions (scatter)
    ax3 = axes[1, 0]
    ax3.scatter(expected[:, 1], actual[:, 1], alpha=0.5, label='X positions')
    ax3.scatter(expected[:, 0], actual[:, 0], alpha=0.5, label='Y positions')
    max_val = max(expected.max(), actual.max())
    ax3.plot([0, max_val], [0, max_val], 'k--', label='Perfect fit')
    ax3.set_xlabel('Expected position (pixels)')
    ax3.set_ylabel('Actual position (pixels)')
    ax3.set_title('Expected vs Actual Positions')
    ax3.legend()

    # 4. Residuals distribution
    ax4 = axes[1, 1]
    ax4.hist(residuals['residual_magnitude'], bins=30, edgecolor='black', alpha=0.7)
    ax4.axvline(residuals['mean_residual'], color='red', linestyle='--',
                label=f'Mean: {residuals["mean_residual"]:.1f} px')
    ax4.set_xlabel('Residual magnitude (pixels)')
    ax4.set_ylabel('Count')
    ax4.set_title('Residual Distribution (after scale correction)')
    ax4.legend()

    slice_label = f" (Slice {slice_id})" if slice_id else ""
    fig.suptitle(f'Tile Dilation Analysis{slice_label}', fontsize=14)
    plt.tight_layout()

    plot_path = output_dir / 'dilation_analysis.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()

    logger.info(f"Plots saved to {plot_path}")
    return plot_path


def main():
    p = _build_arg_parser()
    args = p.parse_args()

    input_file = Path(args.input_volume)
    transform_file = Path(args.input_transform)
    output_dir = Path(args.out_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load mosaic grid to get tile shape
    logger.info(f"Loading mosaic grid metadata from {input_file}")
    volume, resolution = read_omezarr(str(input_file), level=0)
    tile_shape = volume.chunks

    nx = volume.shape[1] // tile_shape[1]
    ny = volume.shape[2] // tile_shape[2]
    logger.info(f"Grid: {nx} x {ny} tiles, tile shape: {tile_shape}")

    # Load transform
    transform = np.load(transform_file)
    logger.info(f"Transform matrix:\n{transform}")

    # Compute positions
    expected = compute_expected_positions(nx, ny, tile_shape[1], tile_shape[2], args.overlap_fraction)
    actual = compute_actual_positions(nx, ny, transform)

    # Analysis
    scale_factors = estimate_scale_factors(expected, actual)
    residuals = analyze_residuals(expected, actual, scale_factors)
    distortions = detect_local_distortions(expected, actual, nx, ny)

    # Compile full analysis
    analysis = {
        'slice_id': args.slice_id,
        'grid_size': [nx, ny],
        'tile_shape': list(tile_shape),
        'resolution_um': args.resolution,
        'overlap_fraction': args.overlap_fraction,
        'scale_factors': scale_factors,
        'residuals': {k: v for k, v in residuals.items() if not isinstance(v, np.ndarray)},
        'distortions': distortions,
    }

    # Save JSON
    json_path = output_dir / 'dilation_analysis.json'
    with open(json_path, 'w') as f:
        json.dump(analysis, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Analysis JSON saved to {json_path}")

    # Generate outputs
    generate_report(analysis, scale_factors, residuals, distortions, output_dir, args.slice_id)
    generate_plots(expected, actual, residuals, nx, ny, output_dir, args.slice_id)

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Mean scale factor: {scale_factors['mean_scale']:.4f}")
    if abs(scale_factors['mean_scale'] - 1.0) > 0.001:
        deviation = (scale_factors['mean_scale'] - 1.0) * 100
        direction = "expansion" if deviation > 0 else "contraction"
        print(f"⚠ {abs(deviation):.2f}% {direction} detected")
    else:
        print("✓ No significant dilation")

    if scale_factors['anisotropy'] > 0.005:
        print(f"⚠ Anisotropic scaling: X/Y differ by {scale_factors['anisotropy']:.4f}")


if __name__ == '__main__':
    main()
