#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a stitched mosaic using only motor positions (bypassing image-based registration).

This diagnostic tool creates stitched mosaics that use ONLY the motor/stage positions
recorded during acquisition. By comparing this "motor-only" reconstruction against
the fully-registered reconstruction, you can identify:

1. **Dilation/scaling issues**: If motor positions don't match actual image positions,
   the motor-only stitch will show systematic offsets or gaps
2. **Registration drift**: By comparing motor-only vs registered, you can see
   how much the registration is "correcting" the motor positions
3. **Stage repeatability**: Systematic patterns in motor-only errors indicate
   stage calibration issues

For troubleshooting 45° oblique-cut samples where edges don't match up.
"""
import linumpy._thread_config  # noqa: F401

import argparse
import logging
from pathlib import Path

import numpy as np

from linumpy.io.zarr import read_omezarr, OmeZarrWriter
from linumpy.utils.mosaic_grid import addVolumeToMosaic
from linumpy.utils.io import add_overwrite_arg
from linumpy.utils.metrics import collect_stitch_3d_metrics

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('input_volume',
                   help='Full path to a 3D mosaic grid volume (.ome.zarr)')
    p.add_argument('output_volume',
                   help='Output stitched mosaic filename (.ome.zarr)')

    p.add_argument('--overlap_fraction', type=float, default=0.1,
                   help='Expected overlap fraction between tiles [%(default)s]')
    p.add_argument('--blending_method', type=str, default='diffusion',
                   choices=['none', 'average', 'diffusion'],
                   help='Blending method [%(default)s]')
    p.add_argument('--scale_factor', type=float, default=1.0,
                   help='Scale factor to apply to motor positions (to test dilation) [%(default)s]')
    p.add_argument('--rotation_deg', type=float, default=0.0,
                   help='Global rotation to apply to tile grid (degrees) [%(default)s]')
    p.add_argument('--compare_transform', type=str, default=None,
                   help='Path to registration transform .npy file for comparison output')
    p.add_argument('--output_comparison', type=str, default=None,
                   help='Output path for comparison metrics JSON')

    add_overwrite_arg(p)
    return p


def compute_motor_positions(nx, ny, tile_shape, overlap_fraction, scale_factor=1.0, rotation_deg=0.0):
    """
    Compute tile positions based on motor grid (ideal positions).

    This assumes a regular grid where tiles are spaced by (1 - overlap) * tile_size.
    """
    tile_height, tile_width = tile_shape[1], tile_shape[2]  # Assuming [z, y, x] or [z, row, col]

    # Effective step between tiles
    step_y = int(tile_height * (1.0 - overlap_fraction))
    step_x = int(tile_width * (1.0 - overlap_fraction))

    # Apply scale factor (to test dilation hypothesis)
    step_y = int(step_y * scale_factor)
    step_x = int(step_x * scale_factor)

    # Rotation matrix (for global grid rotation)
    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    positions = []
    for i in range(nx):
        for j in range(ny):
            # Base position in grid
            pos = np.array([i * step_y, j * step_x])

            # Apply rotation if specified
            if rotation_deg != 0.0:
                pos = np.dot(rotation_matrix, pos)

            positions.append(pos.astype(int))

    return positions


def compute_registration_positions(nx, ny, transform):
    """Compute tile positions using registration transform."""
    positions = []
    for i in range(nx):
        for j in range(ny):
            pos = np.dot(transform, [i, j]).astype(int)
            positions.append(pos)
    return positions


def compare_positions(motor_positions, reg_positions, output_path=None):
    """Compare motor-based positions with registration-based positions."""
    import json

    motor_arr = np.array(motor_positions)
    reg_arr = np.array(reg_positions)

    # Compute differences
    diff = reg_arr - motor_arr

    # Statistics
    comparison = {
        'n_tiles': len(motor_positions),
        'mean_diff_y': float(np.mean(diff[:, 0])),
        'mean_diff_x': float(np.mean(diff[:, 1])),
        'std_diff_y': float(np.std(diff[:, 0])),
        'std_diff_x': float(np.std(diff[:, 1])),
        'max_diff_y': float(np.max(np.abs(diff[:, 0]))),
        'max_diff_x': float(np.max(np.abs(diff[:, 1]))),
        'mean_magnitude': float(np.mean(np.sqrt(diff[:, 0]**2 + diff[:, 1]**2))),
        'max_magnitude': float(np.max(np.sqrt(diff[:, 0]**2 + diff[:, 1]**2))),
    }

    # Check for systematic offset (would indicate motor calibration issue)
    if abs(comparison['mean_diff_y']) > 5 or abs(comparison['mean_diff_x']) > 5:
        comparison['systematic_offset'] = True
        comparison['offset_warning'] = f"Systematic offset detected: ({comparison['mean_diff_y']:.1f}, {comparison['mean_diff_x']:.1f}) pixels"
    else:
        comparison['systematic_offset'] = False

    # Check for scaling/dilation (increasing error with tile index)
    tile_indices = np.arange(len(motor_positions))
    diff_magnitude = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)

    # Correlation between tile index and error magnitude
    if len(tile_indices) > 10:
        correlation = np.corrcoef(tile_indices, diff_magnitude)[0, 1]
        comparison['index_error_correlation'] = float(correlation)
        if abs(correlation) > 0.5:
            comparison['dilation_indicator'] = True
            comparison['dilation_warning'] = f"Error increases with tile index (r={correlation:.2f}), suggesting dilation/scaling"
        else:
            comparison['dilation_indicator'] = False

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Comparison saved to {output_path}")

    return comparison


def main():
    p = _build_arg_parser()
    args = p.parse_args()

    input_file = Path(args.input_volume)
    output_file = Path(args.output_volume)

    assert output_file.name.endswith('.zarr'), "output_volume must be a .zarr file"

    if not args.overwrite and output_file.exists():
        raise FileExistsError(f"Output file exists: {output_file}. Use --overwrite to replace.")

    # Load the mosaic grid volume
    logger.info(f"Loading mosaic grid from {input_file}")
    volume, resolution = read_omezarr(str(input_file), level=0)
    tile_shape = volume.chunks

    logger.info(f"Volume shape: {volume.shape}")
    logger.info(f"Tile shape: {tile_shape}")
    logger.info(f"Resolution: {resolution}")

    # Compute grid dimensions
    nx = volume.shape[1] // tile_shape[1]
    ny = volume.shape[2] // tile_shape[2]
    logger.info(f"Grid: {nx} x {ny} tiles")

    # Compute motor-based positions
    motor_positions = compute_motor_positions(
        nx, ny, tile_shape,
        args.overlap_fraction,
        args.scale_factor,
        args.rotation_deg
    )

    # If comparison transform provided, compare positions
    if args.compare_transform:
        transform = np.load(args.compare_transform)
        reg_positions = compute_registration_positions(nx, ny, transform)
        comparison = compare_positions(motor_positions, reg_positions, args.output_comparison)

        logger.info("Position comparison summary:")
        logger.info(f"  Mean offset: ({comparison['mean_diff_y']:.1f}, {comparison['mean_diff_x']:.1f}) px")
        logger.info(f"  Max offset: {comparison['max_magnitude']:.1f} px")
        if comparison.get('dilation_indicator'):
            logger.warning(comparison['dilation_warning'])

    # Compute output mosaic shape
    posx_min = min([pos[0] for pos in motor_positions])
    posx_max = max([pos[0] + tile_shape[1] for pos in motor_positions])
    posy_min = min([pos[1] for pos in motor_positions])
    posy_max = max([pos[1] + tile_shape[2] for pos in motor_positions])
    mosaic_shape = (volume.shape[0], int(posx_max - posx_min), int(posy_max - posy_min))

    logger.info(f"Output mosaic shape: {mosaic_shape}")

    # Stitch the mosaic using motor positions only
    logger.info("Stitching mosaic using motor positions...")
    writer = OmeZarrWriter(output_file, mosaic_shape, chunk_shape=(100, 100, 100),
                           dtype=np.float32, overwrite=args.overwrite)

    for i in range(nx):
        for j in range(ny):
            # Extract tile from input
            rmin = i * tile_shape[1]
            rmax = (i + 1) * tile_shape[1]
            cmin = j * tile_shape[2]
            cmax = (j + 1) * tile_shape[2]
            tile = volume[:, rmin:rmax, cmin:cmax]

            if np.any(tile < 0.0):
                tile = tile - tile.min()

            # Get motor-based position
            pos = motor_positions[i * ny + j].copy()
            pos[0] -= posx_min
            pos[1] -= posy_min

            addVolumeToMosaic(tile, pos, writer, blendingMethod=args.blending_method)

    writer.finalize(resolution)

    # Collect metrics
    collect_stitch_3d_metrics(
        input_shape=volume.shape,
        output_shape=mosaic_shape,
        num_tiles=nx * ny,
        resolution=list(resolution),
        output_path=str(output_file),
        input_path=str(input_file),
        blending_method=args.blending_method
    )

    logger.info(f"Motor-only stitched mosaic saved to {output_file}")


if __name__ == '__main__':
    main()
