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
from linumpy.stitching.mosaic_grid import addVolumeToMosaic
from linumpy.utils.io import add_overwrite_arg
from linumpy.utils.metrics import collect_stitch_3d_metrics
from linumpy.stitching.motor import compute_motor_positions, compare_motor_vs_registration

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


def compute_registration_positions(nx, ny, transform):
    """Compute tile positions using registration transform."""
    positions = []
    for i in range(nx):
        for j in range(ny):
            pos = np.dot(transform, [i, j]).astype(int)
            positions.append(pos)
    return positions


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
        comparison = compare_motor_vs_registration(motor_positions, reg_positions, args.output_comparison)

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
