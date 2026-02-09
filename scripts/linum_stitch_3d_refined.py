#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stitch a 3D mosaic grid with registration-refined blending.

This script combines the best of both approaches:
1. Uses MOTOR POSITIONS for tile placement (precise, no systematic drift)
2. Uses REGISTRATION to refine blending transitions in overlap regions

The registration is only used to compute sub-pixel adjustments for smoother
blending at tile boundaries, NOT to change where tiles are positioned.

This produces better seam quality while maintaining correct overall geometry.
"""

# Configure thread limits before numpy/scipy imports
import linumpy._thread_config  # noqa: F401

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import shift as ndi_shift

from linumpy.io.zarr import read_omezarr, OmeZarrWriter
from linumpy.stitching.registration import pairWisePhaseCorrelation
from linumpy.utils.mosaic_grid import addVolumeToMosaic

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_volume",
                   help="Full path to a 3D mosaic grid volume (.ome.zarr)")
    p.add_argument("output_volume",
                   help="Output stitched mosaic filename (.ome.zarr)")

    p.add_argument("--overlap_fraction", type=float, default=0.2,
                   help="Expected tile overlap fraction (0-1). [%(default)s]")
    p.add_argument("--blending_method", type=str, default="diffusion",
                   choices=["none", "average", "diffusion"],
                   help="Blending method for overlap regions. [%(default)s]")
    p.add_argument("--refinement_mode", type=str, default="blend_shift",
                   choices=["none", "blend_shift", "full_shift"],
                   help="How to apply registration refinements:\n"
                        "  none: Pure motor positions, no refinement\n"
                        "  blend_shift: Shift blending weights (recommended)\n"
                        "  full_shift: Apply sub-pixel shifts to tiles [%(default)s]")
    p.add_argument("--max_refinement_px", type=float, default=10.0,
                   help="Maximum allowed refinement shift in pixels. [%(default)s]\n"
                        "Larger shifts are clamped to prevent bad registrations.")
    p.add_argument("--output_refinements", type=str, default=None,
                   help="Output JSON file to save computed refinements for analysis.")
    p.add_argument("--overwrite", "-f", action="store_true",
                   help="Overwrite output if it exists.")
    return p


def compute_motor_positions(nx, ny, tile_shape, overlap_fraction):
    """
    Compute tile positions based on motor grid (expected overlap).

    Parameters
    ----------
    nx, ny : int
        Number of tiles in each direction
    tile_shape : tuple
        Tile dimensions (z, y, x)
    overlap_fraction : float
        Expected overlap (0-1)

    Returns
    -------
    list
        List of (row_pos, col_pos) pixel positions for each tile
    """
    tile_height, tile_width = tile_shape[1], tile_shape[2]
    step_y = int(tile_height * (1.0 - overlap_fraction))
    step_x = int(tile_width * (1.0 - overlap_fraction))

    positions = []
    for i in range(nx):
        for j in range(ny):
            pos_y = i * step_y
            pos_x = j * step_x
            positions.append((pos_y, pos_x))

    return positions, step_y, step_x


def compute_registration_refinements(volume, tile_shape, nx, ny, overlap_fraction, max_refinement_px=10.0):
    """
    Compute sub-pixel refinements by registering overlapping tile regions.

    This doesn't change tile positions - it computes how much the blending
    transition should be adjusted for smoother seams.

    Parameters
    ----------
    volume : array
        The mosaic grid volume
    tile_shape : tuple
        Tile dimensions
    nx, ny : int
        Number of tiles
    overlap_fraction : float
        Expected overlap
    max_refinement_px : float
        Maximum allowed refinement

    Returns
    -------
    dict
        Refinement data for each tile pair
    """
    tile_height, tile_width = tile_shape[1], tile_shape[2]
    overlap_y = int(tile_height * overlap_fraction)
    overlap_x = int(tile_width * overlap_fraction)

    refinements = {
        'horizontal': {},  # Refinements for left-right neighbors
        'vertical': {},    # Refinements for top-bottom neighbors
        'stats': {
            'total_pairs': 0,
            'valid_pairs': 0,
            'clamped_pairs': 0,
            'mean_refinement': 0.0,
            'max_refinement': 0.0
        }
    }

    all_shifts = []

    # Compute horizontal refinements (between columns)
    for i in range(nx):
        for j in range(ny - 1):
            # Get overlapping regions from adjacent tiles
            tile1_idx = i * ny + j
            tile2_idx = i * ny + (j + 1)

            # Extract tiles
            r1_start = i * tile_height
            r1_end = (i + 1) * tile_height
            c1_start = j * tile_width
            c1_end = (j + 1) * tile_width
            c2_start = (j + 1) * tile_width
            c2_end = (j + 2) * tile_width

            # Get overlap regions (right edge of tile1, left edge of tile2)
            # Use middle z-slice for 2D registration
            z_mid = volume.shape[0] // 2

            overlap1 = volume[z_mid, r1_start:r1_end, c1_end - overlap_x:c1_end]
            overlap2 = volume[z_mid, r1_start:r1_end, c2_start:c2_start + overlap_x]

            # Skip if either region is mostly empty
            if np.mean(overlap1 > 0) < 0.1 or np.mean(overlap2 > 0) < 0.1:
                continue

            refinements['stats']['total_pairs'] += 1

            try:
                # Compute sub-pixel shift
                dy, dx = pairWisePhaseCorrelation(overlap1, overlap2)

                # Clamp to maximum
                magnitude = np.sqrt(dx**2 + dy**2)
                if magnitude > max_refinement_px:
                    scale = max_refinement_px / magnitude
                    dx *= scale
                    dy *= scale
                    refinements['stats']['clamped_pairs'] += 1

                refinements['horizontal'][(i, j)] = {'dx': float(dx), 'dy': float(dy)}
                refinements['stats']['valid_pairs'] += 1
                all_shifts.append(magnitude)

            except Exception as e:
                logger.debug(f"Registration failed for h-pair ({i},{j})-({i},{j+1}): {e}")

    # Compute vertical refinements (between rows)
    for i in range(nx - 1):
        for j in range(ny):
            # Get overlapping regions
            r1_start = i * tile_height
            r1_end = (i + 1) * tile_height
            r2_start = (i + 1) * tile_height
            r2_end = (i + 2) * tile_height
            c_start = j * tile_width
            c_end = (j + 1) * tile_width

            z_mid = volume.shape[0] // 2

            overlap1 = volume[z_mid, r1_end - overlap_y:r1_end, c_start:c_end]
            overlap2 = volume[z_mid, r2_start:r2_start + overlap_y, c_start:c_end]

            if np.mean(overlap1 > 0) < 0.1 or np.mean(overlap2 > 0) < 0.1:
                continue

            refinements['stats']['total_pairs'] += 1

            try:
                dy, dx = pairWisePhaseCorrelation(overlap1, overlap2)

                magnitude = np.sqrt(dx**2 + dy**2)
                if magnitude > max_refinement_px:
                    scale = max_refinement_px / magnitude
                    dx *= scale
                    dy *= scale
                    refinements['stats']['clamped_pairs'] += 1

                refinements['vertical'][(i, j)] = {'dx': float(dx), 'dy': float(dy)}
                refinements['stats']['valid_pairs'] += 1
                all_shifts.append(magnitude)

            except Exception as e:
                logger.debug(f"Registration failed for v-pair ({i},{j})-({i+1},{j}): {e}")

    if all_shifts:
        refinements['stats']['mean_refinement'] = float(np.mean(all_shifts))
        refinements['stats']['max_refinement'] = float(np.max(all_shifts))

    return refinements


def apply_blend_shift_refinement(tile, refinements_for_tile, overlap_fraction):
    """
    Apply refinement by slightly shifting the tile data in overlap regions only.

    This creates smoother blending transitions without changing tile positions.
    """
    # For now, apply a small sub-pixel shift to the entire tile
    # A more sophisticated approach would only shift the overlap regions
    if not refinements_for_tile:
        return tile

    # Average the refinements from all neighbors
    total_dy = 0.0
    total_dx = 0.0
    count = 0

    for ref in refinements_for_tile:
        total_dy += ref.get('dy', 0)
        total_dx += ref.get('dx', 0)
        count += 1

    if count == 0:
        return tile

    avg_dy = total_dy / count / 2  # Half shift - meet in the middle
    avg_dx = total_dx / count / 2

    if abs(avg_dy) < 0.1 and abs(avg_dx) < 0.1:
        return tile

    # Apply sub-pixel shift
    shifted = ndi_shift(tile, (0, avg_dy, avg_dx), order=1, mode='constant', cval=0)

    return shifted


def stitch_with_refinements(volume, tile_shape, overlap_fraction, blending_method,
                           refinement_mode, refinements, output_shape):
    """
    Stitch tiles using motor positions with optional registration refinements.
    """
    nz = volume.shape[0]
    tile_height, tile_width = tile_shape[1], tile_shape[2]
    nx = volume.shape[1] // tile_height
    ny = volume.shape[2] // tile_width

    # Compute motor-based positions
    positions, step_y, step_x = compute_motor_positions(nx, ny, tile_shape, overlap_fraction)

    # Initialize output array
    output = np.zeros(output_shape, dtype=np.float32)

    for i in range(nx):
        for j in range(ny):
            # Extract tile
            r_start = i * tile_height
            r_end = (i + 1) * tile_height
            c_start = j * tile_width
            c_end = (j + 1) * tile_width

            tile = volume[:, r_start:r_end, c_start:c_end].copy()

            if np.any(tile < 0):
                tile = tile - tile.min()

            # Get position from motor positions
            pos = list(positions[i * ny + j])

            # Apply refinements if requested
            if refinement_mode == 'blend_shift':
                # Collect refinements for this tile from its neighbors
                tile_refinements = []

                # From horizontal neighbor to the left
                if j > 0 and (i, j-1) in refinements.get('horizontal', {}):
                    ref = refinements['horizontal'][(i, j-1)]
                    tile_refinements.append({'dy': -ref['dy'], 'dx': -ref['dx']})

                # From horizontal neighbor to the right
                if (i, j) in refinements.get('horizontal', {}):
                    ref = refinements['horizontal'][(i, j)]
                    tile_refinements.append(ref)

                # From vertical neighbor above
                if i > 0 and (i-1, j) in refinements.get('vertical', {}):
                    ref = refinements['vertical'][(i-1, j)]
                    tile_refinements.append({'dy': -ref['dy'], 'dx': -ref['dx']})

                # From vertical neighbor below
                if (i, j) in refinements.get('vertical', {}):
                    ref = refinements['vertical'][(i, j)]
                    tile_refinements.append(ref)

                tile = apply_blend_shift_refinement(tile, tile_refinements, overlap_fraction)

            elif refinement_mode == 'full_shift':
                # Apply average refinement as position offset (sub-pixel)
                # This is more aggressive - shifts the entire tile
                tile_refinements = []

                if j > 0 and (i, j-1) in refinements.get('horizontal', {}):
                    tile_refinements.append(refinements['horizontal'][(i, j-1)])
                if (i, j) in refinements.get('horizontal', {}):
                    tile_refinements.append(refinements['horizontal'][(i, j)])
                if i > 0 and (i-1, j) in refinements.get('vertical', {}):
                    tile_refinements.append(refinements['vertical'][(i-1, j)])
                if (i, j) in refinements.get('vertical', {}):
                    tile_refinements.append(refinements['vertical'][(i, j)])

                if tile_refinements:
                    avg_dy = np.mean([r['dy'] for r in tile_refinements]) / 2
                    avg_dx = np.mean([r['dx'] for r in tile_refinements]) / 2
                    pos[0] += avg_dy
                    pos[1] += avg_dx

            # Add tile to mosaic
            addVolumeToMosaic(tile, pos, output, blendingMethod=blending_method)

    return output


def main():
    p = _build_arg_parser()
    args = p.parse_args()

    input_file = Path(args.input_volume)
    output_file = Path(args.output_volume)

    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_file}. Use -f to overwrite.")

    # Load volume
    logger.info(f"Loading mosaic grid: {input_file}")
    volume, resolution = read_omezarr(str(input_file), level=0)
    volume = np.array(volume[:])
    tile_shape = volume.shape[0], volume.shape[1] // (volume.shape[1] // 75), volume.shape[2] // (volume.shape[2] // 75)

    # Try to get tile shape from chunks
    vol_dask, _ = read_omezarr(str(input_file), level=0)
    if hasattr(vol_dask, 'chunks'):
        tile_shape = vol_dask.chunks

    logger.info(f"Volume shape: {volume.shape}")
    logger.info(f"Tile shape: {tile_shape}")
    logger.info(f"Overlap fraction: {args.overlap_fraction}")
    logger.info(f"Refinement mode: {args.refinement_mode}")

    nx = volume.shape[1] // tile_shape[1]
    ny = volume.shape[2] // tile_shape[2]
    logger.info(f"Grid: {nx} x {ny} tiles")

    # Compute registration refinements
    refinements = {}
    if args.refinement_mode != 'none':
        logger.info("Computing registration refinements...")
        refinements = compute_registration_refinements(
            volume, tile_shape, nx, ny,
            args.overlap_fraction, args.max_refinement_px
        )

        stats = refinements['stats']
        logger.info(f"  Total tile pairs: {stats['total_pairs']}")
        logger.info(f"  Valid registrations: {stats['valid_pairs']}")
        logger.info(f"  Clamped (large shifts): {stats['clamped_pairs']}")
        logger.info(f"  Mean refinement: {stats['mean_refinement']:.2f} px")
        logger.info(f"  Max refinement: {stats['max_refinement']:.2f} px")

        if args.output_refinements:
            # Convert tuple keys to strings for JSON serialization
            json_refinements = {
                'horizontal': {f"{k[0]},{k[1]}": v for k, v in refinements['horizontal'].items()},
                'vertical': {f"{k[0]},{k[1]}": v for k, v in refinements['vertical'].items()},
                'stats': refinements['stats'],
                'parameters': {
                    'overlap_fraction': args.overlap_fraction,
                    'max_refinement_px': args.max_refinement_px,
                    'refinement_mode': args.refinement_mode
                }
            }
            with open(args.output_refinements, 'w') as f:
                json.dump(json_refinements, f, indent=2)
            logger.info(f"Refinements saved to: {args.output_refinements}")

    # Compute output shape
    step_y = int(tile_shape[1] * (1.0 - args.overlap_fraction))
    step_x = int(tile_shape[2] * (1.0 - args.overlap_fraction))
    output_height = (nx - 1) * step_y + tile_shape[1]
    output_width = (ny - 1) * step_x + tile_shape[2]
    output_shape = (volume.shape[0], output_height, output_width)

    logger.info(f"Output shape: {output_shape}")

    # Stitch with refinements
    logger.info(f"Stitching with {args.blending_method} blending...")
    output = stitch_with_refinements(
        volume, tile_shape, args.overlap_fraction, args.blending_method,
        args.refinement_mode, refinements, output_shape
    )

    # Save output
    logger.info(f"Saving to: {output_file}")
    from linumpy.io.zarr import save_omezarr
    import dask.array as da
    save_omezarr(da.from_array(output), str(output_file), resolution, n_levels=3)

    logger.info("Done!")


if __name__ == "__main__":
    main()
