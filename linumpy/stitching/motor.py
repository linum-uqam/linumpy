# -*- coding: utf-8 -*-
"""
Motor-position-based tile placement for mosaic stitching.

Consolidated from linum_stitch_3d_refined.py and linum_stitch_motor_only.py.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_motor_positions(nx: int, ny: int, tile_shape: tuple,
                            overlap_fraction: float,
                            scale_factor: float = 1.0,
                            rotation_deg: float = 0.0):
    """Compute tile positions based on motor grid (ideal positions).

    Assumes a regular grid where tiles are spaced by (1 - overlap) * tile_size.
    Optionally applies scale factor and rotation to test hypotheses about
    stage calibration issues.

    Parameters
    ----------
    nx, ny : int
        Number of tiles in each direction.
    tile_shape : tuple
        Tile dimensions (z, height, width).
    overlap_fraction : float
        Expected overlap between tiles (0-1).
    scale_factor : float
        Scale applied to step size (default 1.0 = no scaling).
    rotation_deg : float
        Global grid rotation in degrees (default 0.0).

    Returns
    -------
    positions : list
        List of (row_pos, col_pos) pixel positions for each tile.
    step_y : int
        Y step in pixels.
    step_x : int
        X step in pixels.
    """
    tile_height, tile_width = tile_shape[1], tile_shape[2]

    step_y = int(tile_height * (1.0 - overlap_fraction))
    step_x = int(tile_width * (1.0 - overlap_fraction))

    step_y = int(step_y * scale_factor)
    step_x = int(step_x * scale_factor)

    if rotation_deg != 0.0:
        theta = np.radians(rotation_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    positions = []
    for i in range(nx):
        for j in range(ny):
            pos = np.array([i * step_y, j * step_x])
            if rotation_deg != 0.0:
                pos = np.dot(rotation_matrix, pos)
            positions.append(pos.astype(int) if rotation_deg != 0.0 else (int(pos[0]), int(pos[1])))

    return positions, step_y, step_x


def compute_registration_refinements(volume: np.ndarray,
                                     tile_shape: tuple,
                                     nx: int, ny: int,
                                     overlap_fraction: float,
                                     max_refinement_px: float = 10.0) -> dict:
    """Compute sub-pixel refinements by phase-correlating overlapping tile regions.

    Does not change tile positions — computes how much the blending transition
    should be adjusted for smoother seams at tile boundaries.

    Parameters
    ----------
    volume : np.ndarray
        The mosaic grid volume (Z, nx*tile_h, ny*tile_w).
    tile_shape : tuple
        Tile dimensions (z, height, width).
    nx, ny : int
        Number of tiles in each direction.
    overlap_fraction : float
        Expected overlap fraction (0-1).
    max_refinement_px : float
        Maximum allowed refinement shift. Larger shifts are clamped.

    Returns
    -------
    dict with keys 'horizontal', 'vertical', 'stats'.
    """
    from linumpy.stitching.registration import pairWisePhaseCorrelation

    tile_height, tile_width = tile_shape[1], tile_shape[2]
    overlap_y = int(tile_height * overlap_fraction)
    overlap_x = int(tile_width * overlap_fraction)

    refinements = {
        'horizontal': {},
        'vertical': {},
        'stats': {
            'total_pairs': 0,
            'valid_pairs': 0,
            'clamped_pairs': 0,
            'mean_refinement': 0.0,
            'max_refinement': 0.0
        }
    }

    all_shifts = []
    z_mid = volume.shape[0] // 2

    # Horizontal refinements (between columns)
    for i in range(nx):
        for j in range(ny - 1):
            r1_start = i * tile_height
            r1_end = (i + 1) * tile_height
            c1_end = (j + 1) * tile_width
            c2_start = (j + 1) * tile_width

            overlap1 = volume[z_mid, r1_start:r1_end, c1_end - overlap_x:c1_end]
            overlap2 = volume[z_mid, r1_start:r1_end, c2_start:c2_start + overlap_x]

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

                refinements['horizontal'][(i, j)] = {'dx': float(dx), 'dy': float(dy)}
                refinements['stats']['valid_pairs'] += 1
                all_shifts.append(magnitude)
            except Exception as e:
                logger.debug(f"Registration failed for h-pair ({i},{j})-({i},{j+1}): {e}")

    # Vertical refinements (between rows)
    for i in range(nx - 1):
        for j in range(ny):
            r1_end = (i + 1) * tile_height
            r2_start = (i + 1) * tile_height
            c_start = j * tile_width
            c_end = (j + 1) * tile_width

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


def apply_blend_shift_refinement(tile: np.ndarray,
                                 refinements_for_tile: list,
                                 overlap_fraction: float) -> np.ndarray:
    """Apply registration refinement by shifting tile data in overlap regions.

    Applies a small sub-pixel shift (averaged from all neighbors) to improve
    blending quality without changing the tile's position in the mosaic.

    Parameters
    ----------
    tile : np.ndarray
        3D tile data (Z, Y, X).
    refinements_for_tile : list
        List of dicts with 'dx', 'dy' refinements from neighbors.
    overlap_fraction : float
        Tile overlap fraction (used for context; shift applies to whole tile).

    Returns
    -------
    np.ndarray
        Shifted tile (or unmodified if shift is negligible).
    """
    from scipy.ndimage import shift as ndi_shift

    if not refinements_for_tile:
        return tile

    total_dy = sum(ref.get('dy', 0) for ref in refinements_for_tile)
    total_dx = sum(ref.get('dx', 0) for ref in refinements_for_tile)
    count = len(refinements_for_tile)

    avg_dy = total_dy / count / 2
    avg_dx = total_dx / count / 2

    if abs(avg_dy) < 0.1 and abs(avg_dx) < 0.1:
        return tile

    nonzero_vals = tile[tile > 0]
    cval = float(np.percentile(nonzero_vals, 1)) if len(nonzero_vals) > 0 else 0.0
    shifted = ndi_shift(tile, (0, avg_dy, avg_dx), order=1, mode='constant', cval=cval)
    return shifted


def compare_motor_vs_registration(motor_positions: list,
                                  reg_positions: list,
                                  output_path: str = None) -> dict:
    """Compare motor-based positions with registration-based positions.

    Used diagnostically to identify stage calibration issues (systematic offset,
    dilation/scaling) and registration drift.

    Parameters
    ----------
    motor_positions : list
        List of (row, col) positions from motor grid.
    reg_positions : list
        List of (row, col) positions from image registration.
    output_path : str or None
        If provided, save comparison JSON to this path.

    Returns
    -------
    dict
        Statistics including mean/std/max differences and diagnostic flags.
    """
    import json

    motor_arr = np.array(motor_positions)
    reg_arr = np.array(reg_positions)
    diff = reg_arr - motor_arr

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

    if abs(comparison['mean_diff_y']) > 5 or abs(comparison['mean_diff_x']) > 5:
        comparison['systematic_offset'] = True
        comparison['offset_warning'] = (
            f"Systematic offset detected: "
            f"({comparison['mean_diff_y']:.1f}, {comparison['mean_diff_x']:.1f}) pixels"
        )
    else:
        comparison['systematic_offset'] = False

    tile_indices = np.arange(len(motor_positions))
    diff_magnitude = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)
    if len(tile_indices) > 10:
        correlation = np.corrcoef(tile_indices, diff_magnitude)[0, 1]
        comparison['index_error_correlation'] = float(correlation)
        if abs(correlation) > 0.5:
            comparison['dilation_indicator'] = True
            comparison['dilation_warning'] = (
                f"Error increases with tile index (r={correlation:.2f}), "
                f"suggesting dilation/scaling"
            )
        else:
            comparison['dilation_indicator'] = False

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)

    return comparison
