"""
Motor-position-based tile placement for mosaic stitching.

Consolidated from linum_stitch_3d_refined.py and linum_stitch_motor_only.py.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_motor_positions(
    nx: int, ny: int, tile_shape: tuple, overlap_fraction: float, scale_factor: float = 1.0, rotation_deg: float = 0.0
):
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

    rotation_matrix: np.ndarray | None = None
    if rotation_deg != 0.0:
        theta = np.radians(rotation_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    positions = []
    for i in range(nx):
        for j in range(ny):
            pos = np.array([i * step_y, j * step_x])
            if rotation_deg != 0.0 and rotation_matrix is not None:
                pos = np.dot(rotation_matrix, pos)
            positions.append(pos.astype(int) if rotation_deg != 0.0 else (int(pos[0]), int(pos[1])))

    return positions, step_y, step_x


def compute_registration_refinements(
    volume: np.ndarray, tile_shape: tuple, nx: int, ny: int, overlap_fraction: float, max_refinement_px: float = 10.0
) -> dict:
    """Correlate neighboring tiles within a slice to measure displacement errors.

    Phase-correlates overlapping regions of adjacent tiles (horizontal and
    vertical neighbors) to measure the difference between expected and actual
    tile positions.  Returns both clamped residuals for blend refinement and
    unclamped absolute displacements for fitting the affine displacement model
    (Lefebvre et al. 2017, Eqs 1-6).

    Note: this operates on tiles *within a single slice* — it is entirely
    separate from the Z-slice pairwise registration (``linum_register_pairwise.py``).

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
        Maximum residual shift retained for blend refinement. Larger residuals
        are clamped. Does not affect the absolute displacements in 'pairs'.

    Returns
    -------
    dict with keys 'horizontal', 'vertical', 'pairs', 'stats'.
        'pairs' is a list of dicts with keys 'row_delta', 'col_delta',
        'measured_dy', 'measured_dx' — the absolute observed pixel
        displacements used for affine model estimation.
    """
    from linumpy.registration.transforms import pairWisePhaseCorrelation

    tile_height, tile_width = tile_shape[1], tile_shape[2]
    overlap_y = int(tile_height * overlap_fraction)
    overlap_x = int(tile_width * overlap_fraction)

    # Expected step sizes (what a diagonal model would predict)
    step_y = tile_height * (1.0 - overlap_fraction)
    step_x = tile_width * (1.0 - overlap_fraction)

    refinements = {
        "horizontal": {},
        "vertical": {},
        "pairs": [],  # absolute displacements for affine estimation
        "stats": {"total_pairs": 0, "valid_pairs": 0, "clamped_pairs": 0, "mean_refinement": 0.0, "max_refinement": 0.0},
    }

    all_shifts = []
    z_mid = volume.shape[0] // 2

    # Horizontal refinements (between columns: tile (i,j) → (i,j+1))
    # The expected displacement is (0, step_x); registration measures residual
    for i in range(nx):
        for j in range(ny - 1):
            r1_start = i * tile_height
            r1_end = (i + 1) * tile_height
            c1_end = (j + 1) * tile_width
            c2_start = (j + 1) * tile_width

            overlap1 = volume[z_mid, r1_start:r1_end, c1_end - overlap_x : c1_end]
            overlap2 = volume[z_mid, r1_start:r1_end, c2_start : c2_start + overlap_x]

            if np.mean(overlap1 > 0) < 0.1 or np.mean(overlap2 > 0) < 0.1:
                continue

            refinements["stats"]["total_pairs"] += 1
            try:
                dy, dx = pairWisePhaseCorrelation(overlap1, overlap2)

                # Store absolute displacement for affine estimation (unclamped)
                # Horizontal pair: row_delta=0, col_delta=1
                # Measured position = expected_step + residual
                refinements["pairs"].append(
                    {
                        "row_delta": 0,
                        "col_delta": 1,
                        "measured_dy": float(dy),  # cross-axis residual
                        "measured_dx": float(step_x + dx),  # along-axis: step + residual
                    }
                )

                magnitude = np.sqrt(dx**2 + dy**2)
                if magnitude > max_refinement_px:
                    scale = max_refinement_px / magnitude
                    dx *= scale
                    dy *= scale
                    refinements["stats"]["clamped_pairs"] += 1

                refinements["horizontal"][(i, j)] = {"dx": float(dx), "dy": float(dy)}
                refinements["stats"]["valid_pairs"] += 1
                all_shifts.append(magnitude)
            except Exception as e:
                logger.debug(f"Registration failed for h-pair ({i},{j})-({i},{j + 1}): {e}")

    # Vertical refinements (between rows: tile (i,j) → (i+1,j))
    # The expected displacement is (step_y, 0); registration measures residual
    for i in range(nx - 1):
        for j in range(ny):
            r1_end = (i + 1) * tile_height
            r2_start = (i + 1) * tile_height
            c_start = j * tile_width
            c_end = (j + 1) * tile_width

            overlap1 = volume[z_mid, r1_end - overlap_y : r1_end, c_start:c_end]
            overlap2 = volume[z_mid, r2_start : r2_start + overlap_y, c_start:c_end]

            if np.mean(overlap1 > 0) < 0.1 or np.mean(overlap2 > 0) < 0.1:
                continue

            refinements["stats"]["total_pairs"] += 1
            try:
                dy, dx = pairWisePhaseCorrelation(overlap1, overlap2)

                # Store absolute displacement for affine estimation (unclamped)
                # Vertical pair: row_delta=1, col_delta=0
                refinements["pairs"].append(
                    {
                        "row_delta": 1,
                        "col_delta": 0,
                        "measured_dy": float(step_y + dy),  # along-axis: step + residual
                        "measured_dx": float(dx),  # cross-axis residual
                    }
                )

                magnitude = np.sqrt(dx**2 + dy**2)
                if magnitude > max_refinement_px:
                    scale = max_refinement_px / magnitude
                    dx *= scale
                    dy *= scale
                    refinements["stats"]["clamped_pairs"] += 1

                refinements["vertical"][(i, j)] = {"dx": float(dx), "dy": float(dy)}
                refinements["stats"]["valid_pairs"] += 1
                all_shifts.append(magnitude)
            except Exception as e:
                logger.debug(f"Registration failed for v-pair ({i},{j})-({i + 1},{j}): {e}")

    if all_shifts:
        refinements["stats"]["mean_refinement"] = float(np.mean(all_shifts))
        refinements["stats"]["max_refinement"] = float(np.max(all_shifts))

    return refinements


def estimate_affine_from_pairs(pairs: list, tile_shape: tuple, overlap_fraction: float) -> tuple[np.ndarray, dict]:
    """Estimate a 2x2 affine displacement model from neighbor tile correlations.

    Fits the Lefebvre et al. (2017) motor displacement model using
    least-squares on the absolute (step + residual) displacements returned
    by :func:`compute_registration_refinements`.

    Note: this uses phase correlation between *neighboring tiles within a
    single slice*, not the Z-slice pairwise registration that appears
    elsewhere in the pipeline.

    The model is: ``pixel_pos = A @ [i, j]^T`` where *A* is a general 2x2
    matrix.  Off-diagonal terms capture the scan-to-stage rotation (θ) and
    the non-perpendicularity of the motor axes (φ).

    Parameters
    ----------
    pairs : list of dict
        Each dict has 'row_delta', 'col_delta', 'measured_dy', 'measured_dx'.
    tile_shape : tuple
        Tile dimensions (z, height, width).
    overlap_fraction : float
        Expected overlap fraction (for diagnostics only).

    Returns
    -------
    transform : np.ndarray
        Fitted 2×2 affine matrix mapping tile index to pixel position.
    diagnostics : dict
        Extracted displacement model parameters (θ, φ, Ox, Oy) and fit
        residual statistics.
    """
    if not pairs:
        # Fallback to diagonal model
        step_y = tile_shape[1] * (1.0 - overlap_fraction)
        step_x = tile_shape[2] * (1.0 - overlap_fraction)
        transform = np.array([[step_y, 0.0], [0.0, step_x]])
        return transform, {"fallback": True, "reason": "no pairs"}

    n = len(pairs)
    # System:  A_mat @ x = b_vec
    # For each pair, row_delta and col_delta give the tile index offset,
    # measured_dy and measured_dx give the observed pixel displacement.
    # We solve for the 4 elements of the 2x2 transform:
    #   [row_delta, col_delta, 0,         0        ] [a]   [measured_dy]
    #   [0,         0,         row_delta, col_delta ] [b] = [measured_dx]
    #                                                  [c]
    #                                                  [d]
    a_mat = np.zeros((2 * n, 4))
    b_vec = np.zeros((2 * n, 1))
    for idx, p in enumerate(pairs):
        r, c = p["row_delta"], p["col_delta"]
        a_mat[2 * idx, :] = [r, c, 0, 0]
        b_vec[2 * idx, 0] = p["measured_dy"]
        a_mat[2 * idx + 1, :] = [0, 0, r, c]
        b_vec[2 * idx + 1, 0] = p["measured_dx"]

    result = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    transform = result[0].reshape((2, 2))
    residuals = result[1] if len(result[1]) > 0 else np.array([0.0])

    # Extract Lefebvre displacement model parameters for diagnostics
    diagnostics = _extract_displacement_params(transform, tile_shape, overlap_fraction)
    diagnostics["n_pairs"] = n
    diagnostics["lstsq_residual"] = float(np.sum(residuals))
    diagnostics["fallback"] = False

    return transform, diagnostics


def _extract_displacement_params(transform: np.ndarray, tile_shape: tuple, overlap_fraction: float) -> dict:
    """Extract Lefebvre motor model parameters from a 2x2 affine transform.

    Given the fitted transform ``A`` where ``pixel_pos = A @ [i, j]^T``,
    extract the scan-to-stage rotation θ, the non-perpendicularity angle φ,
    and the effective overlap fractions Ox, Oy.

    References: Lefebvre et al. 2017, Eqs 3–6.

    Parameters
    ----------
    transform : np.ndarray
        2×2 affine matrix.
    tile_shape : tuple
        Tile dimensions (z, height, width).
    overlap_fraction : float
        Expected overlap fraction (for comparison).

    Returns
    -------
    dict with 'theta_deg', 'phi_deg', 'Ox_fraction', 'Oy_fraction',
    'off_diagonal_px'.
    """
    a, b = transform[0, 0], transform[0, 1]
    c, d = transform[1, 0], transform[1, 1]
    tile_h, tile_w = tile_shape[1], tile_shape[2]

    # θ: rotation between scanning and stage reference frames
    # From vertical displacements: tan(θ) = -c / a  (Eq 3)
    theta_rad = np.arctan2(-c, a) if abs(a) > 1e-6 else 0.0

    # φ: non-perpendicularity between motor X and Y axes
    # From horizontal displacements: tan(φ - θ) = -b / d  (Eq 4 rearranged)
    phi_minus_theta = np.arctan2(-b, d) if abs(d) > 1e-6 else 0.0
    phi_rad = phi_minus_theta + theta_rad

    # Effective overlap fractions
    # Ox = 1 - |vertical step along row axis| / tile_height
    vertical_step = np.sqrt(a**2 + c**2)
    Ox_fraction = 1.0 - vertical_step / tile_h

    # Oy = 1 - |horizontal step along col axis| / tile_width
    horizontal_step = np.sqrt(b**2 + d**2)
    Oy_fraction = 1.0 - horizontal_step / tile_w

    return {
        "theta_deg": float(np.degrees(theta_rad)),
        "phi_deg": float(np.degrees(phi_rad)),
        "Ox_fraction": float(Ox_fraction),
        "Oy_fraction": float(Oy_fraction),
        "expected_overlap": float(overlap_fraction),
        "off_diagonal_px": [float(b), float(c)],
        "transform": transform.tolist(),
    }


def compute_affine_positions(nx: int, ny: int, transform: np.ndarray) -> list[tuple[int, int]]:
    """Compute tile positions using a 2x2 affine displacement model.

    This is the corrected version of :func:`compute_motor_positions` that
    accounts for scan-to-stage rotation (θ) and non-perpendicular motor
    axes (φ) via the off-diagonal terms in the transform matrix.

    Parameters
    ----------
    nx, ny : int
        Number of tiles in each direction.
    transform : np.ndarray
        2×2 affine matrix mapping tile index (i, j) to pixel position
        (row_px, col_px).

    Returns
    -------
    positions : list of (int, int)
        Pixel positions for each tile, row-major order.
    """
    positions = []
    for i in range(nx):
        for j in range(ny):
            pos = transform @ np.array([i, j], dtype=float)
            positions.append((round(pos[0]), round(pos[1])))
    return positions


def compute_affine_output_shape(nx: int, ny: int, tile_shape: tuple, transform: np.ndarray) -> tuple[int, int, int]:
    """Compute the output mosaic shape from affine tile positions.

    With off-diagonal terms, tiles may extend beyond what the diagonal model
    predicts.  This computes the bounding box over all tile corner positions.

    Parameters
    ----------
    nx, ny : int
        Number of tiles in each direction.
    tile_shape : tuple
        Tile dimensions (z, height, width).
    transform : np.ndarray
        2×2 affine matrix.

    Returns
    -------
    (nz, output_height, output_width) : tuple of int
    """
    nz = tile_shape[0]
    tile_h, tile_w = tile_shape[1], tile_shape[2]

    # Check all four corner tiles
    corners = [(0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1)]
    max_row, max_col = 0, 0
    min_row, min_col = 0, 0
    for i, j in corners:
        pos = transform @ np.array([i, j], dtype=float)
        # Tile occupies [pos[0], pos[0]+tile_h) x [pos[1], pos[1]+tile_w)
        min_row = min(min_row, pos[0])
        min_col = min(min_col, pos[1])
        max_row = max(max_row, pos[0] + tile_h)
        max_col = max(max_col, pos[1] + tile_w)

    output_height = int(np.ceil(max_row - min_row))
    output_width = int(np.ceil(max_col - min_col))
    return (nz, output_height, output_width)


def apply_blend_shift_refinement(tile: np.ndarray, refinements_for_tile: list, overlap_fraction: float) -> np.ndarray:
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

    total_dy = sum(ref.get("dy", 0) for ref in refinements_for_tile)
    total_dx = sum(ref.get("dx", 0) for ref in refinements_for_tile)
    count = len(refinements_for_tile)

    avg_dy = total_dy / count / 2
    avg_dx = total_dx / count / 2

    if abs(avg_dy) < 0.1 and abs(avg_dx) < 0.1:
        return tile

    nonzero_vals = tile[tile > 0]
    cval = float(np.percentile(nonzero_vals, 1)) if len(nonzero_vals) > 0 else 0.0
    shifted = ndi_shift(tile, (0, avg_dy, avg_dx), order=1, mode="constant", cval=cval)
    return shifted


def compare_motor_vs_registration(motor_positions: list, reg_positions: list, output_path: str | None = None) -> dict:
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

    comparison: dict[str, Any] = {
        "n_tiles": len(motor_positions),
        "mean_diff_y": float(np.mean(diff[:, 0])),
        "mean_diff_x": float(np.mean(diff[:, 1])),
        "std_diff_y": float(np.std(diff[:, 0])),
        "std_diff_x": float(np.std(diff[:, 1])),
        "max_diff_y": float(np.max(np.abs(diff[:, 0]))),
        "max_diff_x": float(np.max(np.abs(diff[:, 1]))),
        "mean_magnitude": float(np.mean(np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2))),
        "max_magnitude": float(np.max(np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2))),
    }

    if abs(comparison["mean_diff_y"]) > 5 or abs(comparison["mean_diff_x"]) > 5:
        comparison["systematic_offset"] = True
        comparison["offset_warning"] = (
            f"Systematic offset detected: ({comparison['mean_diff_y']:.1f}, {comparison['mean_diff_x']:.1f}) pixels"
        )
    else:
        comparison["systematic_offset"] = False

    tile_indices = np.arange(len(motor_positions))
    diff_magnitude = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
    if len(tile_indices) > 10:
        correlation = np.corrcoef(tile_indices, diff_magnitude)[0, 1]
        comparison["index_error_correlation"] = float(correlation)
        if abs(correlation) > 0.5:
            comparison["dilation_indicator"] = True
            comparison["dilation_warning"] = (
                f"Error increases with tile index (r={correlation:.2f}), suggesting dilation/scaling"
            )
        else:
            comparison["dilation_indicator"] = False

    if output_path:
        with Path(output_path).open("w") as f:
            json.dump(comparison, f, indent=2)

    return comparison
