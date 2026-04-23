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
) -> tuple:
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
    volume: np.ndarray,
    tile_shape: tuple,
    nx: int,
    ny: int,
    overlap_fraction: float,
    max_refinement_px: float = 10.0,
    *,
    histogram_match: bool = False,
    max_empty_fraction: float | None = None,
    use_gpu: bool = False,
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
    histogram_match : bool, keyword-only
        If True, match the intensity histogram of the second overlap to the
        first before phase correlation.  Improves robustness when tile-edge
        illumination is uneven; disabled by default to preserve existing
        behaviour.
    max_empty_fraction : float or None, keyword-only
        If set, use an Otsu threshold on the central plane to classify
        tissue vs background, and skip any pair whose overlap contains more
        than this fraction of background pixels (mirrors the behaviour of
        ``linumpy.registration.transforms.estimate_mosaic_transform``).
        When ``None`` (default), the prior ``mean(overlap > 0) < 0.1``
        heuristic is used.
    use_gpu : bool, keyword-only
        If True, run the pairwise phase correlations via
        :func:`linumpy.gpu.fft_ops.phase_correlation` (CuPy-accelerated).
        Falls back silently to the CPU path when CuPy / a CUDA device is
        not available. Default is False.

    Returns
    -------
    dict with keys 'horizontal', 'vertical', 'pairs', 'stats'.
        'pairs' is a list of dicts with keys 'row_delta', 'col_delta',
        'measured_dy', 'measured_dx' — the absolute observed pixel
        displacements used for affine model estimation.
    """
    from linumpy.registration.transforms import pair_wise_phase_correlation

    gpu_phase_correlation: Any = None
    if use_gpu:
        try:
            from linumpy.gpu import GPU_AVAILABLE
            from linumpy.gpu.fft_ops import phase_correlation as _gpu_phase_correlation

            if GPU_AVAILABLE:
                gpu_phase_correlation = _gpu_phase_correlation
            else:
                logger.info("use_gpu=True but no CUDA device detected; falling back to CPU phase correlation")
        except ImportError as e:
            logger.info("use_gpu=True but GPU stack unavailable (%s); falling back to CPU", e)

    def _phase_correlate(ov1: np.ndarray, ov2: np.ndarray) -> tuple[float, float]:
        """Return (axis-0 shift, axis-1 shift) for vol2 relative to vol1."""
        if gpu_phase_correlation is not None:
            translation, _ = gpu_phase_correlation(ov1, ov2, use_gpu=True)
            return float(translation[0]), float(translation[1])
        axis0, axis1 = pair_wise_phase_correlation(ov1, ov2)
        return float(axis0), float(axis1)

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

    empty_threshold: float | None = None
    if max_empty_fraction is not None:
        from skimage.filters import threshold_otsu

        plane = np.asarray(volume[z_mid])
        positive = plane[plane > 0]
        if positive.size > 0:
            empty_threshold = float(threshold_otsu(positive))

    match_histograms_fn = None
    if histogram_match:
        from skimage.exposure import match_histograms as _match_histograms

        match_histograms_fn = _match_histograms

    def _is_empty(ov: np.ndarray) -> bool:
        if empty_threshold is not None and max_empty_fraction is not None:
            return bool(np.sum(ov <= empty_threshold) > max_empty_fraction * ov.size)
        return bool(np.mean(ov > 0) < 0.1)

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

            if _is_empty(overlap1) or _is_empty(overlap2):
                continue

            if match_histograms_fn is not None:
                overlap2 = match_histograms_fn(overlap2, overlap1)

            refinements["stats"]["total_pairs"] += 1
            try:
                dy, dx = _phase_correlate(overlap1, overlap2)

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
                logger.debug("Registration failed for h-pair (%d,%d)-(%d,%d): %s", i, j, i, j + 1, e)

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

            if _is_empty(overlap1) or _is_empty(overlap2):
                continue

            if match_histograms_fn is not None:
                overlap2 = match_histograms_fn(overlap2, overlap1)

            refinements["stats"]["total_pairs"] += 1
            try:
                dy, dx = _phase_correlate(overlap1, overlap2)

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
                logger.debug("Registration failed for v-pair (%d,%d)-(%d,%d): %s", i, j, i + 1, j, e)

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
    # System: A_mat @ x = b_vec, where A_mat has rows [r, c, 0, 0] (for dy) and [0, 0, r, c] (for dx),
    # and x = [a, b, c, d]^T are the four elements of the 2x2 transform matrix.
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


def pool_pairs_and_fit_global_affine(
    volumes: list[tuple[str, Any]],
    overlap_fraction: float,
    *,
    histogram_match: bool = False,
    max_empty_fraction: float | None = None,
    n_samples: int | None = None,
    seed: int = 0,
    use_gpu: bool = False,
) -> tuple[np.ndarray, dict]:
    """Pool neighbor-tile pair measurements across many mosaic grids and fit one affine.

    For each ``(slice_id, path)`` entry, load only the central Z plane of the
    OME-Zarr volume and call :func:`compute_registration_refinements` with the
    supplied options.  All resulting pairs are concatenated, optionally
    sub-sampled with a deterministic seed, and fed to
    :func:`estimate_affine_from_pairs` for a single 2×2 affine fit.

    Parameters
    ----------
    volumes : list of (slice_id, path)
        Each ``path`` must be a string or :class:`pathlib.Path` pointing at a
        ``*.ome.zarr`` mosaic grid.
    overlap_fraction : float
        Expected tile overlap fraction (must match acquisition).
    histogram_match : bool, keyword-only
        Forwarded to :func:`compute_registration_refinements`.
    max_empty_fraction : float or None, keyword-only
        Forwarded to :func:`compute_registration_refinements`.
    n_samples : int or None, keyword-only
        If set and the pooled pair count exceeds this value, a reproducible
        random sub-sample of size ``n_samples`` is drawn before fitting.
    seed : int, keyword-only
        Seed used when sub-sampling.  Ignored when ``n_samples`` is None.
    use_gpu : bool, keyword-only
        Forwarded to :func:`compute_registration_refinements`.

    Returns
    -------
    transform : np.ndarray
        Fitted 2×2 affine matrix.
    diagnostics : dict
        Full diagnostics including per-slice stats, pooled pair count,
        chosen backend label, and the output of
        :func:`estimate_affine_from_pairs`.
    """
    import random as _random

    from linumpy.io.zarr import read_omezarr

    tile_shape_ref: tuple | None = None
    all_pairs: list[dict] = []
    per_slice_stats: list[dict] = []

    for slice_id, zarr_path in volumes:
        vol, _ = read_omezarr(str(zarr_path), level=0)
        tile_shape = tuple(vol.chunks)
        if len(tile_shape) != 3:
            logger.warning("slice %s: unexpected chunks %s, skipping", slice_id, tile_shape)
            continue
        if tile_shape_ref is None:
            tile_shape_ref = tile_shape
        elif tile_shape[1:] != tile_shape_ref[1:]:
            logger.warning(
                "slice %s: tile shape %s differs from reference %s — pooling across different "
                "tile sizes is not supported. Skipping.",
                slice_id,
                tile_shape,
                tile_shape_ref,
            )
            continue

        nx = vol.shape[1] // tile_shape[1]
        ny = vol.shape[2] // tile_shape[2]
        if nx == 0 or ny == 0:
            logger.warning("slice %s: too few tiles (nx=%d ny=%d), skipping", slice_id, nx, ny)
            continue

        z_mid_full = vol.shape[0] // 2
        logger.info(
            "slice %s: shape=%s tile=%s grid=%dx%d z_mid=%d (hist_match=%s empty_frac=%s use_gpu=%s)",
            slice_id,
            tuple(vol.shape),
            tile_shape,
            nx,
            ny,
            z_mid_full,
            histogram_match,
            max_empty_fraction,
            use_gpu,
        )
        z_plane = np.asarray(vol[z_mid_full : z_mid_full + 1])

        refinements = compute_registration_refinements(
            z_plane,
            tile_shape,
            nx,
            ny,
            overlap_fraction,
            histogram_match=histogram_match,
            max_empty_fraction=max_empty_fraction,
            use_gpu=use_gpu,
        )
        pairs = refinements["pairs"]
        stats = dict(refinements["stats"])
        stats["slice_id"] = slice_id
        stats["nx"] = int(nx)
        stats["ny"] = int(ny)
        per_slice_stats.append(stats)
        logger.info(
            "slice %s: %d valid pairs collected (total=%d)",
            slice_id,
            stats["valid_pairs"],
            stats["total_pairs"],
        )
        all_pairs.extend(pairs)

    if tile_shape_ref is None:
        raise ValueError("No usable mosaic grids produced pair measurements.")

    total_pooled = len(all_pairs)
    logger.info("pooled pair count: %d", total_pooled)

    sampled = False
    if n_samples is not None and total_pooled > n_samples:
        rng = _random.Random(seed)
        all_pairs = rng.sample(all_pairs, n_samples)
        sampled = True
        logger.info("random-sampled to %d pairs (seed=%d)", len(all_pairs), seed)

    transform, fit_diag = estimate_affine_from_pairs(all_pairs, tile_shape_ref, overlap_fraction)
    diagnostics: dict[str, Any] = {
        "n_volumes": len(per_slice_stats),
        "n_pairs_pooled_total": total_pooled,
        "n_pairs_used": len(all_pairs),
        "tile_shape": list(tile_shape_ref),
        "overlap_fraction": overlap_fraction,
        "histogram_match": bool(histogram_match),
        "max_empty_fraction": max_empty_fraction,
        "sampled_n": n_samples,
        "seed": seed if sampled else None,
        "backend": "gpu" if use_gpu else "cpu",
        "transform": transform.tolist(),
        "displacement_model": _extract_displacement_params(transform, tile_shape_ref, overlap_fraction),
        "lstsq_residual": fit_diag.get("lstsq_residual"),
        "fallback": fit_diag.get("fallback", False),
        "per_slice_stats": per_slice_stats,
    }
    return transform, diagnostics


def _extract_displacement_params(transform: np.ndarray, tile_shape: tuple, overlap_fraction: float) -> dict:
    """Extract Lefebvre motor model parameters from a 2x2 affine transform.

    Given the fitted transform ``A`` where ``(dy, dx) = A @ (row_delta, col_delta)``,
    recover the scan-to-stage rotation θ, the motor-axis angle φ, and the
    effective per-direction overlap fractions Ox, Oy.

    Derivation (Lefebvre et al. 2017, Eqs. 1–6).  In image coordinates
    (y-down, x-right) the horizontal motor step (``col_delta = 1``) has
    image displacement

        (dy, dx) = (b, d) = nx·(1 - Ox)·(-sin θ, cos θ)

    so that ``θ = arctan2(-b, d)`` and ``Ox = 1 - sqrt(b**2 + d**2) / nx``
    with ``nx = tile_w``.  The vertical motor step (``row_delta = 1``) has

        (dy, dx) = (a, c) = ny·(1 - Oy)·(sin(φ - θ), cos(φ - θ))

    so that ``φ - θ = arctan2(a, c)`` and ``Oy = 1 - sqrt(a**2 + c**2) / ny`` with
    ``ny = tile_h``.  Perfectly perpendicular motors correspond to
    ``φ = 90°`` (not zero).

    Parameters
    ----------
    transform : np.ndarray
        2×2 affine matrix fitted by :func:`estimate_affine_from_pairs`.
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

    # θ: scan-to-stage rotation, from the horizontal motor step (b, d) (Eq. 3).
    # tan(θ) = -b / d
    theta_rad = np.arctan2(-b, d) if abs(d) > 1e-6 else 0.0

    # φ - θ: from the vertical motor step (a, c) (Eq. 4).
    # tan(φ - θ) = a / c  (image-frame y-down convention folds the paper's
    # negative-sine into the atan2 arguments).
    phi_minus_theta = np.arctan2(a, c) if abs(c) > 1e-6 else np.pi / 2.0
    phi_rad = phi_minus_theta + theta_rad

    # Ox: overlap along the horizontal motor axis (Eq. 5).
    horizontal_step = np.sqrt(b**2 + d**2)
    ox_fraction = 1.0 - horizontal_step / tile_w

    # Oy: overlap along the vertical motor axis (Eq. 6).
    vertical_step = np.sqrt(a**2 + c**2)
    oy_fraction = 1.0 - vertical_step / tile_h

    return {
        "theta_deg": float(np.degrees(theta_rad)),
        "phi_deg": float(np.degrees(phi_rad)),
        "Ox_fraction": float(ox_fraction),
        "Oy_fraction": float(oy_fraction),
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


def apply_blend_shift_refinement(tile: np.ndarray, refinements_for_tile: list) -> np.ndarray:
    """Apply registration refinement by shifting tile data in overlap regions.

    Applies a small sub-pixel shift (averaged from all neighbors) to improve
    blending quality without changing the tile's position in the mosaic.

    Parameters
    ----------
    tile : np.ndarray
        3D tile data (Z, Y, X).
    refinements_for_tile : list
        List of dicts with 'dx', 'dy' refinements from neighbors.

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


def compare_motor_vs_registration(
    motor_positions: list | tuple, reg_positions: list | tuple, output_path: str | None = None
) -> dict:
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
