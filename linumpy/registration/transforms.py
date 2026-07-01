"""Transform construction and mosaic-level transform estimation."""

import random
from collections.abc import Sequence
from typing import Any

import numpy as np
import SimpleITK as sitk
from skimage.exposure import match_histograms
from skimage.filters import threshold_otsu

from linumpy.registration.phase_correlation import pair_wise_phase_correlation


def create_transform(tx: float, ty: float, angle_deg: float, center: Sequence[float]) -> sitk.Euler3DTransform:
    """Create a 3D SimpleITK Euler transform from 2D parameters.

    Parameters
    ----------
    tx, ty : float
        Translation in pixels.
    angle_deg : float
        Rotation angle in degrees (around Z axis).
    center : sequence
        (cx, cy) rotation center.

    Returns
    -------
    sitk.Euler3DTransform
    """
    transform = sitk.Euler3DTransform()
    transform.SetCenter([center[0], center[1], 0.0])
    transform.SetRotation(0.0, 0.0, np.radians(angle_deg))
    transform.SetTranslation([tx, ty, 0.0])
    return transform


def compute_motor_transform(tile_shape: Sequence[int], overlap_fraction: float) -> np.ndarray:
    """Compute the transform matrix for motor-based tile positions.

    Creates a diagonal transform where tile index (i, j) maps to a pixel
    position based on the expected overlap, corresponding to precise
    motor/stage positions from acquisition.

    Parameters
    ----------
    tile_shape : tuple or list
        Tile shape as (height, width) in pixels.
    overlap_fraction : float
        Expected overlap between tiles (0-1).

    Returns
    -------
    np.ndarray
        2x2 transform matrix where ``transform @ [i, j]`` gives the pixel
        position of tile ``(i, j)``.
    """
    step_y = tile_shape[0] * (1.0 - overlap_fraction)
    step_x = tile_shape[1] * (1.0 - overlap_fraction)
    return np.array([[step_y, 0.0], [0.0, step_x]])


def estimate_mosaic_transform(
    mosaics: list[Any], max_empty_fraction: float = 0.9, n_samples: int = 512, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, int]:
    """Estimate the 2x2 mosaic transform from pairwise phase-correlation registration.

    For each mosaic, neighbouring tile pairs are registered with
    :func:`pair_wise_phase_correlation` and the resulting pixel displacements are
    assembled into a least-squares system to recover the underlying affine
    transform.

    Parameters
    ----------
    mosaics : list of MosaicGrid
        Loaded mosaic grids to use for estimation.
    max_empty_fraction : float, optional
        Maximum fraction of empty pixels in an overlap region to still use the
        pair (default 0.9).
    n_samples : int, optional
        Maximum number of tile pairs to sample across all mosaics (default 512).
    seed : int, optional
        Random seed for reproducible tile-pair sampling.

    Returns
    -------
    transform : np.ndarray
        2x2 transform matrix.
    residuals : np.ndarray
        Residuals from the least-squares fit.
    tile_count : int
        Number of tile pairs actually used.
    """
    rows, rows_px, cols, cols_px = [], [], [], []
    tile_count = 0

    if seed is not None:
        random.seed(seed)
    mosaic_idx = list(range(len(mosaics)))
    random.shuffle(mosaic_idx)

    thresholds = [threshold_otsu(m.image) for m in mosaics]

    for m_id in mosaic_idx:
        mosaic = mosaics[m_id]
        thresh = thresholds[m_id]

        for i in range(mosaic.n_tiles_x):
            for j in range(mosaic.n_tiles_y):
                if tile_count > n_samples:
                    break

                neighbors, tiles = mosaic.get_neighbors_around_tile(i, j)
                for _n, t in zip(neighbors, tiles, strict=False):
                    r = t[0] - i
                    c = t[1] - j

                    o1, o2, p1, _p2 = mosaic.get_neighbor_overlap_from_pos((i, j), t)

                    o1_empty = np.sum(o1 <= thresh) > max_empty_fraction * o1.size
                    o2_empty = np.sum(o2 <= thresh) > max_empty_fraction * o2.size
                    if o1_empty or o2_empty:
                        continue

                    o2 = match_histograms(o2, o1)
                    dx, dy = pair_wise_phase_correlation(o1, o2)

                    r_px = p1[2] - mosaic.tile_size_x + dx if r == -1 else p1[0] + dx
                    c_px = p1[3] - mosaic.tile_size_y + dy if c == -1 else p1[1] + dy

                    rows.append(r)
                    cols.append(c)
                    rows_px.append(r_px)
                    cols_px.append(c_px)
                    tile_count += 1

    # Build and solve the least-squares system
    a = np.zeros((len(rows) * 2, 4))
    b = np.zeros((len(rows) * 2, 1))
    for i in range(len(rows)):
        a[2 * i, :] = [rows[i], cols[i], 0, 0]
        b[2 * i, 0] = rows_px[i]
        a[2 * i + 1, :] = [0, 0, rows[i], cols[i]]
        b[2 * i + 1, 0] = cols_px[i]

    result = np.linalg.lstsq(a, b, rcond=None)
    transform = result[0].reshape((2, 2))
    residuals = result[1] if len(result[1]) > 0 else np.array([0.0])

    return transform, residuals, tile_count
